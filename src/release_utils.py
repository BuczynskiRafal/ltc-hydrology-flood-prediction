from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.logger import get_console_logger

logger = get_console_logger(__name__)


CONFIG_SCHEMA_VERSION = 2

COMMON_TOP_LEVEL_KEYS = {
    "schema_version",
    "runtime",
    "model",
    "training",
    "loss",
    "data",
    "evaluation",
    "output",
}

COMMON_RUNTIME_KEYS = {"seed", "device", "deterministic"}
COMMON_TRAINING_KEYS = {
    "batch_size",
    "epochs",
    "gradient_clip",
    "num_workers",
    "optimizer",
    "scheduler",
    "early_stopping",
}
COMMON_OPTIMIZER_KEYS = {"type", "learning_rate"}
COMMON_SCHEDULER_KEYS = {
    "type",
    "patience",
    "factor",
    "min_lr",
    "eta_min",
    "T_0",
    "T_mult",
    "warmup_epochs",
    "warmup_start_factor",
}
COMMON_EARLY_STOPPING_KEYS = {"patience", "min_delta"}
COMMON_DATA_KEYS = {"use_reduced", "data_dir"}
COMMON_EVALUATION_KEYS = {"threshold_artifact"}
COMMON_OUTPUT_KEYS = {"checkpoint_dir"}
LNN_OPTIONAL_MODEL_KEYS = {
    "encoder_input_size",
    "use_fast_path",
    "use_slow_path",
    "use_attention",
    "use_learnable_slow_tau",
    "slow_tau_init",
    "use_path_layer_norm",
    "per_neuron_tau",
    "fast_tau_min",
    "fast_tau_max",
    "use_separate_depth_heads",
    "depth_head_hidden_size",
    "pump_head_target_index",
    "pump_head_feature_indices",
    "use_pump_branch",
    "pump_branch_input_indices",
    "pump_branch_fast_units",
    "pump_branch_slow_units",
    "pump_branch_hidden_size",
    "pump_branch_use_attention",
}

MODEL_REQUIRED_KEYS = {
    "gru": {"input_size", "hidden_size", "num_depth_outputs", "num_layers", "dropout"},
    "lstm": {"input_size", "hidden_size", "num_depth_outputs", "num_layers", "dropout"},
    "tcn": {
        "input_size",
        "hidden_size",
        "num_depth_outputs",
        "kernel_size",
        "num_layers",
        "dropout",
    },
    "mlp": {
        "input_size",
        "seq_len",
        "hidden_dims",
        "num_depth_outputs",
        "dropout",
        "use_batch_norm",
    },
    "lnn": {
        "input_size",
        "fast_units",
        "slow_units",
        "hidden_size",
        "num_depth_outputs",
        "dropout",
        "tau_mode",
    },
}

LOSS_REQUIRED_KEYS = {
    "gru": {"depth_weight", "overflow_weight", "flood_weight"},
    "lstm": {"depth_weight", "overflow_weight", "flood_weight"},
    "tcn": {"depth_weight", "overflow_weight", "flood_weight"},
    "mlp": {"depth_weight", "overflow_weight", "flood_weight"},
    "lnn": {"depth_weight", "overflow_weight", "intensity_weight"},
}

LOSS_OPTIONAL_KEYS = {
    "gru": {"pos_weight"},
    "lstm": {"pos_weight"},
    "tcn": {"pos_weight"},
    "mlp": {"pos_weight"},
    "lnn": {"flood_weight", "pos_weight"},
}


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _validate_section_keys(
    section_name: str,
    section_value: Any,
    required_keys: set[str],
    *,
    optional_keys: set[str] | None = None,
    allow_extra: bool,
) -> None:
    if not isinstance(section_value, dict):
        raise ValueError(f"Config section '{section_name}' must be a dictionary.")

    allowed_keys = required_keys | (optional_keys or set())
    missing = sorted(required_keys - set(section_value))
    extra = sorted(set(section_value) - allowed_keys)
    if missing or (extra and not allow_extra):
        problems = []
        if missing:
            problems.append(f"missing={missing}")
        if extra and not allow_extra:
            problems.append(f"extra={extra}")
        raise ValueError(
            f"Config section '{section_name}' does not match the canonical schema: "
            f"{', '.join(problems)}"
        )


def validate_model_config(
    model_name: str,
    config: dict[str, Any],
    *,
    source_label: str,
    allow_extra: bool = False,
) -> None:
    if not isinstance(config, dict):
        raise ValueError(f"{source_label} config must be a dictionary.")

    missing_top = sorted(COMMON_TOP_LEVEL_KEYS - set(config))
    extra_top = sorted(set(config) - COMMON_TOP_LEVEL_KEYS)
    if missing_top or (extra_top and not allow_extra):
        problems = []
        if missing_top:
            problems.append(f"missing={missing_top}")
        if extra_top and not allow_extra:
            problems.append(f"extra={extra_top}")
        raise ValueError(
            f"{source_label} top-level config does not match the canonical schema: "
            f"{', '.join(problems)}"
        )

    if int(config["schema_version"]) != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{source_label} schema_version must be {CONFIG_SCHEMA_VERSION}, "
            f"got {config['schema_version']}."
        )

    _validate_section_keys(
        "runtime", config["runtime"], COMMON_RUNTIME_KEYS, allow_extra=allow_extra
    )
    _validate_section_keys(
        "model",
        config["model"],
        MODEL_REQUIRED_KEYS[model_name],
        optional_keys=LNN_OPTIONAL_MODEL_KEYS if model_name == "lnn" else set(),
        allow_extra=allow_extra,
    )
    _validate_section_keys(
        "training",
        config["training"],
        COMMON_TRAINING_KEYS,
        allow_extra=allow_extra,
    )
    _validate_section_keys(
        "training.optimizer",
        config["training"]["optimizer"],
        COMMON_OPTIMIZER_KEYS
        | ({"betas", "eps", "weight_decay"} if model_name == "lnn" else set()),
        allow_extra=allow_extra,
    )
    _validate_section_keys(
        "training.scheduler",
        config["training"]["scheduler"],
        {"type"}
        | (
            {"eta_min"}
            if str(config["training"]["scheduler"]["type"]).lower()
            in {
                "cosine_annealing",
                "cosineannealinglr",
                "cosine_warm_restarts",
                "cosineannealingwarmrestarts",
            }
            else {"patience", "factor", "min_lr"}
        ),
        optional_keys=(
            {
                "patience",
                "factor",
                "min_lr",
                "T_0",
                "T_mult",
                "warmup_epochs",
                "warmup_start_factor",
            }
            if str(config["training"]["scheduler"]["type"]).lower()
            in {
                "cosine_annealing",
                "cosineannealinglr",
                "cosine_warm_restarts",
                "cosineannealingwarmrestarts",
            }
            else {"eta_min", "T_0", "T_mult", "warmup_epochs", "warmup_start_factor"}
        ),
        allow_extra=allow_extra,
    )
    _validate_section_keys(
        "training.early_stopping",
        config["training"]["early_stopping"],
        COMMON_EARLY_STOPPING_KEYS,
        allow_extra=allow_extra,
    )
    _validate_section_keys(
        "loss",
        config["loss"],
        LOSS_REQUIRED_KEYS[model_name],
        optional_keys=LOSS_OPTIONAL_KEYS[model_name],
        allow_extra=allow_extra,
    )
    _validate_section_keys(
        "data", config["data"], COMMON_DATA_KEYS, allow_extra=allow_extra
    )
    _validate_section_keys(
        "evaluation",
        config["evaluation"],
        COMMON_EVALUATION_KEYS,
        allow_extra=allow_extra,
    )
    _validate_section_keys(
        "output", config["output"], COMMON_OUTPUT_KEYS, allow_extra=allow_extra
    )


def _filter_nested_dict(
    data: dict[str, Any], allowed_schema: dict[str, Any], prefix: str = ""
) -> tuple[dict[str, Any], list[str]]:
    filtered: dict[str, Any] = {}
    ignored_paths: list[str] = []
    for key, value in data.items():
        if key not in allowed_schema:
            ignored_paths.append(f"{prefix}{key}")
            continue

        allowed_value = allowed_schema[key]
        if isinstance(allowed_value, dict) and isinstance(value, dict):
            child_filtered, child_ignored = _filter_nested_dict(
                value, allowed_value, prefix=f"{prefix}{key}."
            )
            filtered[key] = child_filtered
            ignored_paths.extend(child_ignored)
        else:
            filtered[key] = deepcopy(value)
    return filtered, ignored_paths


def extract_supported_config(
    model_name: str, raw_config: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    allowed_schema = {
        "schema_version": None,
        "runtime": {key: None for key in COMMON_RUNTIME_KEYS},
        "model": {
            key: None
            for key in (
                MODEL_REQUIRED_KEYS[model_name]
                | (LNN_OPTIONAL_MODEL_KEYS if model_name == "lnn" else set())
            )
        },
        "training": {
            "batch_size": None,
            "epochs": None,
            "gradient_clip": None,
            "num_workers": None,
            "optimizer": {
                key: None
                for key in (
                    COMMON_OPTIMIZER_KEYS
                    | (
                        {"betas", "eps", "weight_decay"}
                        if model_name == "lnn"
                        else set()
                    )
                )
            },
            "scheduler": {key: None for key in COMMON_SCHEDULER_KEYS},
            "early_stopping": {key: None for key in COMMON_EARLY_STOPPING_KEYS},
        },
        "loss": {
            key: None
            for key in (LOSS_REQUIRED_KEYS[model_name] | LOSS_OPTIONAL_KEYS[model_name])
        },
        "data": {key: None for key in COMMON_DATA_KEYS},
        "evaluation": {key: None for key in COMMON_EVALUATION_KEYS},
        "output": {key: None for key in COMMON_OUTPUT_KEYS},
    }
    return _filter_nested_dict(raw_config, allowed_schema)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.lower()
    if normalized == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(normalized)


def get_git_sha() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        PermissionError,
        OSError,
    ) as exc:
        warnings.warn(
            f"Could not resolve git SHA for runtime metadata: {exc}",
            stacklevel=2,
        )
        return None
    return output.strip()


def collect_library_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def build_dataset_fingerprint(split_descriptions: list[dict[str, Any]]) -> str:
    payload = json.dumps(split_descriptions, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
