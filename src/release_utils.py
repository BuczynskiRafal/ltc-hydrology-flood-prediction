from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import warnings
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
}
COMMON_EARLY_STOPPING_KEYS = {"patience", "min_delta"}
COMMON_DATA_KEYS = {"use_reduced"}
COMMON_EVALUATION_KEYS = {"threshold_artifact"}
COMMON_OUTPUT_KEYS = {"checkpoint_dir"}

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
        optional_keys=(
            {"use_fast_path", "use_slow_path", "use_attention"}
            if model_name == "lnn"
            else set()
        ),
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
            {"patience", "factor", "min_lr", "T_0", "T_mult"}
            if str(config["training"]["scheduler"]["type"]).lower()
            in {
                "cosine_annealing",
                "cosineannealinglr",
                "cosine_warm_restarts",
                "cosineannealingwarmrestarts",
            }
            else {"eta_min", "T_0", "T_mult"}
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
        optional_keys={"pos_weight"} if model_name != "lnn" else set(),
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
