from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from experiments.eval_utils import (
    evaluate_depths,
    evaluate_overflow,
    find_optimal_threshold,
    resolve_checkpoint_path,
)
from src.data.load_regression_data import describe_regression_data, load_regression_data
from src.data.regression_dataloader import RegressionDataset
from src.evaluation.hydrological_metrics import compute_hydrological_metrics
from src.logger import get_console_logger
from src.models.gru_regression import GRURegression
from src.models.lnn_regression import LNNRegression
from src.models.lstm_regression import LSTMRegression
from src.models.mlp_regression import MLPRegression
from src.models.tcn_regression import TCNRegression
from src.release_utils import (
    build_dataset_fingerprint,
    collect_library_versions,
    get_git_sha,
    resolve_device,
    set_global_seed,
    validate_model_config,
    write_json,
)
from src.training.regression_losses import MultiTaskRegressionLoss

logger = get_console_logger(__name__)


CANONICAL_CONFIG_PATHS = {
    "gru": Path("configs/gru_regression_config.yaml"),
    "lstm": Path("configs/lstm_regression_config.yaml"),
    "tcn": Path("configs/tcn_regression_config.yaml"),
    "mlp": Path("configs/mlp_regression_config.yaml"),
    "lnn": Path("configs/lnn_regression_config.yaml"),
}

MODEL_TITLES = {
    "gru": "GRU Regression",
    "lstm": "LSTM Regression",
    "tcn": "TCN Regression",
    "mlp": "MLP Regression",
    "lnn": "LNN Regression",
}

ENSEMBLE_DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r") as handle:
        content = yaml.safe_load(handle)

    if content is None or not isinstance(content, dict):
        raise ValueError(f"Config file is empty or invalid: {config_path}")
    return content


def load_model_config(
    model_name: str, config_path: str | Path | None = None
) -> dict[str, Any]:
    path = (
        Path(config_path)
        if config_path is not None
        else CANONICAL_CONFIG_PATHS[model_name]
    )
    config = load_yaml_config(path)
    validate_model_config(model_name, config, source_label=str(path))
    return config


def resolve_runtime_config(
    model_name: str,
    canonical_config: dict[str, Any],
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        logger.info(
            f"[{model_name}] runtime config source: canonical "
            "(checkpoint is missing embedded config)"
        )
        return deepcopy(canonical_config), "canonical"

    validate_model_config(
        model_name,
        checkpoint_config,
        source_label=f"{model_name}:checkpoint_runtime",
    )
    logger.info(f"[{model_name}] runtime config source: checkpoint")
    return deepcopy(checkpoint_config), "checkpoint"


def get_runtime_seed(config: dict[str, Any]) -> int:
    return int(config["runtime"]["seed"])


def get_runtime_device(
    config: dict[str, Any], device: torch.device | None = None
) -> torch.device:
    if device is not None:
        return device
    return resolve_device(config["runtime"]["device"])


def get_device() -> torch.device:
    return resolve_device("auto")


def build_run_metadata(
    *,
    config: dict[str, Any],
    model_name: str,
    device: torch.device,
    split_descriptions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "seed": get_runtime_seed(config),
        "device": str(device),
        "schema_version": int(config["schema_version"]),
        "git_sha": get_git_sha(),
        "library_versions": collect_library_versions(),
        "dataset_fingerprint": build_dataset_fingerprint(split_descriptions),
        "data_splits": split_descriptions,
    }


def build_ensemble_run_metadata(
    *,
    config: dict[str, Any],
    model_name: str,
    device: torch.device,
    split_descriptions: list[dict[str, Any]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    metadata = build_run_metadata(
        config=config,
        model_name=model_name,
        device=device,
        split_descriptions=split_descriptions,
    )
    metadata.pop("seed", None)
    metadata["seeds"] = [int(seed) for seed in seeds]
    return metadata


def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def create_model(model_name: str, config: dict[str, Any]) -> torch.nn.Module:
    model_config = config["model"]

    if model_name == "gru":
        return GRURegression(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_depth_outputs=model_config["num_depth_outputs"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
        )
    if model_name == "lstm":
        return LSTMRegression(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_depth_outputs=model_config["num_depth_outputs"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
        )
    if model_name == "tcn":
        return TCNRegression(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_depth_outputs=model_config["num_depth_outputs"],
            kernel_size=model_config["kernel_size"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
        )
    if model_name == "mlp":
        return MLPRegression(
            input_size=model_config["input_size"],
            seq_len=model_config["seq_len"],
            hidden_dims=model_config["hidden_dims"],
            num_depth_outputs=model_config["num_depth_outputs"],
            dropout=model_config["dropout"],
            use_batch_norm=model_config.get("use_batch_norm", True),
        )
    if model_name == "lnn":
        return LNNRegression(
            input_size=model_config["input_size"],
            fast_units=model_config["fast_units"],
            slow_units=model_config["slow_units"],
            hidden_size=model_config["hidden_size"],
            num_depth_outputs=model_config["num_depth_outputs"],
            dropout=model_config["dropout"],
            tau_mode=model_config.get("tau_mode", "stepwise"),
            use_fast_path=model_config.get("use_fast_path", True),
            use_slow_path=model_config.get("use_slow_path", True),
            use_attention=model_config.get("use_attention", True),
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def create_regression_loss(config: dict[str, Any]) -> MultiTaskRegressionLoss:
    loss_config = config["loss"]
    pos_weight = loss_config.get("pos_weight")
    return MultiTaskRegressionLoss(
        depth_weight=float(loss_config["depth_weight"]),
        overflow_weight=float(loss_config["overflow_weight"]),
        flood_weight=float(loss_config["flood_weight"]),
        pos_weight=None if pos_weight is None else float(pos_weight),
    )


def get_lnn_loss_weights(config: dict[str, Any]) -> dict[str, float]:
    return {
        "depth_weight": float(config["loss"]["depth_weight"]),
        "overflow_weight": float(config["loss"]["overflow_weight"]),
        "intensity_weight": float(config["loss"]["intensity_weight"]),
    }


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> optim.Optimizer:
    training_config = config["training"]
    optimizer_config = training_config.get("optimizer", {})
    optimizer_type = str(optimizer_config.get("type", "adam")).lower()
    learning_rate = optimizer_config.get(
        "learning_rate", training_config.get("learning_rate")
    )
    if learning_rate is None:
        raise ValueError("Optimizer learning rate is not configured.")

    kwargs: dict[str, Any] = {"lr": float(learning_rate)}
    if "betas" in optimizer_config:
        kwargs["betas"] = tuple(float(value) for value in optimizer_config["betas"])
    if "eps" in optimizer_config:
        kwargs["eps"] = float(optimizer_config["eps"])
    if "weight_decay" in optimizer_config:
        kwargs["weight_decay"] = float(optimizer_config["weight_decay"])

    if optimizer_type == "adam":
        return optim.Adam(model.parameters(), **kwargs)
    if optimizer_type == "adamw":
        return optim.AdamW(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def build_scheduler(
    optimizer: optim.Optimizer, config: dict[str, Any]
) -> optim.lr_scheduler.LRScheduler | optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_config = config["training"].get("scheduler")
    if not scheduler_config:
        return None

    scheduler_type = str(scheduler_config["type"]).lower()
    if scheduler_type in {
        "reduce_on_plateau",
        "reduce_lr_on_plateau",
        "reducelronplateau",
    }:
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_config["factor"]),
            patience=int(scheduler_config["patience"]),
            min_lr=float(scheduler_config["min_lr"]),
        )
    if scheduler_type in {"cosine_annealing", "cosineannealinglr"}:
        eta_min = float(
            scheduler_config.get("eta_min", scheduler_config.get("min_lr", 0.0))
        )
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["training"]["epochs"]),
            eta_min=eta_min,
        )
    if scheduler_type in {"cosine_warm_restarts", "cosineannealingwarmrestarts"}:
        eta_min = float(
            scheduler_config.get("eta_min", scheduler_config.get("min_lr", 0.0))
        )
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(scheduler_config.get("T_0", 10)),
            T_mult=int(scheduler_config.get("T_mult", 2)),
            eta_min=eta_min,
        )
    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def scheduler_requires_metric(config: dict[str, Any]) -> bool:
    scheduler_config = config["training"].get("scheduler")
    if not scheduler_config:
        return False

    scheduler_type = str(scheduler_config["type"]).lower()
    return scheduler_type in {
        "reduce_on_plateau",
        "reduce_lr_on_plateau",
        "reducelronplateau",
    }


def get_early_stopping_settings(config: dict[str, Any]) -> dict[str, float] | None:
    training_config = config["training"]
    early_stopping = training_config.get("early_stopping")
    if early_stopping is None:
        return None

    return {
        "patience": int(early_stopping["patience"]),
        "min_delta": float(early_stopping.get("min_delta", 0.0)),
    }


def load_split_data_for_config(config: dict[str, Any], split: str) -> dict[str, Any]:
    use_reduced = config["data"].get("use_reduced", True)
    data = load_regression_data(split, use_reduced=use_reduced)
    validate_split_runtime_contract(split, data, config)
    return data


def describe_split_data_for_config(
    config: dict[str, Any], split: str
) -> dict[str, Any]:
    use_reduced = config["data"].get("use_reduced", True)
    return describe_regression_data(split, use_reduced=use_reduced)


def validate_split_runtime_contract(
    split: str, data: dict[str, Any], config: dict[str, Any]
) -> None:
    X = data["X"]
    y_depths = data["y_depths"]
    y_overflow = data["y_overflow"]
    flood_mask = data["flood_mask"]

    expected_input_size = int(config["model"]["input_size"])
    if X.shape[2] != expected_input_size:
        raise ValueError(
            f"{split}: configured input_size={expected_input_size} does not match "
            f"loaded data channels={X.shape[2]}"
        )

    expected_seq_len = int(config["model"].get("seq_len", X.shape[1]))
    if X.shape[1] != expected_seq_len:
        raise ValueError(
            f"{split}: configured seq_len={expected_seq_len} does not match "
            f"loaded data length={X.shape[1]}"
        )

    expected_depths = int(config["model"]["num_depth_outputs"])
    if y_depths.shape[1] != expected_depths:
        raise ValueError(
            f"{split}: configured num_depth_outputs={expected_depths} does not match "
            f"loaded depth targets={y_depths.shape[1]}"
        )

    n_samples = X.shape[0]
    if y_overflow.shape[0] != n_samples or flood_mask.shape[0] != n_samples:
        raise ValueError(
            f"{split}: inconsistent sample count after load: X={n_samples}, "
            f"y_overflow={y_overflow.shape[0]}, flood_mask={flood_mask.shape[0]}"
        )


def create_dataloader(
    data: dict[str, Any],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    effective_num_workers = int(num_workers)
    if effective_num_workers > 0 and platform.system() == "Darwin":
        logger.info(
            "Falling back to num_workers=0 on macOS for publication-safe "
            "DataLoader compatibility."
        )
        effective_num_workers = 0
    return DataLoader(
        RegressionDataset(**data),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=effective_num_workers,
    )


def create_train_and_validation_loaders(
    config: dict[str, Any],
    train_data: dict[str, Any],
    val_data: dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    return (
        create_dataloader(train_data, batch_size, num_workers, shuffle=True),
        create_dataloader(val_data, batch_size, num_workers, shuffle=False),
    )


def create_test_loader(config: dict[str, Any], test_data: dict[str, Any]) -> DataLoader:
    return create_dataloader(
        test_data,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        shuffle=False,
    )


def _move_batch_to_device(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def _compute_lnn_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    pred_depths, pred_overflow, pred_intensity = outputs
    depth_loss = F.mse_loss(pred_depths, batch["y_depths"])
    overflow_loss = F.binary_cross_entropy(
        pred_overflow, batch["y_overflow"].unsqueeze(1)
    )
    intensity_loss = F.mse_loss(
        pred_intensity, batch["y_overflow"].unsqueeze(1).float()
    )
    total_loss = (
        loss_weights["depth_weight"] * depth_loss
        + loss_weights["overflow_weight"] * overflow_loss
        + loss_weights["intensity_weight"] * intensity_loss
    )
    return (
        total_loss,
        {
            "depth": depth_loss,
            "overflow": overflow_loss,
            "intensity": intensity_loss,
        },
        pred_overflow,
    )


def _compute_baseline_loss(
    outputs: tuple[torch.Tensor, torch.Tensor],
    batch: dict[str, torch.Tensor],
    criterion: MultiTaskRegressionLoss,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    pred_depths, pred_overflow = outputs
    total_loss, depth_loss, overflow_loss = criterion(
        pred_depths,
        pred_overflow,
        batch["y_depths"],
        batch["y_overflow"],
        batch["flood_mask"],
    )
    return (
        total_loss,
        {"depth": depth_loss, "overflow": overflow_loss},
        pred_overflow,
    )


def run_epoch(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    loss_spec: Any,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    gradient_clip: float | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    component_names = ["depth", "overflow"]
    if model_name == "lnn":
        component_names.append("intensity")

    sums = {"loss": 0.0, "accuracy": 0.0, **{name: 0.0 for name in component_names}}
    correct = 0
    total = 0

    if training:
        model.train()
    else:
        model.eval()

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            if training:
                optimizer.zero_grad()

            outputs = model(batch["X"])
            if model_name == "lnn":
                loss, components, pred_overflow = _compute_lnn_loss(
                    outputs, batch, loss_spec
                )
            else:
                loss, components, pred_overflow = _compute_baseline_loss(
                    outputs, batch, loss_spec
                )

            if training:
                loss.backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

            sums["loss"] += loss.item()
            for component_name in component_names:
                sums[component_name] += components[component_name].item()

            pred_binary = (pred_overflow.squeeze(1) > 0.5).float()
            correct += (pred_binary == batch["y_overflow"]).sum().item()
            total += batch["y_overflow"].size(0)

    batch_count = len(loader)
    metrics = {
        "loss": sums["loss"] / batch_count,
        "accuracy": 100.0 * correct / total,
    }
    for component_name in component_names:
        metrics[component_name] = sums[component_name] / batch_count
    return metrics


def format_component_metrics(model_name: str, metrics: dict[str, float]) -> str:
    ordered_names = ["depth", "overflow"]
    if model_name == "lnn":
        ordered_names.append("intensity")
    return "/".join(f"{metrics[name]:.4f}" for name in ordered_names)


def train_model(
    model_name: str,
    config: dict[str, Any],
    device: torch.device | None = None,
    train_data: dict[str, Any] | None = None,
    val_data: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    max_epochs: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    set_global_seed(
        get_runtime_seed(config),
        deterministic=bool(config["runtime"]["deterministic"]),
    )
    device = get_runtime_device(config, device)
    train_data = train_data or load_split_data_for_config(config, "train")
    val_data = val_data or load_split_data_for_config(config, "val")
    split_descriptions = [
        describe_split_data_for_config(config, "train"),
        describe_split_data_for_config(config, "val"),
    ]
    train_loader, val_loader = create_train_and_validation_loaders(
        config, train_data, val_data
    )

    model = create_model(model_name, config).to(device)
    loss_spec = (
        get_lnn_loss_weights(config)
        if model_name == "lnn"
        else create_regression_loss(config)
    )
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    gradient_clip = float(config["training"]["gradient_clip"])
    epochs = int(max_epochs if max_epochs is not None else config["training"]["epochs"])
    early_stopping = get_early_stopping_settings(config)

    best_val_loss = float("inf")
    best_epoch = 0
    best_val_accuracy = 0.0
    best_state_dict = deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_metrics = run_epoch(
            model_name,
            model,
            train_loader,
            loss_spec,
            device,
            optimizer=optimizer,
            gradient_clip=gradient_clip,
        )
        val_metrics = run_epoch(model_name, model, val_loader, loss_spec, device)

        if scheduler is not None:
            if scheduler_requires_metric(config):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train: {train_metrics['loss']:.4f} ({format_component_metrics(model_name, train_metrics)}) "
            f"{train_metrics['accuracy']:.1f}% | "
            f"Val: {val_metrics['loss']:.4f} ({format_component_metrics(model_name, val_metrics)}) "
            f"{val_metrics['accuracy']:.1f}%"
        )

        min_delta = 0.0 if early_stopping is None else early_stopping["min_delta"]
        improved = val_metrics["loss"] < (best_val_loss - min_delta)
        if improved:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            best_val_accuracy = val_metrics["accuracy"]
            best_state_dict = deepcopy(model.state_dict())
            epochs_without_improvement = 0

            if checkpoint_path is not None:
                checkpoint_path = Path(checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_payload = {
                    "epoch": best_epoch,
                    "model_state_dict": deepcopy(model.state_dict()),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_accuracy": best_val_accuracy,
                    "n_params": count_parameters(model),
                    "config": deepcopy(config),
                    "runtime_metadata": build_run_metadata(
                        config=config,
                        model_name=model_name,
                        device=device,
                        split_descriptions=split_descriptions,
                    ),
                }
                if scheduler is not None:
                    checkpoint_payload["scheduler_state_dict"] = scheduler.state_dict()
                torch.save(checkpoint_payload, checkpoint_path)
        elif early_stopping is not None:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping["patience"]:
                logger.info(
                    f"Early stopping triggered after {epoch+1} epochs "
                    f"(patience={early_stopping['patience']}, min_delta={early_stopping['min_delta']})."
                )
                break

    model.load_state_dict(best_state_dict)
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "epochs_ran": epoch + 1,
        "num_parameters": count_parameters(model),
        "runtime_metadata": build_run_metadata(
            config=config,
            model_name=model_name,
            device=device,
            split_descriptions=split_descriptions,
        ),
    }
    return model, summary


def train_configured_model(
    model_name: str, config_path: str | Path | None = None
) -> None:
    config = load_model_config(model_name, config_path)
    device = get_runtime_device(config)
    checkpoint_path = Path(config["output"]["checkpoint_dir"]) / "best_model.pt"

    logger.info(f"Using device: {device}")
    logger.info(
        f"Using seed={config['runtime']['seed']} "
        f"(deterministic={config['runtime']['deterministic']})"
    )
    logger.info(
        f"Loading training data (use_reduced={config['data'].get('use_reduced', True)})..."
    )
    train_model(model_name, config, device=device, checkpoint_path=checkpoint_path)
    logger.info(f"Best checkpoint saved to: {checkpoint_path}")


def train_baseline_model(
    model_name: str, config_path: str | Path | None = None
) -> None:
    train_configured_model(model_name, config_path=config_path)


def load_trained_model(
    model_name: str,
    device: torch.device | None = None,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    canonical_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    canonical_path = (
        Path(config_path)
        if config_path is not None
        else CANONICAL_CONFIG_PATHS[model_name]
    )
    if canonical_config is None:
        canonical_config = load_model_config(model_name, canonical_path)
    else:
        canonical_config = deepcopy(canonical_config)
        validate_model_config(
            model_name,
            canonical_config,
            source_label=str(config_path or canonical_path),
        )
    resolved_checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else resolve_checkpoint_path(canonical_config["output"]["checkpoint_dir"])
    )
    checkpoint = torch.load(
        resolved_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    runtime_config, config_source = resolve_runtime_config(
        model_name, canonical_config, checkpoint
    )
    set_global_seed(
        get_runtime_seed(runtime_config),
        deterministic=bool(runtime_config["runtime"]["deterministic"]),
    )
    device = get_runtime_device(runtime_config, device)

    model = create_model(model_name, runtime_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "device": device,
        "canonical_config": canonical_config,
        "runtime_config": runtime_config,
        "config_source": config_source,
        "checkpoint": checkpoint,
        "checkpoint_path": resolved_checkpoint_path,
        "canonical_config_path": canonical_path,
    }


def collect_predictions(
    model_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    all_pred_depths = []
    all_true_depths = []
    all_pred_overflow = []
    all_true_overflow = []
    all_pred_intensity = []

    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            y_depths = batch["y_depths"].cpu().numpy()
            y_overflow = batch["y_overflow"].cpu().numpy()

            outputs = model(X)
            if model_name == "lnn":
                pred_depths, pred_overflow, pred_intensity = outputs
                all_pred_intensity.append(pred_intensity.cpu().numpy().reshape(-1))
            else:
                pred_depths, pred_overflow = outputs

            all_pred_depths.append(pred_depths.cpu().numpy())
            all_true_depths.append(y_depths)
            all_pred_overflow.append(pred_overflow.cpu().numpy().reshape(-1))
            all_true_overflow.append(y_overflow)

    predictions = {
        "pred_depths": np.concatenate(all_pred_depths, axis=0),
        "true_depths": np.concatenate(all_true_depths, axis=0),
        "pred_overflow": np.concatenate(all_pred_overflow, axis=0),
        "true_overflow": np.concatenate(all_true_overflow, axis=0),
    }
    if model_name == "lnn":
        predictions["pred_intensity"] = np.concatenate(all_pred_intensity, axis=0)
    return predictions


def build_metrics_payload_from_predictions(
    model_name: str,
    predictions: dict[str, np.ndarray],
    *,
    split: str,
    overflow_threshold: float,
    threshold_source: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    resolved_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    depth_metrics = evaluate_depths(
        predictions["true_depths"], predictions["pred_depths"]
    )
    hydro_metrics = compute_hydrological_metrics(
        predictions["true_depths"], predictions["pred_depths"]
    )
    overflow_metrics = evaluate_overflow(
        predictions["true_overflow"],
        predictions["pred_overflow"],
        threshold=overflow_threshold,
    )

    metrics_payload: dict[str, Any] = {
        "timestamp": resolved_timestamp,
        "evaluation_split": split,
        "depth_metrics": {
            "aggregated": {
                key: float(value) for key, value in depth_metrics["aggregated"].items()
            },
            "per_sensor": {
                key: value.tolist()
                for key, value in depth_metrics["per_sensor"].items()
            },
        },
        "hydrological_metrics": {
            "aggregated": {
                key: float(value) for key, value in hydro_metrics["aggregated"].items()
            },
            "per_sensor": {
                key: value.tolist()
                for key, value in hydro_metrics["per_sensor"].items()
            },
        },
        "overflow_metrics": {
            key: float(value) for key, value in overflow_metrics.items()
        },
        "overflow_threshold": float(overflow_threshold),
        "overflow_threshold_source": threshold_source,
    }
    if model_name == "lnn":
        metrics_payload["intensity_metrics"] = {
            "MAE": float(
                np.mean(
                    np.abs(predictions["pred_intensity"] - predictions["true_overflow"])
                )
            ),
            "RMSE": float(
                np.sqrt(
                    np.mean(
                        (predictions["pred_intensity"] - predictions["true_overflow"])
                        ** 2
                    )
                )
            ),
        }
    return metrics_payload


def _validate_prediction_targets(
    reference_predictions: dict[str, np.ndarray],
    candidate_predictions: dict[str, np.ndarray],
) -> None:
    np.testing.assert_allclose(
        reference_predictions["true_depths"],
        candidate_predictions["true_depths"],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        reference_predictions["true_overflow"],
        candidate_predictions["true_overflow"],
        atol=1e-8,
    )


def aggregate_prediction_sets(
    model_name: str,
    prediction_sets: Sequence[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not prediction_sets:
        raise ValueError("Cannot aggregate an empty set of prediction payloads.")

    reference_predictions = prediction_sets[0]
    for candidate_predictions in prediction_sets[1:]:
        _validate_prediction_targets(reference_predictions, candidate_predictions)

    aggregated_predictions = {
        "true_depths": reference_predictions["true_depths"],
        "true_overflow": reference_predictions["true_overflow"],
        "pred_depths": np.mean(
            np.stack(
                [prediction["pred_depths"] for prediction in prediction_sets], axis=0
            ),
            axis=0,
        ),
        "pred_overflow": np.mean(
            np.stack(
                [prediction["pred_overflow"] for prediction in prediction_sets], axis=0
            ),
            axis=0,
        ),
    }
    if model_name == "lnn":
        aggregated_predictions["pred_intensity"] = np.mean(
            np.stack(
                [
                    prediction["pred_intensity"]
                    for prediction in prediction_sets
                    if "pred_intensity" in prediction
                ],
                axis=0,
            ),
            axis=0,
        )
    return aggregated_predictions


def describe_model_architecture(model_name: str, config: dict[str, Any]) -> str:
    model_config = config["model"]
    if model_name == "gru":
        return (
            f"input={model_config['input_size']}, hidden={model_config['hidden_size']}, "
            f"layers={model_config['num_layers']}"
        )
    if model_name == "lstm":
        return (
            f"input={model_config['input_size']}, hidden={model_config['hidden_size']}, "
            f"layers={model_config['num_layers']}"
        )
    if model_name == "tcn":
        return (
            f"input={model_config['input_size']}, hidden={model_config['hidden_size']}, "
            f"layers={model_config['num_layers']}, kernel={model_config['kernel_size']}"
        )
    if model_name == "mlp":
        return (
            f"input={model_config['input_size']}, seq_len={model_config['seq_len']}, "
            f"hidden_dims={model_config['hidden_dims']}, "
            f"batch_norm={model_config.get('use_batch_norm', True)}"
        )
    if model_name == "lnn":
        return (
            f"fast={model_config['fast_units']}, slow={model_config['slow_units']}, "
            f"hidden={model_config['hidden_size']}"
        )
    return model_name


def resolve_ensemble_seeds(seeds: Sequence[int] | None = None) -> list[int]:
    resolved = [int(seed) for seed in (seeds or ENSEMBLE_DEFAULT_SEEDS)]
    if not resolved:
        raise ValueError("Ensemble seed list cannot be empty.")
    return resolved


def build_ensemble_checkpoint_path(
    model_name: str,
    seed: int,
    *,
    checkpoint_root: str | Path = "artifacts/checkpoints/ensemble",
) -> Path:
    return Path(checkpoint_root) / model_name / f"seed_{int(seed)}" / "best_model.pt"


def resolve_ensemble_results_dir(
    model_name: str,
    *,
    results_dir: str | Path | None = None,
) -> Path:
    if results_dir is not None:
        return Path(results_dir)
    return Path("artifacts/results/ensemble") / model_name


def resolve_threshold_artifact_path(
    config: dict[str, Any], threshold_artifact_path: str | Path | None = None
) -> Path:
    return (
        Path(threshold_artifact_path)
        if threshold_artifact_path is not None
        else Path(config["evaluation"]["threshold_artifact"])
    )


def load_threshold_artifact(path: str | Path) -> dict[str, Any]:
    threshold_path = Path(path)
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold artifact does not exist: {threshold_path}")
    with open(threshold_path, "r") as handle:
        return json.load(handle)


def select_overflow_threshold(
    model_name: str,
    split: str = "val",
    *,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    artifact = load_trained_model(
        model_name,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    runtime_config = artifact["runtime_config"]
    device = artifact["device"]
    split_data = load_split_data_for_config(runtime_config, split)
    split_description = describe_split_data_for_config(runtime_config, split)
    split_loader = create_test_loader(runtime_config, split_data)
    predictions = collect_predictions(
        model_name, artifact["model"], split_loader, device
    )
    threshold = float(
        find_optimal_threshold(
            predictions["true_overflow"], predictions["pred_overflow"]
        )
    )
    metrics = evaluate_overflow(
        predictions["true_overflow"],
        predictions["pred_overflow"],
        threshold=threshold,
    )
    threshold_payload = {
        "model": model_name,
        "selection_split": split,
        "threshold": threshold,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "checkpoint_path": str(artifact["checkpoint_path"]),
        "canonical_config_path": str(artifact["canonical_config_path"]),
        "config_source": artifact["config_source"],
        "runtime_metadata": build_run_metadata(
            config=runtime_config,
            model_name=model_name,
            device=device,
            split_descriptions=[split_description],
        ),
    }
    resolved_output_path = resolve_threshold_artifact_path(
        runtime_config, threshold_artifact_path=output_path
    )
    write_json(resolved_output_path, threshold_payload)
    return threshold_payload, resolved_output_path


def build_evaluation_payload(
    model_name: str,
    *,
    split: str = "test",
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    threshold_artifact_path: str | Path | None = None,
    overflow_threshold: float | None = None,
) -> dict[str, Any]:
    artifact = load_trained_model(
        model_name,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    runtime_config = artifact["runtime_config"]
    device = artifact["device"]

    split_data = load_split_data_for_config(runtime_config, split)
    split_description = describe_split_data_for_config(runtime_config, split)
    if split_data["X"].shape[-1] != runtime_config["model"]["input_size"]:
        raise ValueError(
            "Configured input_size does not match evaluation data channels: "
            f"{runtime_config['model']['input_size']} != {split_data['X'].shape[-1]}"
        )
    if (
        "seq_len" in runtime_config["model"]
        and split_data["X"].shape[1] != runtime_config["model"]["seq_len"]
    ):
        raise ValueError(
            "Configured seq_len does not match evaluation data length: "
            f"{runtime_config['model']['seq_len']} != {split_data['X'].shape[1]}"
        )

    applied_threshold = 0.5
    threshold_source = "default_0.5"
    if overflow_threshold is not None:
        applied_threshold = float(overflow_threshold)
        threshold_source = "explicit_argument"
    else:
        resolved_threshold_path = resolve_threshold_artifact_path(
            runtime_config, threshold_artifact_path=threshold_artifact_path
        )
        if not resolved_threshold_path.exists():
            raise FileNotFoundError(
                f"Threshold artifact not found at {resolved_threshold_path}. "
                f"Run select_overflow_threshold() first, or pass "
                f"--overflow-threshold explicitly."
            )
        threshold_payload = load_threshold_artifact(resolved_threshold_path)
        applied_threshold = float(threshold_payload["threshold"])
        threshold_source = f"artifact:{resolved_threshold_path}"

    split_loader = create_test_loader(runtime_config, split_data)
    predictions = collect_predictions(
        model_name, artifact["model"], split_loader, device
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_payload = build_metrics_payload_from_predictions(
        model_name,
        predictions,
        split=split,
        overflow_threshold=applied_threshold,
        threshold_source=threshold_source,
        timestamp=timestamp,
    )

    manifest_payload = {
        "model": model_name,
        "title": MODEL_TITLES[model_name],
        "architecture": describe_model_architecture(model_name, runtime_config),
        "timestamp": timestamp,
        "canonical_config_path": str(artifact["canonical_config_path"]),
        "checkpoint_path": str(artifact["checkpoint_path"]),
        "config_source": artifact["config_source"],
        "use_reduced": bool(runtime_config["data"].get("use_reduced", True)),
        "input_size": int(runtime_config["model"]["input_size"]),
        "runtime_metadata": build_run_metadata(
            config=runtime_config,
            model_name=model_name,
            device=device,
            split_descriptions=[split_description],
        ),
    }

    return {
        "metrics": metrics_payload,
        "manifest": manifest_payload,
        "predictions": predictions,
    }


def save_evaluation_payload(
    model_name: str,
    payload: dict[str, Any],
    *,
    results_dir: str | Path = "artifacts/results/release",
) -> tuple[Path, Path]:
    output_dir = Path(results_dir)
    timestamp = payload["manifest"]["timestamp"]
    metrics_path = (
        output_dir
        / f"{model_name}_{payload['metrics']['evaluation_split']}_{timestamp}_metrics.json"
    )
    manifest_path = (
        output_dir
        / f"{model_name}_{payload['metrics']['evaluation_split']}_{timestamp}_manifest.json"
    )
    write_json(metrics_path, payload["metrics"])
    write_json(manifest_path, payload["manifest"])
    return metrics_path, manifest_path


def train_lnn_ensemble(
    *,
    config_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    seeds: Sequence[int] | None = None,
    checkpoint_root: str | Path = "artifacts/checkpoints/ensemble",
    device: torch.device | None = None,
    train_data: dict[str, Any] | None = None,
    val_data: dict[str, Any] | None = None,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    model_name = "lnn"
    base_config = (
        deepcopy(config)
        if config is not None
        else load_model_config(model_name, config_path=config_path)
    )
    validate_model_config(
        model_name,
        base_config,
        source_label=str(config_path or f"{model_name}:ensemble_config"),
    )
    resolved_seeds = resolve_ensemble_seeds(seeds)
    resolved_device = get_runtime_device(base_config, device)
    resolved_train_data = train_data or load_split_data_for_config(base_config, "train")
    resolved_val_data = val_data or load_split_data_for_config(base_config, "val")

    member_payloads = []
    for seed in resolved_seeds:
        member_config = deepcopy(base_config)
        member_config["runtime"]["seed"] = int(seed)
        checkpoint_path = build_ensemble_checkpoint_path(
            model_name,
            seed,
            checkpoint_root=checkpoint_root,
        )
        member_config["output"]["checkpoint_dir"] = str(checkpoint_path.parent)
        _, training_summary = train_model(
            model_name,
            member_config,
            device=resolved_device,
            train_data=resolved_train_data,
            val_data=resolved_val_data,
            checkpoint_path=checkpoint_path,
            max_epochs=max_epochs,
        )
        member_payloads.append(
            {
                "seed": int(seed),
                "checkpoint_path": str(checkpoint_path),
                "best_epoch": int(training_summary["best_epoch"]),
                "best_val_loss": float(training_summary["best_val_loss"]),
                "best_val_accuracy": float(training_summary["best_val_accuracy"]),
                "epochs_ran": int(training_summary["epochs_ran"]),
                "num_parameters": int(training_summary["num_parameters"]),
                "runtime_metadata": training_summary["runtime_metadata"],
            }
        )

    return {
        "model": model_name,
        "ensemble_size": len(member_payloads),
        "seeds": [int(seed) for seed in resolved_seeds],
        "members": member_payloads,
        "checkpoint_root": str(checkpoint_root),
    }


def evaluate_lnn_ensemble(
    *,
    config_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    seeds: Sequence[int] | None = None,
    checkpoint_root: str | Path = "artifacts/checkpoints/ensemble",
    results_dir: str | Path | None = None,
    device: torch.device | None = None,
    val_data: dict[str, Any] | None = None,
    test_data: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    model_name = "lnn"
    base_config = (
        deepcopy(config)
        if config is not None
        else load_model_config(model_name, config_path=config_path)
    )
    validate_model_config(
        model_name,
        base_config,
        source_label=str(config_path or f"{model_name}:ensemble_config"),
    )
    resolved_seeds = resolve_ensemble_seeds(seeds)
    checkpoint_paths = [
        build_ensemble_checkpoint_path(
            model_name,
            seed,
            checkpoint_root=checkpoint_root,
        )
        for seed in resolved_seeds
    ]
    artifacts = [
        load_trained_model(
            model_name,
            device=device,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            canonical_config=base_config,
        )
        for checkpoint_path in checkpoint_paths
    ]
    reference_artifact = artifacts[0]
    reference_config = reference_artifact["runtime_config"]
    resolved_val_data = val_data or load_split_data_for_config(reference_config, "val")
    resolved_test_data = test_data or load_split_data_for_config(
        reference_config, "test"
    )
    val_description = describe_split_data_for_config(reference_config, "val")
    test_description = describe_split_data_for_config(reference_config, "test")
    val_loader = create_test_loader(reference_config, resolved_val_data)
    test_loader = create_test_loader(reference_config, resolved_test_data)

    member_val_predictions = []
    member_test_predictions = []
    member_payloads = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for seed, checkpoint_path, artifact in zip(
        resolved_seeds, checkpoint_paths, artifacts
    ):
        val_predictions = collect_predictions(
            model_name,
            artifact["model"],
            val_loader,
            artifact["device"],
        )
        test_predictions = collect_predictions(
            model_name,
            artifact["model"],
            test_loader,
            artifact["device"],
        )
        member_val_predictions.append(val_predictions)
        member_test_predictions.append(test_predictions)
        member_payloads.append(
            {
                "seed": int(seed),
                "checkpoint_path": str(checkpoint_path),
                "config_source": artifact["config_source"],
                "metrics": None,
                "checkpoint_epoch": int(artifact["checkpoint"].get("epoch", 0)),
                "num_parameters": int(artifact["checkpoint"].get("n_params", 0)),
            }
        )

    ensemble_val_predictions = aggregate_prediction_sets(
        model_name, member_val_predictions
    )
    applied_threshold = float(
        find_optimal_threshold(
            ensemble_val_predictions["true_overflow"],
            ensemble_val_predictions["pred_overflow"],
        )
    )
    selection_metrics = evaluate_overflow(
        ensemble_val_predictions["true_overflow"],
        ensemble_val_predictions["pred_overflow"],
        threshold=applied_threshold,
    )
    ensemble_test_predictions = aggregate_prediction_sets(
        model_name,
        member_test_predictions,
    )
    threshold_source = "ensemble:val_average_probabilities"
    ensemble_metrics = build_metrics_payload_from_predictions(
        model_name,
        ensemble_test_predictions,
        split="test",
        overflow_threshold=applied_threshold,
        threshold_source=threshold_source,
        timestamp=timestamp,
    )
    for member_payload, test_predictions in zip(
        member_payloads, member_test_predictions
    ):
        member_payload["metrics"] = build_metrics_payload_from_predictions(
            model_name,
            test_predictions,
            split="test",
            overflow_threshold=applied_threshold,
            threshold_source=threshold_source,
            timestamp=timestamp,
        )

    payload = {
        "model": model_name,
        "title": "LNN Regression Ensemble",
        "architecture": describe_model_architecture(model_name, reference_config),
        "timestamp": timestamp,
        "evaluation_split": "test",
        "selection_split": "val",
        "ensemble_size": len(resolved_seeds),
        "seeds": [int(seed) for seed in resolved_seeds],
        "canonical_config_path": str(reference_artifact["canonical_config_path"]),
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "config_sources": [artifact["config_source"] for artifact in artifacts],
        "overflow_threshold": float(applied_threshold),
        "overflow_threshold_source": threshold_source,
        "selection_metrics": {
            key: float(value) for key, value in selection_metrics.items()
        },
        "ensemble_metrics": ensemble_metrics,
        "member_metrics": member_payloads,
        "use_reduced": bool(reference_config["data"].get("use_reduced", True)),
        "input_size": int(reference_config["model"]["input_size"]),
        "runtime_metadata": build_ensemble_run_metadata(
            config=reference_config,
            model_name=model_name,
            device=reference_artifact["device"],
            split_descriptions=[val_description, test_description],
            seeds=resolved_seeds,
        ),
    }
    output_dir = resolve_ensemble_results_dir(model_name, results_dir=results_dir)
    output_path = output_dir / f"{model_name}_ensemble_test_{timestamp}.json"
    write_json(output_path, payload)
    return payload, output_path


def evaluate_regression_model(
    model_name: str,
    config_path: str | Path | None = None,
    include_optimal_threshold: bool = False,
) -> None:
    if include_optimal_threshold:
        raise ValueError(
            "Publication-safe evaluation no longer supports fitting thresholds on the "
            "test split. Use select_overflow_threshold(..., split='val') first."
        )

    payload = build_evaluation_payload(
        model_name, split="test", config_path=config_path
    )
    metrics = payload["metrics"]
    manifest = payload["manifest"]

    logger.info(f"Loading model from: {manifest['checkpoint_path']}")
    logger.info(
        f"Threshold source: {metrics['overflow_threshold_source']} "
        f"(threshold={metrics['overflow_threshold']:.6f})"
    )

    logger.info("\n" + "=" * 50)
    logger.info("DEPTH REGRESSION METRICS (Aggregated)")
    logger.info("=" * 50)
    for metric, value in metrics["depth_metrics"]["aggregated"].items():
        logger.info(f"{metric:12s}: {value:.4f}")

    logger.info("\n" + "=" * 50)
    logger.info("HYDROLOGICAL METRICS (Aggregated)")
    logger.info("=" * 50)
    for metric, value in metrics["hydrological_metrics"]["aggregated"].items():
        logger.info(f"{metric:25s}: {value:.4f}")

    logger.info("\n" + "=" * 50)
    logger.info("OVERFLOW CLASSIFICATION METRICS")
    logger.info("=" * 50)
    for metric, value in metrics["overflow_metrics"].items():
        logger.info(f"{metric:12s}: {value:.4f}")

    results_dir = Path("artifacts/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    combined_results_path = results_dir / (
        f"{model_name}_regression_evaluation_{manifest['timestamp']}.json"
    )
    combined_payload = {
        "manifest": manifest,
        "metrics": metrics,
    }
    write_json(combined_results_path, combined_payload)
    logger.info(f"\nResults saved to: {combined_results_path}")
