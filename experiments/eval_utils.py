"""Shared helpers for deterministic model evaluation on the article test set."""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_nse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)


def evaluate_depths(y_true, y_pred):
    mae_per_sensor = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse_per_sensor = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    nse_per_sensor = np.array(
        [compute_nse(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    )

    return {
        "aggregated": {
            "MAE": np.mean(mae_per_sensor),
            "RMSE": np.mean(rmse_per_sensor),
            "NSE": np.mean(nse_per_sensor),
        },
        "per_sensor": {
            "MAE": mae_per_sensor,
            "RMSE": rmse_per_sensor,
            "NSE": nse_per_sensor,
        },
    }


def find_optimal_threshold(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def evaluate_overflow(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred > threshold).astype(int)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred)
    return {"F1": f1, "Precision": precision, "Recall": recall, "ROC-AUC": roc_auc}


def resolve_checkpoint_path(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    flat_path = checkpoint_dir / "best_model.pt"

    if flat_path.exists():
        return flat_path

    raise FileNotFoundError(f"Could not find checkpoint in '{flat_path}'.")
