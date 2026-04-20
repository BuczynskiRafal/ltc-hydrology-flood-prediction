from typing import Callable, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.regression_dataloader import RegressionDataset


def time_series_cv_torch(
    X: np.ndarray,
    y_depths: np.ndarray,
    y_overflow: np.ndarray,
    flood_mask: np.ndarray,
    model_factory: Callable,
    train_fn: Callable,
    n_folds: int = 5,
    device: str = "cuda",
) -> Dict:
    """
    Time-series cross-validation with expanding window for PyTorch models.
    """
    n = len(X)
    fold_size = n // (n_folds + 1)

    fold_results = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 2)
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        X_train = X[:train_end]
        y_depths_train = y_depths[:train_end]
        y_overflow_train = y_overflow[:train_end]
        flood_mask_train = flood_mask[:train_end]

        X_test = X[test_start:test_end]
        y_depths_test = y_depths[test_start:test_end]
        y_overflow_test = y_overflow[test_start:test_end]
        flood_mask_test = flood_mask[test_start:test_end]

        train_data = {
            "X": X_train,
            "y_depths": y_depths_train,
            "y_overflow": y_overflow_train,
            "flood_mask": flood_mask_train,
        }
        test_data = {
            "X": X_test,
            "y_depths": y_depths_test,
            "y_overflow": y_overflow_test,
            "flood_mask": flood_mask_test,
        }

        model = model_factory()
        model = model.to(device)

        train_fn(model, train_data, device)

        test_loader = DataLoader(
            RegressionDataset(**test_data), batch_size=512, shuffle=False
        )

        nse = evaluate_nse(model, test_loader, device)

        fold_results.append(
            {
                "fold": fold + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "nse": float(nse),
            }
        )

    nse_scores = np.array([r["nse"] for r in fold_results])
    mean_nse = float(np.mean(nse_scores))
    std_nse = float(np.std(nse_scores, ddof=1))
    cv = float(std_nse / mean_nse) if mean_nse != 0 else float("inf")

    return {
        "fold_scores": nse_scores.tolist(),
        "fold_details": fold_results,
        "mean_nse": mean_nse,
        "std_nse": std_nse,
        "cv": cv,
        "n_folds": n_folds,
    }


def evaluate_nse(model, loader, device):
    """Evaluate NSE for a PyTorch model."""
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            y_depths = batch["y_depths"].to(device)

            outputs = model(X)
            pred_depths = outputs[0] if isinstance(outputs, tuple) else outputs

            all_true.append(y_depths.cpu().numpy())
            all_pred.append(pred_depths.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    numerator = np.sum((y_true_flat - y_pred_flat) ** 2)
    denominator = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)

    return 1 - (numerator / denominator)
