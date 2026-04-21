import json
from datetime import datetime
from pathlib import Path

import numpy as np

from experiments.regression_pipeline import (
    collect_predictions,
    create_dataloader,
    get_device,
    load_model_config,
    load_split_data_for_config,
    train_model,
)
from src.logger import get_console_logger

logger = get_console_logger(__name__)


MODEL_NAMES = ("lnn", "gru", "lstm", "tcn", "mlp")
VALIDATION_FRACTION = 0.1


def compute_nse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)


def slice_data(data, start, end):
    return {key: value[start:end] for key, value in data.items()}


def split_train_and_validation_data(train_data):
    n_samples = len(train_data["X"])
    val_size = max(1, int(n_samples * VALIDATION_FRACTION))
    val_size = min(val_size, n_samples - 1)
    split_index = n_samples - val_size
    return slice_data(train_data, 0, split_index), slice_data(
        train_data, split_index, n_samples
    )


def evaluate_fold(model_name, model, test_data, config, device):
    test_loader = create_dataloader(
        test_data,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        shuffle=False,
    )
    predictions = collect_predictions(model_name, model, test_loader, device)
    y_true = predictions["true_depths"].flatten()
    y_pred = predictions["pred_depths"].flatten()
    return float(compute_nse(y_true, y_pred))


def time_series_cv(model_name, config, data, device, n_folds=5):
    n_samples = len(data["X"])
    fold_size = n_samples // (n_folds + 1)
    fold_results = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 2)
        test_start = train_end
        test_end = min(test_start + fold_size, n_samples)

        train_fold = slice_data(data, 0, train_end)
        test_fold = slice_data(data, test_start, test_end)
        train_data, val_data = split_train_and_validation_data(train_fold)

        model, summary = train_model(
            model_name,
            config,
            device=device,
            train_data=train_data,
            val_data=val_data,
            checkpoint_path=None,
        )
        nse = evaluate_fold(model_name, model, test_fold, config, device)
        fold_results.append(
            {
                "fold": fold + 1,
                "train_size": len(train_data["X"]),
                "val_size": len(val_data["X"]),
                "test_size": len(test_fold["X"]),
                "best_epoch": summary["best_epoch"],
                "best_val_loss": summary["best_val_loss"],
                "nse": nse,
            }
        )

    scores = np.array([result["nse"] for result in fold_results])
    mean_nse = float(np.mean(scores))
    std_nse = float(np.std(scores, ddof=1))
    return {
        "fold_scores": scores.tolist(),
        "fold_details": fold_results,
        "mean_nse": mean_nse,
        "std_nse": std_nse,
        "cv": float(std_nse / mean_nse) if mean_nse != 0 else float("inf"),
        "n_folds": n_folds,
    }


def main():
    device = get_device()
    logger.info(f"Using device: {device}")

    results = {}
    for model_name in MODEL_NAMES:
        config = load_model_config(model_name)
        data = load_split_data_for_config(config, "train")
        logger.info(f"\nRunning time-series CV for {model_name.upper()}...")
        results[model_name] = time_series_cv(
            model_name, config, data, device, n_folds=5
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("artifacts/results/cross_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"cv_results_{timestamp}.json"
    with open(output_path, "w") as handle:
        json.dump(results, handle, indent=2)

    logger.info(f"\nResults saved to: {output_path}")
    for model_name, cv_results in results.items():
        logger.info(f"\n{model_name.upper()}:")
        logger.info(
            f"  Mean NSE: {cv_results['mean_nse']:.4f} ± {cv_results['std_nse']:.4f}"
        )
        logger.info(f"  CV: {cv_results['cv']:.4f}")
        logger.info(f"  Fold scores: {cv_results['fold_scores']}")


if __name__ == "__main__":
    main()
