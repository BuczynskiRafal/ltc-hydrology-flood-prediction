"""Enforce the canonical 31-feature regression tensor contract."""

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.logger import get_console_logger
from src.project_config import TARGET_SENSORS
from src.selected_features import SELECTED_FEATURES

logger = get_console_logger(__name__)


DATA_REGRESSION = Path("data/final_regression")
OUTPUT_DIR = Path("output/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_regression_tensors():
    train_X = np.load(DATA_REGRESSION / "train_X.npy")
    val_X = np.load(DATA_REGRESSION / "val_X.npy")
    test_X = np.load(DATA_REGRESSION / "test_X.npy")

    with open(DATA_REGRESSION / "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    return train_X, val_X, test_X, feature_names


def validate_tensor_shapes(train_X, val_X, test_X, feature_names):
    if any(tensor.ndim != 3 for tensor in (train_X, val_X, test_X)):
        raise ValueError("Regression tensors must be 3D arrays with shape (N, T, C).")

    expected_channels = len(feature_names)
    split_channels = {
        "train": train_X.shape[2],
        "val": val_X.shape[2],
        "test": test_X.shape[2],
    }

    if len(set(split_channels.values())) != 1:
        raise ValueError(
            "Regression splits do not share the same channel count: "
            f"{split_channels}"
        )

    mismatched = {
        split: channels
        for split, channels in split_channels.items()
        if channels != expected_channels
    }
    if mismatched:
        raise ValueError(
            "Regression tensor channels do not match feature_names.pkl: "
            f"expected {expected_channels}, got {mismatched}."
        )


def validate_feature_contract(feature_names):
    duplicates = [name for name, count in Counter(feature_names).items() if count > 1]
    if duplicates:
        raise ValueError(
            f"Duplicate feature names found in feature_names.pkl: {duplicates}"
        )

    missing_selected = [name for name in SELECTED_FEATURES if name not in feature_names]
    if missing_selected:
        raise ValueError(
            "Missing canonical selected features in feature_names.pkl: "
            f"{missing_selected}"
        )

    removed_features = [name for name in feature_names if name not in SELECTED_FEATURES]
    if set(removed_features) != set(TARGET_SENSORS) or len(removed_features) != len(
        TARGET_SENSORS
    ):
        raise ValueError(
            "Unexpected regression channels outside the canonical 31 predictors. "
            f"Expected only target sensors {TARGET_SENSORS}, got {removed_features}."
        )

    return removed_features


def get_selected_indices(feature_names):
    return [feature_names.index(name) for name in SELECTED_FEATURES]


def build_feature_categories(feature_names):
    temporal_features = {
        "API",
        "tau_event",
        "hour_sin",
        "hour_cos",
        "dow_0",
        "dow_1",
        "dow_2",
        "dow_3",
        "dow_4",
        "dow_5",
        "dow_6",
        "month_sin",
        "month_cos",
    }

    return {
        "Normalized Sensors": [name for name in feature_names if "_norm" in name],
        "Rain Features": [
            name
            for name in feature_names
            if "rain" in name.lower() or name in {"I_t", "P_t"}
        ],
        "Flow (Q_in)": [name for name in feature_names if "Q_in" in name],
        "Velocities": [name for name in feature_names if "_velocity" in name],
        "Gradients": [name for name in feature_names if "gradient" in name],
        "Accelerations": [name for name in feature_names if "_acceleration" in name],
        "Temporal": [name for name in feature_names if name in temporal_features],
    }


def build_feature_comparison(feature_names):
    comparison_rows = []
    for name in feature_names:
        if name in SELECTED_FEATURES:
            comparison_rows.append(
                {
                    "Feature": name,
                    "Status": "KEPT",
                    "Reason": "Canonical predictor",
                }
            )
        else:
            comparison_rows.append(
                {
                    "Feature": name,
                    "Status": "REMOVED",
                    "Reason": "Regression target sensor channel (not a predictor)",
                }
            )
    return pd.DataFrame(comparison_rows)


def write_summary(feature_names, removed_features):
    summary_lines = [
        "Feature reduction summary",
        f"Original regression channels: {len(feature_names)}",
        f"Canonical predictors: {len(SELECTED_FEATURES)}",
        f"Removed target-sensor channels: {len(removed_features)}",
        f"Removed channels: {', '.join(removed_features)}",
        "",
        "Outputs:",
        "- data/final_regression/train_X_reduced.npy",
        "- data/final_regression/val_X_reduced.npy",
        "- data/final_regression/test_X_reduced.npy",
        "- data/final_regression/feature_names_reduced.pkl",
        "- data/final_regression/feature_categories.pkl",
        "- output/reports/feature_selection_comparison.csv",
    ]

    with open(OUTPUT_DIR / "feature_engineering_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))


def main():
    train_X, val_X, test_X, feature_names = load_regression_tensors()
    validate_tensor_shapes(train_X, val_X, test_X, feature_names)
    removed_features = validate_feature_contract(feature_names)
    selected_indices = get_selected_indices(feature_names)

    train_X_reduced = train_X[:, :, selected_indices]
    val_X_reduced = val_X[:, :, selected_indices]
    test_X_reduced = test_X[:, :, selected_indices]

    categories = build_feature_categories(SELECTED_FEATURES)

    np.save(DATA_REGRESSION / "train_X_reduced.npy", train_X_reduced)
    np.save(DATA_REGRESSION / "val_X_reduced.npy", val_X_reduced)
    np.save(DATA_REGRESSION / "test_X_reduced.npy", test_X_reduced)

    with open(DATA_REGRESSION / "feature_names_reduced.pkl", "wb") as f:
        pickle.dump(SELECTED_FEATURES, f)

    with open(DATA_REGRESSION / "feature_categories.pkl", "wb") as f:
        pickle.dump(categories, f)

    build_feature_comparison(feature_names).to_csv(
        OUTPUT_DIR / "feature_selection_comparison.csv", index=False
    )
    write_summary(feature_names, removed_features)

    logger.info(
        "Reduced regression tensors "
        f"{len(feature_names)} -> {len(SELECTED_FEATURES)} channels; "
        f"removed target-sensor channels: {', '.join(removed_features)}"
    )


if __name__ == "__main__":
    main()
