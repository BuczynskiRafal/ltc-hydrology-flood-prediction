import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data_utils import (
    load_labeled_data,
    save_regression_arrays,
    save_regression_metadata,
)
from src.logger import get_console_logger
from src.project_config import DATA_INTERIM, DATA_REGRESSION, TARGET_SENSORS
from src.selected_features import SELECTED_FEATURES
from src.window_utils import (
    create_sequences_with_regression_targets,
    print_sequence_stats,
)

logger = get_console_logger(__name__)


REQUIRED_COLUMNS = ["time", "year", "flash_flood", "target", *TARGET_SENSORS]
EXPECTED_FEATURE_COLUMNS = [*SELECTED_FEATURES, *TARGET_SENSORS]
EXPECTED_TABULAR_COLUMNS = [
    "time",
    "year",
    *SELECTED_FEATURES,
    *TARGET_SENSORS,
    "flash_flood",
    "target",
]
AUGMENTATION_COLUMNS = ["synthetic", "augmentation_method"]


def get_feature_columns(df):
    missing = [col for col in EXPECTED_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing canonical regression columns: {missing}")
    return list(EXPECTED_FEATURE_COLUMNS)


def validate_required_columns(split_name, df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{split_name} is missing required columns: {missing}")


def validate_tabular_schema(split_name, df, *, allow_augmentation_columns=False):
    allowed = set(EXPECTED_TABULAR_COLUMNS)
    if allow_augmentation_columns:
        allowed |= set(AUGMENTATION_COLUMNS)
    missing = [col for col in EXPECTED_TABULAR_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in allowed]

    if missing or extra:
        problems = []
        if missing:
            problems.append(f"missing={missing}")
        if extra:
            problems.append(f"extra={extra}")
        raise ValueError(
            f"{split_name} tabular schema does not match canonical contract: "
            f"{', '.join(problems)}"
        )


def validate_time_schema(split_name, df):
    duplicate_count = int(df["time"].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"{split_name} has {duplicate_count:,} duplicate time rows. "
            "Canonical windowing requires unique timestamps."
        )

    if df["time"].isna().any():
        raise ValueError(f"{split_name} contains missing timestamps.")

    if not df["time"].is_monotonic_increasing:
        raise ValueError(
            f"{split_name} time column is not monotonic increasing. "
            "Canonical windowing requires pre-sorted timestamps."
        )


def validate_predictor_missingness(split_name, df):
    predictor_frame = df[SELECTED_FEATURES]
    if predictor_frame.isna().any().any():
        missing_cols = predictor_frame.columns[predictor_frame.isna().any()].tolist()
        raise ValueError(
            f"{split_name} contains missing values in canonical predictors: "
            f"{missing_cols}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create regression windows for the publication-safe single-horizon task."
    )
    parser.add_argument(
        "--use-augmented-train",
        action="store_true",
        help=(
            "Load train_labeled_augmented.pkl instead of train_labeled.pkl. "
            "This path is experimental and not part of the canonical release pipeline."
        ),
    )
    return parser.parse_args()


def load_flash_floods():
    swmm_labels = pd.read_csv(DATA_INTERIM / "swmm_floods_timestep_all_years.csv")
    swmm_labels["timestamp"] = pd.to_datetime(swmm_labels["timestamp"])
    swmm_labels["period_end"] = pd.to_datetime(swmm_labels["period_end"])
    return swmm_labels[swmm_labels["flash_flood"]]


def warn_empty_sequences(split_name, X):
    if len(X) == 0:
        logger.info(
            f"WARNING: {split_name} generated 0 sequences; "
            "empty regression arrays will be saved."
        )


def main():
    args = parse_args()
    DATA_REGRESSION.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_labeled_data(
        use_augmented_train=args.use_augmented_train
    )
    splits = {"train": train_df, "val": val_df, "test": test_df}

    for split_name, df in splits.items():
        df["time"] = pd.to_datetime(df["time"])
        validate_required_columns(split_name, df)
        is_augmented_split = args.use_augmented_train and split_name == "train"
        validate_tabular_schema(
            split_name, df, allow_augmentation_columns=is_augmented_split
        )
        validate_time_schema(split_name, df)
        validate_predictor_missingness(split_name, df)
        if is_augmented_split:
            df = df.drop(columns=[c for c in AUGMENTATION_COLUMNS if c in df.columns])
        splits[split_name] = df.sort_values("time").reset_index(drop=True)

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]
    feature_cols = get_feature_columns(train_df)

    flash_floods = load_flash_floods()

    train_X, train_y_depths, train_y_overflow, train_flood_mask = (
        create_sequences_with_regression_targets(
            train_df,
            flash_floods,
            feature_cols,
            TARGET_SENSORS,
        )
    )

    val_X, val_y_depths, val_y_overflow, val_flood_mask = (
        create_sequences_with_regression_targets(
            val_df,
            flash_floods,
            feature_cols,
            TARGET_SENSORS,
        )
    )

    test_X, test_y_depths, test_y_overflow, test_flood_mask = (
        create_sequences_with_regression_targets(
            test_df,
            flash_floods,
            feature_cols,
            TARGET_SENSORS,
        )
    )

    warn_empty_sequences("train", train_X)
    warn_empty_sequences("val", val_X)
    warn_empty_sequences("test", test_X)

    save_regression_arrays(
        "train", train_X, train_y_depths, train_y_overflow, train_flood_mask
    )
    save_regression_arrays("val", val_X, val_y_depths, val_y_overflow, val_flood_mask)
    save_regression_arrays(
        "test", test_X, test_y_depths, test_y_overflow, test_flood_mask
    )
    save_regression_metadata(TARGET_SENSORS, feature_cols)

    print_sequence_stats(
        "train",
        train_X,
        train_y_depths,
        train_y_overflow,
        train_flood_mask,
        TARGET_SENSORS,
    )
    print_sequence_stats(
        "val", val_X, val_y_depths, val_y_overflow, val_flood_mask, TARGET_SENSORS
    )
    print_sequence_stats(
        "test",
        test_X,
        test_y_depths,
        test_y_overflow,
        test_flood_mask,
        TARGET_SENSORS,
    )


if __name__ == "__main__":
    main()
