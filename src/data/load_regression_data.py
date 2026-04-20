from src.data_utils import (
    describe_regression_arrays,
    load_feature_names,
    load_regression_arrays,
)
from src.project_config import TARGET_SENSORS, WINDOW_T_IN
from src.selected_features import SELECTED_FEATURES


def load_regression_data(split="train", use_reduced=True):
    arrays = load_regression_arrays(split, use_reduced)
    validate_regression_split_contract(split, arrays, use_reduced=use_reduced)
    return arrays


def describe_regression_data(split="train", use_reduced=True):
    return describe_regression_arrays(split, use_reduced)


def validate_regression_split_contract(split, arrays, use_reduced=True):
    X = arrays["X"]
    y_depths = arrays["y_depths"]
    y_overflow = arrays["y_overflow"]
    flood_mask = arrays["flood_mask"]

    if X.ndim != 3:
        raise ValueError(f"{split}: X must be 3D [N, T, C], got shape {X.shape}")
    if X.shape[1] != WINDOW_T_IN:
        raise ValueError(
            f"{split}: X second dimension must equal WINDOW_T_IN={WINDOW_T_IN}, "
            f"got {X.shape[1]}"
        )

    expected_channels = (
        len(SELECTED_FEATURES)
        if use_reduced
        else len(load_feature_names(use_reduced=False))
    )
    if X.shape[2] != expected_channels:
        raise ValueError(
            f"{split}: X channel count must be {expected_channels}, got {X.shape[2]}"
        )

    if y_depths.ndim != 2 or y_depths.shape[1] != len(TARGET_SENSORS):
        raise ValueError(
            f"{split}: y_depths must be 2D [N, {len(TARGET_SENSORS)}], "
            f"got shape {y_depths.shape}"
        )
    if y_overflow.ndim != 1:
        raise ValueError(
            f"{split}: y_overflow must be 1D [N], got shape {y_overflow.shape}"
        )
    if flood_mask.ndim != 1:
        raise ValueError(
            f"{split}: flood_mask must be 1D [N], got shape {flood_mask.shape}"
        )

    n_samples = X.shape[0]
    if (
        y_depths.shape[0] != n_samples
        or y_overflow.shape[0] != n_samples
        or flood_mask.shape[0] != n_samples
    ):
        raise ValueError(
            f"{split}: inconsistent sample counts across arrays: "
            f"X={X.shape[0]}, y_depths={y_depths.shape[0]}, "
            f"y_overflow={y_overflow.shape[0]}, flood_mask={flood_mask.shape[0]}"
        )
