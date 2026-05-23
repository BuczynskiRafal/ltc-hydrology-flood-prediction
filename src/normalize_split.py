import numpy as np
import pandas as pd

from src.data_utils import load_features, save_pickle
from src.logger import get_console_logger
from src.project_config import (
    DATA_INTERIM,
    LOG_EPSILON,
    OUTPUT_REPORTS,
    TARGET_SENSORS,
    TEST_YEARS,
    TRAIN_YEARS,
    VAL_YEARS,
    is_bounded_feature,
    is_rain_feature,
    is_temporal_feature,
)
from src.selected_features import SELECTED_FEATURES, filter_features

logger = get_console_logger(__name__)


data = load_features()
required_columns = ["time", *SELECTED_FEATURES, *TARGET_SENSORS]
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(
        "normalize_split is missing canonical columns from features.pkl: "
        + ", ".join(missing_columns)
    )

data = data[required_columns].copy()
for col in [*SELECTED_FEATURES, *TARGET_SENSORS]:
    data[col] = pd.to_numeric(data[col], downcast="float")

data["year"] = data["time"].dt.year.astype(np.int16)

train_mask = data["year"].isin(TRAIN_YEARS)
val_mask = data["year"].isin(VAL_YEARS)
test_mask = data["year"].isin(TEST_YEARS)

feature_cols = list(SELECTED_FEATURES)

temporal_features = [c for c in feature_cols if is_temporal_feature(c)]
bounded_features = [c for c in feature_cols if is_bounded_feature(c)]
rain_features = [c for c in feature_cols if is_rain_feature(c)]
continuous_features = [
    c
    for c in feature_cols
    if c not in temporal_features + bounded_features + rain_features
]

norm_params = {}

for col in continuous_features:
    values = data.loc[train_mask, col].dropna()
    if len(values) > 0:
        sigma = values.std()
        if sigma == 0:
            logger.info(f"WARNING: Recording {col} as unnormalized because sigma == 0.")
            norm_params[col] = {"type": "none"}
        else:
            norm_params[col] = {
                "type": "zscore",
                "mu": values.mean(),
                "sigma": sigma,
            }

for col in bounded_features:
    values = data.loc[train_mask, col].dropna()
    if len(values) > 0:
        min_value = values.min()
        max_value = values.max()
        if max_value == min_value:
            logger.info(f"WARNING: Recording {col} as unnormalized because max == min.")
            norm_params[col] = {"type": "none"}
        else:
            norm_params[col] = {
                "type": "minmax",
                "min": min_value,
                "max": max_value,
            }

for col in rain_features:
    norm_params[col] = {"type": "log", "epsilon": LOG_EPSILON}

for col in temporal_features:
    norm_params[col] = {"type": "none"}


def normalize_data(df, params):
    for col, param in params.items():
        if col not in df.columns:
            continue
        if param["type"] == "zscore":
            if param["sigma"] == 0:
                logger.info(
                    f"WARNING: Skipping z-score normalization for {col} because sigma == 0."
                )
                continue
            df[col] = (df[col] - param["mu"]) / param["sigma"]
        elif param["type"] == "minmax":
            if param["max"] == param["min"]:
                logger.info(
                    f"WARNING: Skipping min-max normalization for {col} because max == min."
                )
                continue
            df[col] = (df[col] - param["min"]) / (param["max"] - param["min"])
        elif param["type"] == "log":
            df[col] = np.log(df[col] + param["epsilon"])
    return df


def save_normalized_split(split_name, mask):
    logger.info(f"Normalizing {split_name} split...")
    split_frame = data.loc[
        mask, ["time", "year", *SELECTED_FEATURES, *TARGET_SENSORS]
    ].copy()
    split_frame = normalize_data(split_frame, norm_params)
    split_frame = filter_features(split_frame)
    save_pickle(split_frame, DATA_INTERIM / f"{split_name}_normalized.pkl")
    logger.info(f"Saved {split_name} split with shape={split_frame.shape}")


save_normalized_split("train", train_mask)
save_normalized_split("val", val_mask)
save_normalized_split("test", test_mask)
save_pickle(norm_params, DATA_INTERIM / "norm_params.pkl")
OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)
pd.DataFrame.from_dict(norm_params, orient="index").to_csv(
    OUTPUT_REPORTS / "normalization_params.csv"
)
