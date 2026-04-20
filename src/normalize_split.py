import numpy as np
import pandas as pd

from src.data_utils import load_features, save_normalized_data
from src.logger import get_console_logger
from src.project_config import (
    LOG_EPSILON,
    OUTPUT_REPORTS,
    TEST_YEARS,
    TRAIN_YEARS,
    VAL_YEARS,
    is_bounded_feature,
    is_rain_feature,
    is_temporal_feature,
)
from src.selected_features import filter_features

logger = get_console_logger(__name__)


data = load_features()
data["year"] = data["time"].dt.year

train = data[data["year"].isin(TRAIN_YEARS)].copy()
val = data[data["year"].isin(VAL_YEARS)].copy()
test = data[data["year"].isin(TEST_YEARS)].copy()

feature_cols = [
    c for c in data.columns if c not in ["time", "year", "event_active", "event_change"]
]

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
    values = train[col].dropna()
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
    values = train[col].dropna()
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
    df_norm = df.copy()
    for col, param in params.items():
        if col not in df_norm.columns:
            continue
        if param["type"] == "zscore":
            if param["sigma"] == 0:
                logger.info(
                    f"WARNING: Skipping z-score normalization for {col} because sigma == 0."
                )
                continue
            df_norm[col] = (df_norm[col] - param["mu"]) / param["sigma"]
        elif param["type"] == "minmax":
            if param["max"] == param["min"]:
                logger.info(
                    f"WARNING: Skipping min-max normalization for {col} because max == min."
                )
                continue
            df_norm[col] = (df_norm[col] - param["min"]) / (param["max"] - param["min"])
        elif param["type"] == "log":
            df_norm[col] = np.log(df_norm[col] + param["epsilon"])
    return df_norm


train_norm = filter_features(normalize_data(train, norm_params))
val_norm = filter_features(normalize_data(val, norm_params))
test_norm = filter_features(normalize_data(test, norm_params))

save_normalized_data(train_norm, val_norm, test_norm, norm_params)
pd.DataFrame.from_dict(norm_params, orient="index").to_csv(
    OUTPUT_REPORTS / "normalization_params.csv"
)
