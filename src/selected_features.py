"""Canonical final 31 predictors used in normalized tables and reduced tensors."""

from src.logger import get_console_logger
from src.project_config import TARGET_SENSORS

logger = get_console_logger(__name__)


SELECTED_FEATURES = [
    "rain_5425",
    "rain_5427",
    "rain_aabakken",
    "rain_avg",
    "I_t",
    "P_t",
    "G71F05R_position_norm",
    "G71F04R_Level2_norm",
    "G71F05R_LevelBasin_norm",
    "G71F04R_Level1_norm",
    "G80F13P_LevelPS_norm",
    "G71F68Y_LevelPS_norm",
    "G71F05R_LevelInlet_norm",
    "G71F06R_LevelInlet_velocity",
    "gradient_0",
    "gradient_3",
    "gradient_4",
    "gradient_5",
    "gradient_9",
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
]

assert len(SELECTED_FEATURES) == 31


def filter_features(df):
    metadata_cols = ["time", "year", "target", "flash_flood"]
    missing_features = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing_features:
        logger.info(
            f"WARNING: Missing selected features: {', '.join(missing_features)}"
        )
    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    available_metadata = [c for c in metadata_cols if c in df.columns]
    available_targets = [c for c in TARGET_SENSORS if c in df.columns]

    cols_to_keep = available_metadata + available_features + available_targets
    return df[cols_to_keep]
