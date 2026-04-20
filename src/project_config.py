"""
Canonical project configuration for the publication-safe pipeline.

This module lives inside `src` so package imports stay unambiguous and do not
depend on mutating `sys.path` at runtime.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DOWNLOADED: Path = PROJECT_ROOT / "downloaded"
DATA_RAW: Path = PROJECT_ROOT / "data/raw"
DATA_INTERIM: Path = PROJECT_ROOT / "data/interim"
DATA_PROCESSED: Path = PROJECT_ROOT / "data/processed"
DATA_REGRESSION: Path = PROJECT_ROOT / "data/final_regression"
OUTPUT: Path = PROJECT_ROOT / "output"
OUTPUT_REPORTS: Path = OUTPUT / "reports"
OUTPUT_VIZ: Path = OUTPUT / "visualizations"
ARTIFACTS: Path = PROJECT_ROOT / "artifacts"
ARTIFACTS_CHECKPOINTS: Path = ARTIFACTS / "checkpoints"
ARTIFACTS_LOGS: Path = ARTIFACTS / "logs"


RAIN_GAUGE_IDS: list[str] = ["5425", "5427", "Aabakken"]
RAIN_GAUGE_FILES: dict[str, str] = {
    "5425": "3a_Raingauges/5425_ts.txt",
    "5427": "3a_Raingauges/5427_ts.txt",
    "Aabakken": "3a_Raingauges/Aabakken_bellinge_vandvaerk_v2_ts.txt",
}
MET_VARIABLES: list[str] = ["precip_past10min", "temp_dry", "humidity", "wind_speed"]
ASSET_FILES: dict[str, str] = {
    "links": "1_Assetdata/Links.shp",
    "nodes": "1_Assetdata/Manholes.shp",
}


PIPE_DIAMETERS: dict[str, float] = {
    "G71F04R": 0.8,
    "G71F05R": 1.2,
    "G71F06R": 0.6,
    "G71F68Y": 1.0,
    "G73F010": 0.8,
    "G80F11B": 1.5,
    "G80F13P": 1.2,
    "G80F66Y": 0.9,
}

TARGET_SENSORS: list[str] = [
    "G71F05R_LevelBasin",
    "G71F05R_LevelInlet",
    "G71F05R_position",
    "G71F68Y_LevelPS",
    "G80F13P_LevelPS",
]


RAIN_COLS: list[str] = ["rain_5425", "rain_5427", "rain_aabakken"]
RAINFALL_INTENSITY_WINDOW: int = 5
CUMULATIVE_PRECIP_WINDOW: int = 60
API_DECAY_FACTOR: float = 0.85
RAIN_EVENT_THRESHOLD: float = 0.1
RAIN_MAX_INTENSITY: float = 200.0


TRAIN_YEARS: list[int] = list(range(2010, 2018))
VAL_YEARS: list[int] = [2018]
TEST_YEARS: list[int] = [2019]


LOG_EPSILON: float = 1e-6
TEMPORAL_PATTERNS: list[str] = ["sin", "cos", "dow_"]
BOUNDED_PATTERNS: list[str] = ["norm"]
RAIN_PATTERNS: list[str] = ["rain", "I_t", "P_t", "API"]


SWMM_MODEL_FILE: str = "downloaded/7_SWMM/BellingeSWMM_v021_nopervious.inp"
SWMM_RAIN_FILE: str = "downloaded/7_SWMM/rg_bellinge_Jun2010_Aug2021.dat"
SWMM_RAIN_PERCENTILE: int = 90
SWMM_BUFFER_HOURS: int = 2
FLASH_FLOOD_MIN_INTENSITY: float = 20.0
FLASH_FLOOD_MIN_DURATION: int = 10


WINDOW_T_IN: int = 45
WINDOW_T_OUT: int = 10
WINDOW_STRIDE: int = 10
CI_Z_SCORE: float = 1.96
FLOOD_WEIGHT_NORMAL: float = 1.0
FLOOD_WEIGHT_FLOOD: float = 5.0
FLOOD_BUFFER_MINUTES: int = 30


VARIANCE_THRESHOLD: float = 0.001
CORRELATION_THRESHOLD: float = 0.95
CONSTANT_THRESHOLD: float = 1e-8


EXCLUDE_COLUMNS: list[str] = [
    "time",
    "year",
    "target",
    "rainfall_event",
    "flash_flood",
    "is_rain",
    "rain_change",
    "high_intensity",
    "event_active",
    "event_change",
    "day_of_week",
    "hour",
    "month",
    "augmentation_method",
    "synthetic",
]


def get_train_val_test_mask(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    train_mask = df["year"].isin(TRAIN_YEARS)
    val_mask = df["year"].isin(VAL_YEARS)
    test_mask = df["year"].isin(TEST_YEARS)
    return train_mask, val_mask, test_mask


def get_pipe_diameter(sensor_id: str) -> float:
    base_id = sensor_id.split("_")[0]
    return PIPE_DIAMETERS.get(base_id, 1.0)


def is_temporal_feature(col_name: str) -> bool:
    return any(pattern in col_name for pattern in TEMPORAL_PATTERNS)


def is_bounded_feature(col_name: str) -> bool:
    return any(pattern in col_name for pattern in BOUNDED_PATTERNS)


def is_rain_feature(col_name: str) -> bool:
    return (
        any(pattern in col_name for pattern in RAIN_PATTERNS) or col_name in RAIN_COLS
    )
