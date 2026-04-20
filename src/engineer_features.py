import numpy as np

from src.data_utils import load_cleaned_data, save_features
from src.logger import get_console_logger
from src.project_config import (
    API_DECAY_FACTOR,
    CUMULATIVE_PRECIP_WINDOW,
    PIPE_DIAMETERS,
    RAIN_COLS,
    RAIN_EVENT_THRESHOLD,
    RAINFALL_INTENSITY_WINDOW,
)

logger = get_console_logger(__name__)


sensor_data, rain_data = load_cleaned_data()

data = (
    sensor_data.merge(rain_data, on="time", how="left")
    .sort_values("time")
    .reset_index(drop=True)
)

data["rain_avg"] = data[RAIN_COLS].mean(axis=1)
data["I_t"] = (
    data["rain_avg"].rolling(window=RAINFALL_INTENSITY_WINDOW, min_periods=1).mean()
)
data["P_t"] = (
    data["rain_avg"].rolling(window=CUMULATIVE_PRECIP_WINDOW, min_periods=1).sum()
)

flowmeter_col = [c for c in data.columns if "G71F68Yp1" in c and "power" not in c]
if flowmeter_col:
    data["Q_in"] = data[flowmeter_col[0]]
else:
    logger.info(
        "WARNING: Flowmeter column not found. Falling back to Q_in = 0; velocity and acceleration features may be zero-derived."
    )
    data["Q_in"] = 0

level_cols = [
    c
    for c in data.columns
    if c not in ["time"] + RAIN_COLS + ["rain_avg", "I_t", "P_t", "Q_in"]
]

for col in level_cols:
    sensor_id = col.split("_")[0]
    diameter = PIPE_DIAMETERS.get(sensor_id, 1.0)
    data[f"{col}_norm"] = data[col] / diameter
    data[f"{col}_velocity"] = data["Q_in"] / (np.pi * (diameter / 2) ** 2)

for i in range(len(level_cols) - 1):
    data[f"gradient_{i}"] = data[level_cols[i + 1]] - data[level_cols[i]]

velocity_cols = [c for c in data.columns if "velocity" in c]
for col in velocity_cols:
    data[col.replace("velocity", "acceleration")] = data[col].diff()

data["API"] = 0.0
for i in range(1, len(data)):
    data.loc[i, "API"] = (
        API_DECAY_FACTOR * data.loc[i - 1, "API"] + data.loc[i, "rain_avg"]
    )

data["event_active"] = (data["rain_avg"] > RAIN_EVENT_THRESHOLD).astype(int)
data["tau_event"] = 0
current_tau = 0
for i in range(len(data)):
    if data.loc[i, "event_active"] == 1:
        current_tau += 1
        data.loc[i, "tau_event"] = current_tau
    else:
        current_tau = 0

data["hour"] = data["time"].dt.hour
data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

data["day_of_week"] = data["time"].dt.dayofweek
for i in range(7):
    data[f"dow_{i}"] = (data["day_of_week"] == i).astype(int)

data["month"] = data["time"].dt.month
data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

feature_names = [c for c in data.columns if c != "time"]
save_features(data, feature_names)
