import pickle

import numpy as np
import pandas as pd

from src.project_config import DATA_INTERIM

RAIN_THRESHOLD = 0.1
BUFFER_BEFORE = 120
BUFFER_AFTER = 180
SPLIT_FILES = {
    "train": "train_normalized.pkl",
    "val": "val_normalized.pkl",
    "test": "test_normalized.pkl",
}


def load_flash_flood_timesteps():
    swmm_periods = pd.read_csv(DATA_INTERIM / "swmm_floods_timestep_all_years.csv")
    swmm_periods["timestamp"] = pd.to_datetime(swmm_periods["timestamp"])
    swmm_periods["period_end"] = pd.to_datetime(swmm_periods["period_end"])
    flash_periods = swmm_periods[swmm_periods["flash_flood"] == 1]

    flood_timesteps = []
    for _, period in flash_periods.iterrows():
        period_times = pd.date_range(
            start=period["timestamp"], end=period["period_end"], freq="1min"
        )
        flood_timesteps.extend(period_times)

    return pd.DataFrame({"time": flood_timesteps, "flash_flood": 1}).drop_duplicates(
        subset=["time"]
    )


def build_event_mask(df, rain_col="rain_avg", rain_threshold=RAIN_THRESHOLD):
    keep_mask = np.zeros(len(df), dtype=bool)

    rain_indices = np.where(df[rain_col] > rain_threshold)[0]
    overflow_indices = np.where(df["target"] == 1)[0]

    for idx in rain_indices:
        start_idx = max(0, idx - BUFFER_BEFORE)
        end_idx = min(len(df), idx + BUFFER_AFTER + 1)
        keep_mask[start_idx:end_idx] = True

    for idx in overflow_indices:
        start_idx = max(0, idx - BUFFER_BEFORE)
        end_idx = min(len(df), idx + BUFFER_AFTER + 1)
        keep_mask[start_idx:end_idx] = True

    return keep_mask


def create_labeled_split(split_name, flood_timesteps):
    data = pd.read_pickle(DATA_INTERIM / SPLIT_FILES[split_name])
    data = data.sort_values("time").reset_index(drop=True)
    data = data.merge(flood_timesteps, on="time", how="left")
    data["flash_flood"] = data["flash_flood"].fillna(0).astype(int)
    data["target"] = data["flash_flood"]

    keep_mask = build_event_mask(data)
    return data[keep_mask].copy()


def save_labeled_split(split_name, data):
    with open(DATA_INTERIM / f"{split_name}_labeled.pkl", "wb") as f:
        pickle.dump(data, f)


def main():
    flood_timesteps = load_flash_flood_timesteps()

    for split_name in SPLIT_FILES:
        labeled_split = create_labeled_split(split_name, flood_timesteps)
        save_labeled_split(split_name, labeled_split)


if __name__ == "__main__":
    main()
