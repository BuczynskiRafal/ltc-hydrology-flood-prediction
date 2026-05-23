"""Backward-compatible aliases for raw-data loading helpers.

Prefer importing these from `src.load_raw_data` in new code.
"""

import pandas as pd

from src.load_raw_data import load_radar, load_rain_gauge, load_sensor_csv


def merge_sensor_parts(sensor_files):
    dfs = [load_sensor_csv(filepath) for filepath in sensor_files]
    merged = pd.concat(dfs, ignore_index=False)
    return merged.sort_values("time").reset_index(drop=True)


__all__ = ["load_sensor_csv", "load_rain_gauge", "load_radar", "merge_sensor_parts"]
