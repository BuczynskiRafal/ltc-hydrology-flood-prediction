"""Aggregate SWMM node floods into period-level flash-flood labels."""

from pathlib import Path

import pandas as pd

from src.project_config import (
    DATA_INTERIM,
    FLASH_FLOOD_MIN_DURATION,
    FLASH_FLOOD_MIN_INTENSITY,
    SWMM_RAIN_FILE,
)

OUTPUT_DIR = DATA_INTERIM
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAIN_DATA = Path(SWMM_RAIN_FILE)
TARGET_YEARS = (2017, 2018, 2019)

rain_raw = pd.read_csv(
    RAIN_DATA,
    sep=r"\s+",
    names=["station", "year", "month", "day", "hour", "minute", "rainfall_mm"],
    header=None,
)

rain_raw["timestamp"] = pd.to_datetime(
    rain_raw[["year", "month", "day", "hour", "minute"]]
)

rain_data = rain_raw.groupby("timestamp")["rainfall_mm"].max().reset_index()
rain_data.columns = ["timestamp", "rainfall_mm"]
rain_data = rain_data.sort_values("timestamp").set_index("timestamp")

rain_data["intensity_mm_h"] = rain_data["rainfall_mm"] * 60


def check_sustained_high_intensity(
    period_start,
    period_end,
    rain_data,
    threshold=FLASH_FLOOD_MIN_INTENSITY,
    min_duration=FLASH_FLOOD_MIN_DURATION,
):
    mask = (rain_data.index >= period_start) & (rain_data.index <= period_end)
    period_rain = rain_data.loc[mask, "intensity_mm_h"].copy()

    if len(period_rain) == 0:
        return False, 0

    high_intensity = (period_rain >= threshold).astype(int)

    changes = high_intensity.diff().fillna(0)
    run_starts = changes[changes == 1].index
    run_ends = changes[changes == -1].index

    if len(high_intensity) > 0 and high_intensity.iloc[0] == 1:
        run_starts = pd.Index([high_intensity.index[0]]).append(run_starts)

    if len(high_intensity) > 0 and high_intensity.iloc[-1] == 1:
        run_ends = run_ends.append(pd.Index([high_intensity.index[-1]]))

    max_consecutive = 0
    if len(run_starts) > 0 and len(run_ends) > 0:
        for start, end in zip(run_starts, run_ends):
            duration = ((end - start).total_seconds() / 60) + 1
            max_consecutive = max(max_consecutive, duration)

    is_flash_flood = max_consecutive >= min_duration
    return is_flash_flood, int(max_consecutive)


all_periods = []

for year in TARGET_YEARS:
    flood_file = OUTPUT_DIR / f"swmm_floods_{year}.csv"
    floods = pd.read_csv(flood_file)
    floods["period_start"] = pd.to_datetime(floods["period_start"])
    floods["period_end"] = pd.to_datetime(floods["period_end"])
    floods["year"] = year

    period_floods = (
        floods.groupby("period_start")
        .agg(
            {
                "period_end": "first",
                "flood_volume_m3": "sum",
                "node_id": "count",
                "max_rainfall_intensity": "first",
                "time_to_peak_min": "first",
            }
        )
        .reset_index()
    )

    period_floods.columns = [
        "timestamp",
        "period_end",
        "total_flood_volume",
        "num_flooded_nodes",
        "max_rainfall_intensity",
        "time_to_peak_min",
    ]

    flash_flood_results = []
    max_consecutive_list = []

    for idx, row in period_floods.iterrows():
        is_flash, max_consec = check_sustained_high_intensity(
            row["timestamp"],
            row["period_end"],
            rain_data,
        )
        flash_flood_results.append(is_flash)
        max_consecutive_list.append(max_consec)

    period_floods["flash_flood"] = flash_flood_results
    period_floods["max_consecutive_high_intensity_min"] = max_consecutive_list
    period_floods["year"] = year

    all_periods.append(period_floods)

combined = pd.concat(all_periods, ignore_index=True).sort_values("timestamp")

combined.to_csv(OUTPUT_DIR / "swmm_floods_timestep_all_years.csv", index=False)
