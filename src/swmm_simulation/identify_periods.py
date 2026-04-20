"""Build the SWMM rainfall schedule used for historical flood labeling."""

from pathlib import Path

import pandas as pd

from src.project_config import (
    DATA_INTERIM,
    SWMM_BUFFER_HOURS,
    SWMM_RAIN_FILE,
    SWMM_RAIN_PERCENTILE,
)

RAIN_DATA = Path(SWMM_RAIN_FILE)
OUTPUT_DIR = DATA_INTERIM
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rain_data = pd.read_csv(
    RAIN_DATA,
    sep=r"\s+",
    names=["station", "year", "month", "day", "hour", "minute", "rainfall_mm"],
    header=None,
)

rain_data["timestamp"] = pd.to_datetime(
    rain_data[["year", "month", "day", "hour", "minute"]]
)

rain_avg = rain_data.groupby("timestamp")["rainfall_mm"].mean().reset_index()
rain_avg.columns = ["timestamp", "rainfall_mm"]
rain_avg = rain_avg.sort_values("timestamp").reset_index(drop=True)

rain_avg["intensity_mm_h"] = (
    rain_avg["rainfall_mm"].rolling(window=5, min_periods=1).mean() * 60
)

rainfall_threshold = rain_avg["intensity_mm_h"].quantile(SWMM_RAIN_PERCENTILE / 100)
rain_avg["is_high_rainfall"] = rain_avg["intensity_mm_h"] > rainfall_threshold

high_rainfall_periods = rain_avg[rain_avg["is_high_rainfall"]].copy()

buffer = pd.Timedelta(hours=SWMM_BUFFER_HOURS)

windows = []
for ts in high_rainfall_periods["timestamp"]:
    windows.append({"start": ts - buffer, "end": ts + buffer})

windows_df = pd.DataFrame(windows).sort_values("start").reset_index(drop=True)

merged_windows = []
current_start = windows_df.iloc[0]["start"]
current_end = windows_df.iloc[0]["end"]

for idx in range(1, len(windows_df)):
    next_start = windows_df.iloc[idx]["start"]
    next_end = windows_df.iloc[idx]["end"]

    if next_start <= current_end:
        current_end = max(current_end, next_end)
    else:
        merged_windows.append({"start": current_start, "end": current_end})
        current_start = next_start
        current_end = next_end

merged_windows.append({"start": current_start, "end": current_end})

merged_windows_df = pd.DataFrame(merged_windows)

schedule = []

for idx, row in merged_windows_df.iterrows():
    schedule.append(
        {
            "start": row["start"],
            "end": row["end"],
            "timestep_min": 1,
            "priority": "high",
        }
    )

full_start = rain_avg["timestamp"].min()
full_end = rain_avg["timestamp"].max()

merged_windows_sorted = merged_windows_df.sort_values("start").reset_index(drop=True)

if merged_windows_sorted.iloc[0]["start"] > full_start:
    schedule.append(
        {
            "start": full_start,
            "end": merged_windows_sorted.iloc[0]["start"],
            "timestep_min": 5,
            "priority": "low",
        }
    )

for idx in range(len(merged_windows_sorted) - 1):
    gap_start = merged_windows_sorted.iloc[idx]["end"]
    gap_end = merged_windows_sorted.iloc[idx + 1]["start"]

    if gap_end > gap_start:
        schedule.append(
            {"start": gap_start, "end": gap_end, "timestep_min": 5, "priority": "low"}
        )

if merged_windows_sorted.iloc[-1]["end"] < full_end:
    schedule.append(
        {
            "start": merged_windows_sorted.iloc[-1]["end"],
            "end": full_end,
            "timestep_min": 5,
            "priority": "low",
        }
    )

schedule_df = pd.DataFrame(schedule).sort_values("start").reset_index(drop=True)

total_minutes = (full_end - full_start).total_seconds() / 60
high_res_minutes = sum(
    (row["end"] - row["start"]).total_seconds() / 60
    for _, row in schedule_df.iterrows()
    if row["priority"] == "high"
)
low_res_minutes = sum(
    (row["end"] - row["start"]).total_seconds() / 60
    for _, row in schedule_df.iterrows()
    if row["priority"] == "low"
)

schedule_df.to_csv(OUTPUT_DIR / "simulation_schedule.csv", index=False)
high_rainfall_periods.to_csv(OUTPUT_DIR / "high_rainfall_periods.csv", index=False)

summary = {
    "total_periods": len(schedule_df),
    "high_priority_periods": (schedule_df["priority"] == "high").sum(),
    "low_priority_periods": (schedule_df["priority"] == "low").sum(),
    "total_days": total_minutes / 60 / 24,
    "high_res_days": high_res_minutes / 60 / 24,
    "low_res_days": low_res_minutes / 60 / 24,
    "high_res_ratio_pct": 100 * high_res_minutes / total_minutes,
    "low_res_ratio_pct": 100 * low_res_minutes / total_minutes,
    "estimated_time_savings_pct": 100 * (low_res_minutes * 0.8) / total_minutes,
    "rainfall_threshold_90p": rainfall_threshold,
}

pd.DataFrame([summary]).to_csv(
    OUTPUT_DIR / "simulation_schedule_summary.csv", index=False
)
