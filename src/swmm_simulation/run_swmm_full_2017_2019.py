"""Run SWMM as a labeling oracle for the historical 2017-2019 periods."""

import shutil
import time
from pathlib import Path

import pandas as pd
import swmmio
from pyswmm import Simulation

from src.project_config import DATA_INTERIM, SWMM_MODEL_FILE, SWMM_RAIN_FILE

OUTPUT_DIR = DATA_INTERIM
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SWMM_MODEL = Path(SWMM_MODEL_FILE)
SCHEDULE = DATA_INTERIM / "simulation_schedule.csv"
RAIN_DATA_PATH = Path(SWMM_RAIN_FILE).absolute()
TARGET_YEARS = (2017, 2018, 2019)

rain = pd.read_csv(
    RAIN_DATA_PATH,
    sep=r"\s+",
    names=["station", "year", "month", "day", "hour", "minute", "rainfall_mm"],
    header=None,
)
rain["timestamp"] = pd.to_datetime(rain[["year", "month", "day", "hour", "minute"]])
rain = rain.set_index("timestamp").sort_index()
rain["intensity_mm_h"] = rain["rainfall_mm"] * 60

schedule = pd.read_csv(SCHEDULE)
schedule["start"] = pd.to_datetime(schedule["start"])
schedule["end"] = pd.to_datetime(schedule["end"])
schedule["year"] = schedule["start"].dt.year

periods = schedule[
    (schedule["year"].isin(TARGET_YEARS)) & (schedule["priority"] == "high")
].copy()
periods = periods.sort_values("start").reset_index(drop=True)

results_by_year = {year: [] for year in TARGET_YEARS}
sim_start = time.time()

for period_num, (_, period) in enumerate(periods.iterrows(), 1):
    start, end, timestep, year = (
        period["start"],
        period["end"],
        period["timestep_min"],
        period["year"],
    )

    temp_inp = OUTPUT_DIR / f"temp_period_{period_num}.inp"
    shutil.copy(SWMM_MODEL, temp_inp)

    with open(temp_inp, "r") as f:
        content = f.read()

    content = content.replace(
        'FILE       "rg_bellinge_Jun2010_Aug2021.dat"', f'FILE       "{RAIN_DATA_PATH}"'
    )
    content = content.replace(
        "START_DATE           06/29/2012",
        f'START_DATE           {start.strftime("%m/%d/%Y")}',
    )
    content = content.replace(
        "END_DATE             06/30/2012",
        f'END_DATE             {end.strftime("%m/%d/%Y")}',
    )
    content = content.replace(
        "START_TIME           00:01:00",
        f'START_TIME           {start.strftime("%H:%M:%S")}',
    )
    content = content.replace(
        "END_TIME             23:59:00",
        f'END_TIME             {end.strftime("%H:%M:%S")}',
    )
    content = content.replace(
        "REPORT_STEP          00:01:00",
        f"REPORT_STEP          00:{int(timestep):02d}:00",
    )

    with open(temp_inp, "w") as f:
        f.write(content)

    with Simulation(str(temp_inp)) as sim:
        for step in sim:
            pass

    model = swmmio.Model(in_file_path=str(temp_inp))
    flooding_summary = model.rpt.node_flooding_summary

    period_rain = rain.loc[start:end]
    if len(period_rain) > 0:
        max_intensity = period_rain["intensity_mm_h"].max()
        peak_idx = period_rain["intensity_mm_h"].idxmax()
        time_to_peak = (peak_idx - start).total_seconds() / 60
    else:
        max_intensity = 0
        time_to_peak = 0

    if flooding_summary is not None and len(flooding_summary) > 0:
        floods = flooding_summary[flooding_summary["TotalFloodVol"] > 0]

        if len(floods) > 0:
            for node_id, row in floods.iterrows():
                results_by_year[year].append(
                    {
                        "period_start": start,
                        "period_end": end,
                        "node_id": node_id,
                        "flood_volume_m3": row["TotalFloodVol"],
                        "max_flood_rate": row.get("MaxRate", 0),
                        "flood_duration_hrs": row.get("MaxHr_Node_Flooding_Summary", 0),
                        "max_rainfall_intensity": max_intensity,
                        "time_to_peak_min": time_to_peak,
                    }
                )

    temp_inp.unlink()

    if period_num % 20 == 0:
        for yr in TARGET_YEARS:
            if results_by_year[yr]:
                checkpoint = (
                    OUTPUT_DIR / f"swmm_floods_checkpoint_{yr}_period_{period_num}.csv"
                )
                pd.DataFrame(results_by_year[yr]).to_csv(checkpoint, index=False)

for year in TARGET_YEARS:
    if results_by_year[year]:
        output = OUTPUT_DIR / f"swmm_floods_{year}.csv"
        pd.DataFrame(results_by_year[year]).to_csv(output, index=False)

for f in OUTPUT_DIR.glob("swmm_floods_checkpoint_*.csv"):
    f.unlink()
