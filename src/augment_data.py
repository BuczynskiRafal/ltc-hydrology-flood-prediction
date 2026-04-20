import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import swmmio
from pyswmm import Simulation

from src.logger import get_console_logger
from src.project_config import DATA_INTERIM

logger = get_console_logger(__name__)


SWMM_MODEL = Path("downloaded/7_SWMM/BellingeSWMM_v021_nopervious.inp")
RAIN_DATA_PATH = Path("downloaded/7_SWMM/rg_bellinge_Jun2010_Aug2021.dat")

TARGET_NEW_FLOODS = 500
MIN_FLOOD_VOLUME = 3.0
INTENSITY_THRESHOLD = 15.0
MIN_DURATION = 10

SCALING_FACTORS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
CHICAGO_DURATIONS = [20, 30, 45, 60, 90, 120, 150]
CHICAGO_INTENSITIES = [40, 60, 80, 100, 120, 150, 180]
UNIFORM_DURATIONS = [20, 30, 40, 60, 80]
UNIFORM_INTENSITIES = [40, 50, 70, 90, 120, 150]

OUTPUT_DIR = Path("data/interim/augmented")
TEMP_FILE_GLOBS = ("temp_*.inp", "temp_*.out", "temp_*.rpt")
SUMMARY_COLUMNS = [
    "idx",
    "period_start",
    "period_end",
    "total_flood_volume",
    "num_flooded_nodes",
    "peak_intensity",
    "duration_min",
    "method",
    "params",
]
FAILURE_PREVIEW_LIMIT = 5


class RainPattern:
    def __init__(self, timestamps, rainfall_mm, method, params):
        self.timestamps = timestamps
        self.rainfall_mm = rainfall_mm
        self.method = method
        self.params = params
        self.duration_min = len(timestamps)
        self.peak_intensity = (rainfall_mm * 60).max()


def generate_scaled_rain(period_start, period_end, rain_data, scale_factor):
    mask = (rain_data.index >= period_start) & (rain_data.index <= period_end)
    period_rain = rain_data.loc[mask].copy()
    scaled_rain = period_rain["rainfall_mm"] * scale_factor
    return RainPattern(
        period_rain.index, scaled_rain.values, "scaled", {"factor": scale_factor}
    )


def generate_chicago_storm(start_time, duration_min, peak_intensity_mmh):
    timestamps = pd.date_range(start_time, periods=duration_min, freq="1min")
    peak_pos = int(duration_min * 0.4)
    rising = np.linspace(0, peak_intensity_mmh, peak_pos)
    falling = np.linspace(peak_intensity_mmh, 0, duration_min - peak_pos)
    intensity_mmh = np.concatenate([rising, falling])
    rainfall_mm = intensity_mmh / 60.0
    return RainPattern(
        timestamps,
        rainfall_mm,
        "chicago",
        {"duration_min": duration_min, "peak_intensity_mmh": peak_intensity_mmh},
    )


def generate_uniform_rain(start_time, duration_min, intensity_mmh):
    timestamps = pd.date_range(start_time, periods=duration_min, freq="1min")
    rainfall_mm = np.full(duration_min, intensity_mmh / 60.0)
    return RainPattern(
        timestamps,
        rainfall_mm,
        "uniform",
        {"duration_min": duration_min, "intensity_mmh": intensity_mmh},
    )


def cleanup_stale_temp_files():
    for pattern in TEMP_FILE_GLOBS:
        for temp_file in OUTPUT_DIR.glob(pattern):
            temp_file.unlink(missing_ok=True)


def cleanup_temp_simulation_files(temp_inp):
    for temp_file in (
        temp_inp,
        temp_inp.with_suffix(".out"),
        temp_inp.with_suffix(".rpt"),
    ):
        temp_file.unlink(missing_ok=True)


def load_swmm_rain_data():
    rain = pd.read_csv(
        RAIN_DATA_PATH,
        sep=r"\s+",
        names=["station", "year", "month", "day", "hour", "minute", "rainfall_mm"],
        header=None,
    )
    rain["timestamp"] = pd.to_datetime(rain[["year", "month", "day", "hour", "minute"]])
    rain = rain.sort_values("timestamp")

    rain_max = rain.groupby("timestamp")["rainfall_mm"].max().reset_index()
    rain_max.columns = ["timestamp", "rainfall_mm"]
    rain_max = rain_max.set_index("timestamp").sort_index()
    return rain, rain_max


def load_augmentation_candidates():
    swmm_2017 = pd.read_csv(DATA_INTERIM / "swmm_floods_2017.csv")
    swmm_2017["period_start"] = pd.to_datetime(swmm_2017["period_start"])
    swmm_2017["period_end"] = pd.to_datetime(swmm_2017["period_end"])

    period_floods = (
        swmm_2017.groupby("period_start")
        .agg({"flood_volume_m3": "sum", "period_end": "first"})
        .reset_index()
    )
    dry_periods = period_floods[period_floods["flood_volume_m3"] < 1.0]

    return dry_periods.sample(n=min(len(dry_periods), 100), random_state=42)


def build_synthetic_patterns(candidates, rain_max):
    synthetic_patterns = []

    for _, row in candidates.iterrows():
        for scale in SCALING_FACTORS:
            pattern = generate_scaled_rain(
                row["period_start"], row["period_end"], rain_max, scale
            )
            if pattern.peak_intensity >= INTENSITY_THRESHOLD:
                synthetic_patterns.append(
                    (row["period_start"], row["period_end"], pattern)
                )

    for _, row in candidates.head(50).iterrows():
        for duration in CHICAGO_DURATIONS:
            for intensity in CHICAGO_INTENSITIES:
                pattern = generate_chicago_storm(
                    row["period_start"], duration, intensity
                )
                synthetic_patterns.append(
                    (row["period_start"], row["period_end"], pattern)
                )

    for _, row in candidates.head(30).iterrows():
        for duration in UNIFORM_DURATIONS:
            for intensity in UNIFORM_INTENSITIES:
                pattern = generate_uniform_rain(
                    row["period_start"], duration, intensity
                )
                synthetic_patterns.append(
                    (row["period_start"], row["period_end"], pattern)
                )

    return synthetic_patterns


def write_synthetic_rain_files(synthetic_patterns, rain):
    synthetic_rain_files = []

    for idx, (period_start, period_end, pattern) in enumerate(synthetic_patterns):
        rain_file = OUTPUT_DIR / f"synthetic_rain_{idx:04d}.dat"
        mask = (rain["timestamp"] >= period_start) & (rain["timestamp"] <= period_end)
        original_rain = rain[mask].copy()

        if len(original_rain) == 0:
            continue

        synthetic_df = pd.DataFrame(
            {"timestamp": pattern.timestamps, "synthetic_rainfall": pattern.rainfall_mm}
        )
        merged = original_rain.merge(synthetic_df, on="timestamp", how="left")
        merged["rainfall_mm"] = merged["synthetic_rainfall"].fillna(0)

        with open(rain_file, "w") as f:
            for _, row in merged.iterrows():
                f.write(
                    f"{row['station']} {row['year']} {row['month']} "
                    f"{row['day']} {row['hour']} {row['minute']} {row['rainfall_mm']}\n"
                )

        synthetic_rain_files.append(
            {
                "idx": idx,
                "rain_file": rain_file,
                "period_start": period_start,
                "period_end": period_end,
                "pattern": pattern,
            }
        )

    return synthetic_rain_files


def run_single_simulation(rain_info):
    idx = rain_info["idx"]
    rain_file = rain_info["rain_file"]
    start = rain_info["period_start"]
    end = rain_info["period_end"]
    pattern = rain_info["pattern"]

    temp_inp = OUTPUT_DIR / f"temp_{idx:04d}.inp"

    try:
        shutil.copy(SWMM_MODEL, temp_inp)

        with open(temp_inp, "r") as f:
            content = f.read()

        content = content.replace(
            'FILE       "rg_bellinge_Jun2010_Aug2021.dat"',
            f'FILE       "{rain_file.absolute()}"',
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

        with open(temp_inp, "w") as f:
            f.write(content)

        with Simulation(str(temp_inp)) as sim:
            for _ in sim:
                pass

        model = swmmio.Model(in_file_path=str(temp_inp))
        flooding_summary = model.rpt.node_flooding_summary

        if flooding_summary is None or len(flooding_summary) == 0:
            return None, None

        total_flood = flooding_summary["TotalFloodVol"].sum()
        if total_flood < MIN_FLOOD_VOLUME:
            return None, None

        return (
            {
                "idx": idx,
                "period_start": start,
                "period_end": end,
                "total_flood_volume": total_flood,
                "num_flooded_nodes": len(flooding_summary),
                "peak_intensity": pattern.peak_intensity,
                "duration_min": pattern.duration_min,
                "method": pattern.method,
                "params": pattern.params,
            },
            None,
        )
    except Exception as exc:
        if isinstance(exc, MemoryError):
            raise

        return (
            None,
            {
                "idx": idx,
                "method": pattern.method,
                "period_start": start,
                "period_end": end,
                "error": str(exc),
            },
        )
    finally:
        cleanup_temp_simulation_files(temp_inp)


def ensure_time_index(df):
    df = df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    else:
        df.index = pd.to_datetime(df.index)
        df.index.name = "time"

    return df.sort_index()


def build_augmented_sensor_data(successful_floods, sensor_data):
    sensor_with_time_index = ensure_time_index(sensor_data)
    augmented_records = []

    for flood in successful_floods:
        start, end = flood["period_start"], flood["period_end"]
        mask = (sensor_with_time_index.index >= start) & (
            sensor_with_time_index.index <= end
        )
        period_sensors = sensor_with_time_index.loc[mask].copy()
        if len(period_sensors) == 0:
            continue

        period_sensors["flash_flood"] = 1
        period_sensors["synthetic"] = True
        period_sensors["augmentation_method"] = flood["method"]
        augmented_records.append(period_sensors)

    if len(augmented_records) > 0:
        return pd.concat(augmented_records, axis=0).sort_index()

    empty = sensor_with_time_index.iloc[0:0].copy()
    empty["flash_flood"] = pd.Series(index=empty.index, dtype="int64")
    empty["synthetic"] = pd.Series(index=empty.index, dtype="bool")
    empty["augmentation_method"] = pd.Series(index=empty.index, dtype="object")
    return empty


def assemble_augmented_train(train_labeled, augmented_sensor_data, rain_data):
    train_original = train_labeled.copy()
    if "synthetic" not in train_original.columns:
        train_original["synthetic"] = False

    augmented_with_time = augmented_sensor_data.copy()
    if "time" not in augmented_with_time.columns:
        augmented_with_time = augmented_with_time.reset_index()
    augmented_with_time["time"] = pd.to_datetime(augmented_with_time["time"])

    rain_with_time_index = ensure_time_index(rain_data)
    augmented_merged = (
        augmented_with_time.set_index("time")
        .join(rain_with_time_index, how="left")
        .reset_index()
    )
    augmented_merged["synthetic"] = True
    augmented_merged["flash_flood"] = 1
    augmented_merged["target"] = 1

    final_cols = train_original.columns.tolist()
    for col in final_cols:
        if col not in augmented_merged.columns:
            augmented_merged[col] = np.nan

    train_combined = pd.concat(
        [train_original[final_cols], augmented_merged[final_cols]],
        axis=0,
        ignore_index=True,
    )
    train_combined["time"] = pd.to_datetime(train_combined["time"])
    train_combined["year"] = train_combined["time"].dt.year.astype(int)
    train_combined["flash_flood"] = train_combined["flash_flood"].fillna(0).astype(int)
    train_combined["target"] = train_combined["target"].fillna(0).astype(int)
    train_combined["synthetic"] = train_combined["synthetic"].fillna(False).astype(bool)
    return train_combined.sort_values("time").reset_index(drop=True)


def save_summary(successful_floods):
    pd.DataFrame(successful_floods, columns=SUMMARY_COLUMNS).to_csv(
        OUTPUT_DIR / "synthetic_floods_summary.csv", index=False
    )


def print_failure_summary(attempted_runs, successful_floods, failed_runs):
    logger.info(f"Attempted simulations: {attempted_runs}")
    logger.info(f"Successful synthetic floods: {len(successful_floods)}")
    logger.info(f"Failed simulations: {len(failed_runs)}")

    if len(failed_runs) == 0:
        return

    logger.info("Failure examples:")
    for failed_run in failed_runs[:FAILURE_PREVIEW_LIMIT]:
        logger.info(
            "  "
            f"idx={failed_run['idx']} "
            f"method={failed_run['method']} "
            f"start={failed_run['period_start']} "
            f"end={failed_run['period_end']} "
            f"error={failed_run['error']}"
        )


def run_augmentation():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    cleanup_stale_temp_files()

    rain, rain_max = load_swmm_rain_data()
    candidates = load_augmentation_candidates()
    train_labeled = pd.read_pickle(DATA_INTERIM / "train_labeled.pkl")
    sensor_data = pd.read_pickle(DATA_INTERIM / "sensor_cleaned.pkl")
    rain_cleaned = pd.read_pickle(DATA_INTERIM / "rain_cleaned.pkl")
    synthetic_patterns = build_synthetic_patterns(candidates, rain_max)
    synthetic_rain_files = write_synthetic_rain_files(synthetic_patterns, rain)
    successful_floods = []
    failed_runs = []
    attempted_runs = 0

    for rain_info in synthetic_rain_files[: TARGET_NEW_FLOODS * 3]:
        if len(successful_floods) >= TARGET_NEW_FLOODS:
            break

        attempted_runs += 1
        successful_flood, failed_run = run_single_simulation(rain_info)
        if successful_flood is not None:
            successful_floods.append(successful_flood)
        if failed_run is not None:
            failed_runs.append(failed_run)

    save_summary(successful_floods)
    print_failure_summary(attempted_runs, successful_floods, failed_runs)

    augmented_data = build_augmented_sensor_data(successful_floods, sensor_data)
    with open(OUTPUT_DIR / "augmented_sensor_data.pkl", "wb") as f:
        pickle.dump(augmented_data, f)

    train_augmented = assemble_augmented_train(
        train_labeled, augmented_data, rain_cleaned
    )
    with open(DATA_INTERIM / "train_labeled_augmented.pkl", "wb") as f:
        pickle.dump(train_augmented, f)

    logger.info(f"Augmented train rows: {len(train_augmented)}")
    logger.info(f"Synthetic rows: {int(train_augmented['synthetic'].sum())}")


if __name__ == "__main__":
    run_augmentation()
