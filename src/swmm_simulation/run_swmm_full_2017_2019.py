"""Run SWMM as a labeling oracle for the historical 2017-2019 periods."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import swmmio
    from pyswmm import Simulation
except ModuleNotFoundError as exc:  # pragma: no cover - exercised at runtime only
    swmmio = None
    Simulation = None
    SWMM_IMPORT_ERROR = exc
else:
    SWMM_IMPORT_ERROR = None

from src.project_config import DATA_INTERIM, SWMM_MODEL_FILE, SWMM_RAIN_FILE

OUTPUT_DIR = DATA_INTERIM
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SWMM_MODEL = Path(SWMM_MODEL_FILE)
SCHEDULE = DATA_INTERIM / "simulation_schedule.csv"
RAIN_DATA_PATH = Path(SWMM_RAIN_FILE).absolute()
TARGET_YEARS = (2017, 2018, 2019)

DEFAULT_ARTIFACTS_DIR = OUTPUT_DIR / "swmm_runs" / "historical_2017_2019"
PERIOD_ARTIFACTS_DIRNAME = "period_artifacts"
CHECKPOINTS_DIRNAME = "checkpoints"
RESUME_STATUS_FILENAME = "resume_status.json"
CHECKPOINT_INTERVAL = 20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run historical SWMM labeling simulations for 2017-2019."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint inside the artifacts directory.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help=(
            "Directory for per-period SWMM artifacts and resume checkpoints "
            f"(default: {DEFAULT_ARTIFACTS_DIR})."
        ),
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep per-period .inp/.out/.rpt files instead of cleaning them up.",
    )
    return parser.parse_args(argv)


def ensure_swmm_dependencies() -> None:
    if Simulation is not None and swmmio is not None:
        return
    raise ModuleNotFoundError(
        "run_swmm_full_2017_2019.py requires both `pyswmm` and `swmmio`. "
        "Install project requirements for the SWMM labeling stage first."
    ) from SWMM_IMPORT_ERROR


def resolve_artifacts_dir(artifacts_dir: str | Path | None = None) -> Path:
    resolved = (
        Path(artifacts_dir) if artifacts_dir is not None else DEFAULT_ARTIFACTS_DIR
    )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_period_artifact_paths(
    artifacts_dir: str | Path, period_num: int
) -> dict[str, Path]:
    artifacts_root = resolve_artifacts_dir(artifacts_dir)
    base = artifacts_root / PERIOD_ARTIFACTS_DIRNAME / f"period_{int(period_num):04d}"
    base.parent.mkdir(parents=True, exist_ok=True)
    return {
        "inp": base.with_suffix(".inp"),
        "out": base.with_suffix(".out"),
        "rpt": base.with_suffix(".rpt"),
    }


def resolve_checkpoints_dir(artifacts_dir: str | Path) -> Path:
    checkpoints_dir = resolve_artifacts_dir(artifacts_dir) / CHECKPOINTS_DIRNAME
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir


def resolve_resume_status_path(artifacts_dir: str | Path) -> Path:
    return resolve_checkpoints_dir(artifacts_dir) / RESUME_STATUS_FILENAME


def resolve_latest_checkpoint_path(artifacts_dir: str | Path, year: int) -> Path:
    return (
        resolve_checkpoints_dir(artifacts_dir)
        / f"swmm_floods_checkpoint_latest_{int(year)}.csv"
    )


def resolve_versioned_checkpoint_path(
    artifacts_dir: str | Path, year: int, period_num: int
) -> Path:
    return (
        resolve_checkpoints_dir(artifacts_dir)
        / f"swmm_floods_checkpoint_{int(year)}_period_{int(period_num):04d}.csv"
    )


def load_rain_data(rain_data_path: str | Path = RAIN_DATA_PATH) -> pd.DataFrame:
    rain = pd.read_csv(
        Path(rain_data_path),
        sep=r"\s+",
        names=["station", "year", "month", "day", "hour", "minute", "rainfall_mm"],
        header=None,
    )
    rain["timestamp"] = pd.to_datetime(rain[["year", "month", "day", "hour", "minute"]])
    rain = rain.set_index("timestamp").sort_index()
    rain["intensity_mm_h"] = rain["rainfall_mm"] * 60
    return rain


def load_periods(schedule_path: str | Path = SCHEDULE) -> pd.DataFrame:
    schedule = pd.read_csv(schedule_path)
    schedule["start"] = pd.to_datetime(schedule["start"])
    schedule["end"] = pd.to_datetime(schedule["end"])
    schedule["year"] = schedule["start"].dt.year

    periods = schedule[
        (schedule["year"].isin(TARGET_YEARS)) & (schedule["priority"] == "high")
    ].copy()
    return periods.sort_values("start").reset_index(drop=True)


def _deserialize_checkpoint(df: pd.DataFrame) -> list[dict[str, Any]]:
    if "period_start" in df.columns:
        df["period_start"] = pd.to_datetime(df["period_start"])
    if "period_end" in df.columns:
        df["period_end"] = pd.to_datetime(df["period_end"])
    return df.to_dict(orient="records")


def load_resume_checkpoint(
    artifacts_dir: str | Path, *, years: tuple[int, ...] = TARGET_YEARS
) -> tuple[int, dict[int, list[dict[str, Any]]]]:
    status_path = resolve_resume_status_path(artifacts_dir)
    last_completed_period = 0
    if status_path.exists():
        with open(status_path, "r") as handle:
            status = json.load(handle)
        last_completed_period = int(status.get("last_completed_period", 0))

    results_by_year = {int(year): [] for year in years}
    for year in years:
        checkpoint_path = resolve_latest_checkpoint_path(artifacts_dir, year)
        if not checkpoint_path.exists():
            continue
        checkpoint_df = pd.read_csv(checkpoint_path)
        results_by_year[int(year)] = _deserialize_checkpoint(checkpoint_df)

    return last_completed_period, results_by_year


def write_resume_checkpoint(
    artifacts_dir: str | Path,
    results_by_year: dict[int, list[dict[str, Any]]],
    *,
    period_num: int,
    total_periods: int,
    sim_start: float,
) -> None:
    resolve_checkpoints_dir(artifacts_dir)
    for year, results in results_by_year.items():
        if not results:
            continue
        checkpoint_df = pd.DataFrame(results)
        checkpoint_df.to_csv(
            resolve_latest_checkpoint_path(artifacts_dir, year), index=False
        )
        checkpoint_df.to_csv(
            resolve_versioned_checkpoint_path(artifacts_dir, year, period_num),
            index=False,
        )

    status = {
        "last_completed_period": int(period_num),
        "total_periods": int(total_periods),
        "updated_at_unix": float(time.time()),
        "elapsed_seconds": float(time.time() - sim_start),
    }
    with open(resolve_resume_status_path(artifacts_dir), "w") as handle:
        json.dump(status, handle, indent=2)


def cleanup_period_artifacts(period_paths: dict[str, Path]) -> None:
    for path in period_paths.values():
        path.unlink(missing_ok=True)


def prepare_period_input(
    period_paths: dict[str, Path],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timestep: int,
    rain_data_path: str | Path,
) -> None:
    period_paths["inp"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SWMM_MODEL, period_paths["inp"])

    with open(period_paths["inp"], "r") as handle:
        content = handle.read()

    content = content.replace(
        'FILE       "rg_bellinge_Jun2010_Aug2021.dat"',
        f'FILE       "{Path(rain_data_path).absolute()}"',
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

    with open(period_paths["inp"], "w") as handle:
        handle.write(content)


def build_period_flood_rows(
    *,
    model: Any,
    rain: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict[str, Any]]:
    flooding_summary = model.rpt.node_flooding_summary
    if flooding_summary is None or len(flooding_summary) == 0:
        return []

    floods = flooding_summary[flooding_summary["TotalFloodVol"] > 0]
    if len(floods) == 0:
        return []

    period_rain = rain.loc[start:end]
    if len(period_rain) > 0:
        max_intensity = float(period_rain["intensity_mm_h"].max())
        peak_idx = period_rain["intensity_mm_h"].idxmax()
        time_to_peak = float((peak_idx - start).total_seconds() / 60)
    else:
        max_intensity = 0.0
        time_to_peak = 0.0

    rows = []
    for node_id, row in floods.iterrows():
        rows.append(
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
    return rows


def run_single_period(
    *,
    period_num: int,
    period: pd.Series,
    rain: pd.DataFrame,
    artifacts_dir: str | Path,
    keep_artifacts: bool,
) -> tuple[int, list[dict[str, Any]]]:
    ensure_swmm_dependencies()

    start = pd.Timestamp(period["start"])
    end = pd.Timestamp(period["end"])
    timestep = int(period["timestep_min"])
    year = int(period["year"])

    period_paths = resolve_period_artifact_paths(artifacts_dir, period_num)
    prepare_period_input(
        period_paths,
        start=start,
        end=end,
        timestep=timestep,
        rain_data_path=RAIN_DATA_PATH,
    )

    try:
        with Simulation(str(period_paths["inp"])) as sim:
            for _ in sim:
                pass

        model = swmmio.Model(in_file_path=str(period_paths["inp"]))
        return year, build_period_flood_rows(
            model=model, rain=rain, start=start, end=end
        )
    finally:
        if not keep_artifacts:
            cleanup_period_artifacts(period_paths)


def save_final_outputs(results_by_year: dict[int, list[dict[str, Any]]]) -> None:
    for year in TARGET_YEARS:
        if not results_by_year[int(year)]:
            continue
        output_path = OUTPUT_DIR / f"swmm_floods_{int(year)}.csv"
        pd.DataFrame(results_by_year[int(year)]).to_csv(output_path, index=False)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    artifacts_dir = resolve_artifacts_dir(args.artifacts_dir)
    rain = load_rain_data()
    periods = load_periods()

    if args.resume:
        last_completed_period, results_by_year = load_resume_checkpoint(artifacts_dir)
    else:
        last_completed_period = 0
        results_by_year = {int(year): [] for year in TARGET_YEARS}

    sim_start = time.time()
    total_periods = len(periods)

    for period_num, (_, period) in enumerate(periods.iterrows(), 1):
        if period_num <= last_completed_period:
            continue

        year, flood_rows = run_single_period(
            period_num=period_num,
            period=period,
            rain=rain,
            artifacts_dir=artifacts_dir,
            keep_artifacts=args.keep_artifacts,
        )
        results_by_year[year].extend(flood_rows)

        if period_num % CHECKPOINT_INTERVAL == 0:
            write_resume_checkpoint(
                artifacts_dir,
                results_by_year,
                period_num=period_num,
                total_periods=total_periods,
                sim_start=sim_start,
            )

    write_resume_checkpoint(
        artifacts_dir,
        results_by_year,
        period_num=total_periods,
        total_periods=total_periods,
        sim_start=sim_start,
    )
    save_final_outputs(results_by_year)

    return {
        "artifacts_dir": str(artifacts_dir),
        "total_periods": int(total_periods),
        "last_completed_period": int(total_periods),
        "years_with_results": [
            int(year) for year, rows in results_by_year.items() if rows
        ],
        "keep_artifacts": bool(args.keep_artifacts),
    }


if __name__ == "__main__":
    main()
