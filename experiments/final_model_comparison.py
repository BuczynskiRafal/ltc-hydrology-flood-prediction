from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.project_config import TARGET_SENSORS

DEFAULT_PUMP_SENSOR = "G80F13P_LevelPS"


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build final per-sensor and grouped comparison tables from saved "
            "metrics.json artifacts."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Model label and metrics path in the form "
            "'Label=/abs/or/relative/path/to/metrics.json'. Repeat for each model."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where CSV/JSON comparison artifacts will be written.",
    )
    parser.add_argument(
        "--copy-to",
        default=None,
        help="Optional directory where generated artifacts will also be copied.",
    )
    parser.add_argument(
        "--pump-sensor",
        default=DEFAULT_PUMP_SENSOR,
        help=f"Sensor treated as pump-only. Defaults to {DEFAULT_PUMP_SENSOR}.",
    )
    return parser.parse_args()


def parse_model_specs(specs: list[str]) -> list[dict[str, str]]:
    parsed_specs = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --model value '{spec}'. Expected 'Label=/path/to/metrics.json'."
            )
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        metrics_path = Path(raw_path.strip())
        if not label:
            raise ValueError(f"Missing model label in --model '{spec}'.")
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file does not exist: {metrics_path}")
        parsed_specs.append({"label": label, "metrics_path": str(metrics_path)})
    return parsed_specs


def load_metrics_payload(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as handle:
        payload = json.load(handle)
    required = {"timestamp", "depth_metrics"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    return payload


def _safe_get(mapping: dict[str, Any], *keys: str) -> float:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return math.nan
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return math.nan


def _infer_manifest_path(metrics_path: Path) -> str | None:
    candidate = Path(str(metrics_path).replace("_metrics.json", "_manifest.json"))
    return str(candidate) if candidate.exists() else None


def build_model_summary_row(
    label: str, metrics_path: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    return {
        "model": label,
        "timestamp": payload.get("timestamp"),
        "metrics_path": str(metrics_path),
        "manifest_path": _infer_manifest_path(metrics_path),
        "agg_nse": _safe_get(payload, "depth_metrics", "aggregated", "NSE"),
        "agg_rmse": _safe_get(payload, "depth_metrics", "aggregated", "RMSE"),
        "agg_mae": _safe_get(payload, "depth_metrics", "aggregated", "MAE"),
        "agg_peak_flow_error": _safe_get(
            payload, "hydrological_metrics", "aggregated", "Peak_Flow_Error"
        ),
        "agg_time_to_peak_error": _safe_get(
            payload, "hydrological_metrics", "aggregated", "Time_to_Peak_Error"
        ),
        "agg_volume_error": _safe_get(
            payload, "hydrological_metrics", "aggregated", "Volume_Error"
        ),
        "agg_lag_time": _safe_get(
            payload, "hydrological_metrics", "aggregated", "Lag_Time"
        ),
        "overflow_f1_global": _safe_get(payload, "overflow_metrics", "F1"),
        "overflow_roc_auc_global": _safe_get(payload, "overflow_metrics", "ROC-AUC"),
    }


def build_per_sensor_rows(
    label: str,
    payload: dict[str, Any],
    *,
    pump_sensor: str,
) -> list[dict[str, Any]]:
    depth = payload["depth_metrics"]["per_sensor"]
    hydro = payload.get("hydrological_metrics", {}).get("per_sensor", {})
    rows = []
    for sensor_index, sensor_name in enumerate(TARGET_SENSORS):
        rows.append(
            {
                "model": label,
                "sensor": sensor_name,
                "group": "pump_only" if sensor_name == pump_sensor else "non_pump_mean",
                "NSE": float(depth["NSE"][sensor_index]),
                "RMSE": float(depth["RMSE"][sensor_index]),
                "MAE": float(depth["MAE"][sensor_index]),
                "Peak_Flow_Error": float(
                    hydro.get("Peak_Flow_Error", [math.nan] * len(TARGET_SENSORS))[
                        sensor_index
                    ]
                ),
                "Time_to_Peak_Error": float(
                    hydro.get("Time_to_Peak_Error", [math.nan] * len(TARGET_SENSORS))[
                        sensor_index
                    ]
                ),
                "Volume_Error": float(
                    hydro.get("Volume_Error", [math.nan] * len(TARGET_SENSORS))[
                        sensor_index
                    ]
                ),
                "Lag_Time": float(
                    hydro.get("Lag_Time", [math.nan] * len(TARGET_SENSORS))[
                        sensor_index
                    ]
                ),
            }
        )
    return rows


def build_grouped_summary(
    model_summary: pd.DataFrame, per_sensor: pd.DataFrame
) -> pd.DataFrame:
    group_metrics = [
        "NSE",
        "RMSE",
        "MAE",
        "Peak_Flow_Error",
        "Time_to_Peak_Error",
        "Volume_Error",
        "Lag_Time",
    ]
    grouped_rows: list[dict[str, Any]] = []

    for _, row in model_summary.iterrows():
        grouped_rows.append(
            {
                "model": row["model"],
                "group": "all_sensors",
                "n_sensors": len(TARGET_SENSORS),
                "NSE": row["agg_nse"],
                "RMSE": row["agg_rmse"],
                "MAE": row["agg_mae"],
                "Peak_Flow_Error": row["agg_peak_flow_error"],
                "Time_to_Peak_Error": row["agg_time_to_peak_error"],
                "Volume_Error": row["agg_volume_error"],
                "Lag_Time": row["agg_lag_time"],
            }
        )

    for (model, group), frame in per_sensor.groupby(["model", "group"], sort=False):
        grouped_rows.append(
            {
                "model": model,
                "group": group,
                "n_sensors": int(len(frame)),
                **{
                    metric: float(frame[metric].mean(skipna=True))
                    for metric in group_metrics
                },
            }
        )

    grouped = pd.DataFrame(grouped_rows)
    return grouped.sort_values(["group", "NSE"], ascending=[True, False]).reset_index(
        drop=True
    )


def _format_markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            "| "
            + " | ".join(_format_markdown_cell(row[column]) for column in columns)
            + " |"
        )
    return "\n".join([header, separator, *rows])


def build_publication_tables(
    model_summary: pd.DataFrame,
    grouped: pd.DataFrame,
    per_sensor: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    overall = model_summary[
        [
            "model",
            "agg_nse",
            "agg_rmse",
            "agg_mae",
            "agg_peak_flow_error",
            "agg_time_to_peak_error",
            "agg_volume_error",
        ]
    ].rename(
        columns={
            "agg_nse": "NSE",
            "agg_rmse": "RMSE",
            "agg_mae": "MAE",
            "agg_peak_flow_error": "Peak_Flow_Error",
            "agg_time_to_peak_error": "Time_to_Peak_Error",
            "agg_volume_error": "Volume_Error",
        }
    )

    grouped_publication = grouped[
        [
            "model",
            "group",
            "n_sensors",
            "NSE",
            "RMSE",
            "MAE",
            "Peak_Flow_Error",
            "Time_to_Peak_Error",
            "Volume_Error",
        ]
    ]

    per_sensor_nse = per_sensor.pivot(index="model", columns="sensor", values="NSE")
    per_sensor_rmse = per_sensor.pivot(index="model", columns="sensor", values="RMSE")
    per_sensor_mae = per_sensor.pivot(index="model", columns="sensor", values="MAE")

    return {
        "overall": overall.reset_index(drop=True),
        "grouped": grouped_publication.reset_index(drop=True),
        "per_sensor_nse": per_sensor_nse.reset_index(),
        "per_sensor_rmse": per_sensor_rmse.reset_index(),
        "per_sensor_mae": per_sensor_mae.reset_index(),
    }


def build_article_main_table(grouped: pd.DataFrame) -> pd.DataFrame:
    overall = (
        grouped[grouped["group"] == "all_sensors"]
        .set_index("model")
        .rename(
            columns={
                "NSE": "all_nse",
                "RMSE": "all_rmse",
                "MAE": "all_mae",
                "Peak_Flow_Error": "all_peak_flow_error",
            }
        )[["all_nse", "all_rmse", "all_mae", "all_peak_flow_error"]]
    )
    non_pump = (
        grouped[grouped["group"] == "non_pump_mean"]
        .set_index("model")
        .rename(
            columns={
                "NSE": "non_pump_nse",
                "RMSE": "non_pump_rmse",
                "MAE": "non_pump_mae",
            }
        )[["non_pump_nse", "non_pump_rmse", "non_pump_mae"]]
    )
    pump_only = (
        grouped[grouped["group"] == "pump_only"]
        .set_index("model")
        .rename(
            columns={
                "NSE": "pump_nse",
                "RMSE": "pump_rmse",
                "MAE": "pump_mae",
                "Peak_Flow_Error": "pump_peak_flow_error",
                "Volume_Error": "pump_volume_error",
            }
        )[
            [
                "pump_nse",
                "pump_rmse",
                "pump_mae",
                "pump_peak_flow_error",
                "pump_volume_error",
            ]
        ]
    )
    article = overall.join(non_pump, how="left").join(pump_only, how="left")
    article = article.reset_index().rename(columns={"index": "model"})
    return article.sort_values("all_nse", ascending=False).reset_index(drop=True)


def build_discussion_note(
    model_summary: pd.DataFrame,
    grouped: pd.DataFrame,
    per_sensor: pd.DataFrame,
    *,
    pump_sensor: str,
) -> str:
    overall = grouped[grouped["group"] == "all_sensors"].sort_values(
        "NSE", ascending=False
    )
    non_pump = grouped[grouped["group"] == "non_pump_mean"].sort_values(
        "NSE", ascending=False
    )
    pump_only = grouped[grouped["group"] == "pump_only"].sort_values(
        "NSE", ascending=False
    )

    best_overall = overall.iloc[0]
    best_non_pump = non_pump.iloc[0]
    best_pump = pump_only.iloc[0]

    pump_gap = float(best_non_pump["NSE"]) - float(best_pump["NSE"])

    stable_candidates = grouped[
        (grouped["group"] == "pump_only")
        & grouped["model"].str.contains("stable", case=False, regex=False)
    ]
    pump_delta_sentence = ""
    if not stable_candidates.empty:
        stable_pump = stable_candidates.iloc[0]
        pump_delta = float(best_pump["NSE"]) - float(stable_pump["NSE"])
        pump_delta_sentence = (
            f" Relative to the stable reference, the best pump-focused variant "
            f"improves `{pump_sensor}` by `{pump_delta:.4f}` NSE."
        )

    head_only_candidates = grouped[
        (grouped["group"] == "pump_only")
        & grouped["model"].str.contains("head", case=False, regex=False)
    ]
    head_only_sentence = ""
    if not head_only_candidates.empty:
        best_head_only = head_only_candidates.sort_values("NSE", ascending=False).iloc[
            0
        ]
        if best_head_only["model"] != best_pump["model"]:
            head_only_sentence = (
                f" A head-only pump specialization reaches `{best_head_only['NSE']:.4f}` "
                "NSE on the pump sensor but does not match the best pump-aware variant, "
                "which indicates that the control signal must enter the temporal "
                "representation rather than only the final readout."
            )

    pump_row = per_sensor[per_sensor["sensor"] == pump_sensor].sort_values(
        "NSE", ascending=False
    )
    pump_best_hydro_sentence = ""
    if not pump_row.empty and "Volume_Error" in pump_row.columns:
        best_pump_sensor = pump_row.iloc[0]
        pump_best_hydro_sentence = (
            f" For `{pump_sensor}`, the best recorded configuration reaches "
            f"`NSE={best_pump_sensor['NSE']:.4f}`, `RMSE={best_pump_sensor['RMSE']:.4f}`, "
            f"`MAE={best_pump_sensor['MAE']:.4f}`, and "
            f"`Volume_Error={best_pump_sensor['Volume_Error']:.4f}`."
        )

    return "\n".join(
        [
            "## Discussion: Hybrid Pump Dynamics",
            "",
            (
                f"The grouped evaluation distinguishes the four passive / gravitational "
                f"sensors from the pump-controlled sensor `{pump_sensor}`. This decomposition "
                f"shows that the aggregate error is not caused by uniformly weak performance "
                f"across the network. Instead, the best non-pump mean reaches "
                f"`{best_non_pump['NSE']:.4f}` NSE (`{best_non_pump['model']}`), whereas the "
                f"best pump-only score reaches only `{best_pump['NSE']:.4f}` NSE "
                f"(`{best_pump['model']}`). The resulting gap of `{pump_gap:.4f}` NSE indicates "
                "that the pump sensor remains the principal source of residual error."
            ),
            "",
            (
                f"The best aggregate model is `{best_overall['model']}` with "
                f"`NSE={best_overall['NSE']:.4f}` across all sensors."
                f"{pump_delta_sentence}"
            ),
            "",
            (
                "This pattern is consistent with a hybrid-dynamics interpretation of the "
                "pump station. The passive sensors are dominated by smoother storage and "
                "conveyance processes, which are well aligned with the continuous-time "
                "inductive bias of the liquid architecture. By contrast, the pump sensor is "
                "affected by thresholded on/off control and hysteretic switching, which "
                "introduce discrete state transitions on top of the underlying continuous "
                "hydraulic response. Explicit pump-state features therefore improve the model, "
                "but they do not remove the performance gap entirely."
                f"{head_only_sentence}"
            ),
            "",
            pump_best_hydro_sentence,
            "",
            (
                "For manuscript reporting, the main benchmark should therefore separate "
                "`all_sensors`, `non_pump_mean`, and `pump_only`, rather than relying solely "
                "on a single aggregate score. This framing makes clear that the liquid model "
                "is highly competitive on the passive sensors, while the remaining error is "
                "concentrated in the pump-controlled regime. If one additional architectural "
                "experiment is pursued, the most defensible next step is a dual-branch pump "
                "encoder that injects the pump-control signal into a dedicated temporal "
                "sub-encoder, rather than another round of broad loss tuning or additional "
                "global feature engineering."
            ),
        ]
    ).strip()


def build_publication_report(
    model_summary: pd.DataFrame,
    grouped: pd.DataFrame,
    per_sensor: pd.DataFrame,
    *,
    pump_sensor: str,
) -> str:
    tables = build_publication_tables(model_summary, grouped, per_sensor)
    article_main_table = build_article_main_table(grouped)
    discussion = build_discussion_note(
        model_summary,
        grouped,
        per_sensor,
        pump_sensor=pump_sensor,
    )
    sections = [
        "# Final Model Comparison",
        "",
        "## Overall Summary",
        "",
        dataframe_to_markdown(tables["overall"]),
        "",
        "## Grouped Summary",
        "",
        dataframe_to_markdown(tables["grouped"]),
        "",
        "## Article Main Table",
        "",
        dataframe_to_markdown(article_main_table),
        "",
        "## Per-Sensor NSE",
        "",
        dataframe_to_markdown(tables["per_sensor_nse"]),
        "",
        "## Per-Sensor RMSE",
        "",
        dataframe_to_markdown(tables["per_sensor_rmse"]),
        "",
        "## Per-Sensor MAE",
        "",
        dataframe_to_markdown(tables["per_sensor_mae"]),
        "",
        discussion,
        "",
    ]
    return "\n".join(sections)


def copy_outputs(output_paths: list[Path], copy_to: Path) -> None:
    copy_to.mkdir(parents=True, exist_ok=True)
    for path in output_paths:
        destination = copy_to / path.name
        shutil.copy2(path, destination)
        print(f"copied -> {destination}")


def main() -> None:
    args = parse_args()
    model_specs = parse_model_specs(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog_rows = []
    model_summary_rows = []
    per_sensor_rows = []

    for spec in model_specs:
        metrics_path = Path(spec["metrics_path"])
        payload = load_metrics_payload(metrics_path)
        catalog_rows.append(
            {
                "model": spec["label"],
                "timestamp": payload.get("timestamp"),
                "metrics_path": str(metrics_path),
                "manifest_path": _infer_manifest_path(metrics_path),
            }
        )
        model_summary_rows.append(
            build_model_summary_row(spec["label"], metrics_path, payload)
        )
        per_sensor_rows.extend(
            build_per_sensor_rows(spec["label"], payload, pump_sensor=args.pump_sensor)
        )

    catalog_df = pd.DataFrame(catalog_rows)
    model_summary_df = pd.DataFrame(model_summary_rows).sort_values(
        "agg_nse", ascending=False
    )
    per_sensor_df = pd.DataFrame(per_sensor_rows)
    grouped_df = build_grouped_summary(model_summary_df, per_sensor_df)
    article_main_table_df = build_article_main_table(grouped_df)

    nse_pivot = per_sensor_df.pivot(index="model", columns="sensor", values="NSE")
    rmse_pivot = per_sensor_df.pivot(index="model", columns="sensor", values="RMSE")
    mae_pivot = per_sensor_df.pivot(index="model", columns="sensor", values="MAE")

    summary_payload = {
        "pump_sensor": args.pump_sensor,
        "models": catalog_rows,
        "leaderboard": model_summary_df[
            ["model", "agg_nse", "agg_rmse", "agg_mae", "agg_peak_flow_error"]
        ].to_dict(orient="records"),
    }

    outputs = {
        "model_catalog.csv": catalog_df,
        "model_summary.csv": model_summary_df,
        "grouped_summary.csv": grouped_df,
        "article_main_table.csv": article_main_table_df,
        "per_sensor_metrics.csv": per_sensor_df,
        "nse_per_sensor_pivot.csv": nse_pivot.reset_index(),
        "rmse_per_sensor_pivot.csv": rmse_pivot.reset_index(),
        "mae_per_sensor_pivot.csv": mae_pivot.reset_index(),
    }

    output_paths: list[Path] = []
    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        output_paths.append(path)

    summary_path = output_dir / "comparison_summary.json"
    write_json(summary_path, summary_payload)
    output_paths.append(summary_path)

    publication_report_path = output_dir / "publication_report.md"
    publication_report_path.write_text(
        build_publication_report(
            model_summary_df,
            grouped_df,
            per_sensor_df,
            pump_sensor=args.pump_sensor,
        )
    )
    output_paths.append(publication_report_path)

    article_main_table_path = output_dir / "article_main_table.md"
    article_main_table_path.write_text(dataframe_to_markdown(article_main_table_df))
    output_paths.append(article_main_table_path)

    discussion_path = output_dir / "discussion_pump_dynamics.md"
    discussion_path.write_text(
        build_discussion_note(
            model_summary_df,
            grouped_df,
            per_sensor_df,
            pump_sensor=args.pump_sensor,
        )
    )
    output_paths.append(discussion_path)

    if args.copy_to is not None:
        copy_outputs(output_paths, Path(args.copy_to))

    print("Saved locally:")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
