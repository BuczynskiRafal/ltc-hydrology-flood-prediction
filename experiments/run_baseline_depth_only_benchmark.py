from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.final_model_comparison import (
    DEFAULT_PUMP_SENSOR,
    build_article_main_table,
    build_grouped_summary,
    build_model_summary_row,
    build_per_sensor_rows,
)
from src.evaluation.hydrological_metrics import compute_hydrological_metrics
from src.project_config import TARGET_SENSORS

DEFAULT_MODELS = ("gru", "lstm", "tcn", "mlp")
MODEL_TITLES = {
    "gru": "GRU Regression",
    "lstm": "LSTM Regression",
    "tcn": "TCN Regression",
    "mlp": "MLP Regression",
}
DEFAULT_CONFIGS = {
    "gru": Path("configs/baseline_depth_only/gru_depth_only_config.yaml"),
    "lstm": Path("configs/baseline_depth_only/lstm_depth_only_config.yaml"),
    "tcn": Path("configs/baseline_depth_only/tcn_depth_only_config.yaml"),
    "mlp": Path("configs/baseline_depth_only/mlp_depth_only_config.yaml"),
}
DEFAULT_MULTITASK_METRICS = {
    "gru": Path(
        "notebooks/article_materials_20260419/baselines/GRU/"
        "gru_test_20260416_121123_metrics.json"
    ),
    "lstm": Path(
        "notebooks/article_materials_20260419/baselines/LSTM/"
        "lstm_test_20260416_121317_metrics.json"
    ),
    "tcn": Path(
        "notebooks/article_materials_20260419/baselines/TCN/"
        "tcn_test_20260416_121518_metrics.json"
    ),
    "mlp": Path(
        "notebooks/article_materials_20260419/baselines/MLP/"
        "mlp_test_20260416_121641_metrics.json"
    ),
}
DEFAULT_LNN_GROUPED = Path(
    "notebooks/article_materials_20260419/benchmark/grouped_summary.csv"
)
DEFAULT_LNN_NSE = Path(
    "notebooks/article_materials_20260419/benchmark/nse_per_sensor_pivot.csv"
)
DEFAULT_LNN_RMSE = Path(
    "notebooks/article_materials_20260419/benchmark/rmse_per_sensor_pivot.csv"
)
DEFAULT_LNN_MAE = Path(
    "notebooks/article_materials_20260419/benchmark/mae_per_sensor_pivot.csv"
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate depth-only GRU/LSTM/TCN/MLP baselines, then "
            "build fair comparison tables against existing multitask baselines "
            "and depth-only LNN results."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=list(DEFAULT_MODELS),
        help="Baseline models to run. Defaults to all four baselines.",
    )
    parser.add_argument(
        "--output-dir",
        default="notebooks/article_materials_20260428/baseline_depth_only",
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse existing checkpoints from each depth-only config.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Reuse existing depth-only metrics JSON files under output-dir.",
    )
    parser.add_argument(
        "--lnn-label",
        default="dual-branch LNN",
        help="LNN depth-only row to use from the existing article benchmark.",
    )
    parser.add_argument(
        "--pump-sensor",
        default=DEFAULT_PUMP_SENSOR,
        help=f"Pump-only target sensor. Defaults to {DEFAULT_PUMP_SENSOR}.",
    )
    return parser.parse_args()


def copy_configs(models: list[str], output_dir: Path) -> None:
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for model_name in models:
        shutil.copy2(
            DEFAULT_CONFIGS[model_name],
            config_dir / DEFAULT_CONFIGS[model_name].name,
        )


def evaluate_depth_only_model(
    model_name: str,
    config_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    from experiments.eval_utils import evaluate_depths
    from experiments.regression_pipeline import (
        build_run_metadata,
        collect_predictions,
        count_parameters,
        create_test_loader,
        describe_model_architecture,
        describe_split_data_for_config,
        load_split_data_for_config,
        load_trained_model,
    )

    artifact = load_trained_model(model_name, config_path=config_path)
    runtime_config = artifact["runtime_config"]
    device = artifact["device"]
    split = "test"
    split_data = load_split_data_for_config(runtime_config, split)
    split_description = describe_split_data_for_config(runtime_config, split)
    test_loader = create_test_loader(runtime_config, split_data)
    predictions = collect_predictions(
        model_name, artifact["model"], test_loader, device
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    depth_metrics = evaluate_depths(
        predictions["true_depths"], predictions["pred_depths"]
    )
    hydro_metrics = compute_hydrological_metrics(
        predictions["true_depths"], predictions["pred_depths"]
    )
    metrics_payload = {
        "timestamp": timestamp,
        "evaluation_split": split,
        "task": "depth_only",
        "depth_metrics": {
            "aggregated": {
                key: float(value) for key, value in depth_metrics["aggregated"].items()
            },
            "per_sensor": {
                key: value.tolist()
                for key, value in depth_metrics["per_sensor"].items()
            },
        },
        "hydrological_metrics": {
            "aggregated": {
                key: float(value) for key, value in hydro_metrics["aggregated"].items()
            },
            "per_sensor": {
                key: value.tolist()
                for key, value in hydro_metrics["per_sensor"].items()
            },
        },
    }
    manifest_payload = {
        "model": model_name,
        "title": MODEL_TITLES[model_name],
        "task": "depth_only",
        "architecture": describe_model_architecture(model_name, runtime_config),
        "timestamp": timestamp,
        "canonical_config_path": str(artifact["canonical_config_path"]),
        "checkpoint_path": str(artifact["checkpoint_path"]),
        "config_source": artifact["config_source"],
        "use_reduced": bool(runtime_config["data"].get("use_reduced", True)),
        "input_size": int(runtime_config["model"]["input_size"]),
        "num_parameters": int(
            artifact["checkpoint"].get("n_params", count_parameters(artifact["model"]))
        ),
        "runtime_metadata": build_run_metadata(
            config=runtime_config,
            model_name=model_name,
            device=device,
            split_descriptions=[split_description],
        ),
    }

    model_dir = output_dir / "models" / model_name
    metrics_path = model_dir / f"{model_name}_depth_only_test_{timestamp}_metrics.json"
    manifest_path = (
        model_dir / f"{model_name}_depth_only_test_{timestamp}_manifest.json"
    )
    predictions_path = (
        output_dir
        / "predictions"
        / f"{model_name}_depth_only_test_{timestamp}_predictions.npz"
    )
    write_json(metrics_path, metrics_payload)
    write_json(manifest_path, manifest_payload)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(predictions_path, **predictions)
    return metrics_path, manifest_path, predictions_path


def find_existing_depth_only_metrics(output_dir: Path, model_name: str) -> Path:
    candidates = sorted(
        (output_dir / "models" / model_name).glob(
            f"{model_name}_depth_only_test_*_metrics.json"
        )
    )
    if not candidates:
        raise FileNotFoundError(
            f"No existing depth-only metrics found for {model_name} under {output_dir}."
        )
    return candidates[-1]


def load_metrics(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def build_depth_only_comparison(
    *,
    models: list[str],
    depth_only_metrics: dict[str, Path],
    output_dir: Path,
    pump_sensor: str,
    lnn_label: str,
) -> None:
    model_rows = []
    per_sensor_rows = []

    for model_name in models:
        label = f"{model_name.upper()} depth-only"
        payload = load_metrics(depth_only_metrics[model_name])
        model_rows.append(
            build_model_summary_row(label, depth_only_metrics[model_name], payload)
        )
        per_sensor_rows.extend(
            build_per_sensor_rows(label, payload, pump_sensor=pump_sensor)
        )

    model_summary = pd.DataFrame(model_rows).sort_values("agg_nse", ascending=False)
    per_sensor = pd.DataFrame(per_sensor_rows)
    grouped = build_grouped_summary(model_summary, per_sensor)
    article = build_article_main_table(grouped)

    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    model_summary.to_csv(comparison_dir / "depth_only_model_summary.csv", index=False)
    grouped.to_csv(comparison_dir / "depth_only_grouped_summary.csv", index=False)
    per_sensor.to_csv(comparison_dir / "depth_only_per_sensor_metrics.csv", index=False)
    article.to_csv(comparison_dir / "depth_only_article_main_table.csv", index=False)

    write_multitask_depth_lnn_tables(
        models=models,
        depth_only_grouped=grouped,
        depth_only_per_sensor=per_sensor,
        output_dir=comparison_dir,
        pump_sensor=pump_sensor,
        lnn_label=lnn_label,
    )


def _summary_rows_for_group(
    frame: pd.DataFrame, source: str, group: str, name_map: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    selected = frame[frame["group"] == group].copy()
    rows = []
    for _, row in selected.iterrows():
        model = str(row["model"])
        rows.append(
            {
                "model": name_map.get(model, model) if name_map else model,
                "source": source,
                "group": group,
                "NSE": float(row["NSE"]),
                "RMSE": float(row["RMSE"]),
                "MAE": float(row["MAE"]),
            }
        )
    return rows


def write_multitask_depth_lnn_tables(
    *,
    models: list[str],
    depth_only_grouped: pd.DataFrame,
    depth_only_per_sensor: pd.DataFrame,
    output_dir: Path,
    pump_sensor: str,
    lnn_label: str,
) -> None:
    multitask_rows = []
    multitask_per_sensor = []
    for model_name in models:
        payload = load_metrics(DEFAULT_MULTITASK_METRICS[model_name])
        label = model_name.upper()
        multitask_rows.append(
            build_model_summary_row(
                label,
                DEFAULT_MULTITASK_METRICS[model_name],
                payload,
            )
        )
        multitask_per_sensor.extend(
            build_per_sensor_rows(label, payload, pump_sensor=pump_sensor)
        )
    multitask_grouped = build_grouped_summary(
        pd.DataFrame(multitask_rows), pd.DataFrame(multitask_per_sensor)
    )

    existing_grouped = pd.read_csv(DEFAULT_LNN_GROUPED)
    lnn_grouped = existing_grouped[existing_grouped["model"] == lnn_label].copy()
    if lnn_grouped.empty:
        raise ValueError(f"LNN label '{lnn_label}' not found in {DEFAULT_LNN_GROUPED}.")

    source_rows = []
    name_map = {f"{model.upper()} depth-only": model.upper() for model in models}
    for group in ("all_sensors", "non_pump_mean", "pump_only"):
        source_rows.extend(
            _summary_rows_for_group(multitask_grouped, "baseline_multitask", group)
        )
        source_rows.extend(
            _summary_rows_for_group(
                depth_only_grouped, "baseline_depth_only", group, name_map=name_map
            )
        )
        source_rows.extend(
            _summary_rows_for_group(lnn_grouped, "lnn_depth_only", group)
        )

    summary = pd.DataFrame(source_rows)
    summary.to_csv(
        output_dir / "baseline_multitask_vs_depth_only_vs_lnn_grouped.csv",
        index=False,
    )

    deltas = []
    for model_name in models:
        model_label = model_name.upper()
        for group in ("all_sensors", "non_pump_mean", "pump_only"):
            multitask = summary[
                (summary["model"] == model_label)
                & (summary["source"] == "baseline_multitask")
                & (summary["group"] == group)
            ].iloc[0]
            depth_only = summary[
                (summary["model"] == model_label)
                & (summary["source"] == "baseline_depth_only")
                & (summary["group"] == group)
            ].iloc[0]
            deltas.append(
                {
                    "model": model_label,
                    "group": group,
                    "multitask_NSE": float(multitask["NSE"]),
                    "depth_only_NSE": float(depth_only["NSE"]),
                    "delta_NSE": float(depth_only["NSE"] - multitask["NSE"]),
                    "multitask_RMSE": float(multitask["RMSE"]),
                    "depth_only_RMSE": float(depth_only["RMSE"]),
                    "delta_RMSE": float(depth_only["RMSE"] - multitask["RMSE"]),
                    "multitask_MAE": float(multitask["MAE"]),
                    "depth_only_MAE": float(depth_only["MAE"]),
                    "delta_MAE": float(depth_only["MAE"] - multitask["MAE"]),
                }
            )
    pd.DataFrame(deltas).to_csv(
        output_dir / "baseline_depth_only_minus_multitask_deltas.csv",
        index=False,
    )
    write_improvement_assessment(pd.DataFrame(deltas), output_dir)

    write_per_sensor_comparison(
        models=models,
        multitask_per_sensor=pd.DataFrame(multitask_per_sensor),
        depth_only_per_sensor=depth_only_per_sensor,
        output_dir=output_dir,
        lnn_label=lnn_label,
    )


def write_per_sensor_comparison(
    *,
    models: list[str],
    multitask_per_sensor: pd.DataFrame,
    depth_only_per_sensor: pd.DataFrame,
    output_dir: Path,
    lnn_label: str,
) -> None:
    lnn_pivots = {
        "NSE": pd.read_csv(DEFAULT_LNN_NSE).set_index("model"),
        "RMSE": pd.read_csv(DEFAULT_LNN_RMSE).set_index("model"),
        "MAE": pd.read_csv(DEFAULT_LNN_MAE).set_index("model"),
    }
    rows = []
    for model_name in models:
        model_label = model_name.upper()
        depth_label = f"{model_label} depth-only"
        for sensor in TARGET_SENSORS:
            multitask = multitask_per_sensor[
                (multitask_per_sensor["model"] == model_label)
                & (multitask_per_sensor["sensor"] == sensor)
            ].iloc[0]
            depth_only = depth_only_per_sensor[
                (depth_only_per_sensor["model"] == depth_label)
                & (depth_only_per_sensor["sensor"] == sensor)
            ].iloc[0]
            rows.append(
                {
                    "model": model_label,
                    "sensor": sensor,
                    "multitask_NSE": float(multitask["NSE"]),
                    "depth_only_NSE": float(depth_only["NSE"]),
                    "lnn_depth_only_NSE": float(
                        lnn_pivots["NSE"].loc[lnn_label, sensor]
                    ),
                    "multitask_RMSE": float(multitask["RMSE"]),
                    "depth_only_RMSE": float(depth_only["RMSE"]),
                    "lnn_depth_only_RMSE": float(
                        lnn_pivots["RMSE"].loc[lnn_label, sensor]
                    ),
                    "multitask_MAE": float(multitask["MAE"]),
                    "depth_only_MAE": float(depth_only["MAE"]),
                    "lnn_depth_only_MAE": float(
                        lnn_pivots["MAE"].loc[lnn_label, sensor]
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(
        output_dir / "baseline_multitask_vs_depth_only_vs_lnn_per_sensor.csv",
        index=False,
    )


def write_improvement_assessment(deltas: pd.DataFrame, output_dir: Path) -> None:
    all_sensors = deltas[deltas["group"] == "all_sensors"].copy()
    substantial_threshold = 0.02
    all_sensors["substantial_NSE_improvement"] = (
        all_sensors["delta_NSE"] >= substantial_threshold
    )
    gru_rows = all_sensors[all_sensors["model"] == "GRU"]
    payload = {
        "substantial_delta_nse_threshold": substantial_threshold,
        "all_sensors": all_sensors.to_dict(orient="records"),
        "gru": None if gru_rows.empty else gru_rows.iloc[0].to_dict(),
    }
    write_json(output_dir / "improvement_assessment.json", payload)


def write_run_manifest(
    *,
    output_dir: Path,
    models: list[str],
    metrics_paths: dict[str, Path],
    predictions_paths: dict[str, Path],
    skipped_train: bool,
    skipped_eval: bool,
) -> None:
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "task": "baseline_depth_only_benchmark",
        "models": models,
        "objective": {
            "depth_weight": 1.0,
            "overflow_weight": 0.0,
            "flood_weight": 1.0,
        },
        "config_paths": {
            model_name: str(DEFAULT_CONFIGS[model_name]) for model_name in models
        },
        "metrics_paths": {
            model_name: str(path) for model_name, path in metrics_paths.items()
        },
        "predictions_paths": {
            model_name: str(path) for model_name, path in predictions_paths.items()
        },
        "skipped_train": skipped_train,
        "skipped_eval": skipped_eval,
    }
    write_json(output_dir / "run_manifest.json", payload)


def main() -> None:
    args = parse_args()
    models = list(args.models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_configs(models, output_dir)

    if not args.skip_train:
        from experiments.regression_pipeline import train_configured_model

        for model_name in models:
            train_configured_model(model_name, config_path=DEFAULT_CONFIGS[model_name])

    metrics_paths: dict[str, Path] = {}
    predictions_paths: dict[str, Path] = {}
    if args.skip_eval:
        for model_name in models:
            metrics_paths[model_name] = find_existing_depth_only_metrics(
                output_dir, model_name
            )
    else:
        for model_name in models:
            metrics_path, _manifest_path, predictions_path = evaluate_depth_only_model(
                model_name,
                DEFAULT_CONFIGS[model_name],
                output_dir,
            )
            metrics_paths[model_name] = metrics_path
            predictions_paths[model_name] = predictions_path

    build_depth_only_comparison(
        models=models,
        depth_only_metrics=metrics_paths,
        output_dir=output_dir,
        pump_sensor=args.pump_sensor,
        lnn_label=args.lnn_label,
    )
    write_run_manifest(
        output_dir=output_dir,
        models=models,
        metrics_paths=metrics_paths,
        predictions_paths=predictions_paths,
        skipped_train=args.skip_train,
        skipped_eval=args.skip_eval,
    )

    print(f"Saved baseline depth-only benchmark under: {output_dir}")


if __name__ == "__main__":
    main()
