import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.eval_utils import evaluate_depths  # noqa: E402
from experiments.regression_pipeline import (  # noqa: E402
    MODEL_TITLES,
    build_run_metadata,
    collect_predictions,
    count_parameters,
    create_test_loader,
    describe_model_architecture,
    describe_split_data_for_config,
    load_split_data_for_config,
    load_trained_model,
)
from src.evaluation.hydrological_metrics import (
    compute_hydrological_metrics,  # noqa: E402
)
from src.release_utils import write_json  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a regression model on depth-only metrics and save JSON/NPZ artifacts."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["gru", "lstm", "tcn", "mlp", "lnn"],
        help="Model family to evaluate.",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Optional config path."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Optional checkpoint path."
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory for metrics and manifest JSON files.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Optional directory for predictions .npz. Defaults to <results-dir>/../predictions.",
    )
    parser.add_argument(
        "--artifact-prefix",
        type=str,
        default=None,
        help="Filename prefix for saved artifacts. Defaults to the model name.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    artifact = load_trained_model(
        args.model,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )
    runtime_config = artifact["runtime_config"]
    device = artifact["device"]
    split_data = load_split_data_for_config(runtime_config, args.split)
    split_description = describe_split_data_for_config(runtime_config, args.split)
    split_loader = create_test_loader(runtime_config, split_data)
    predictions = collect_predictions(
        args.model,
        artifact["model"],
        split_loader,
        device,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.artifact_prefix or args.model
    results_dir = Path(args.results_dir)
    predictions_dir = (
        Path(args.predictions_dir)
        if args.predictions_dir is not None
        else results_dir.parent / "predictions"
    )

    depth_metrics = evaluate_depths(
        predictions["true_depths"], predictions["pred_depths"]
    )
    hydro_metrics = compute_hydrological_metrics(
        predictions["true_depths"], predictions["pred_depths"]
    )

    metrics_payload = {
        "timestamp": timestamp,
        "evaluation_split": args.split,
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
        "model": args.model,
        "title": MODEL_TITLES[args.model],
        "task": "depth_only",
        "architecture": describe_model_architecture(args.model, runtime_config),
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
            model_name=args.model,
            device=device,
            split_descriptions=[split_description],
            model=artifact["model"],
        ),
    }

    metrics_path = results_dir / f"{prefix}_{args.split}_{timestamp}_metrics.json"
    manifest_path = results_dir / f"{prefix}_{args.split}_{timestamp}_manifest.json"
    predictions_path = (
        predictions_dir / f"{prefix}_{args.split}_{timestamp}_predictions.npz"
    )

    write_json(metrics_path, metrics_payload)
    write_json(manifest_path, manifest_payload)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(predictions_path, **predictions)

    print(f"Saved metrics:      {metrics_path}")
    print(f"Saved manifest:     {manifest_path}")
    print(f"Saved predictions:  {predictions_path}")


if __name__ == "__main__":
    main()
