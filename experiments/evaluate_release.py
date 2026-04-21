import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.regression_pipeline import (
    build_evaluation_payload,
    save_evaluation_payload,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the canonical publication-safe evaluation pipeline."
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
        "--threshold-artifact",
        type=str,
        default=None,
        help="Optional path to a saved validation threshold artifact.",
    )
    parser.add_argument(
        "--overflow-threshold",
        type=float,
        default=None,
        help="Explicit overflow threshold override.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="artifacts/results/release",
        help="Directory for manifest and metrics JSON files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_evaluation_payload(
        args.model,
        split=args.split,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        threshold_artifact_path=args.threshold_artifact,
        overflow_threshold=args.overflow_threshold,
    )
    metrics_path, manifest_path = save_evaluation_payload(
        args.model, payload, results_dir=args.results_dir
    )
    print(f"Metrics saved to: {metrics_path}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
