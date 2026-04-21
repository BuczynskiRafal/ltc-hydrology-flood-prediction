import argparse
import sys
from pathlib import Path

from src.logger import get_console_logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.regression_pipeline import select_overflow_threshold

logger = get_console_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit the overflow probability threshold on the validation split."
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
        "--output",
        type=str,
        default=None,
        help="Optional output path. Defaults to the path configured in YAML.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    payload, output_path = select_overflow_threshold(
        args.model,
        split="val",
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )
    logger.info(f"Threshold: {payload['threshold']:.6f}")
    logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
