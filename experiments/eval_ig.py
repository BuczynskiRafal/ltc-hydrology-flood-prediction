import argparse
import json
from pathlib import Path

import torch

from experiments.regression_pipeline import (
    load_split_data_for_config,
    load_trained_model,
)
from src.data_utils import load_feature_names
from src.logger import get_console_logger
from src.release_utils import write_json

logger = get_console_logger(__name__)


try:
    from captum.attr import IntegratedGradients
except ModuleNotFoundError as exc:  # pragma: no cover - exercised at runtime only
    IntegratedGradients = None
    CAPTUM_IMPORT_ERROR = exc
else:
    CAPTUM_IMPORT_ERROR = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Integrated Gradients feature attributions for the overflow head."
    )
    parser.add_argument(
        "--model",
        default="lnn",
        choices=["lnn"],
        help="Currently supported model family.",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Optional config path."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Optional checkpoint path."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="Maximum number of test samples to attribute.",
    )
    return parser.parse_args()


def main():
    if IntegratedGradients is None:
        raise ModuleNotFoundError(
            "Captum is required for eval_ig.py. Install project dependencies from "
            "`requirements.txt` or install `captum` manually."
        ) from CAPTUM_IMPORT_ERROR

    args = parse_args()
    artifact = load_trained_model(
        args.model,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )
    runtime_config = artifact["runtime_config"]
    device = artifact["device"]
    test_data = load_split_data_for_config(runtime_config, "test")
    feature_names = load_feature_names(
        use_reduced=bool(runtime_config["data"].get("use_reduced", True))
    )

    max_samples = min(args.max_samples, len(test_data["X"]))
    inputs = torch.from_numpy(test_data["X"][:max_samples]).float().to(device)
    baselines = torch.zeros_like(inputs)

    def overflow_forward(x):
        return artifact["model"](x)[1].squeeze(1)

    ig = IntegratedGradients(overflow_forward)
    attributions, delta = ig.attribute(
        inputs,
        baselines=baselines,
        return_convergence_delta=True,
    )
    abs_attr = attributions.abs()
    per_feature = abs_attr.mean(dim=(0, 1)).detach().cpu().numpy()
    per_timestep = abs_attr.mean(dim=(0, 2)).detach().cpu().numpy()

    output = {
        "model": args.model,
        "checkpoint_path": str(artifact["checkpoint_path"]),
        "config_source": artifact["config_source"],
        "num_samples": int(max_samples),
        "mean_convergence_delta": float(delta.abs().mean().item()),
        "feature_importance": {
            feature_name: float(score)
            for feature_name, score in zip(feature_names, per_feature)
        },
        "timestep_importance": [float(value) for value in per_timestep],
    }
    output_path = (
        Path("artifacts/results/ig") / f"{args.model}_integrated_gradients.json"
    )
    write_json(output_path, output)
    logger.info(json.dumps(output, indent=2))
    logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
