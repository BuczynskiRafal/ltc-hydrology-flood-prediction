import json
from datetime import datetime
from pathlib import Path

import numpy as np

from experiments.regression_pipeline import (
    create_test_loader,
    get_device,
    load_split_data_for_config,
    load_trained_model,
)
from src.evaluation.uncertainty_analysis import (
    DeltaMethodUncertainty,
    MCDropoutUncertainty,
)
from src.evaluation.uncertainty_metrics import compute_all_uncertainty_metrics
from src.logger import get_console_logger

logger = get_console_logger(__name__)

MC_DROPOUT_SAMPLES = 100
DELTA_RELATIVE_INPUT_ERROR = 0.05
DELTA_INPUT_EPS = 1e-6


def analyze_flood_uncertainty(results):
    flood_mask = results["overflow_true"] == 1
    normal_mask = results["overflow_true"] == 0

    if flood_mask.sum() == 0:
        return None

    return {
        "overflow_uncertainty_flood": float(results["overflow_std"][flood_mask].mean()),
        "overflow_uncertainty_normal": float(
            results["overflow_std"][normal_mask].mean()
        ),
        "depth_uncertainty_flood": float(results["depths_std"][flood_mask].mean()),
        "depth_uncertainty_normal": float(results["depths_std"][normal_mask].mean()),
        "flood_events_count": int(flood_mask.sum()),
        "normal_events_count": int(normal_mask.sum()),
    }


def save_results(
    mc_results,
    delta_results,
    mc_metrics,
    delta_metrics,
    mc_flood_analysis,
    delta_flood_analysis,
    config,
    config_path,
    checkpoint_path,
    test_data,
    use_reduced,
    config_source,
    results_dir=None,
):
    results_dir = Path(results_dir or "artifacts/results/uncertainty")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_data = {
        "model": "LNN_Regression",
        "architecture": (
            f"fast={config['model']['fast_units']}, "
            f"slow={config['model']['slow_units']}, "
            f"hidden={config['model']['hidden_size']}"
        ),
        "timestamp": timestamp,
        "config_path": config_path,
        "checkpoint_path": str(checkpoint_path),
        "config_source": config_source,
        "use_reduced": bool(use_reduced),
        "input_size": int(config["model"]["input_size"]),
        "evaluation_split": "test",
        "confidence_level": 0.95,
        "test_samples": len(test_data["X"]),
        "mc_dropout": {
            "samples": MC_DROPOUT_SAMPLES,
            "uncertainty_metrics": mc_metrics,
            "flood_analysis": mc_flood_analysis,
        },
        "delta_method": {
            "relative_input_error": DELTA_RELATIVE_INPUT_ERROR,
            "eps": DELTA_INPUT_EPS,
            "uncertainty_metrics": delta_metrics,
            "flood_analysis": delta_flood_analysis,
        },
    }

    with open(results_dir / f"lnn_uncertainty_metrics_{timestamp}.json", "w") as f:
        json.dump(output_data, f, indent=2)

    np.savez(
        results_dir / f"lnn_uncertainty_predictions_{timestamp}.npz",
        mc_depths_mean=mc_results["depths_mean"],
        mc_depths_std=mc_results["depths_std"],
        mc_depths_ci_lower=mc_results["depths_ci_lower"],
        mc_depths_ci_upper=mc_results["depths_ci_upper"],
        mc_depths_true=mc_results["depths_true"],
        mc_overflow_mean=mc_results["overflow_mean"],
        mc_overflow_std=mc_results["overflow_std"],
        mc_overflow_ci_lower=mc_results["overflow_ci_lower"],
        mc_overflow_ci_upper=mc_results["overflow_ci_upper"],
        mc_overflow_true=mc_results["overflow_true"],
        delta_depths_mean=delta_results["depths_mean"],
        delta_depths_std=delta_results["depths_std"],
        delta_depths_ci_lower=delta_results["depths_ci_lower"],
        delta_depths_ci_upper=delta_results["depths_ci_upper"],
        delta_depths_true=delta_results["depths_true"],
        delta_overflow_mean=delta_results["overflow_mean"],
        delta_overflow_std=delta_results["overflow_std"],
        delta_overflow_ci_lower=delta_results["overflow_ci_lower"],
        delta_overflow_ci_upper=delta_results["overflow_ci_upper"],
        delta_overflow_true=delta_results["overflow_true"],
    )


def main():
    device = get_device()
    artifact = load_trained_model("lnn", device=device)
    runtime_config = artifact["runtime_config"]

    use_reduced = runtime_config["data"].get("use_reduced", True)
    logger.info(
        f"Loading test data (use_reduced={use_reduced}, "
        f"config_source={artifact['config_source']})..."
    )
    test_data = load_split_data_for_config(runtime_config, "test")
    if test_data["X"].shape[-1] != runtime_config["model"]["input_size"]:
        raise ValueError(
            "Configured input_size does not match test data channels: "
            f"{runtime_config['model']['input_size']} != {test_data['X'].shape[-1]}"
        )

    test_loader = create_test_loader(runtime_config, test_data)
    mc_uncertainty_analyzer = MCDropoutUncertainty(
        model=artifact["model"],
        n_samples=MC_DROPOUT_SAMPLES,
        device=device,
    )
    delta_uncertainty_analyzer = DeltaMethodUncertainty(
        model=artifact["model"],
        relative_input_error=DELTA_RELATIVE_INPUT_ERROR,
        eps=DELTA_INPUT_EPS,
        device=device,
    )
    mc_results = mc_uncertainty_analyzer.predict_batch_with_uncertainty(test_loader)
    delta_results = delta_uncertainty_analyzer.predict_batch_with_uncertainty(
        test_loader
    )
    mc_metrics = compute_all_uncertainty_metrics(mc_results)
    delta_metrics = compute_all_uncertainty_metrics(delta_results)
    mc_flood_analysis = analyze_flood_uncertainty(mc_results)
    delta_flood_analysis = analyze_flood_uncertainty(delta_results)

    save_results(
        mc_results,
        delta_results,
        mc_metrics,
        delta_metrics,
        mc_flood_analysis,
        delta_flood_analysis,
        runtime_config,
        str(artifact["canonical_config_path"]),
        artifact["checkpoint_path"],
        test_data,
        use_reduced,
        artifact["config_source"],
    )


if __name__ == "__main__":
    main()
