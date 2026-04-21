import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

from experiments.regression_pipeline import (
    collect_predictions,
    create_test_loader,
    get_device,
    load_split_data_for_config,
    load_trained_model,
)
from src.statistical_tests import (
    bootstrap_ci,
    calculate_nse,
    diebold_mariano_test,
    wilcoxon_test,
)

MODEL_NAMES = ("lnn", "gru", "lstm", "tcn", "mlp")


def load_predictions_for_model(model_name, device):
    artifact = load_trained_model(model_name, device=device)
    test_data = load_split_data_for_config(artifact["runtime_config"], "test")
    test_loader = create_test_loader(artifact["runtime_config"], test_data)
    predictions = collect_predictions(
        model_name, artifact["model"], test_loader, device
    )
    y_true = predictions["true_depths"].flatten()
    y_pred = predictions["pred_depths"].flatten()
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "checkpoint_path": str(artifact["checkpoint_path"]),
        "config_source": artifact["config_source"],
        "use_reduced": bool(
            artifact["runtime_config"]["data"].get("use_reduced", True)
        ),
    }


def main():
    device = get_device()
    print(f"Using device: {device}")

    predictions = {
        model_name: load_predictions_for_model(model_name, device)
        for model_name in MODEL_NAMES
    }

    reference_truth = predictions[MODEL_NAMES[0]]["y_true"]
    for model_name in MODEL_NAMES[1:]:
        if not np.allclose(reference_truth, predictions[model_name]["y_true"]):
            raise ValueError(
                f"Ground-truth mismatch detected for model '{model_name}'."
            )

    results = {}
    for model_a, model_b in combinations(MODEL_NAMES, 2):
        pred_a = predictions[model_a]["y_pred"]
        pred_b = predictions[model_b]["y_pred"]

        wilcoxon = wilcoxon_test(reference_truth, pred_a, pred_b)
        bootstrap = bootstrap_ci(reference_truth, pred_a, pred_b, n_iterations=1000)
        dm = diebold_mariano_test(reference_truth, pred_a, pred_b)

        nse_a = calculate_nse(reference_truth, pred_a)
        nse_b = calculate_nse(reference_truth, pred_b)

        results[f"{model_a}_vs_{model_b}"] = {
            "models": {
                model_a: {
                    "checkpoint_path": predictions[model_a]["checkpoint_path"],
                    "config_source": predictions[model_a]["config_source"],
                    "use_reduced": predictions[model_a]["use_reduced"],
                },
                model_b: {
                    "checkpoint_path": predictions[model_b]["checkpoint_path"],
                    "config_source": predictions[model_b]["config_source"],
                    "use_reduced": predictions[model_b]["use_reduced"],
                },
            },
            "nse": {
                model_a: float(nse_a),
                model_b: float(nse_b),
                "delta": float(nse_a - nse_b),
            },
            "wilcoxon": wilcoxon,
            "bootstrap_ci": {
                "delta_nse_mean": bootstrap["delta_nse_mean"],
                "ci_lower": bootstrap["ci_lower"],
                "ci_upper": bootstrap["ci_upper"],
                "n_iterations": bootstrap["n_iterations"],
            },
            "diebold_mariano": dm,
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("artifacts/results/statistical_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"pairwise_comparisons_{timestamp}.json"
    with open(output_path, "w") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nResults saved to: {output_path}")
    for comparison, metrics in results.items():
        print(f"\n{comparison}:")
        print(
            f"  ΔNSE = {metrics['nse']['delta']:.4f}, "
            f"95% CI [{metrics['bootstrap_ci']['ci_lower']:.4f}, {metrics['bootstrap_ci']['ci_upper']:.4f}]"
        )
        print(
            f"  Wilcoxon: W={metrics['wilcoxon']['W']:.0f}, "
            f"p={metrics['wilcoxon']['p_value']:.4f}, "
            f"r={metrics['wilcoxon']['effect_size_r']:.3f}"
        )
        print(
            f"  DM: stat={metrics['diebold_mariano']['dm_statistic']:.3f}, "
            f"p={metrics['diebold_mariano']['p_value']:.4f}, "
            f"lag1_autocorr={metrics['diebold_mariano']['lag1_autocorr']:.3f}"
        )


if __name__ == "__main__":
    main()
