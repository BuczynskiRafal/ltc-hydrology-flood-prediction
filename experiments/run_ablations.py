import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from experiments.eval_utils import (
    evaluate_depths,
    evaluate_overflow,
    find_optimal_threshold,
)
from experiments.regression_pipeline import (
    collect_predictions,
    create_test_loader,
    get_runtime_device,
    load_model_config,
    load_split_data_for_config,
    train_model,
)
from src.release_utils import write_json

ABLATIONS = {
    "full": {
        "use_fast_path": True,
        "use_slow_path": True,
        "use_attention": True,
    },
    "no_fast_path": {
        "use_fast_path": False,
        "use_slow_path": True,
        "use_attention": True,
    },
    "no_slow_path": {
        "use_fast_path": True,
        "use_slow_path": False,
        "use_attention": True,
    },
    "no_attention": {
        "use_fast_path": True,
        "use_slow_path": True,
        "use_attention": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train publication-safe LNN ablations for 50 epochs."
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Optional config path."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Number of epochs to train each ablation.",
    )
    return parser.parse_args()


def evaluate_ablation(model_name, model, config, split_data, device, threshold):
    loader = create_test_loader(config, split_data)
    predictions = collect_predictions(model_name, model, loader, device)
    depth_metrics = evaluate_depths(
        predictions["true_depths"], predictions["pred_depths"]
    )
    overflow_metrics = evaluate_overflow(
        predictions["true_overflow"],
        predictions["pred_overflow"],
        threshold=threshold,
    )
    return {
        "depth_nse": float(depth_metrics["aggregated"]["NSE"]),
        "depth_rmse": float(depth_metrics["aggregated"]["RMSE"]),
        "overflow_f1": float(overflow_metrics["F1"]),
        "overflow_roc_auc": float(overflow_metrics["ROC-AUC"]),
    }


def main():
    args = parse_args()
    base_config = load_model_config("lnn", args.config)
    device = get_runtime_device(base_config)
    train_data = load_split_data_for_config(base_config, "train")
    val_data = load_split_data_for_config(base_config, "val")
    test_data = load_split_data_for_config(base_config, "test")

    results = {}
    for ablation_name, flags in ABLATIONS.items():
        config = deepcopy(base_config)
        config["training"]["epochs"] = int(args.max_epochs)
        config["model"].update(flags)
        config["output"]["checkpoint_dir"] = (
            f"artifacts/checkpoints/ablations/{ablation_name}"
        )
        checkpoint_path = Path(config["output"]["checkpoint_dir"]) / "best_model.pt"

        model, summary = train_model(
            "lnn",
            config,
            device=device,
            train_data=train_data,
            val_data=val_data,
            checkpoint_path=checkpoint_path,
        )
        val_loader = create_test_loader(config, val_data)
        val_predictions = collect_predictions("lnn", model, val_loader, device)
        threshold = float(
            find_optimal_threshold(
                val_predictions["true_overflow"], val_predictions["pred_overflow"]
            )
        )
        results[ablation_name] = {
            "flags": flags,
            "summary": summary,
            "val_threshold": threshold,
            "test_metrics": evaluate_ablation(
                "lnn", model, config, test_data, device, threshold
            ),
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path("artifacts/results/ablations") / f"lnn_ablations_{timestamp}.json"
    )
    write_json(
        output_path,
        {
            "model": "lnn",
            "max_epochs": int(args.max_epochs),
            "device": str(device),
            "results": results,
        },
    )
    print(json.dumps(results, indent=2))
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
