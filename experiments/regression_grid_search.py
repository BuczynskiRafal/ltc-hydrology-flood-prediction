from __future__ import annotations

import json
import traceback
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from experiments.regression_pipeline import (
    count_parameters,
    create_model,
    get_device,
    load_model_config,
    train_model,
)
from src.logger import get_console_logger

logger = get_console_logger(__name__)


def apply_trial_overrides(
    base_config: dict[str, Any], params: dict[str, Any], max_epochs: int
) -> dict[str, Any]:
    trial_config = deepcopy(base_config)
    trial_config["training"]["epochs"] = int(max_epochs)

    for key, value in params.items():
        if key == "learning_rate":
            trial_config["training"].setdefault("optimizer", {})
            trial_config["training"]["optimizer"]["learning_rate"] = value
            continue
        if key in trial_config["model"]:
            trial_config["model"][key] = value
            continue
        if key in trial_config["training"]:
            trial_config["training"][key] = value
            continue
        if key in trial_config["loss"]:
            trial_config["loss"][key] = value
            continue
        raise KeyError(f"Unsupported grid-search override: {key}")

    return trial_config


def create_search_space(variable_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = tuple(variable_grid.keys())
    return [dict(zip(keys, values)) for values in product(*variable_grid.values())]


def run_grid_search(
    model_name: str,
    variable_grid: dict[str, list[Any]],
    max_epochs: int,
    header: str | None = None,
) -> None:
    device = get_device()
    base_config = load_model_config(model_name)
    combinations = create_search_space(variable_grid)

    logger.info("=" * 70)
    logger.info(header or f"{model_name.upper()} GRID SEARCH")
    logger.info("=" * 70)
    logger.info(f"Using device: {device}")
    logger.info(f"Total combinations: {len(combinations)}")
    logger.info(f"Training epochs per run: {max_epochs}")

    results = []
    best_result = None

    for index, params in enumerate(combinations, start=1):
        logger.info(f"\n{'#' * 70}")
        logger.info(f"EXPERIMENT {index}/{len(combinations)}")
        logger.info(f"{'#' * 70}")
        logger.info(params)

        try:
            trial_config = apply_trial_overrides(base_config, params, max_epochs)
            model, summary = train_model(
                model_name,
                trial_config,
                device=device,
                checkpoint_path=None,
                max_epochs=max_epochs,
            )
            result = {
                "params": params,
                "trial_config": trial_config,
                "best_epoch": summary["best_epoch"],
                "best_val_loss": summary["best_val_loss"],
                "best_val_accuracy": summary["best_val_accuracy"],
                "epochs_ran": summary["epochs_ran"],
                "n_params": count_parameters(model),
                "model_state_dict": deepcopy(model.state_dict()),
            }
            if model_name == "tcn":
                result["receptive_field"] = create_model(
                    model_name, trial_config
                ).get_receptive_field()
            results.append(result)
            if (
                best_result is None
                or result["best_val_loss"] < best_result["best_val_loss"]
            ):
                best_result = result
            logger.info(
                f"\nBest val_loss={result['best_val_loss']:.4f}, "
                f"val_acc={result['best_val_accuracy']:.2f}%"
            )
        except Exception as exc:
            logger.info(f"\nFailed: {exc}")
            traceback.print_exc()
            results.append({"params": params, "error": str(exc)})
            if index == 1:
                raise RuntimeError(
                    "Grid search aborted after the first trial failed; "
                    "this likely indicates a structural configuration or data issue."
                ) from exc

    successful = [result for result in results if "best_val_loss" in result]
    successful.sort(key=lambda item: item["best_val_loss"])

    logger.info("\n" + "=" * 70)
    logger.info("GRID SEARCH COMPLETE")
    logger.info("=" * 70)
    if not successful:
        logger.info("No successful runs.")
        return

    for rank, result in enumerate(successful, start=1):
        logger.info(
            f"\n{rank}. val_loss={result['best_val_loss']:.4f} | "
            f"val_acc={result['best_val_accuracy']:.2f}% | "
            f"params={result['n_params']:,}"
        )
        logger.info(result["params"])

    tuning_dir = Path("artifacts/tuning") / model_name
    tuning_dir.mkdir(parents=True, exist_ok=True)

    best_trial_config = deepcopy(best_result["trial_config"])
    best_trial_config["output"]["checkpoint_dir"] = str(tuning_dir)

    checkpoint_payload = {
        "epoch": best_result["best_epoch"],
        "model_state_dict": best_result["model_state_dict"],
        "val_loss": best_result["best_val_loss"],
        "val_acc": best_result["best_val_accuracy"],
        "n_params": best_result["n_params"],
        "config": best_trial_config,
    }
    if "receptive_field" in best_result:
        checkpoint_payload["receptive_field"] = best_result["receptive_field"]

    checkpoint_path = tuning_dir / "best_model.pt"
    torch.save(checkpoint_payload, checkpoint_path)
    logger.info(f"\nBest model saved to: {checkpoint_path}")

    config_path = tuning_dir / "best_config.yaml"
    with open(config_path, "w") as handle:
        yaml.safe_dump(best_trial_config, handle, sort_keys=False)
    logger.info(f"Best config saved to: {config_path}")

    results_path = tuning_dir / "grid_search_results.json"
    results_for_json = []
    for result in results:
        serializable = {
            key: value for key, value in result.items() if key != "model_state_dict"
        }
        results_for_json.append(serializable)
    with open(results_path, "w") as handle:
        json.dump(results_for_json, handle, indent=2)
    logger.info(f"All results saved to: {results_path}")

    if variable_grid:
        logger.info("\n" + "=" * 70)
        logger.info("PARAMETER IMPACT")
        logger.info("=" * 70)
        for parameter_name, values in variable_grid.items():
            logger.info(f"\n{parameter_name}:")
            for value in values:
                matching = [
                    result
                    for result in successful
                    if result["params"].get(parameter_name) == value
                ]
                if not matching:
                    continue
                average_loss = np.mean([item["best_val_loss"] for item in matching])
                average_acc = np.mean([item["best_val_accuracy"] for item in matching])
                logger.info(
                    f"  {value}: avg_val_loss={average_loss:.4f}, "
                    f"avg_val_acc={average_acc:.2f}% (n={len(matching)})"
                )
