"""Unified CLI for training, evaluation, and grid search experiments.

Usage::

    python -m experiments.cli train gru
    python -m experiments.cli evaluate lnn --config configs/custom.yaml
    python -m experiments.cli ensemble-train lnn
    python -m experiments.cli ensemble-evaluate lnn
    python -m experiments.cli evaluate --all --split test
    python -m experiments.cli grid-search gru --profile full
    python -m experiments train tcn          # shorthand via __main__.py

Replaces the former per-model thin wrappers (train_gru.py, evaluate_lstm.py, etc.).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

SUPPORTED_MODELS = ("gru", "lstm", "tcn", "mlp", "lnn")

GRID_PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "gru": {
        "default": {
            "grid": {
                "hidden_size": [64, 128],
                "num_layers": [2, 3],
                "dropout": [0.2],
                "learning_rate": [0.001, 0.01],
                "batch_size": [512],
            },
            "max_epochs": 20,
            "header": "GRU HYPERPARAMETER GRID SEARCH",
        },
        "full": {
            "grid": {
                "hidden_size": [32, 64, 128],
                "num_layers": [1, 2, 3],
                "dropout": [0.1, 0.2, 0.3],
                "learning_rate": [0.0001, 0.001, 0.01],
                "batch_size": [256, 512],
            },
            "max_epochs": 20,
            "header": "GRU FULL HYPERPARAMETER GRID SEARCH",
        },
    },
    "lstm": {
        "default": {
            "grid": {
                "hidden_size": [64, 128],
                "num_layers": [2, 3],
                "learning_rate": [0.001, 0.01],
            },
            "max_epochs": 30,
            "header": "LSTM MINIMAL GRID SEARCH",
        },
    },
    "mlp": {
        "default": {
            "grid": {
                "hidden_dims": [[256, 128, 64], [512, 256, 128]],
                "learning_rate": [0.001, 0.01],
            },
            "max_epochs": 30,
            "header": "MLP MINIMAL GRID SEARCH",
        },
    },
    "tcn": {
        "default": {
            "grid": {
                "hidden_size": [64, 128],
                "num_layers": [2, 3],
                "learning_rate": [0.001, 0.01],
            },
            "max_epochs": 30,
            "header": "TCN MINIMAL GRID SEARCH",
        },
    },
}


def _resolve_grid_profile(model: str, profile_name: str) -> dict[str, Any]:
    """Look up a grid-search profile. Raises ValueError on miss."""
    model_profiles = GRID_PROFILES.get(model)
    if model_profiles is None:
        available = ", ".join(sorted(GRID_PROFILES.keys()))
        raise ValueError(
            f"no grid search profiles defined for '{model}'. "
            f"Available models with grids: {available}"
        )

    profile = model_profiles.get(profile_name)
    if profile is None:
        available = ", ".join(sorted(model_profiles.keys()))
        raise ValueError(
            f"unknown profile '{profile_name}' for '{model}'. "
            f"Available profiles: {available}"
        )

    return profile


def handle_train(args: argparse.Namespace) -> None:
    from experiments.regression_pipeline import train_configured_model

    train_configured_model(args.model, config_path=args.config)


def handle_evaluate(args: argparse.Namespace) -> None:
    from experiments.regression_pipeline import evaluate_regression_model

    evaluate_regression_model(args.model, config_path=args.config)


def handle_evaluate_all(args: argparse.Namespace) -> None:
    from experiments.regression_pipeline import (
        CANONICAL_CONFIG_PATHS,
        build_evaluation_payload,
        save_evaluation_payload,
    )

    for model_name in CANONICAL_CONFIG_PATHS:
        payload = build_evaluation_payload(model_name, split=args.split)
        metrics_path, manifest_path = save_evaluation_payload(
            model_name, payload, results_dir=args.results_dir
        )
        print(f"[{model_name}] metrics={metrics_path}")
        print(f"[{model_name}] manifest={manifest_path}")


def handle_grid_search(args: argparse.Namespace) -> None:
    profile = _resolve_grid_profile(args.model, args.profile)
    max_epochs = (
        args.max_epochs if args.max_epochs is not None else profile["max_epochs"]
    )

    from experiments.regression_grid_search import run_grid_search

    run_grid_search(
        args.model,
        profile["grid"],
        max_epochs=max_epochs,
        header=profile["header"],
    )


def handle_ensemble_train(args: argparse.Namespace) -> None:
    from experiments.regression_pipeline import train_lnn_ensemble

    payload = train_lnn_ensemble(
        config_path=args.config,
        seeds=args.seeds,
        checkpoint_root=args.checkpoint_root,
    )
    for member in payload["members"]:
        print(
            f"[lnn][seed={member['seed']}] checkpoint={member['checkpoint_path']} "
            f"best_epoch={member['best_epoch']}"
        )


def handle_ensemble_evaluate(args: argparse.Namespace) -> None:
    from experiments.regression_pipeline import evaluate_lnn_ensemble

    payload, output_path = evaluate_lnn_ensemble(
        config_path=args.config,
        seeds=args.seeds,
        checkpoint_root=args.checkpoint_root,
        results_dir=args.results_dir,
    )
    print(
        f"[lnn][ensemble] threshold={payload['overflow_threshold']:.6f} "
        f"results={output_path}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiments.cli",
        description="Unified CLI for Bellinge experiment workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser(
        "train", help="Train a model using its canonical config."
    )
    train_parser.add_argument(
        "model", choices=SUPPORTED_MODELS, help="Model architecture to train."
    )
    train_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a custom YAML config (overrides the canonical config).",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a trained model on the test split."
    )
    evaluate_parser.add_argument(
        "model", choices=SUPPORTED_MODELS, help="Model architecture to evaluate."
    )
    evaluate_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a custom YAML config.",
    )

    evaluate_all_parser = subparsers.add_parser(
        "evaluate-all",
        help="Evaluate all canonical models and save release manifests.",
    )
    evaluate_all_parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    evaluate_all_parser.add_argument(
        "--results-dir",
        type=str,
        default="artifacts/results/release",
        help="Output directory for results (default: artifacts/results/release).",
    )

    grid_parser = subparsers.add_parser(
        "grid-search", help="Run hyperparameter grid search for a model."
    )
    grid_parser.add_argument(
        "model", choices=SUPPORTED_MODELS, help="Model architecture to search."
    )
    grid_parser.add_argument(
        "--profile",
        type=str,
        default="default",
        help="Grid profile name (default: 'default'). Use '--profile full' for GRU.",
    )
    grid_parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override the max training epochs from the profile.",
    )

    ensemble_train_parser = subparsers.add_parser(
        "ensemble-train",
        help="Train a 5-member LNN ensemble with distinct seeds.",
    )
    ensemble_train_parser.add_argument(
        "model",
        choices=["lnn"],
        help="Currently only LNN ensemble training is supported.",
    )
    ensemble_train_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a custom YAML config (overrides the canonical config).",
    )
    ensemble_train_parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[42, 43, 44, 45, 46],
        help="Ensemble member seeds (default: 42 43 44 45 46).",
    )
    ensemble_train_parser.add_argument(
        "--checkpoint-root",
        type=str,
        default="artifacts/checkpoints/ensemble",
        help=(
            "Root directory for ensemble checkpoints "
            "(default: artifacts/checkpoints/ensemble)."
        ),
    )

    ensemble_evaluate_parser = subparsers.add_parser(
        "ensemble-evaluate",
        help="Evaluate a trained LNN ensemble on the test split.",
    )
    ensemble_evaluate_parser.add_argument(
        "model",
        choices=["lnn"],
        help="Currently only LNN ensemble evaluation is supported.",
    )
    ensemble_evaluate_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a custom YAML config.",
    )
    ensemble_evaluate_parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[42, 43, 44, 45, 46],
        help="Ensemble member seeds (default: 42 43 44 45 46).",
    )
    ensemble_evaluate_parser.add_argument(
        "--checkpoint-root",
        type=str,
        default="artifacts/checkpoints/ensemble",
        help=(
            "Root directory for ensemble checkpoints "
            "(default: artifacts/checkpoints/ensemble)."
        ),
    )
    ensemble_evaluate_parser.add_argument(
        "--results-dir",
        type=str,
        default="artifacts/results/ensemble/lnn",
        help=(
            "Output directory for ensemble manifests "
            "(default: artifacts/results/ensemble/lnn)."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "train":
            handle_train(args)
        elif args.command == "evaluate":
            handle_evaluate(args)
        elif args.command == "evaluate-all":
            handle_evaluate_all(args)
        elif args.command == "grid-search":
            handle_grid_search(args)
        elif args.command == "ensemble-train":
            handle_ensemble_train(args)
        elif args.command == "ensemble-evaluate":
            handle_ensemble_evaluate(args)
    except ValueError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
