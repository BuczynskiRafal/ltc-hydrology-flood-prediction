import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from src.logger import get_console_logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from experiments.eval_utils import evaluate_depths
from experiments.regression_pipeline import (
    build_run_metadata,
    create_test_loader,
    describe_split_data_for_config,
    load_split_data_for_config,
    load_trained_model,
)
from src.release_utils import write_json

logger = get_console_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate robustness under Gaussian noise, missing-data masking, "
            "and FGSM perturbations."
        )
    )
    parser.add_argument(
        "--model",
        default="lnn",
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
        "--noise-db",
        type=float,
        nargs="*",
        default=[40.0, 30.0, 20.0, 10.0],
        help="Gaussian noise SNR levels in dB.",
    )
    parser.add_argument(
        "--fgsm-eps",
        type=float,
        nargs="*",
        default=[0.001, 0.005, 0.01],
        help="FGSM epsilon values.",
    )
    parser.add_argument(
        "--missing-rates",
        type=float,
        nargs="*",
        default=[0.05, 0.10, 0.15, 0.20],
        help="Element-wise Bernoulli masking rates for missing-data robustness.",
    )
    return parser.parse_args()


def _predict_with_transform(model_name, model, loader, device, transform=None):
    pred_depths = []
    true_depths = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["X"].to(device)
            if transform is not None:
                x = transform(x)
            outputs = model(x)
            pred_depths.append(outputs[0].detach().cpu().numpy())
            true_depths.append(batch["y_depths"].cpu().numpy())
    return (
        np.concatenate(true_depths, axis=0),
        np.concatenate(pred_depths, axis=0),
    )


def _make_gaussian_transform(reference_batch, snr_db, seed=0):
    signal_std = float(reference_batch.std().item())
    noise_std = signal_std / (10.0 ** (snr_db / 20.0))
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    def transform(x):
        noise = torch.randn(
            x.shape,
            generator=generator,
            dtype=x.dtype,
        ).to(device=x.device)
        return x + noise * noise_std

    return transform


def _interpolate_masked_series(
    values: np.ndarray,
    observed_mask: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    if observed_mask.all():
        return values
    if not observed_mask.any():
        return np.full_like(values, fill_value)

    time_index = np.arange(values.shape[0])
    observed_index = time_index[observed_mask]
    observed_values = values[observed_mask]
    return np.interp(time_index, observed_index, observed_values).astype(values.dtype)


def _apply_missing_data_mask(
    x: torch.Tensor,
    *,
    mask_probability: float,
    feature_means: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= mask_probability <= 1.0:
        raise ValueError(f"mask_probability must be in [0, 1], got {mask_probability}.")

    cpu_mask = torch.rand(x.shape, generator=generator) < mask_probability
    x_np = x.detach().cpu().numpy().copy()
    mask_np = cpu_mask.detach().cpu().numpy()
    feature_means_np = feature_means.detach().cpu().numpy()

    batch_size, seq_len, n_features = x_np.shape
    for batch_idx in range(batch_size):
        for feature_idx in range(n_features):
            observed_mask = ~mask_np[batch_idx, :, feature_idx]
            series = x_np[batch_idx, :, feature_idx]
            x_np[batch_idx, :, feature_idx] = _interpolate_masked_series(
                series,
                observed_mask,
                float(feature_means_np[feature_idx]),
            )

    filled_tensor = torch.from_numpy(x_np).to(device=x.device, dtype=x.dtype)
    return filled_tensor, cpu_mask.to(device=x.device)


def _make_missing_data_transform(
    feature_means: torch.Tensor,
    *,
    mask_probability: float,
    seed: int = 0,
):
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    def transform(x):
        masked_x, _ = _apply_missing_data_mask(
            x,
            mask_probability=mask_probability,
            feature_means=feature_means,
            generator=generator,
        )
        return masked_x

    return transform


def _derive_perturbation_seed(base_seed: int, *, offset: int, value: float) -> int:
    return int(base_seed + offset + round(value * 10_000))


def _predict_with_fgsm(model_name, model, loader, device, epsilon):
    model.eval()
    pred_depths = []
    true_depths = []

    for batch in loader:
        x = batch["X"].to(device).clone().detach().requires_grad_(True)
        y_depths = batch["y_depths"].to(device)
        y_overflow = batch["y_overflow"].to(device)

        outputs = model(x)
        pred_depth, pred_overflow = outputs[:2]
        loss = F.mse_loss(pred_depth, y_depths) + F.binary_cross_entropy(
            pred_overflow.squeeze(1), y_overflow
        )
        if model_name == "lnn":
            loss = loss + F.mse_loss(outputs[2].squeeze(1), y_overflow)

        model.zero_grad(set_to_none=True)
        loss.backward()
        x_adv = (x + epsilon * x.grad.sign()).detach()

        adv_outputs = model(x_adv)
        pred_depths.append(adv_outputs[0].detach().cpu().numpy())
        true_depths.append(batch["y_depths"].cpu().numpy())

    return (
        np.concatenate(true_depths, axis=0),
        np.concatenate(pred_depths, axis=0),
    )


def main():
    args = parse_args()
    artifact = load_trained_model(
        args.model,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )
    runtime_config = artifact["runtime_config"]
    device = artifact["device"]
    runtime_seed = int(runtime_config["runtime"]["seed"])
    test_data = load_split_data_for_config(runtime_config, "test")
    test_split_description = describe_split_data_for_config(runtime_config, "test")
    test_loader = create_test_loader(runtime_config, test_data)
    model = artifact["model"]
    feature_means = torch.from_numpy(
        test_data["X"].mean(axis=(0, 1)).astype(np.float32)
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    baseline_true, baseline_pred = _predict_with_transform(
        args.model, model, test_loader, device
    )
    baseline_nse = float(
        evaluate_depths(baseline_true, baseline_pred)["aggregated"]["NSE"]
    )

    reference_batch = torch.from_numpy(
        test_data["X"][: min(64, len(test_data["X"]))]
    ).float()
    gaussian_results = []
    for snr_db in args.noise_db:
        gaussian_seed = _derive_perturbation_seed(
            runtime_seed,
            offset=10_000,
            value=snr_db,
        )
        y_true, y_pred = _predict_with_transform(
            args.model,
            model,
            test_loader,
            device,
            transform=_make_gaussian_transform(
                reference_batch,
                snr_db,
                seed=gaussian_seed,
            ),
        )
        nse = float(evaluate_depths(y_true, y_pred)["aggregated"]["NSE"])
        gaussian_results.append(
            {
                "snr_db": float(snr_db),
                "seed": gaussian_seed,
                "nse": nse,
                "nse_drop": baseline_nse - nse,
            }
        )

    missing_data_results = []
    for mask_rate in args.missing_rates:
        mask_seed = _derive_perturbation_seed(
            runtime_seed,
            offset=20_000,
            value=mask_rate,
        )
        y_true, y_pred = _predict_with_transform(
            args.model,
            model,
            test_loader,
            device,
            transform=_make_missing_data_transform(
                feature_means,
                mask_probability=mask_rate,
                seed=mask_seed,
            ),
        )
        nse = float(evaluate_depths(y_true, y_pred)["aggregated"]["NSE"])
        missing_data_results.append(
            {
                "mask_rate": float(mask_rate),
                "seed": mask_seed,
                "nse": nse,
                "nse_drop": baseline_nse - nse,
                "imputation": (
                    "linear_interpolation_with_edge_fill_and_feature_mean_fallback"
                ),
                "imputation_mean_source": "test_split_mean",
            }
        )

    fgsm_results = []
    for epsilon in args.fgsm_eps:
        y_true, y_pred = _predict_with_fgsm(
            args.model, model, test_loader, device, epsilon
        )
        nse = float(evaluate_depths(y_true, y_pred)["aggregated"]["NSE"])
        fgsm_results.append(
            {
                "epsilon": float(epsilon),
                "nse": nse,
                "nse_drop": baseline_nse - nse,
            }
        )

    output = {
        "model": args.model,
        "timestamp": timestamp,
        "checkpoint_path": str(artifact["checkpoint_path"]),
        "config_source": artifact["config_source"],
        "evaluation_split": "test",
        "robustness_scope": "canonical_test_split",
        "robustness_seed": runtime_seed,
        "baseline_nse": baseline_nse,
        "gaussian_noise": gaussian_results,
        "missing_data": missing_data_results,
        "fgsm": fgsm_results,
        "device": str(device),
        "checkpoint_epoch": artifact["checkpoint"].get("epoch", "current"),
        "imputation_mean_source": "test_split_mean",
        "runtime_metadata": build_run_metadata(
            config=runtime_config,
            model_name=args.model,
            device=device,
            split_descriptions=[test_split_description],
        ),
    }
    output_path = Path("artifacts/results/robustness") / f"{args.model}_robustness.json"
    write_json(output_path, output)
    logger.info(json.dumps(output, indent=2))
    logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
