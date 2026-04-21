import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr

UNCERTAINTY_RESULT_KEYS = (
    "depths_mean",
    "depths_std",
    "depths_ci_lower",
    "depths_ci_upper",
    "depths_true",
    "overflow_mean",
    "overflow_std",
    "overflow_ci_lower",
    "overflow_ci_upper",
    "overflow_true",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize LNN uncertainty artifacts.")
    parser.add_argument(
        "predictions_path",
        nargs="?",
        default=None,
        help="Optional NPZ artifact path. Defaults to the latest file in artifacts/results/uncertainty.",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "mc", "delta"],
        default="auto",
        help="Which uncertainty view to visualize for multi-method artifacts.",
    )
    return parser.parse_args()


def _resolve_predictions_path(predictions_path: str | None) -> Path:
    if predictions_path is not None:
        return Path(predictions_path)

    results_dir = Path("artifacts/results/uncertainty")
    pred_files = sorted(results_dir.glob("lnn_uncertainty_predictions_*.npz"))
    if not pred_files:
        raise FileNotFoundError(
            f"No uncertainty prediction artifacts found in {results_dir}."
        )
    return pred_files[-1]


def _extract_timestamp(predictions_path: Path) -> str:
    prefix = "lnn_uncertainty_predictions_"
    if predictions_path.stem.startswith(prefix):
        return predictions_path.stem[len(prefix) :]
    return predictions_path.stem


def _resolve_uncertainty_view(results, method="auto"):
    result_keys = set(results.keys())
    available_methods = [
        candidate
        for candidate in ("mc", "delta")
        if f"{candidate}_depths_mean" in result_keys
    ]
    if not available_methods:
        raise KeyError(
            "Could not find supported multi-method uncertainty arrays in the NPZ artifact."
        )

    resolved_method = available_methods[0] if method == "auto" else method
    prefix = f"{resolved_method}_"
    missing_keys = [
        f"{prefix}{key}"
        for key in UNCERTAINTY_RESULT_KEYS
        if f"{prefix}{key}" not in result_keys
    ]
    if missing_keys:
        raise KeyError(
            f"NPZ artifact does not contain the requested '{resolved_method}' view. "
            f"Missing keys: {', '.join(missing_keys)}"
        )

    return (
        {key: results[f"{prefix}{key}"] for key in UNCERTAINTY_RESULT_KEYS},
        resolved_method,
    )


def plot_timeline(results, output_path, n_samples=500, sensor_idx=0):
    depths_mean = results["depths_mean"][:n_samples, sensor_idx]
    depths_ci_lower = results["depths_ci_lower"][:n_samples, sensor_idx]
    depths_ci_upper = results["depths_ci_upper"][:n_samples, sensor_idx]
    depths_true = results["depths_true"][:n_samples, sensor_idx]
    timesteps = np.arange(n_samples)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(timesteps, depths_true, "k-", linewidth=1, alpha=0.7)
    ax.plot(timesteps, depths_mean, "b-", linewidth=1.5)
    ax.fill_between(
        timesteps, depths_ci_lower, depths_ci_upper, color="blue", alpha=0.2
    )
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_uncertainty_vs_error(results, output_path):
    abs_error = np.abs(results["depths_true"] - results["depths_mean"]).flatten()
    uncertainty = results["depths_std"].flatten()

    n_plot = min(5000, len(abs_error))
    indices = np.random.choice(len(abs_error), n_plot, replace=False)
    abs_error = abs_error[indices]
    uncertainty = uncertainty[indices]

    r, p = pearsonr(uncertainty, abs_error)
    z = np.polyfit(uncertainty, abs_error, 1)
    p_fit = np.poly1d(z)
    x_fit = np.linspace(uncertainty.min(), uncertainty.max(), 100)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hexbin(uncertainty, abs_error, gridsize=30, cmap="Blues", mincnt=1, alpha=0.8)
    ax.plot(x_fit, p_fit(x_fit), "r--", linewidth=2, label=f"r={r:.3f}, p={p:.2e}")
    ax.set_xlabel("Prediction Uncertainty (std)", fontsize=12)
    ax.set_ylabel("Absolute Error", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(results, output_path):
    depths_std = results["depths_std"]
    n_sensors = depths_std.shape[1]
    n_samples = depths_std.shape[0]
    n_time_bins = 50
    bin_size = n_samples // n_time_bins

    uncertainty_matrix = np.zeros((n_sensors, n_time_bins))
    for i in range(n_time_bins):
        start_idx = i * bin_size
        end_idx = min((i + 1) * bin_size, n_samples)
        uncertainty_matrix[:, i] = depths_std[start_idx:end_idx, :].mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        uncertainty_matrix,
        cmap="YlOrRd",
        cbar_kws={"label": "Mean Uncertainty (std)"},
        yticklabels=[f"Sensor {i+1}" for i in range(n_sensors)],
        xticklabels=[f"{i*bin_size}" for i in range(n_time_bins)][::5],
        ax=ax,
    )
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Sensor", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    predictions_path = _resolve_predictions_path(args.predictions_path)

    results = dict(np.load(predictions_path))
    resolved_results, resolved_method = _resolve_uncertainty_view(
        results,
        method=args.method,
    )
    output_dir = predictions_path.parent
    timestamp = _extract_timestamp(predictions_path)
    output_suffix = f"{resolved_method}_{timestamp}"

    plot_timeline(
        resolved_results,
        output_dir / f"uncertainty_timeline_{output_suffix}.png",
    )
    plot_uncertainty_vs_error(
        resolved_results,
        output_dir / f"uncertainty_vs_error_{output_suffix}.png",
    )
    plot_heatmap(
        resolved_results,
        output_dir / f"sensor_uncertainty_heatmap_{output_suffix}.png",
    )


if __name__ == "__main__":
    main()
