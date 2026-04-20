import numpy as np
from scipy.stats import pearsonr


def prediction_interval_coverage_probability(y_true, ci_lower, ci_upper):
    in_interval = (y_true >= ci_lower) & (y_true <= ci_upper)
    picp_per_output = np.mean(in_interval, axis=0) * 100
    picp_aggregated = np.mean(picp_per_output)
    return picp_per_output, picp_aggregated


def mean_prediction_interval_width(ci_lower, ci_upper):
    interval_width = ci_upper - ci_lower
    mpiw_per_output = np.mean(interval_width, axis=0)
    mpiw_aggregated = np.mean(mpiw_per_output)
    return mpiw_per_output, mpiw_aggregated


def uncertainty_error_correlation(y_true, y_pred, y_std):
    n_outputs = y_true.shape[1]
    correlations = np.zeros(n_outputs)
    p_values = np.zeros(n_outputs)

    for i in range(n_outputs):
        abs_error = np.abs(y_true[:, i] - y_pred[:, i])
        uncertainty = y_std[:, i]
        r, p = pearsonr(uncertainty, abs_error)
        correlations[i] = r
        p_values[i] = p

    correlation_aggregated = np.mean(correlations)
    return correlations, correlation_aggregated, p_values


def uncertainty_decomposition(y_true, all_samples):
    epistemic = np.var(all_samples, axis=0)
    aleatoric = np.zeros_like(epistemic)
    total = epistemic + aleatoric
    return epistemic, aleatoric, total


def calibration_error(y_true, y_pred, y_std, n_bins=10):
    n_outputs = y_true.shape[1]
    ece_values = np.zeros(n_outputs)

    for i in range(n_outputs):
        abs_error = np.abs(y_true[:, i] - y_pred[:, i])
        uncertainty = y_std[:, i]

        bins = np.percentile(uncertainty, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(uncertainty, bins[1:-1])

        ece = 0.0
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() > 0:
                mean_uncertainty = uncertainty[mask].mean()
                mean_error = abs_error[mask].mean()
                ece += (mask.sum() / len(uncertainty)) * np.abs(
                    mean_uncertainty - mean_error
                )

        ece_values[i] = ece

    ece_aggregated = np.mean(ece_values)
    return ece_values, ece_aggregated


def compute_all_uncertainty_metrics(results):
    y_true_depths = results["depths_true"]
    y_pred_depths = results["depths_mean"]
    y_std_depths = results["depths_std"]
    ci_lower_depths = results["depths_ci_lower"]
    ci_upper_depths = results["depths_ci_upper"]

    y_true_overflow = results["overflow_true"]
    y_pred_overflow = results["overflow_mean"]
    y_std_overflow = results["overflow_std"]
    ci_lower_overflow = results["overflow_ci_lower"]
    ci_upper_overflow = results["overflow_ci_upper"]

    picp_depths_per, picp_depths_agg = prediction_interval_coverage_probability(
        y_true_depths, ci_lower_depths, ci_upper_depths
    )
    mpiw_depths_per, mpiw_depths_agg = mean_prediction_interval_width(
        ci_lower_depths, ci_upper_depths
    )
    corr_depths_per, corr_depths_agg, p_depths = uncertainty_error_correlation(
        y_true_depths, y_pred_depths, y_std_depths
    )

    picp_overflow_per, picp_overflow_agg = prediction_interval_coverage_probability(
        y_true_overflow.reshape(-1, 1), ci_lower_overflow, ci_upper_overflow
    )
    mpiw_overflow_per, mpiw_overflow_agg = mean_prediction_interval_width(
        ci_lower_overflow, ci_upper_overflow
    )
    corr_overflow_per, corr_overflow_agg, p_overflow = uncertainty_error_correlation(
        y_true_overflow.reshape(-1, 1), y_pred_overflow, y_std_overflow
    )

    return {
        "depths": {
            "PICP_per_sensor": picp_depths_per.tolist(),
            "PICP_aggregated": float(picp_depths_agg),
            "MPIW_per_sensor": mpiw_depths_per.tolist(),
            "MPIW_aggregated": float(mpiw_depths_agg),
            "Uncertainty_Error_Correlation_per_sensor": corr_depths_per.tolist(),
            "Uncertainty_Error_Correlation_aggregated": float(corr_depths_agg),
            "Correlation_p_values": p_depths.tolist(),
        },
        "overflow": {
            "PICP": float(picp_overflow_agg),
            "MPIW": float(mpiw_overflow_agg),
            "Uncertainty_Error_Correlation": float(corr_overflow_agg),
            "Correlation_p_value": float(p_overflow[0]),
        },
    }
