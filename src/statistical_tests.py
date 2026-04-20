from typing import Dict

import numpy as np
from scipy import stats


def wilcoxon_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict:
    """
    Wilcoxon signed-rank test comparing two models.
    Returns W statistic, p-value, and rank-biserial correlation (effect size).
    """
    errors_a = np.abs(y_true - pred_a)
    errors_b = np.abs(y_true - pred_b)

    result = stats.wilcoxon(errors_a, errors_b, alternative="two-sided")

    n = len(errors_a)
    r = 1 - (2 * result.statistic) / (n * (n + 1) / 2)

    return {
        "W": float(result.statistic),
        "p_value": float(result.pvalue),
        "effect_size_r": float(r),
        "n_samples": int(n),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_iterations: int = 10000,
    ci: float = 0.95,
) -> Dict:
    """
    Bootstrap confidence intervals for ΔNSE between two models.
    """
    n = len(y_true)
    deltas = np.zeros(n_iterations)

    nse_a = calculate_nse(y_true, pred_a)
    nse_b = calculate_nse(y_true, pred_b)

    for i in range(n_iterations):
        idx = np.random.choice(n, size=n, replace=True)

        nse_a_boot = calculate_nse(y_true[idx], pred_a[idx])
        nse_b_boot = calculate_nse(y_true[idx], pred_b[idx])

        deltas[i] = nse_a_boot - nse_b_boot

    alpha = 1 - ci
    lower = np.percentile(deltas, alpha / 2 * 100)
    upper = np.percentile(deltas, (1 - alpha / 2) * 100)

    return {
        "delta_nse_mean": float(nse_a - nse_b),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_deltas": deltas,
        "n_iterations": n_iterations,
        "confidence_level": ci,
    }


def diebold_mariano_test(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray
) -> Dict:
    """
    Diebold-Mariano test for predictive accuracy comparison.
    Includes lag-1 autocorrelation in residuals.
    """
    errors_a = y_true - pred_a
    errors_b = y_true - pred_b

    d = errors_a**2 - errors_b**2

    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)

    lag1_autocorr = calculate_autocorr_lag1(d)
    variance_corrected = d_var * (1 + lag1_autocorr) / (1 - lag1_autocorr)

    n = len(d)
    dm_stat = d_mean / np.sqrt(variance_corrected / n)

    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return {
        "dm_statistic": float(dm_stat),
        "p_value": float(p_value),
        "lag1_autocorr": float(lag1_autocorr),
        "mean_loss_diff": float(d_mean),
        "n_samples": int(n),
    }


def calculate_nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency."""
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)


def calculate_autocorr_lag1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation."""
    x_mean = np.mean(x)
    x_centered = x - x_mean

    numerator = np.sum(x_centered[:-1] * x_centered[1:])
    denominator = np.sum(x_centered**2)

    return numerator / denominator


def time_series_cv(
    data: np.ndarray, targets: np.ndarray, model_fn, n_folds: int = 5
) -> Dict:
    """
    Time-series cross-validation with expanding window.
    Returns fold-specific NSE values, mean, std, and CV.
    """
    n = len(data)
    fold_size = n // (n_folds + 1)

    nse_scores = []
    fold_details = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 2)
        test_start = train_end
        test_end = test_start + fold_size

        X_train = data[:train_end]
        y_train = targets[:train_end]
        X_test = data[test_start:test_end]
        y_test = targets[test_start:test_end]

        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        nse = calculate_nse(y_test, y_pred)
        nse_scores.append(nse)

        fold_details.append(
            {
                "fold": fold + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "nse": float(nse),
            }
        )

    nse_scores = np.array(nse_scores)
    mean_nse = np.mean(nse_scores)
    std_nse = np.std(nse_scores, ddof=1)
    cv = std_nse / mean_nse if mean_nse != 0 else np.inf

    return {
        "fold_scores": nse_scores,
        "fold_details": fold_details,
        "mean_nse": float(mean_nse),
        "std_nse": float(std_nse),
        "cv": float(cv),
        "n_folds": n_folds,
    }
