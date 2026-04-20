import numpy as np
from scipy.signal import correlate


def peak_flow_error(y_true, y_pred):
    peaks_true = np.max(y_true, axis=0)
    peaks_pred = np.max(y_pred, axis=0)
    return np.abs(peaks_pred - peaks_true) / peaks_true * 100


def time_to_peak_error(y_true, y_pred):
    time_true = np.argmax(y_true, axis=0)
    time_pred = np.argmax(y_pred, axis=0)
    return np.abs(time_pred - time_true)


def volume_error(y_true, y_pred):
    volume_true = np.sum(y_true, axis=0)
    volume_pred = np.sum(y_pred, axis=0)
    return np.abs(volume_pred - volume_true) / volume_true * 100


def lag_time(y_true, y_pred):
    n_sensors = y_true.shape[1]
    lags = np.zeros(n_sensors)
    for i in range(n_sensors):
        correlation = correlate(y_true[:, i], y_pred[:, i], mode="full")
        lag_idx = np.argmax(correlation)
        lags[i] = lag_idx - (len(y_true) - 1)
    return np.abs(lags)


def compute_hydrological_metrics(y_true, y_pred):
    peak_error = peak_flow_error(y_true, y_pred)
    time_error = time_to_peak_error(y_true, y_pred)
    vol_error = volume_error(y_true, y_pred)
    lag = lag_time(y_true, y_pred)

    return {
        "per_sensor": {
            "Peak_Flow_Error": peak_error,
            "Time_to_Peak_Error": time_error,
            "Volume_Error": vol_error,
            "Lag_Time": lag,
        },
        "aggregated": {
            "Peak_Flow_Error": np.mean(peak_error),
            "Time_to_Peak_Error": np.mean(time_error),
            "Volume_Error": np.mean(vol_error),
            "Lag_Time": np.mean(lag),
        },
    }
