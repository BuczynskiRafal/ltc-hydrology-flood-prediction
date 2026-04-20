import numpy as np
import pandas as pd

from src.logger import get_console_logger
from src.project_config import (
    FLOOD_BUFFER_MINUTES,
    FLOOD_WEIGHT_FLOOD,
    FLOOD_WEIGHT_NORMAL,
    WINDOW_STRIDE,
    WINDOW_T_IN,
    WINDOW_T_OUT,
)

logger = get_console_logger(__name__)


def create_flood_mask(df, flash_flood_periods, buffer_minutes=FLOOD_BUFFER_MINUTES):
    mask = np.ones(len(df), dtype=np.float32) * FLOOD_WEIGHT_NORMAL

    for _, period in flash_flood_periods.iterrows():
        start_buffer = period["timestamp"] - pd.Timedelta(minutes=buffer_minutes)
        end_buffer = period["period_end"] + pd.Timedelta(minutes=buffer_minutes)
        idx_mask = (df["time"] >= start_buffer) & (df["time"] <= end_buffer)
        mask[idx_mask] = FLOOD_WEIGHT_FLOOD

    return mask


def create_sliding_windows(
    df,
    feature_cols,
    target_cols,
    target_col="target",
    T_in=WINDOW_T_IN,
    T_out=WINDOW_T_OUT,
    stride=WINDOW_STRIDE,
    timestep_weights=None,
    allow_missing_features=False,
):
    df_sorted = df.sort_values("time").reset_index(drop=True)
    n_samples = len(df_sorted)

    if timestep_weights is None:
        timestep_weights = np.ones(n_samples, dtype=np.float32)

    X_list, y_values_list, y_binary_list, weights_list = [], [], [], []
    skipped_nan_windows = 0

    max_start = n_samples - T_in - T_out + 1
    for start_idx in range(0, max_start, stride):
        window_df = df_sorted.iloc[start_idx : start_idx + T_in]
        target_idx = start_idx + T_in + T_out - 1

        if target_idx >= n_samples:
            break

        feature_frame = window_df[feature_cols]
        if not allow_missing_features and feature_frame.isna().any().any():
            missing_cols = feature_frame.columns[feature_frame.isna().any()].tolist()
            raise ValueError(
                "Found missing predictor values inside a regression window for "
                f"columns {missing_cols}. Canonical windowing requires complete "
                "predictor windows."
            )
        X = feature_frame.values.astype(np.float32)

        target_row = df_sorted.iloc[target_idx]
        y_values = [target_row[col] for col in target_cols]

        if any(pd.isna(v) for v in y_values):
            skipped_nan_windows += 1
            continue

        y_binary = target_row[target_col]

        if pd.isna(y_binary):
            skipped_nan_windows += 1
            continue

        weight = timestep_weights[target_idx]

        X_list.append(X)
        y_values_list.append(y_values)
        y_binary_list.append(y_binary)
        weights_list.append(weight)

    if skipped_nan_windows:
        logger.info(
            "WARNING: "
            f"Skipped {skipped_nan_windows:,} windows due to NaN target values or "
            f"missing overflow label in '{target_col}'."
        )

    if not X_list:
        return (
            np.empty((0, T_in, len(feature_cols)), dtype=np.float32),
            np.empty((0, len(target_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_values_list, dtype=np.float32),
        np.array(y_binary_list, dtype=np.int32),
        np.array(weights_list, dtype=np.float32),
    )


def split_dataframe_by_time_gaps(df, max_gap_minutes=1):
    df_sorted = df.sort_values("time").reset_index(drop=True)
    if df_sorted.empty:
        return [df_sorted]

    time_delta = df_sorted["time"].diff()
    gap_mask = time_delta > pd.Timedelta(minutes=max_gap_minutes)
    segment_ids = gap_mask.cumsum()
    return [
        segment.reset_index(drop=True)
        for _, segment in df_sorted.groupby(segment_ids, sort=False)
    ]


def create_sequences_with_regression_targets(
    df,
    flash_flood_periods,
    feature_cols,
    target_sensors,
    gap_minutes=1,
    allow_missing_features=False,
):
    split_sequences = []
    for segment in split_dataframe_by_time_gaps(df, max_gap_minutes=gap_minutes):
        if len(segment) < (WINDOW_T_IN + WINDOW_T_OUT):
            continue
        flood_mask = create_flood_mask(segment, flash_flood_periods)
        split_sequences.append(
            create_sliding_windows(
                df=segment,
                feature_cols=feature_cols,
                target_cols=target_sensors,
                target_col="target",
                timestep_weights=flood_mask,
                allow_missing_features=allow_missing_features,
            )
        )

    if not split_sequences:
        return (
            np.empty((0, WINDOW_T_IN, len(feature_cols)), dtype=np.float32),
            np.empty((0, len(target_sensors)), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    X_list, y_depths_list, y_overflow_list, weights_list = zip(*split_sequences)
    return (
        np.concatenate(X_list, axis=0),
        np.concatenate(y_depths_list, axis=0),
        np.concatenate(y_overflow_list, axis=0),
        np.concatenate(weights_list, axis=0),
    )


def print_sequence_stats(split_name, X, y_values, y_binary, weights, target_names):
    if len(X) == 0:
        logger.info(f"{split_name}: 0 sequences generated")
        return

    overflow_count = int(np.sum(y_binary))
    overflow_pct = 100.0 * overflow_count / len(y_binary)
    unique_weights, counts = np.unique(weights, return_counts=True)
    weight_summary = ", ".join(
        f"{float(weight):.1f}={int(count)}"
        for weight, count in zip(unique_weights, counts)
    )

    logger.info(
        f"{split_name}: {len(X):,} sequences, "
        f"{overflow_count:,} overflow ({overflow_pct:.2f}%), "
        f"X={X.shape}, y_depths={y_values.shape}, "
        f"weights[{weight_summary}]"
    )
