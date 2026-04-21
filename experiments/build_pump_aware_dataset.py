from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data_utils import (
    load_feature_names,
    load_norm_params,
    load_regression_arrays,
    load_target_sensors,
    save_pickle,
    save_regression_arrays,
    save_regression_metadata,
)
from src.project_config import PIPE_DIAMETERS

PUMP_FEATURE_NAMES = [
    "g80_pump_above_startup",
    "g80_pump_below_shutoff",
    "g80_pump_hysteresis_band",
    "g80_pump_margin_to_startup",
    "g80_pump_margin_to_shutoff",
]


def write_json(path: str | Path, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a lightweight pump-aware branch from existing "
            "data/final_regression reduced tensors."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="data/final_regression",
        help="Directory containing existing reduced regression tensors.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/final_regression_pump_aware",
        help="Directory where the augmented reduced tensors will be written.",
    )
    parser.add_argument(
        "--pump-feature",
        default="G80F13P_LevelPS_norm",
        help="Reduced feature channel used to derive pump-aware indicators.",
    )
    parser.add_argument(
        "--pump-sensor-id",
        default="G80F13P",
        help="Base sensor id used to look up pipe diameter.",
    )
    parser.add_argument(
        "--startup-depth",
        type=float,
        default=1.331,
        help="Pump startup depth in meters from the SWMM model.",
    )
    parser.add_argument(
        "--shutoff-depth",
        type=float,
        default=0.581,
        help="Pump shutoff depth in meters from the SWMM model.",
    )
    return parser.parse_args()


def normalize_threshold_for_feature(
    *,
    feature_name: str,
    raw_divided_threshold: float,
) -> float:
    norm_params = load_norm_params()
    if feature_name not in norm_params:
        raise KeyError(f"Normalization params missing feature: {feature_name}")

    params = norm_params[feature_name]
    norm_type = params["type"]
    if norm_type == "minmax":
        denominator = float(params["max"]) - float(params["min"])
        if denominator == 0.0:
            raise ValueError(
                f"Cannot normalize threshold for {feature_name}: max == min."
            )
        return float((raw_divided_threshold - float(params["min"])) / denominator)
    if norm_type == "zscore":
        sigma = float(params["sigma"])
        if sigma == 0.0:
            raise ValueError(
                f"Cannot normalize threshold for {feature_name}: sigma == 0."
            )
        return float((raw_divided_threshold - float(params["mu"])) / sigma)
    if norm_type == "none":
        return float(raw_divided_threshold)
    raise ValueError(f"Unsupported normalization type for {feature_name}: {norm_type}")


def build_pump_aware_channels(
    pump_signal: np.ndarray,
    *,
    startup_threshold: float,
    shutoff_threshold: float,
) -> np.ndarray:
    above_startup = (pump_signal >= startup_threshold).astype(np.float32)
    below_shutoff = (pump_signal <= shutoff_threshold).astype(np.float32)
    hysteresis_band = (
        (pump_signal > shutoff_threshold) & (pump_signal < startup_threshold)
    ).astype(np.float32)
    margin_to_startup = (pump_signal - startup_threshold).astype(np.float32)
    margin_to_shutoff = (pump_signal - shutoff_threshold).astype(np.float32)
    return np.stack(
        [
            above_startup,
            below_shutoff,
            hysteresis_band,
            margin_to_startup,
            margin_to_shutoff,
        ],
        axis=-1,
    )


def augment_reduced_split(
    *,
    split: str,
    input_dir: Path,
    output_dir: Path,
    pump_feature_name: str,
    startup_threshold: float,
    shutoff_threshold: float,
) -> tuple[list[str], dict[str, int]]:
    feature_names = load_feature_names(use_reduced=True, data_dir=input_dir)
    if pump_feature_name not in feature_names:
        raise ValueError(
            f"Pump feature '{pump_feature_name}' not found in {input_dir}/feature_names_reduced.pkl."
        )

    arrays = load_regression_arrays(split, use_reduced=True, data_dir=input_dir)
    pump_index = feature_names.index(pump_feature_name)
    pump_signal = arrays["X"][:, :, pump_index]
    pump_aware_channels = build_pump_aware_channels(
        pump_signal,
        startup_threshold=startup_threshold,
        shutoff_threshold=shutoff_threshold,
    )
    augmented_X = np.concatenate([arrays["X"], pump_aware_channels], axis=-1).astype(
        np.float32
    )
    save_regression_arrays(
        split,
        augmented_X,
        arrays["y_depths"],
        arrays["y_overflow"],
        arrays["flood_mask"],
        suffix="_reduced",
        data_dir=output_dir,
    )
    return feature_names, {
        "original_channels": int(arrays["X"].shape[-1]),
        "added_channels": int(pump_aware_channels.shape[-1]),
        "augmented_channels": int(augmented_X.shape[-1]),
    }


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input regression directory does not exist: {input_dir}"
        )

    pipe_diameter = float(PIPE_DIAMETERS[args.pump_sensor_id])
    startup_divided = float(args.startup_depth) / pipe_diameter
    shutoff_divided = float(args.shutoff_depth) / pipe_diameter
    startup_threshold = normalize_threshold_for_feature(
        feature_name=args.pump_feature,
        raw_divided_threshold=startup_divided,
    )
    shutoff_threshold = normalize_threshold_for_feature(
        feature_name=args.pump_feature,
        raw_divided_threshold=shutoff_divided,
    )

    feature_names, channel_summary = augment_reduced_split(
        split="train",
        input_dir=input_dir,
        output_dir=output_dir,
        pump_feature_name=args.pump_feature,
        startup_threshold=startup_threshold,
        shutoff_threshold=shutoff_threshold,
    )
    for split in ("val", "test"):
        augment_reduced_split(
            split=split,
            input_dir=input_dir,
            output_dir=output_dir,
            pump_feature_name=args.pump_feature,
            startup_threshold=startup_threshold,
            shutoff_threshold=shutoff_threshold,
        )

    augmented_feature_names = [*feature_names, *PUMP_FEATURE_NAMES]
    save_regression_metadata(
        load_target_sensors(data_dir=input_dir),
        augmented_feature_names,
        suffix="_reduced",
        data_dir=output_dir,
    )
    save_pickle(
        {
            "Pump Logic": PUMP_FEATURE_NAMES,
            "Original Features": feature_names,
        },
        output_dir / "feature_categories.pkl",
    )

    summary_payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "pump_feature": args.pump_feature,
        "pump_sensor_id": args.pump_sensor_id,
        "pipe_diameter_m": pipe_diameter,
        "startup_depth_m": float(args.startup_depth),
        "shutoff_depth_m": float(args.shutoff_depth),
        "startup_threshold_feature_space": float(startup_threshold),
        "shutoff_threshold_feature_space": float(shutoff_threshold),
        "feature_names_added": PUMP_FEATURE_NAMES,
        **channel_summary,
    }
    write_json(output_dir / "pump_aware_summary.json", summary_payload)

    print(f"Created pump-aware reduced dataset in: {output_dir}")
    print(f"Original channels: {channel_summary['original_channels']}")
    print(f"Added channels:    {channel_summary['added_channels']}")
    print(f"Augmented channels:{channel_summary['augmented_channels']}")
    print(f"Startup threshold in feature space: {startup_threshold:.6f}")
    print(f"Shutoff threshold in feature space: {shutoff_threshold:.6f}")


if __name__ == "__main__":
    main()
