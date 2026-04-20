import hashlib
import pickle
from pathlib import Path

import numpy as np

from src.project_config import DATA_INTERIM, DATA_REGRESSION


def load_pickle(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Required pickle file does not exist: {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_unified_data():
    sensor_data = load_pickle(DATA_INTERIM / "sensor_unified.pkl")
    rain_data = load_pickle(DATA_INTERIM / "rain_unified.pkl")
    return sensor_data, rain_data


def load_cleaned_data():
    sensor_data = load_pickle(DATA_INTERIM / "sensor_cleaned.pkl")
    rain_data = load_pickle(DATA_INTERIM / "rain_cleaned.pkl")
    return sensor_data, rain_data


def load_features():
    return load_pickle(DATA_INTERIM / "features.pkl")


def load_normalized_data():
    train = load_pickle(DATA_INTERIM / "train_normalized.pkl")
    val = load_pickle(DATA_INTERIM / "val_normalized.pkl")
    test = load_pickle(DATA_INTERIM / "test_normalized.pkl")
    return train, val, test


def load_labeled_data(use_augmented_train=False):
    train_file = (
        "train_labeled_augmented.pkl" if use_augmented_train else "train_labeled.pkl"
    )
    train = load_pickle(DATA_INTERIM / train_file)
    val = load_pickle(DATA_INTERIM / "val_labeled.pkl")
    test = load_pickle(DATA_INTERIM / "test_labeled.pkl")
    return train, val, test


def load_norm_params():
    return load_pickle(DATA_INTERIM / "norm_params.pkl")


def get_regression_array_paths(split="train", use_reduced=True):
    suffix = "_reduced" if use_reduced else ""
    return {
        "X": DATA_REGRESSION / f"{split}_X{suffix}.npy",
        "y_depths": DATA_REGRESSION / f"{split}_y_depths.npy",
        "y_overflow": DATA_REGRESSION / f"{split}_y_overflow.npy",
        "flood_mask": DATA_REGRESSION / f"{split}_flood_mask.npy",
    }


def _ensure_required_files_exist(filepaths):
    missing = [str(path) for path in filepaths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing regression artifacts. Expected files:\n- " + "\n- ".join(missing)
        )


def load_regression_arrays(split="train", use_reduced=True):
    paths = get_regression_array_paths(split=split, use_reduced=use_reduced)
    _ensure_required_files_exist(paths.values())
    return {name: np.load(path) for name, path in paths.items()}


def _sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def describe_regression_arrays(split="train", use_reduced=True):
    paths = get_regression_array_paths(split=split, use_reduced=use_reduced)
    _ensure_required_files_exist(paths.values())
    description = {
        "split": split,
        "use_reduced": bool(use_reduced),
        "files": {},
    }
    for name, path in paths.items():
        array = np.load(path, mmap_mode="r")
        description["files"][name] = {
            "path": str(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": _sha256_file(path),
        }
    return description


def load_target_sensors():
    return load_pickle(DATA_REGRESSION / "target_sensors.pkl")


def load_feature_names(use_reduced=True):
    suffix = "_reduced" if use_reduced else ""
    return load_pickle(DATA_REGRESSION / f"feature_names{suffix}.pkl")


def save_unified_data(sensor_data, rain_data):
    save_pickle(sensor_data, DATA_INTERIM / "sensor_unified.pkl")
    save_pickle(rain_data, DATA_INTERIM / "rain_unified.pkl")


def save_cleaned_data(sensor_data, rain_data):
    save_pickle(sensor_data, DATA_INTERIM / "sensor_cleaned.pkl")
    save_pickle(rain_data, DATA_INTERIM / "rain_cleaned.pkl")


def save_features(data, feature_names):
    save_pickle(data, DATA_INTERIM / "features.pkl")
    with open(DATA_INTERIM / "feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))


def save_normalized_data(train, val, test, norm_params):
    save_pickle(train, DATA_INTERIM / "train_normalized.pkl")
    save_pickle(val, DATA_INTERIM / "val_normalized.pkl")
    save_pickle(test, DATA_INTERIM / "test_normalized.pkl")
    save_pickle(norm_params, DATA_INTERIM / "norm_params.pkl")


def save_labeled_data(train, val, test):
    save_pickle(train, DATA_INTERIM / "train_labeled.pkl")
    save_pickle(val, DATA_INTERIM / "val_labeled.pkl")
    save_pickle(test, DATA_INTERIM / "test_labeled.pkl")


def save_regression_arrays(split, X, y_depths, y_overflow, flood_mask, suffix=""):
    np.save(DATA_REGRESSION / f"{split}_X{suffix}.npy", X)
    np.save(DATA_REGRESSION / f"{split}_y_depths.npy", y_depths)
    np.save(DATA_REGRESSION / f"{split}_y_overflow.npy", y_overflow)
    np.save(DATA_REGRESSION / f"{split}_flood_mask.npy", flood_mask)


def save_regression_metadata(target_sensors, feature_names, suffix=""):
    save_pickle(target_sensors, DATA_REGRESSION / "target_sensors.pkl")
    save_pickle(feature_names, DATA_REGRESSION / f"feature_names{suffix}.pkl")
