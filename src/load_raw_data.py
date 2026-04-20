import geopandas as gpd
import pandas as pd

from src.data_utils import save_pickle, save_unified_data
from src.project_config import (
    ASSET_FILES,
    DATA_INTERIM,
    DOWNLOADED,
    MET_VARIABLES,
    RAIN_GAUGE_FILES,
    RAIN_GAUGE_IDS,
)


def load_sensor_csv(filepath):
    df = pd.read_csv(
        filepath,
        usecols=["time", "depth_s", "level"],
        parse_dates=["time"],
    )
    return df.dropna(subset=["depth_s"])


def load_rain_gauge(filepath):
    df = pd.read_csv(filepath, sep=";", skiprows=2, parse_dates=[0])
    df.columns = ["time", "intensity"]
    return df


def load_radar(filepath):
    df = pd.read_csv(filepath, sep=";", skiprows=3, parse_dates=[0])
    df.columns = ["time", "intensity"]
    return df


def load_merged_sensors():
    sensors = {}

    for filepath in sorted((DOWNLOADED / "2_cleaned_data").glob("*.csv")):
        sensor_id = filepath.stem.replace("_proc_v6", "")
        base_name = "_".join(sensor_id.split("_")[:-1])
        sensors.setdefault(base_name, []).append(load_sensor_csv(filepath))

    merged_sensors = {}
    for base_name, parts in sensors.items():
        if len(parts) == 1:
            merged_sensors[base_name] = parts[0]
            continue

        merged_sensors[base_name] = (
            pd.concat(parts, ignore_index=False)
            .sort_values("time")
            .drop_duplicates(subset=["time"])
        )

    return merged_sensors


def load_rain_gauges():
    return {
        gauge_id: load_rain_gauge(DOWNLOADED / RAIN_GAUGE_FILES[gauge_id])
        for gauge_id in RAIN_GAUGE_IDS
    }


def load_radar_collection(directory):
    return {
        filepath.stem.split("_")[-1]: load_radar(filepath)
        for filepath in sorted(directory.glob("*.txt"))
    }


def load_meteorological_data():
    met_dir = DOWNLOADED / "3b_Meterologicalstation"
    return {
        var: pd.concat(
            [
                pd.read_pickle(filepath)
                for filepath in sorted(met_dir.glob(f"dmi_{var}_*.p"))
            ]
        ).sort_index()
        for var in MET_VARIABLES
    }


def build_sensor_unified(merged_sensors):
    sensor_frames = []

    for sensor_name, sensor_df in merged_sensors.items():
        sensor_frame = sensor_df[["time", "depth_s"]].copy()
        sensor_frame.columns = ["time", sensor_name]
        sensor_frames.append(sensor_frame.dropna(subset=[sensor_name]))

    sensor_unified = sensor_frames[0]
    for sensor_frame in sensor_frames[1:]:
        sensor_unified = sensor_unified.merge(sensor_frame, on="time", how="outer")

    return sensor_unified.sort_values("time").reset_index(drop=True)


def rain_column_name(gauge_id):
    return "rain_aabakken" if gauge_id.lower() == "aabakken" else f"rain_{gauge_id}"


def build_rain_unified(rain_gauges):
    gauge_frames = [
        rain_gauges[gauge_id].rename(columns={"intensity": rain_column_name(gauge_id)})
        for gauge_id in RAIN_GAUGE_IDS
    ]

    rain_unified = gauge_frames[0]
    for gauge_frame in gauge_frames[1:]:
        rain_unified = rain_unified.merge(gauge_frame, on="time", how="outer")

    return rain_unified.sort_values("time").reset_index(drop=True)


def main():
    merged_sensors = load_merged_sensors()
    rain_gauges = load_rain_gauges()
    xband_data = load_radar_collection(DOWNLOADED / "Local_X-band")
    cband_data = load_radar_collection(DOWNLOADED / "DMI_C-band")
    met_data = load_meteorological_data()

    links = gpd.read_file(DOWNLOADED / ASSET_FILES["links"])
    nodes = gpd.read_file(DOWNLOADED / ASSET_FILES["nodes"])

    sensor_unified = build_sensor_unified(merged_sensors)
    rain_unified = build_rain_unified(rain_gauges)

    save_unified_data(sensor_unified, rain_unified)
    save_pickle(merged_sensors, DATA_INTERIM / "merged_sensors.pkl")
    save_pickle(rain_gauges, DATA_INTERIM / "rain_gauges.pkl")
    save_pickle(xband_data, DATA_INTERIM / "xband_data.pkl")
    save_pickle(cband_data, DATA_INTERIM / "cband_data.pkl")
    save_pickle(met_data, DATA_INTERIM / "met_data.pkl")

    links.to_file(DATA_INTERIM / "links.shp")
    nodes.to_file(DATA_INTERIM / "nodes.shp")


if __name__ == "__main__":
    main()
