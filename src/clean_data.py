import numpy as np
import pandas as pd

from src.data_utils import load_unified_data, save_cleaned_data
from src.project_config import OUTPUT_REPORTS, RAIN_MAX_INTENSITY

sensor_data, rain_data = load_unified_data()

sensor_cols = [c for c in sensor_data.columns if c != "time"]
completeness = {
    col: sensor_data[col].notna().sum() / len(sensor_data) for col in sensor_cols
}
pd.DataFrame.from_dict(completeness, orient="index", columns=["completeness"]).to_csv(
    OUTPUT_REPORTS / "completeness_report.csv"
)

rain_data_cleaned = rain_data.copy()
rain_cols = [c for c in rain_data.columns if c != "time"]
for col in rain_cols:
    rain_data_cleaned.loc[rain_data_cleaned[col] < 0, col] = 0
    rain_data_cleaned.loc[rain_data_cleaned[col] > RAIN_MAX_INTENSITY, col] = np.nan

save_cleaned_data(sensor_data.copy(), rain_data_cleaned)
