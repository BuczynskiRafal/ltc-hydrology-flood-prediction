import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_REGRESSION = Path("data/final_regression")
OUTPUT_DIR = Path("output/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR = Path("output/visualizations")
VIZ_DIR.mkdir(parents=True, exist_ok=True)

train_X = np.load(DATA_REGRESSION / "train_X.npy")
train_y_depths = np.load(DATA_REGRESSION / "train_y_depths.npy")
train_y_overflow = np.load(DATA_REGRESSION / "train_y_overflow.npy")
train_flood_mask = np.load(DATA_REGRESSION / "train_flood_mask.npy")

val_X = np.load(DATA_REGRESSION / "val_X.npy")
val_y_depths = np.load(DATA_REGRESSION / "val_y_depths.npy")
val_y_overflow = np.load(DATA_REGRESSION / "val_y_overflow.npy")
val_flood_mask = np.load(DATA_REGRESSION / "val_flood_mask.npy")

test_X = np.load(DATA_REGRESSION / "test_X.npy")
test_y_depths = np.load(DATA_REGRESSION / "test_y_depths.npy")
test_y_overflow = np.load(DATA_REGRESSION / "test_y_overflow.npy")
test_flood_mask = np.load(DATA_REGRESSION / "test_flood_mask.npy")

with open(DATA_REGRESSION / "target_sensors.pkl", "rb") as f:
    target_sensors = pickle.load(f)
with open(DATA_REGRESSION / "feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

stats_list = []

for split_name, y_depths in [
    ("Train", train_y_depths),
    ("Val", val_y_depths),
    ("Test", test_y_depths),
]:
    for i, sensor in enumerate(target_sensors):
        depths = y_depths[:, i]
        depths_clean = depths[~np.isnan(depths)]

        if len(depths_clean) == 0:
            continue

        stats_list.append(
            {
                "Split": split_name,
                "Sensor": sensor,
                "Count": len(depths_clean),
                "Mean": np.mean(depths_clean),
                "Std": np.std(depths_clean),
                "Min": np.min(depths_clean),
                "P5": np.percentile(depths_clean, 5),
                "Q1": np.percentile(depths_clean, 25),
                "Median": np.percentile(depths_clean, 50),
                "Q3": np.percentile(depths_clean, 75),
                "P95": np.percentile(depths_clean, 95),
                "Max": np.max(depths_clean),
                "NaN_count": np.sum(np.isnan(depths)),
                "NaN_pct": 100 * np.sum(np.isnan(depths)) / len(depths),
            }
        )

pd.DataFrame(stats_list).to_csv(
    OUTPUT_DIR / "regression_depth_statistics.csv", index=False
)

train_depths_df = pd.DataFrame(train_y_depths, columns=target_sensors)
corr_matrix = train_depths_df.corr()
corr_matrix.to_csv(OUTPUT_DIR / "regression_sensor_correlation.csv")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, sensor in enumerate(target_sensors):
    ax = axes[i]
    all_depths = np.concatenate(
        [train_y_depths[:, i], val_y_depths[:, i], test_y_depths[:, i]]
    )
    all_depths = all_depths[~np.isnan(all_depths)]

    ax.hist(all_depths, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax.set_title(f"{sensor}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Depth value", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.grid(True, alpha=0.3)

    stats_text = f"Mean: {np.mean(all_depths):.3f}\nStd: {np.std(all_depths):.3f}\nMedian: {np.median(all_depths):.3f}"
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=8,
    )

plt.tight_layout()
plt.savefig(
    VIZ_DIR / "regression_depth_distributions.png", dpi=150, bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"label": "Pearson correlation"},
)
plt.title(
    "Correlation between Target Sensors (Train split)", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(
    VIZ_DIR / "regression_sensor_correlation_heatmap.png", dpi=150, bbox_inches="tight"
)
plt.close()

train_overflow_count = np.sum(train_y_overflow)
val_overflow_count = np.sum(val_y_overflow)
test_overflow_count = np.sum(test_y_overflow)

overflow_data = {
    "Split": ["Train", "Val", "Test"],
    "Total": [len(train_y_overflow), len(val_y_overflow), len(test_y_overflow)],
    "Overflow": [train_overflow_count, val_overflow_count, test_overflow_count],
    "No Overflow": [
        len(train_y_overflow) - train_overflow_count,
        len(val_y_overflow) - val_overflow_count,
        len(test_y_overflow) - test_overflow_count,
    ],
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x_pos = np.arange(len(overflow_data["Split"]))
ax1.bar(x_pos, overflow_data["No Overflow"], label="No Overflow", color="lightblue")
ax1.bar(
    x_pos,
    overflow_data["Overflow"],
    bottom=overflow_data["No Overflow"],
    label="Overflow",
    color="salmon",
)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(overflow_data["Split"])
ax1.set_ylabel("Number of sequences")
ax1.set_title("Overflow Distribution Across Splits", fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

overflow_pct = [
    100 * overflow_data["Overflow"][i] / overflow_data["Total"][i] for i in range(3)
]
ax2.bar(x_pos, overflow_pct, color=["steelblue", "seagreen", "coral"])
ax2.set_xticks(x_pos)
ax2.set_xticklabels(overflow_data["Split"])
ax2.set_ylabel("Overflow %")
ax2.set_title("Overflow Percentage by Split", fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")

for i, pct in enumerate(overflow_pct):
    ax2.text(i, pct + 0.01, f"{pct:.3f}%", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig(
    VIZ_DIR / "regression_overflow_distribution.png", dpi=150, bbox_inches="tight"
)
plt.close()

total_sequences = train_X.shape[0] + val_X.shape[0] + test_X.shape[0]
summary_lines = [
    f"Total sequences: {total_sequences:,}",
    f"Train: {train_X.shape[0]:,} ({100*train_X.shape[0]/total_sequences:.1f}%)",
    f"Val: {val_X.shape[0]:,} ({100*val_X.shape[0]/total_sequences:.1f}%)",
    f"Test: {test_X.shape[0]:,} ({100*test_X.shape[0]/total_sequences:.1f}%)",
]

with open(OUTPUT_DIR / "regression_data_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))
