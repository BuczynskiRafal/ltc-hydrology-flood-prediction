#!/usr/bin/env bash
# prepare_colab_data.sh — Pack data/final_regression/ for Colab

# Usage:
#   cd /path/to/bellinge
#   bash notebooks/prepare_colab_data.sh
#
# Output:
#   bellinge_final_regression.tar.gz  (≈700 MB)
#
# Upload the resulting archive to your Google Drive.
# The Colab notebook expects it at:
#   My Drive/bellinge/bellinge_final_regression.tar.gz

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/final_regression"
OUTPUT_FILE="${PROJECT_ROOT}/bellinge_final_regression.tar.gz"

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: data/final_regression/ not found at ${DATA_DIR}"
    echo "Run the data pipeline first (see downloaded/readme.md)."
    exit 1
fi

# Verify critical files exist
REQUIRED_FILES=(
    "train_X_reduced.npy"
    "train_y_depths.npy"
    "train_y_overflow.npy"
    "train_flood_mask.npy"
    "val_X_reduced.npy"
    "val_y_depths.npy"
    "val_y_overflow.npy"
    "val_flood_mask.npy"
    "test_X_reduced.npy"
    "test_y_depths.npy"
    "test_y_overflow.npy"
    "test_flood_mask.npy"
    "feature_names_reduced.pkl"
    "target_sensors.pkl"
)

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${DATA_DIR}/${f}" ]; then
        echo "MISSING: ${f}"
        MISSING=$((MISSING + 1))
    fi
done

if [ "${MISSING}" -gt 0 ]; then
    echo "ERROR: ${MISSING} required file(s) missing. Cannot create archive."
    exit 1
fi

echo "Packing data/final_regression/ → ${OUTPUT_FILE}"
echo "Files:"
ls -lh "${DATA_DIR}"/*.npy "${DATA_DIR}"/*.pkl 2>/dev/null | awk '{print "  " $5 "\t" $NF}'

tar -czf "${OUTPUT_FILE}" -C "${PROJECT_ROOT}" \
    data/final_regression/train_X_reduced.npy \
    data/final_regression/train_y_depths.npy \
    data/final_regression/train_y_overflow.npy \
    data/final_regression/train_flood_mask.npy \
    data/final_regression/val_X_reduced.npy \
    data/final_regression/val_y_depths.npy \
    data/final_regression/val_y_overflow.npy \
    data/final_regression/val_flood_mask.npy \
    data/final_regression/test_X_reduced.npy \
    data/final_regression/test_y_depths.npy \
    data/final_regression/test_y_overflow.npy \
    data/final_regression/test_flood_mask.npy \
    data/final_regression/feature_names_reduced.pkl \
    data/final_regression/target_sensors.pkl \
    data/final_regression/feature_categories.pkl \
    data/final_regression/feature_names.pkl \
    2>/dev/null || true

ARCHIVE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
echo ""
echo "Done. Archive: ${OUTPUT_FILE} (${ARCHIVE_SIZE})"
echo ""
echo "Next steps:"
echo "  1. Upload ${OUTPUT_FILE} to Google Drive at: My Drive/bellinge/"
echo "  2. Open the Colab notebook: notebooks/bellinge_full_pipeline.ipynb"
