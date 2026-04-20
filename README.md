# LTC Hydrology Flood Prediction

This repository is a publication-oriented snapshot of the Bellinge flash-flood study.
It separates manuscript-ready code and results from the broader working repository used
for iterative experimentation.

## What Is Included

- `src/`:
  canonical data pipeline, model definitions, evaluation utilities, and training losses
- `experiments/`:
  CLI entrypoints, regression pipeline, benchmark scripts, ablation tooling, and
  publication utilities
- `configs/`:
  model configuration files used by the regression experiments
- `tests/`:
  regression and publication-safety checks
- `notebooks/bellinge_full_pipeline.ipynb`:
  Colab-oriented notebook for rebuilding the pipeline
- `notebooks/prepare_colab_data.sh`:
  helper script for Colab/session setup
- `notebooks/article_materials_20260419/`:
  final tables, benchmark summaries, baseline metrics/manifests, LNN config snapshots,
  and direct Google Drive links to large experimental artifacts

## Publication Snapshot

The current manuscript-facing main result is the `dual-branch LNN`, which achieved:

- `all_sensors NSE = 0.8489`
- `non_pump_mean NSE = 0.9463`
- `pump_only NSE = 0.4594`

The final article comparison tables are available in:

- `notebooks/article_materials_20260419/benchmark/article_main_table.csv`
- `notebooks/article_materials_20260419/benchmark/article_main_table.md`
- `notebooks/article_materials_20260419/benchmark/grouped_summary.csv`
- `notebooks/article_materials_20260419/benchmark/nse_per_sensor_pivot.csv`

## Reproduction Scope

This repository contains the code and result artifacts that should be versioned in Git.
Large runtime assets are intentionally excluded:

- raw downloaded inputs in `downloaded/`
- generated arrays in `data/final_regression/`
- training artifacts in `artifacts/`
- temporary notebook outputs in `notebooks/release/`
- local tar archives such as `bellinge_final_regression.tar.gz`

Direct Google Drive links for large model outputs and benchmark folders are stored in:

- `notebooks/article_materials_20260419/DRIVE_LINKS.md`

## Minimal Reproduction Flow

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. Rebuild or restore the prepared regression arrays.
   Use the Colab notebook or your local workflow to populate `data/final_regression/`.

3. Train or evaluate canonical models through the shared CLI:

```bash
python -m experiments.cli train gru
python -m experiments.cli train lnn
python -m experiments.evaluate_release --model gru
```

4. Recreate publication tables from saved `metrics.json` files:

```bash
python -m experiments.final_model_comparison \
  --model GRU=/abs/path/to/gru_metrics.json \
  --model "dual-branch LNN"=/abs/path/to/dual_branch_metrics.json \
  --output-dir notebooks/release/final_article_benchmark
```

## Notes About Restored Scripts

Three publication-critical helper scripts were restored into this repository from the
history of the working repository because they were used in the final study but were no
longer present in the latest working tree:

- `experiments/final_model_comparison.py`
- `experiments/build_pump_aware_dataset.py`
- `experiments/evaluate_depth_only_release.py`

They are included here deliberately so the article repository remains self-contained.
