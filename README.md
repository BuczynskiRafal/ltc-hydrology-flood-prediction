# LTC Hydrology Flood Prediction

This repository is the publication snapshot of the Bellinge flash-flood study.
It packages the canonical pipeline, model definitions, evaluation utilities,
and final benchmark tables behind the manuscript's central result: a
**dual-branch Liquid Neural Network** that materially improves pump-sensor
forecasting while preserving accuracy on passive water-level sensors.


## Repository Layout

- `src/`:
  canonical data pipeline, model definitions (`GRU`, `LSTM`, `TCN`, `MLP`,
  `LNN` with `HierarchicalLTC` + `FlashFloodGate`), losses, hydrological
  metrics, and uncertainty utilities
- `experiments/`:
  shared CLI (`cli.py`), regression pipeline, benchmark / ablation /
  robustness / interpretability scripts, and publication helpers
- `configs/`:
  per-model training configs (`gru`, `lstm`, `tcn`, `mlp`, `lnn`)
- `tests/`:
  regression contracts, publication-script smoke tests, CLI tests
- `notebooks/bellinge_full_pipeline.ipynb`:
  Colab-oriented notebook for rebuilding the pipeline
- `notebooks/prepare_colab_data.sh`:
  helper script for Colab/session setup
- `notebooks/article_materials_20260419/`:
  manuscript snapshot — see breakdown below
- `REPRODUCTION_AND_RESULTS.md`:
  manifest of git-tracked vs. Drive-hosted assets

### `notebooks/article_materials_20260419/`

- `benchmark/`:
  final article tables — `article_main_table.csv`, `grouped_summary.csv`,
  per-sensor pivots (`nse_per_sensor_pivot.csv`, `mae_per_sensor_pivot.csv`,
  `rmse_per_sensor_pivot.csv`), `model_summary.csv`, `model_catalog.csv`,
  `comparison_summary.json`
- `baselines/{GRU,LSTM,TCN,MLP}/`:
  metrics and manifests for the four classical baselines
- `lnn_variants/{stable_seed42,pump_aware_global,dual_branch}/config.yaml`:
  exact configs of the three LNN variants discussed in the article
- `DRIVE_LINKS.md`:
  direct Drive links to large artifacts (checkpoints, raw `predictions.npz`,
  full benchmark folder)

## Models

| Family | Config / Snapshot | Notes |
|---|---|---|
| GRU | `configs/gru_regression_config.yaml` | 3 layers, hidden 128, baseline |
| LSTM | `configs/lstm_regression_config.yaml` | 2 layers, hidden 128 |
| TCN | `configs/tcn_regression_config.yaml` | causal kernel 3, 3 blocks |
| MLP | `configs/mlp_regression_config.yaml` | dense [256,128,64], no temporal structure |
| LNN | `configs/lnn_regression_config.yaml` | base LNN reference config |
| LNN — stable seed42 | `notebooks/article_materials_20260419/lnn_variants/stable_seed42/config.yaml` | depth-only LNN, per-neuron tau, separate heads |
| LNN — pump-aware global | `notebooks/article_materials_20260419/lnn_variants/pump_aware_global/config.yaml` | + 5 pump features fed through the shared encoder |
| LNN — **dual-branch** (winner) | `notebooks/article_materials_20260419/lnn_variants/dual_branch/config.yaml` | dedicated pump encoder (`pump_branch_use_attention=true`) joined with the main branch |

## Headline Results

The publication-facing model is the **dual-branch LNN**. Aggregated benchmark
metrics on the held-out test set (all values from
`notebooks/article_materials_20260419/benchmark/article_main_table.csv`):

| Model | all_NSE | non_pump_NSE | pump_NSE |
|---|---|---|---|
| **dual-branch LNN** | **0.8489** | **0.9463** | **0.4594** |
| GRU | 0.7831 | 0.9091 | 0.2792 |
| best stable LNN | 0.7581 | 0.9442 | 0.0135 |
| TCN | 0.7302 | 0.9098 | 0.0115 |
| LSTM | 0.7157 | 0.8634 | 0.1248 |
| MLP | 0.6690 | 0.8440 | -0.0311 |

Full per-sensor breakdowns:

- `notebooks/article_materials_20260419/benchmark/grouped_summary.csv`
- `notebooks/article_materials_20260419/benchmark/nse_per_sensor_pivot.csv`
- `notebooks/article_materials_20260419/benchmark/mae_per_sensor_pivot.csv`
- `notebooks/article_materials_20260419/benchmark/rmse_per_sensor_pivot.csv`

## Dataset & Windowing

- **Targets.** 5 sensors — `G71F05R_LevelBasin`, `G71F05R_LevelInlet`,
  `G71F05R_position`, `G71F68Y_LevelPS`, `G80F13P_LevelPS` (pump).
- **Features.** 31 base inputs; 36 in the pump-aware datasets (5 engineered
  pump features such as margin-to-startup and hysteresis band, indices
  `[31..35]`).
- **Windowing.** `WINDOW_T_IN=45`, `WINDOW_T_OUT=5`, stride `1`, time-gap
  aware splits (no leakage across non-contiguous periods).
- **Pump-aware tensors.** Materialized into `data/final_regression_pump_aware/`
  via `experiments/build_pump_aware_dataset.py`; required by
  `pump_aware_global` and `dual_branch` LNN variants.

## Reproduction

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. Restore prepared regression arrays into `data/final_regression/`
   (Colab notebook or local workflow). The pump-aware variants additionally
   require `data/interim/norm_params.pkl` (used by `src.data_utils.load_norm_params`
   to project pump startup/shutoff thresholds into the normalized feature space)
   and are built with:

```bash
python -m experiments.build_pump_aware_dataset
```

3. Train or evaluate any baseline or LNN variant:

```bash
python -m experiments.cli train gru        # also: lstm, tcn, mlp, lnn
python -m experiments.cli evaluate lnn
python -m experiments.evaluate_release --model gru
```

   To reproduce the winning **dual-branch LNN** end-to-end, point the CLI at
   the variant config snapshot:

```bash
python -m experiments.cli train lnn \
  --config notebooks/article_materials_20260419/lnn_variants/dual_branch/config.yaml

python -m experiments.cli evaluate lnn \
  --config notebooks/article_materials_20260419/lnn_variants/dual_branch/config.yaml
```

   The same `--config` pattern reproduces `stable_seed42` and
   `pump_aware_global` from their respective snapshots.

4. Reproduce the article benchmark table:

```bash
python -m experiments.final_model_comparison \
  --model GRU=/abs/path/to/gru_metrics.json \
  --model "dual-branch LNN"=/abs/path/to/dual_branch_metrics.json \
  --output-dir notebooks/release/final_article_benchmark
```

5. Optional supplementary studies used in the manuscript:

```bash
python -m experiments.run_ablations              # LNN component ablations
python -m experiments.eval_robustness            # input-noise robustness
python -m experiments.eval_ig                    # Integrated Gradients
python -m experiments.evaluate_lnn_uncertainty   # Delta-Method aleatoric
python -m experiments.statistical_comparison     # Wilcoxon / permutation
python -m experiments.cross_validation_comparison
python -m experiments.benchmark_inference_time
```

## Tests

```bash
pytest
```

Coverage:

- `tests/test_publication_safe_refactor.py` — model-shape contracts, LNN tau
  semantics, deterministic seeding, time-gap windowing, schema validation,
  embedded checkpoint config loading.
- `tests/test_publication_scripts.py` — smoke tests for the publication
  scripts (`run_ablations`, `eval_robustness`, `eval_ig`, training pipeline)
  on synthetic data.
- `tests/test_cli.py` — CLI argument parsing and subcommand dispatch.
- `tests/test_env.py` — environment sanity.

## Large Artifacts & Drive Links

These runtime-sized assets are intentionally excluded from git:

- raw downloaded inputs in `downloaded/`
- prepared arrays in `data/final_regression/`,
  `data/final_regression_pump_aware/`
- training artifacts in `artifacts/`
- temporary notebook outputs in `notebooks/release/`
- local tar archives such as `bellinge_final_regression.tar.gz`

Direct Google Drive links for the LNN variant folders, raw `predictions.npz`,
and the full benchmark folder:

- `notebooks/article_materials_20260419/DRIVE_LINKS.md`

A complete manifest of git-tracked vs. Drive-hosted assets:

- `REPRODUCTION_AND_RESULTS.md`
