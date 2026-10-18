# LTC Hydrology Flood Prediction

[![DOI: 10.3390/w18172169](https://img.shields.io/badge/DOI-10.3390%2Fw18172169-007ec6)](https://doi.org/10.3390/w18172169)
[![Journal: MDPI Water 18(17), 2169](https://img.shields.io/badge/Journal-MDPI%20Water%2018(17)%2C%202169-008080)](https://www.mdpi.com/2073-4441/18/17/2169)
[![ORCID: 0000-0002-0956-0325](https://img.shields.io/badge/ORCID-0000--0002--0956--0325-A6CE39)](https://orcid.org/0000-0002-0956-0325)
[![Article License: CC BY 4.0](https://img.shields.io/badge/Article%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)

Official source code, model configurations, evaluation pipeline, and benchmark snapshot accompanying the peer-reviewed open-access article:

> **Liquid Time-Constant Networks for Water Level Forecasting in Urban Drainage: Adaptive Time-Scale Modeling of Hydrological Dynamics**
> **Rafał Buczyński** ([ORCID: 0000-0002-0956-0325](https://orcid.org/0000-0002-0956-0325))
> *Department of Water Supply and Sewage Systems, Faculty of Civil Engineering and Environmental Sciences, Bialystok University of Technology, 15-351 Bialystok, Poland*
> **Published in:** *Water* (MDPI), 2026, Volume 18, Issue 17, Article 2169 (Published: 2 September 2026)
> **DOI:** [`10.3390/w18172169`](https://doi.org/10.3390/w18172169) · **Article Page:** [mdpi.com/2073-4441/18/17/2169](https://www.mdpi.com/2073-4441/18/17/2169) · **Full-Text PDF:** [mdpi.com/2073-4441/18/17/2169/pdf](https://www.mdpi.com/2073-4441/18/17/2169/pdf)

---

## Study Overview

Liquid Time-Constant networks (**LNNs**), recurrent neural architectures with adaptive, input-dependent time constants ($\tau$), are evaluated for short-term water level forecasting in an urban drainage system:

- **Domain & Dataset:** Ten-year measurement dataset (`2010–2019`) from the Bellinge urban drainage catchment (Odense, Denmark), forecasting water levels across 5 target sensors: 4 gravity-driven passive sensors (`G71F05R_LevelBasin`, `G71F05R_LevelInlet`, `G71F05R_position`, `G71F68Y_LevelPS`) and 1 intermittently operated pumping station sensor (`G80F13P_LevelPS`).
- **Baselines & Variants:** Benchmarked against Gated Recurrent Unit (`GRU`), Long Short-Term Memory (`LSTM`), Temporal Convolutional Network (`TCN`), and Multilayer Perceptron (`MLP`) baselines alongside three LNN configurations (`stable_seed42`, `pump_aware_global`, and the dual-branch `DB-LNN` / `dual_branch`).
- **Central Finding:** While single-pathway LNNs accurately represent the continuous response of gravity-driven network elements (`non_pump_NSE = 0.9442`), intermittent threshold-controlled pumping causes pathway interference (`pump_NSE = 0.0135`). Isolating the pump pathway in the **dual-branch LNN (`DB-LNN`)** resolves this interference, achieving the highest overall accuracy in the benchmark (**mean `NSE = 0.8489`**, reported as `0.849` in the manuscript abstract), lifting pump-sensor `NSE` to **`0.4594`** (vs. `0.2792` for `GRU`) while preserving **`0.9463`** mean `NSE` across the four non-pump sensors.

---

## Citation

If you use this codebase, model architectures (`HierarchicalLTC`, `FlashFloodGate`, `DB-LNN`), configurations, or benchmark tables in your research, please cite the published *Water* article (and optionally the software repository). GitHub's native **"Cite this repository"** button is also enabled via [`CITATION.cff`](CITATION.cff).

### 1. Journal Article (Primary Citation)

**BibTeX:**

```bibtex
@article{Buczynski2026Water,
  author       = {Buczy{\'n}ski, Rafa{\l}},
  title        = {Liquid Time-Constant Networks for Water Level Forecasting in Urban Drainage: Adaptive Time-Scale Modeling of Hydrological Dynamics},
  journal      = {Water},
  volume       = {18},
  number       = {17},
  pages        = {2169},
  article-number = {2169},
  year         = {2026},
  month        = sep,
  publisher    = {MDPI AG},
  issn         = {2073-4441},
  doi          = {10.3390/w18172169},
  url          = {https://www.mdpi.com/2073-4441/18/17/2169}
}
```

**MDPI / ACS Style:**

> Buczyński, R. Liquid Time-Constant Networks for Water Level Forecasting in Urban Drainage: Adaptive Time-Scale Modeling of Hydrological Dynamics. *Water* **2026**, *18*, 2169. https://doi.org/10.3390/w18172169

**APA Style:**

> Buczyński, R. (2026). Liquid Time-Constant Networks for Water Level Forecasting in Urban Drainage: Adaptive Time-Scale Modeling of Hydrological Dynamics. *Water*, *18*(17), 2169. https://doi.org/10.3390/w18172169

### 2. Companion Software Repository

```bibtex
@software{Buczynski2026LTCHydrologyRepo,
  author       = {Buczy{\'n}ski, Rafa{\l}},
  title        = {{ltc-hydrology-flood-prediction}: Companion Code and Benchmark Snapshot for Liquid Time-Constant Networks in Urban Drainage Forecasting},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/BuczynskiRafal/ltc-hydrology-flood-prediction},
  note         = {Companion repository to Water 2026, 18(17), 2169 (DOI: 10.3390/w18172169)}
}
```

---

## Manuscript-to-Repository Mapping

The table below links the concepts, models, and experimental analyses reported in *Water* 2026, 18(17), 2169 directly to the files in this repository:

| Manuscript Component | Repository Location | Description |
|---|---|---|
| **LNN Architecture (`HierarchicalLTC`, `FlashFloodGate`, `DB-LNN`)** | `src/models/lnn_regression.py` | Fast/slow multi-scale LTC cells, stepwise adaptive $\tau$, flood-event gating, separate sensor heads, and dedicated attention-based pump branch |
| **Classical Baselines (`GRU`, `LSTM`, `TCN`, `MLP`)** | `src/models/{gru,lstm,tcn,mlp}_regression.py` | Reference sequence and feedforward regression models |
| **Dual-Branch LNN (`DB-LNN` / `dual-branch LNN`) Config** | `notebooks/article_materials_20260419/lnn_variants/dual_branch/config.yaml` | Winning configuration (`input_size=36`, main encoder `31` inputs, pump branch `5` inputs `[31..35]` with attention) |
| **Pump-Aware Global LNN Config** | `notebooks/article_materials_20260419/lnn_variants/pump_aware_global/config.yaml` | Single-pathway LNN receiving all 36 features (`31` base + `5` pump features) |
| **Best Stable Base LNN Config** | `notebooks/article_materials_20260419/lnn_variants/stable_seed42/config.yaml` | Depth-only LNN on 31 base predictors (`per_neuron_tau=true`, separate heads, `seed=42`) |
| **Multitask Baseline Configs & Results** | `configs/{gru,lstm,tcn,mlp}_regression_config.yaml`, `notebooks/article_materials_20260419/baselines/` | Canonical configs, evaluation manifests, and test metrics for `GRU`, `LSTM`, `TCN`, `MLP` |
| **Depth-Only Baseline Control Study** | `configs/baseline_depth_only/`, `experiments/run_baseline_depth_only_benchmark.py` | Isolates objective-function effects (`depth_weight=1.0`, `overflow_weight=0.0`, `flood_weight=1.0`) across classical baselines |
| **Main Benchmark Tables** | `notebooks/article_materials_20260419/benchmark/` | `article_main_table.csv`, `grouped_summary.csv`, per-sensor `NSE`/`MAE`/`RMSE` pivots, and `comparison_summary.json` |
| **Pump-Regime Feature Engineering** | `experiments/build_pump_aware_dataset.py`, `src/engineer_features.py` | Constructs the 5 pump-specific features (margin-to-startup, hysteresis state, rate dynamics) at indices `[31..35]` |
| **Hydrological & Peak-Event Metrics** | `src/evaluation/hydrological_metrics.py` | Computes `NSE`, `RMSE`, `MAE`, `Peak_Flow_Error`, `Time_to_Peak_Error`, `Volume_Error`, and `Lag_Time` |
| **Ablation, Robustness, Attribution & Uncertainty** | `experiments/{run_ablations,eval_robustness,eval_ig,evaluate_lnn_uncertainty}.py` | Component ablations, Gaussian input-noise stress tests, Integrated Gradients, and Delta-Method aleatoric uncertainty |
| **Statistical & Runtime Comparisons** | `experiments/{statistical_comparison,cross_validation_comparison,benchmark_inference_time}.py` | Paired Wilcoxon/permutation tests, temporal cross-validation, and latency/throughput benchmarks |

---

## Headline Results

Aggregated and grouped test-set (`2019`, $n = 9{,}375$ windows) metrics from [`notebooks/article_materials_20260419/benchmark/article_main_table.csv`](notebooks/article_materials_20260419/benchmark/article_main_table.csv):

| Model | `all_NSE` | `all_RMSE` (m) | `all_MAE` (m) | `non_pump_NSE` | `pump_NSE` (`G80F13P`) | `pump_RMSE` (m) | `pump_MAE` (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **dual-branch LNN (`DB-LNN`)** | **0.8489** | **0.1071** | **0.0492** | **0.9463** | **0.4594** | **0.1699** | **0.1251** |
| GRU | 0.7831 | 0.1257 | 0.0701 | 0.9091 | 0.2792 | 0.1962 | 0.1480 |
| best stable LNN | 0.7581 | 0.1191 | 0.0664 | 0.9442 | 0.0135 | 0.2295 | 0.1917 |
| TCN | 0.7302 | 0.1327 | 0.0760 | 0.9098 | 0.0115 | 0.2298 | 0.1958 |
| LSTM | 0.7157 | 0.1412 | 0.0689 | 0.8634 | 0.1248 | 0.2162 | 0.1596 |
| MLP | 0.6690 | 0.1356 | 0.0800 | 0.8440 | -0.0311 | 0.2347 | 0.1979 |

### Per-Sensor Nash–Sutcliffe Efficiency (`NSE`)

Values from [`notebooks/article_materials_20260419/benchmark/nse_per_sensor_pivot.csv`](notebooks/article_materials_20260419/benchmark/nse_per_sensor_pivot.csv):

| Model | `G71F05R_LevelBasin` | `G71F05R_LevelInlet` | `G71F05R_position` | `G71F68Y_LevelPS` | `G80F13P_LevelPS` (Pump) |
|---|---:|---:|---:|---:|---:|
| **dual-branch LNN (`DB-LNN`)** | 0.9785 | **0.8924** | 0.9179 | **0.9966** | **0.4594** |
| best stable LNN | **0.9795** | 0.8802 | 0.9210 | 0.9961 | 0.0135 |
| GRU | 0.9736 | 0.7572 | 0.9125 | 0.9930 | 0.2792 |
| TCN | 0.9740 | 0.7721 | 0.9001 | 0.9931 | 0.0115 |
| LSTM | 0.9723 | 0.7993 | 0.6876 | 0.9945 | 0.1248 |
| MLP | 0.9769 | 0.4714 | **0.9378** | 0.9900 | -0.0311 |

Additional pivot tables tracked in Git:

- [`notebooks/article_materials_20260419/benchmark/grouped_summary.csv`](notebooks/article_materials_20260419/benchmark/grouped_summary.csv)
- [`notebooks/article_materials_20260419/benchmark/mae_per_sensor_pivot.csv`](notebooks/article_materials_20260419/benchmark/mae_per_sensor_pivot.csv)
- [`notebooks/article_materials_20260419/benchmark/rmse_per_sensor_pivot.csv`](notebooks/article_materials_20260419/benchmark/rmse_per_sensor_pivot.csv)
- [`notebooks/article_materials_20260419/benchmark/model_summary.csv`](notebooks/article_materials_20260419/benchmark/model_summary.csv)

---

## Models & Configurations

| Family / Variant | Config Snapshot | Architectural Notes |
|---|---|---|
| **LNN — dual-branch (`DB-LNN`, winner)** | [`notebooks/article_materials_20260419/lnn_variants/dual_branch/config.yaml`](notebooks/article_materials_20260419/lnn_variants/dual_branch/config.yaml) | Main encoder (`31` features, `32` fast + `32` slow units, hidden `128`) + dedicated pump LTC branch (`5` features `[31..35]`, `16` fast + `16` slow units, hidden `32`, `pump_branch_use_attention=true`), per-neuron $\tau \in [0.05, 3.0]$ |
| **LNN — pump-aware global** | [`notebooks/article_materials_20260419/lnn_variants/pump_aware_global/config.yaml`](notebooks/article_materials_20260419/lnn_variants/pump_aware_global/config.yaml) | Shared encoder processing all `36` features (`31` base + `5` pump features), per-neuron $\tau$, separate sensor heads |
| **LNN — best stable (`seed42`)** | [`notebooks/article_materials_20260419/lnn_variants/stable_seed42/config.yaml`](notebooks/article_materials_20260419/lnn_variants/stable_seed42/config.yaml) | Depth-only LNN on `31` base features (`32` fast + `32` slow units, hidden `128`), per-neuron $\tau$, separate sensor heads |
| **LNN (base reference)** | [`configs/lnn_regression_config.yaml`](configs/lnn_regression_config.yaml) | Base `HierarchicalLTC` reference configuration (`128` fast + `128` slow units, hidden `320`) |
| **GRU** | [`configs/gru_regression_config.yaml`](configs/gru_regression_config.yaml) · [`configs/baseline_depth_only/gru_depth_only_config.yaml`](configs/baseline_depth_only/gru_depth_only_config.yaml) | 3 layers, hidden size `128`, dropout `0.2` (multitask and depth-only variants) |
| **LSTM** | [`configs/lstm_regression_config.yaml`](configs/lstm_regression_config.yaml) · [`configs/baseline_depth_only/lstm_depth_only_config.yaml`](configs/baseline_depth_only/lstm_depth_only_config.yaml) | 2 layers, hidden size `128`, dropout `0.2` (multitask and depth-only variants) |
| **TCN** | [`configs/tcn_regression_config.yaml`](configs/tcn_regression_config.yaml) · [`configs/baseline_depth_only/tcn_depth_only_config.yaml`](configs/baseline_depth_only/tcn_depth_only_config.yaml) | 3 causal residual blocks, kernel size `3`, hidden size `128` (multitask and depth-only variants) |
| **MLP** | [`configs/mlp_regression_config.yaml`](configs/mlp_regression_config.yaml) · [`configs/baseline_depth_only/mlp_depth_only_config.yaml`](configs/baseline_depth_only/mlp_depth_only_config.yaml) | Dense layers `[256, 128, 64]` with batch normalization, flattened 45-step input window |

---

## Dataset & Windowing

- **Source Dataset:** Open urban drainage dataset from the Bellinge catchment, Denmark (Pedersen et al., 2021, *Earth Syst. Sci. Data*, 13, 4779–4798, DOI: [`10.5194/essd-13-4779-2021`](https://doi.org/10.5194/essd-13-4779-2021)).
- **Targets (5 sensors):** `G71F05R_LevelBasin`, `G71F05R_LevelInlet`, `G71F05R_position`, `G71F68Y_LevelPS`, and the pumping station sensor `G80F13P_LevelPS`.
- **Predictors:** `31` base meteorological, hydraulic, and temporal features in `data/final_regression/`; `36` features in `data/final_regression_pump_aware/` (adds `5` engineered pump-regime indicators at indices `[31..35]`, including startup/shutoff threshold margins and hysteresis state).
- **Windowing (`src/project_config.py`):** 45-minute input sequence (`WINDOW_T_IN = 45`), 10-minute forecast horizon (`WINDOW_T_OUT = 10`), and 10-minute sliding stride (`WINDOW_STRIDE = 10`) restricted to contiguous 1-minute segments (no interpolation or window leakage across inter-segment time gaps).
- **Chronological Splits:**
  - **Train (`2010–2017`):** $n = 65{,}901$ windows
  - **Validation (`2018`):** $n = 7{,}135$ windows
  - **Test (`2019`):** $n = 9{,}375$ windows

---

## Repository Layout

- `src/` — canonical data preprocessing pipeline, model definitions (`GRU`, `LSTM`, `TCN`, `MLP`, `LNN` with `HierarchicalLTC` + `FlashFloodGate`), composite regression losses, hydrological metrics, and uncertainty analysis utilities
- `experiments/` — shared CLI (`cli.py`), training and evaluation pipeline (`regression_pipeline.py`), pump-aware dataset builder, depth-only control benchmark, ablation/robustness/attribution/uncertainty scripts, and statistical comparison tools
- `configs/` — YAML training configurations for multitask baselines (`configs/*.yaml`) and depth-only control baselines (`configs/baseline_depth_only/*.yaml`)
- `notebooks/article_materials_20260419/` — local snapshot of manuscript benchmark tables (`benchmark/`), baseline evaluation manifests/metrics (`baselines/`), and LNN variant configurations (`lnn_variants/`)
- `notebooks/article_materials_20260428/baseline_depth_only/` — configs and runner documentation for the depth-only baseline control experiment
- `notebooks/bellinge_full_pipeline.ipynb` & `notebooks/bellinge_reproduction_colab.ipynb` — Colab notebooks for end-to-end data preparation and experiment reproduction
- `tests/` — unit and smoke tests covering model shape contracts, LNN $\tau$ dynamics, deterministic seeding, time-gap windowing, CLI dispatch, and publication scripts
- `CITATION.cff` — machine-readable citation metadata for GitHub and reference managers
- `REPRODUCTION_AND_RESULTS.md` — manifest of git-tracked vs. Google Drive-hosted artifacts

---

## Reproduction

1. **Create a Python environment and install dependencies:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. **Prepare regression tensors:**
   Place or build the base regression arrays in `data/final_regression/` (via the Colab pipeline or local preprocessing). For the pump-aware LNN variants (`pump_aware_global` and `dual_branch`), ensure `data/interim/norm_params.pkl` is present and materialize `data/final_regression_pump_aware/`:

```bash
python -m experiments.build_pump_aware_dataset
```

3. **Train and evaluate models via the unified CLI:**

```bash
python -m experiments.cli train gru        # choices: gru, lstm, tcn, mlp, lnn
python -m experiments.cli evaluate lnn
python -m experiments.evaluate_release --model gru
```

   To reproduce the winning **dual-branch LNN (`DB-LNN`)** end-to-end using its exact manuscript configuration:

```bash
python -m experiments.cli train lnn \
  --config notebooks/article_materials_20260419/lnn_variants/dual_branch/config.yaml

python -m experiments.cli evaluate lnn \
  --config notebooks/article_materials_20260419/lnn_variants/dual_branch/config.yaml
```

   The same `--config` flag reproduces `stable_seed42` and `pump_aware_global` from their snapshots under `notebooks/article_materials_20260419/lnn_variants/`.

4. **Rebuild the manuscript benchmark comparison tables:**

```bash
python -m experiments.final_model_comparison \
  --model GRU=/abs/path/to/gru_metrics.json \
  --model "dual-branch LNN"=/abs/path/to/dual_branch_metrics.json \
  --output-dir notebooks/release/final_article_benchmark
```

5. **Run supplementary manuscript analyses:**

```bash
python -m experiments.run_baseline_depth_only_benchmark  # depth-only objective control study
python -m experiments.run_ablations                      # LNN architectural ablations
python -m experiments.eval_robustness                    # input-noise robustness
python -m experiments.eval_ig                            # Integrated Gradients attribution
python -m experiments.evaluate_lnn_uncertainty           # Delta-Method aleatoric uncertainty
python -m experiments.statistical_comparison             # paired Wilcoxon / permutation tests
python -m experiments.cross_validation_comparison        # temporal cross-validation
python -m experiments.benchmark_inference_time           # latency and throughput benchmark
```

---

## Tests

Run the automated test suite with `pytest`:

```bash
pytest
```

Test modules:

- `tests/test_publication_safe_refactor.py` — model-shape contracts, LNN $\tau$ semantics, deterministic seeding, time-gap windowing, schema validation, and embedded checkpoint config loading
- `tests/test_publication_scripts.py` — smoke tests for `run_ablations`, `eval_robustness`, `eval_ig`, and the regression training pipeline on synthetic data
- `tests/test_baseline_depth_only_benchmark.py` — verification of the depth-only baseline benchmark runner and config contracts
- `tests/test_cli.py` — CLI argument parsing, subcommand dispatch, and README/citation consistency checks
- `tests/test_env.py` — environment sanity checks

---

## Large Artifacts & Google Drive Links

Runtime-sized binary assets (raw inputs in `downloaded/`, materialized `.npy` tensors in `data/final_regression/` and `data/final_regression_pump_aware/`, model checkpoints in `artifacts/`, and raw `.npz` prediction arrays) are kept outside Git. Direct links to the archived Google Drive experiment artifacts are listed below (and in [`REPRODUCTION_AND_RESULTS.md`](REPRODUCTION_AND_RESULTS.md)):

- **Final Benchmark Folder (`final_article_benchmark_dual_branch`):** [Google Drive Folder](https://drive.google.com/drive/folders/12p2XcHdqfhX3uWg9Q6Hi6TKpuMWFSA-a)
  - [`per_sensor_metrics.csv`](https://drive.google.com/file/d/19joS3jvEF9Hs_Rss7BA6xc2pbJXL3ZQJ/view?usp=drivesdk)
  - [`comparison_summary.json`](https://drive.google.com/file/d/1ubxNcp2SBURDy08OzaH2wWyoBUmLxdo5/view?usp=drivesdk)
- **Dual-Branch LNN (`DB-LNN`, winner):** [Folder](https://drive.google.com/drive/folders/1NllHUEhvoFVPFT2wMXuu8mzve2ZJcciU) · [Config](https://drive.google.com/file/d/1UJlaql60OgWczRsjiCOpSPg5LY9lIF5Y/view?usp=drivesdk) · [Metrics](https://drive.google.com/file/d/1ocESFst4xzfuLeP579z6w6AjzVJvzTEs/view?usp=drivesdk) · [Manifest](https://drive.google.com/file/d/1cdubpUAVGc-8HDOOswbMlfaY9tjm-Efc/view?usp=drivesdk)
- **Global Pump-Aware LNN (`pump_aware_global`):** [Folder](https://drive.google.com/drive/folders/16u9UnPzE7lqmfHNwIQtziJPA7jqibIGu) · [Config](https://drive.google.com/file/d/189z0sEpybdPrXgt9ZXHA7Hc3kCwhVas2/view?usp=drivesdk) · [Metrics](https://drive.google.com/file/d/1k6RgoprylVX7zhvvi4_CeBfDdm2zfD_r/view?usp=drivesdk) · [Manifest](https://drive.google.com/file/d/1ePwnvKe_no3So5WL8qxo1kpEMi8WE8iz/view?usp=drivesdk)
- **Best Stable LNN (`stable_seed42`):** [Folder](https://drive.google.com/drive/folders/1BQXIxBxBNZKwJ6y6_DZg2QLU8lMNta2y) · [Config](https://drive.google.com/file/d/1WYmQcs4MNwUuTEVDjbXcrm5tqV7LDvt2/view?usp=drivesdk) · [Metrics](https://drive.google.com/file/d/1Z2XwJVerEB-wMBjW6UEaZN6IVLxcQhNL/view?usp=drivesdk) · [Manifest](https://drive.google.com/file/d/1apzfJYiN98JbPxmVz3cySzeGnSGPLt1g/view?usp=drivesdk)

---

## Funding & License

- **Funding:** This research was funded by **Bialystok University of Technology** under grant number **`WI/WB-IIŚ/9/2026`** (financed from the subsidy provided by the Minister of Science and Higher Education of Poland).
- **Code License:** The source code in this repository is released under the [MIT License](LICENSE).
- **Article License:** The associated paper in *Water* ([DOI: `10.3390/w18172169`](https://doi.org/10.3390/w18172169)) is published Open Access under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.
