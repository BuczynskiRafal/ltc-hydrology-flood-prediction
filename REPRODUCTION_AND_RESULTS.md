# Reproduction And Results Manifest

**Associated Publication:**
Rafał Buczyński, *"Liquid Time-Constant Networks for Water Level Forecasting in Urban Drainage: Adaptive Time-Scale Modeling of Hydrological Dynamics"*, **Water** 2026, *18*(17), 2169.
- **DOI:** [`10.3390/w18172169`](https://doi.org/10.3390/w18172169)
- **Article URL:** <https://www.mdpi.com/2073-4441/18/17/2169>
- **Repository:** <https://github.com/BuczynskiRafal/ltc-hydrology-flood-prediction>

## Included Git-Tracked Assets

- canonical source code in `src/`
- experiment orchestration in `experiments/`
- model configs in `configs/` (including `configs/baseline_depth_only/`)
- automated checks in `tests/`
- Colab notebooks and helper script in `notebooks/`
- final manuscript tables and result summaries in `notebooks/article_materials_20260419/`
- depth-only baseline control configs in `notebooks/article_materials_20260428/baseline_depth_only/`
- machine-readable citation metadata in `CITATION.cff`

## Included Result Packages

The repository already contains locally versioned result material for:

- baseline models `GRU`, `LSTM`, `TCN`, `MLP` (`notebooks/article_materials_20260419/baselines/`)
- benchmark summary tables for the manuscript (`notebooks/article_materials_20260419/benchmark/`)
- config snapshots for the key LNN variants:
  `stable_seed42`, `pump_aware_global`, `dual_branch` (`notebooks/article_materials_20260419/lnn_variants/`)

## Large Artifacts Kept Outside Git

The following remain intentionally outside the repository because they are runtime-sized
or were stored on Google Drive during experimentation:

- prepared regression tensors (`data/final_regression/`, `data/final_regression_pump_aware/`)
- checkpoints (`artifacts/`)
- raw predictions `.npz`
- large notebook export directories
- raw benchmark folders on Google Drive

Direct Google Drive links to those archived assets:

- **Final Benchmark Folder (`final_article_benchmark_dual_branch`):** [Drive Folder](https://drive.google.com/drive/folders/12p2XcHdqfhX3uWg9Q6Hi6TKpuMWFSA-a) · [`per_sensor_metrics.csv`](https://drive.google.com/file/d/19joS3jvEF9Hs_Rss7BA6xc2pbJXL3ZQJ/view?usp=drivesdk) · [`comparison_summary.json`](https://drive.google.com/file/d/1ubxNcp2SBURDy08OzaH2wWyoBUmLxdo5/view?usp=drivesdk)
- **Primary Single-Branch LNN (`SB-LNN` / `stable_seed42`):** [Folder](https://drive.google.com/drive/folders/1BQXIxBxBNZKwJ6y6_DZg2QLU8lMNta2y) · [Config](https://drive.google.com/file/d/1WYmQcs4MNwUuTEVDjbXcrm5tqV7LDvt2/view?usp=drivesdk) · [Metrics](https://drive.google.com/file/d/1Z2XwJVerEB-wMBjW6UEaZN6IVLxcQhNL/view?usp=drivesdk) · [Manifest](https://drive.google.com/file/d/1apzfJYiN98JbPxmVz3cySzeGnSGPLt1g/view?usp=drivesdk)
- **Dual-Branch LNN (`DB-LNN` / `dual_branch`, pump-pathway extension):** [Folder](https://drive.google.com/drive/folders/1NllHUEhvoFVPFT2wMXuu8mzve2ZJcciU) · [Config](https://drive.google.com/file/d/1UJlaql60OgWczRsjiCOpSPg5LY9lIF5Y/view?usp=drivesdk) · [Metrics](https://drive.google.com/file/d/1ocESFst4xzfuLeP579z6w6AjzVJvzTEs/view?usp=drivesdk) · [Manifest](https://drive.google.com/file/d/1cdubpUAVGc-8HDOOswbMlfaY9tjm-Efc/view?usp=drivesdk)
- **Global Pump-Aware LNN (`pump_aware_global`):** [Folder](https://drive.google.com/drive/folders/16u9UnPzE7lqmfHNwIQtziJPA7jqibIGu) · [Config](https://drive.google.com/file/d/189z0sEpybdPrXgt9ZXHA7Hc3kCwhVas2/view?usp=drivesdk) · [Metrics](https://drive.google.com/file/d/1k6RgoprylVX7zhvvi4_CeBfDdm2zfD_r/view?usp=drivesdk) · [Manifest](https://drive.google.com/file/d/1ePwnvKe_no3So5WL8qxo1kpEMi8WE8iz/view?usp=drivesdk)

## Key Manuscript Conclusion

The primary **Single-Branch LNN (`SB-LNN` / `stable_seed42`)** accurately models the continuous hydraulic response of the gravity-driven network (`non_pump_NSE = 0.9442`), with adaptive time constants ($\tau$) separating system-wide hydraulic processes from local control actions. To address the switching discontinuity of the threshold-controlled pumping station (`G80F13P_LevelPS`), the **Dual-Branch LNN (`DB-LNN` / `dual_branch`)** extension isolates the pump pathway, raising `pump_NSE` from `0.0135` (`SB-LNN`) to `0.4594` and overall mean `NSE` to `0.8489` while preserving `non_pump_NSE = 0.9463` (*Water* 2026, 18(17), 2169).

