# Reproduction And Results Manifest

## Included Git-Tracked Assets

- canonical source code in `src/`
- experiment orchestration in `experiments/`
- model configs in `configs/`
- automated checks in `tests/`
- Colab notebook and helper script in `notebooks/`
- final manuscript tables and result summaries in
  `notebooks/article_materials_20260419/`

## Included Result Packages

The repository already contains locally versioned result material for:

- baseline models `GRU`, `LSTM`, `TCN`, `MLP`
- benchmark summary tables for the manuscript
- config snapshots for the key LNN variants:
  `stable_seed42`, `pump_aware_global`, `dual_branch`

## Large Artifacts Kept Outside Git

The following remain intentionally outside the repository because they are runtime-sized
or were stored on Google Drive during experimentation:

- prepared regression tensors
- checkpoints
- raw predictions `.npz`
- large notebook export directories
- raw benchmark folders on Google Drive

The current access point for those assets is:

- `notebooks/article_materials_20260419/DRIVE_LINKS.md`

## Key Manuscript Conclusion

For the publication snapshot copied here, the strongest LNN result is the
`dual-branch pump encoder` variant. It materially improves the pump sensor while keeping
the passive sensors strong, which is the central modeling conclusion carried into the
article package.
