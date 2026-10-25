# Changelog

## 2.0.0 — 2026-10-25

ApolloX V2 makes the data-to-generation workflow explicit and checks failures before results are accepted.

- Save a complete, ordered PDM feature schema with the descriptor modes, cutoff and geometry version. Align target features by name and preserve the training schema and external scaler with each new model.
- Parse POSCAR variants with pymatgen and use its periodic shortest distances for non-orthogonal cells. Reject failed structures instead of turning them into zero descriptors.
- Preserve structure identity through generation, extraction and metric calculation. Export every requested evaluation and align reconstruction errors by sample ID and feature name.
- Retain generation results when inference or extraction fails; report failed tasks, pass GPU assignments explicitly, and use distinct job IDs and output directories.
- Honor the PSO optimizer device configuration, stop failed generations, aggregate repeated PDM observations, and retain evaluated particle best positions.
- Repair evaluation entry points, native Feather vector conversion, optional property-head inputs and gradients, scalar condition scaling, inference mode and invalid-cell handling.
- Add behavior regressions for the audited failures and small CPU model integration checks.

V2 does not add a new trained checkpoint or establish an improvement in generation quality. Its descriptor changes can affect feature counts and values. See [V2 migration](docs/V2_MIGRATION.md) before using previous datasets, scalers or checkpoints.
