Run the regression tests from the repository root in an environment with the
Cond-CDVAE dependencies installed:

```bash
python -m pip install pytest
python -m pytest tests -q
```

The suite includes the original PDM-conditioning regressions and V2 checks for
periodic geometry, complete feature schemas, train-only standardization,
Feather/scaler provenance, stable sample IDs, multi-evaluation extraction,
failed subprocesses, batch output retention, PSO particle best positions,
duplicate interpolation inputs, and metric alignment.

`test_v2_model.py` also runs a small real CDVAE forward/backward pass, checks
auxiliary-head gradients, reloads a checkpoint, and runs short Langevin sampling.
The full Cond-CDVAE environment (including torch-geometric, torch-scatter,
torch-sparse, torch-cluster, PyTorch Lightning, Hydra, pymatgen and pyarrow) is
required. Other tests use doubles where indicated to isolate failure handling.

These CPU checks do not measure trained-model quality, GPU behavior, convergence,
CHGNet/DeepMD accuracy, or reproduction of the full PSO experiment.

The PDM-control tests cover the soft periodic descriptor and its gradients,
decoder-frame coordinate supervision, prior/posterior loss masks, fixed-cell
refinement, weights-only training snapshots, and independent generation
validation with failed-attempt denominators and paired randomness checks.

`test_pdm_refine_cli.py` exercises the regular structure-refinement entry point,
including target normalization, file identity, saved-structure PDMs, and failures.
