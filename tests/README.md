Run the regression tests from the repository root in an environment with the
Cond-CDVAE dependencies installed:

```bash
python -m pip install pytest
python -m pytest tests -q
```

The tests cover Feather preprocessing and cache migration errors, real PyG
batching, condition embeddings and gradients, CLI/CSV validation and model-scaler
consistency, legacy composition-only configuration, and the complete PDM vector
passed by PSO scripts. Crystal graph construction, checkpoint loading, and the
expensive structural decoder are replaced by test doubles where indicated.
These CPU regressions do not measure the quality of a trained PDM-conditioned
generator or reproduce the full PSO optimization experiment.
