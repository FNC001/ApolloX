# PDM-control training pilot

`PDMControlCDVAE` is an explicit experimental model class. It adds a differentiable structure-space PDM objective and a prior-latent training branch. An auxiliary loss or a condition-sensitive decoder does not establish accurate conditional generation; evaluate hard PDMs from independently generated structures.

The reproducible pilot runner is `scripts/train_pdm_control.py`. It uses weights-only initialization, a fresh Adam optimizer and global step 0. It never loads or evaluates a test dataset. This protocol fixes 10,000 training structures, 1,000 validation structures, seed 20261026, learning rate 0.0001, batch size 16, a 30-minute fit budget, and a predeclared PDM-loss-cost grid of 1, 10 and 100. Each variant uses the same initialization and data. The model uses `teacher_forcing_lattice=false`, prior probability 0.5, soft-descriptor temperature 0.1 Å, and an auxiliary objective only at noise sigma ≤ 0.5 Å.

Create an independent JSON spec for each variant with absolute paths:

```json
{
  "mode": "pilot",
  "repo": "/work/ApolloX",
  "data": "/work/data",
  "out": "/work/runs/cost10",
  "warmstart_checkpoint": "/work/warmstart/epoch=27-step=8764.ckpt",
  "warmstart_checkpoint_sha256": "<sha256>",
  "warmstart_hparams": "/work/warmstart/hparams.yaml",
  "model_overlay": "/work/plan/model_overlay.json",
  "model_target": "cdvae.pl_modules.pdm_control_model.PDMControlCDVAE",
  "frozen_files": {"/absolute/input/path": "<sha256>"},
  "cost_pdm": 10,
  "seed": 20261026,
  "lr": 0.0001,
  "batch_size": 16,
  "num_workers": 2,
  "cpu_threads": 2,
  "budget_seconds": 1800,
  "midpoint_seconds": 900,
  "expected_counts": {"train": 10000, "val": 1000},
  "gradient_audit_every": 25,
  "accelerator": "gpu"
}
```

The model overlay supplies `pdm_control.version=1`, the exact training `schema`, `external_mean` and `external_std`, `cost`, `temperature`, `max_sigma` and `prior_probability`. These statistics must equal the saved training `scaler_stats.txt`; property and lattice scalers are fitted from the training cache. Freeze the warmstart checkpoint/hparams, overlay, runner/source modules, train/val Feather files, cache files and their metadata sidecars, schema and external scalers in `frozen_files`. The runner verifies these hashes at startup and completion. The pilot is intentionally tied to the declared epoch-27/step-8764 warmstart; other experiments require a separately reviewed protocol.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_pdm_control.py --spec /work/plan/cost10.json
```

For a separate dataflow smoke, use `mode=smoke`, an output directory whose name contains `smoke`, and `--max-steps 50`. `--check-only` performs data/model construction without optimizer updates. A formal pilot rejects `--max-steps`. Output directories must be new; this runner does not resume automatically.

The runner saves complete, loadable model directories at `checkpoints/initial`, `checkpoints/mid15min`, and `checkpoints/final`. Their checkpoint names follow `epoch=N-step=M.ckpt`. `checkpoint_manifest.json` records hashes, steps and actual elapsed times. Time includes full validation and checkpoint IO; checkpoints/stopping occur at safe batch or validation boundaries, so their recorded times may exceed the nominal boundary.

Every epoch evaluates the complete validation split as a diagnostic. Validation uses its own loader generator and restores Python, NumPy, CPU and CUDA RNG states. Auxiliary means use eligible-graph denominators, including separate prior/posterior counts. Training records auxiliary gradient norms for the decoder and PDM embedding, finite losses/gradients, branch counts, time and GPU peak memory. Proxy loss does not select checkpoints, reduce the learning rate or stop training.

Choose among the three snapshots using separately frozen **prior-generation** validation with identical conditions and seeds, including guidance OFF/ON comparisons. Preserve failed and invalid outputs in the denominators. The runner finishes as `training_complete_awaiting_prior_validation` with no selected checkpoint; this status is not a generation-accuracy claim. Short-sampling pilot diagnostics must pass the separately specified full-sampling validation before a longer training run is justified.


## Prior generation and independent validation

Use `scripts/pdm_control/generate_validation.py` for each saved model snapshot. Supply a frozen validation protocol and `generator_inputs` containing only exact compositions, requested raw PDMs, identity/pair plans, schema and training statistics. The generator never loads reference geometry or encodes a validation crystal. It verifies the model's schema/scaler contract and checkpoint hash, then records every planned attempt, including failures.

```bash
python scripts/pdm_control/generate_validation.py \
  --repo /work/ApolloX --model /work/runs/cost10/checkpoints/final \
  --checkpoint-sha256 '<sha256>' --candidate-id pdm-cost-10 \
  --snapshot-name final --expected-cost-pdm 10 \
  --generator-inputs /work/validation/frozen_targets/stages/pilot12/generator_inputs \
  --protocol /work/validation/VALIDATION_PROTOCOL_PROPOSED.json \
  --guidance-config /work/ApolloX/scripts/pdm_control/guidance_config.json \
  --stage pilot12 --gpu 0 --out /work/validation-results/cost10-final-generation
python scripts/pdm_control/evaluate_validation.py \
  --generation /work/validation-results/cost10-final-generation \
  --out /work/validation-results/cost10-final-evaluation
```

Both output directories must be new. Keep logs outside the generation directory; its completion seal inventories every evidence file. The evaluator checks this inventory, planned denominator, input/model transforms, common latent and RNG controls, and the relationship between native and refined structures before calculating metrics from exported POSCARs with pymatgen.

Guidance **OFF** is prior generation from the learned model. Guidance **ON** refines that exact OFF output once with a fixed cell and fixed species, using only the requested PDM. The fixed settings are 200 steps, temperature 0.1 Å, maximum Cartesian step 0.04 Å, minimum-distance penalty threshold 1.0 Å, geometry weight 10, and six backtracking steps. Guidance failures remain in the denominator. ON success supports the model-plus-refinement procedure, not network-only control.

The proposed `pilot12` screen uses 12 real same-composition MD pairs and 12 legacy singletons, 50 noise levels × 10 steps, one seed, and four repeated pair-left controls. It has 80 total OFF/ON attempts and cannot establish final sampling accuracy. `gate64` uses 64 MD pairs plus 64 legacy singletons, 100 steps per noise level, two fixed seeds and eight repeats, totaling 800 attempts. Full validation covers all 1,000 references with 4,032 attempts. The exact selected IDs, seeds, batch order and thresholds belong to the external frozen protocol, not to repository defaults.

For a tiny integration check, use a separate protocol with `development_only: true` and a one-pair/one-singleton input subset. This GPU smoke has eight attempts including repeats and is always ineligible for a scientific acceptance gate. CPU fixtures require `--allow-cpu-fixture` and are also ineligible. Neither changes formal validation targets.

Report geometry validity, all-feature MRE, joint validity-and-MRE-below-20%, and normalized MAE by source and OFF/ON. Same-composition paired advantage compares assigned and swapped PDM targets, with pair and batch cluster bootstrap intervals; failed pairs retain the declared penalty. Preserve zero target components in MRE. Low mean error without positive paired evidence does not establish control. Legacy singletons provide accuracy evidence but no real same-composition counterfactual. Evaluate one shared initial baseline after verifying identical weights, then retain every cost candidate's mid/final result and rank OFF/ON separately. The old test set must not influence these development choices.
