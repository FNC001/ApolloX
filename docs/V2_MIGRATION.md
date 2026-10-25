# ApolloX V2 migration

V2 updates the public Cond-CDVAE implementation and its validation. It does not include a newly trained model or establish a result for a paper.

## Start a new V2 dataset and training run

1. Choose a new `dataset_path` in `prepare_dataset/config.yaml`, for example `~/autodl-tmp/prepare_data_v2`. Structure generation refuses to overwrite a nonempty output folder.
2. Run the data preparation pipeline again from the original POSCAR structures. It creates `pdm_schema.json` and fits `scaler_stats.txt` using only the training split.
3. Set `data.root_path` to the new absolute dataset directory and set `data.pdm_dim` to the length of `features` in `pdm_schema.json`. For six elements in pair mode the complete schema has 21 features, even if some interactions never appear.
4. Start a new model run directory. The recommended ApolloX setup uses `model.predict_property=False`. The training entry point preserves the dataset schema and external scaler beside the checkpoint and model property scalers.
5. Use those same schema and scaler files for generation and PSO. Batch generation and PSO reject a mismatching schema or external scaler. V2 caches carry schema/source fingerprints; regenerate a cache with `force_process: true` if it is missing or does not match the source data.

PDM features are counts of unordered site pairs or higher-order cliques within the cutoff, using each pair's shortest periodic distance. The algorithm does not count every periodic image as a separate site. The schema records `periodic-minimum-image-clique-v2` to distinguish this definition and implementation from previous public data.

The new complete feature set includes zero-count interactions. Parsing scaled Cartesian coordinates and Selective dynamics, and fixing distances in non-orthogonal cells, can also change descriptor values. Equal vector lengths do not establish compatibility with an older checkpoint. Do not assign a new schema file to old data merely to bypass a check.

## Preserve an old experiment

Retain its original code version, dataset, feature order, scaler files, cached graphs and checkpoint together. A configuration edit does not retrain a model or add PDM dependence to composition-only weights.

For precomputed legacy vectors whose ordering and normalization have already been verified, `standardize.py --allow-legacy-order` explicitly permits a scaler without a schema. Batch modes `feather` and `unscaled_pdm` have the analogous `allow_legacy_order: true` option, only when both model and input lack a V2 schema. V2 structure-to-PDM batch generation and PSO require a V2 model and schema because they calculate descriptors with the corrected geometry. Low-level generation still supports compatible legacy checkpoints; omitting PDM remains necessary for composition-only models.

Legacy auxiliary property-prediction heads used target values as inputs. V2 heads consume latent representations and keep gradients. Such heads require retraining; their old weights must not be treated as a validated V2 property predictor. The default ApolloX conditioning-only configuration disables these auxiliary heads.

## Generation and extraction

`use_model/submit_tasks.py` accepts `--config` and `--model-path`. It validates positive integer batch settings before starting jobs. Paths may contain spaces; tasks are invoked with argument lists rather than generated shell commands.

Each invocation writes results to `final_generated_structures/<run_id>/` and logs plus a machine-readable `summary.json` to `parallel_logs/<run_id>/`. `keep_intermediate: true` retains the source `.pt` and extracted files. Setting it to false removes intermediates only after successful, complete extraction, copying and saving that job's identity mapping in `manifests/<job_id>.csv`. These per-job manifests remain available if the parent process stops before the combined manifest is written. Failed tasks return a nonzero exit status and retain their source tensors for inspection.

`extract_gen.py` exports all evaluations and writes `gen/structure_manifest.csv` with `filename`, `material_id`, `eval_index` and `sample_index`. Keep this manifest with the structures. Reconstruction uses the original test-sample IDs; target generation uses the target IDs. The PDM calculator reads the manifest, so directory listing order does not determine sample identity.

Use `compute_pdm_formula.py --schema /path/to/model/pdm_schema.json` when calculating descriptors of generated structures. Its modes and cutoff must match the schema. For raw feature CSV input, names are aligned to the training schema, missing known interactions are filled with zero and unknown features are rejected. Check that absent columns really represent zero counts, rather than unavailable measurements.

## Evaluation

The reconstruction workflow is: sample aligned Feather/cache records, reconstruct, extract with the manifest, calculate PDM using the training schema, inverse-transform the original Feather using its training scaler, then compare by sample ID and feature names.

Keep `pdm_schema.json` and `scaler_stats.txt` beside each V2 Feather file, including sampled subsets. `feather_to_csv.py` verifies that this provenance matches the supplied scaler and its schema before inverse transformation. `--ref_csv` is optional and validates feature names; it does not determine the names of V2 vector entries. Native Arrow vectors and JSON vectors are supported. Invalid or nonfinite values fail explicitly.

`mean_relative_error.py` rejects missing, duplicate or unmatched sample IDs and mismatched feature sets. If the reconstruction CSV contains multiple evaluations, select one with `--eval-index 0` (or another index). Do not sort filenames lexicographically to infer the mapping: `10.vasp` sorts before `2.vasp`.

Coverage, uniqueness and novelty exclude identity/evaluation metadata from their feature matrices and reject invalid numeric input. Their descriptor distances are not a substitute for structural equivalence or a physical stability calculation.

## PSO

Start in a fresh model working directory without previous PSO `temp`, `pdm_and_energy`, `poscars` or `pso_initial` outputs. Set `opt_script` to the CPU or GPU script as appropriate; every generation honors the same setting. A failed relaxation or missing result terminates the generation rather than silently replacing all of its results with backup structures.

The energy interpolator aggregates repeated PDM vectors using the mean of their recorded energies; a single unique vector produces a constant predictor. This is an explicit surrogate-model convention, since distinct structures can share a descriptor. The integer-valued particle search evaluates moves and returns particle best positions, retaining the best candidate it has found. These changes require renewed experimental validation before comparing optimization performance to an earlier run.

## Validation scope

The regression suite covers data/schema alignment, file-format variants, failure propagation, metadata alignment, interpolation degeneracy, model interfaces and export behavior. CPU smoke runs verify that training and generation execute with small artificial inputs. They do not establish convergence, production GPU behavior, CHGNet/DeepMD accuracy, trained-model quality or reproduction of published results.
