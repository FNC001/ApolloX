# PDM refinement

Run the fixed PDM postprocessor after exporting generated structures with `cond-cdvae/scripts/extract_gen.py`. It adjusts atomic coordinates while keeping the cell, species and atom count fixed. The output consists of optimized structures and their raw pair PDMs.

From the repository root, in the Cond-CDVAE environment:

```bash
python scripts/pdm_control/refine_structures.py \
  --input-dir /work/model/eval_gen_sample/gen \
  --output-dir /work/refined_structures \
  --model-path /work/model \
  --target /work/target_pdm.json \
  --target-space raw
```

`--target` is a JSON file containing one complete numeric array in the model's `pdm_schema.json` feature order. It applies to every input structure. The model directory supplies the matching V2 **pair-only** schema and original training `scaler_stats.txt`; the postprocessor does not load or update model weights.

Use `--target-space standardized` when supplying the externally standardized values used by `evaluate.py --element_values`. This reverses the external training mean/std once. Do not apply `prop_scalers.pt` to these inputs. Raw pair counts are the default.

For different targets per structure, replace `--target` with `--targets /work/targets.csv`. The CSV has `filename,element_values` columns; each `element_values` cell is a quoted JSON array. Include each input filename exactly once. Match filenames such as `0.vasp` and `1.vasp`, rather than `material_id`, which may repeat across evaluations.

The output directory must be new. It contains:

- `gen/`: optimized structures with their original filenames; the input identity manifest is preserved when supplied.
- `pdm.csv`: PDM counts recomputed from the saved structures. `label` identifies the filename; descriptor columns follow the saved schema.
- `status.csv`: processing status for every input, including failures.
- `run.json`: the schema, settings and completion status for this run.

The original input structures are preserved. Failures are recorded, and a run containing failures exits with a nonzero status; samples are not silently retried.

The default settings come from `scripts/pdm_control/guidance_config.json`: up to 200 steps, soft-count temperature 0.1 Å, maximum coordinate step 0.04 Å, a short-distance penalty below 1.0 Å with weight 10, and six backtracking attempts. Supply `--config /path/to/config.json` to use an explicit configuration. CPU is the default; `--device cuda:0` uses an available GPU.

The short-distance term is a soft penalty, not a hard minimum-distance constraint. PDM refinement is not energy relaxation; assess physical stability separately when required.
