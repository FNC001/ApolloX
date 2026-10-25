"""Batch generation with matching schemas, stable job IDs and recoverable failures."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apollox_common.pdm import load_schema


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def resolved_path(value):
    return Path(value).expanduser().resolve()


def parse_vector(value):
    if isinstance(value, str):
        value = json.loads(value)
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("element_values must be a nonempty finite one-dimensional vector")
    return vector.tolist()


def matching_schema(model_dir, source_dir, allow_legacy_order=False):
    model_path, source_path = model_dir / "pdm_schema.json", source_dir / "pdm_schema.json"
    if model_path.is_file() and source_path.is_file():
        model_schema, source_schema = load_schema(model_path), load_schema(source_path)
        if model_schema != source_schema:
            raise ValueError("Input PDM schema differs from the trained model schema")
        return model_schema
    if allow_legacy_order and not model_path.exists() and not source_path.exists():
        return None
    raise ValueError(
        "Both the model and input/scaler directory need matching pdm_schema.json files. "
        "For an unchanged legacy model and legacy vectors only, explicitly set "
        "allow_legacy_order: true; do not mix legacy and V2 descriptors."
    )


def run_command(arguments, **kwargs):
    return subprocess.run([str(arg) for arg in arguments], check=True, **kwargs)


def check_external_scaler(model_dir, scaler_path):
    from prepare_dataset.standardize import load_scaler_stats
    saved = model_dir / "scaler_stats.txt"
    if not saved.is_file():
        raise ValueError("V2 model directory is missing its training scaler_stats.txt")
    expected, actual = load_scaler_stats(saved), load_scaler_stats(scaler_path)
    if any(not np.array_equal(a, b) for a, b in zip(expected, actual)):
        raise ValueError("External scaler differs from the statistics saved with the trained model")


def prepare_inputs(config, model_dir, work_dir, root=ROOT):
    mode = config.get("mode_choice")
    legacy = config.get("allow_legacy_order", False)
    if not isinstance(legacy, bool):
        raise ValueError("allow_legacy_order must be true or false")
    if mode == "feather":
        source = resolved_path(config["feather_mode_params"]["feather_path"])
        schema = matching_schema(model_dir, source.parent, legacy)
        if schema is not None:
            check_external_scaler(model_dir, source.parent / "scaler_stats.txt")
        frame = pd.read_feather(source)
        labels = frame["material_id"]
        vectors = [parse_vector(v) for v in frame["element_values"]]
        if "formula" in frame:
            formulas = frame["formula"].astype(str).tolist()
        elif "cif" in frame:
            formulas = [Structure.from_str(cif, fmt="cif").composition.formula.replace(" ", "")
                        for cif in frame["cif"]]
        else:
            raise ValueError("Feather input must include formula or cif; filenames do not establish composition")
    elif mode in {"structure", "unscaled_pdm"}:
        params = config[f"{mode}_mode_params"]
        scaler = resolved_path(params["scaler_path"])
        schema = matching_schema(model_dir, scaler.parent, legacy)
        if schema is not None:
            check_external_scaler(model_dir, scaler)
        if mode == "structure":
            if schema is None:
                raise ValueError("V2 structure calculation requires a V2 schema and retrained model")
            source = work_dir / "raw_pdm.csv"
            command = [sys.executable, root / "use_model/compute_pdm_formula.py",
                       "--input_dir", resolved_path(params["input_dir"]), "--output_csv", source,
                       "--schema", scaler.parent / "pdm_schema.json",
                       "--n_jobs", positive_integer(params.get("n_jobs", 1), "n_jobs"),
                       "--starts_with", params.get("starts_with", "POSCAR"),
                       "--ends_with", params.get("ends_with", "")]
            if "cutoff" in params:
                command += ["--cutoff", params["cutoff"]]
            if "compute_mode" in params:
                modes = params["compute_mode"]
                command += ["--mode", *(modes.split() if isinstance(modes, str) else modes)]
            run_command(command)
        else:
            source = resolved_path(params["input_csv"])
        scaled = work_dir / "standardized.csv"
        command = [sys.executable, root / "prepare_dataset/standardize.py",
                   "--input", source, "--scaler", scaler, "--output", scaled]
        if schema is None:
            command.append("--allow-legacy-order")
        run_command(command)
        frame = pd.read_csv(scaled, dtype={"label": str, "formula": str}, keep_default_na=False)
        labels = frame["label"]
        formulas = frame["formula"].astype(str).tolist()
        metadata = {"label", "formula", "material_id", "cif_file", "Energy", "eval_index", "sample_index"}
        columns = schema["features"] if schema is not None else [c for c in frame if c not in metadata]
        vectors = [parse_vector(row) for row in frame[columns].to_numpy()]
    else:
        raise ValueError("mode_choice must be structure, unscaled_pdm or feather")
    if not len(labels) or labels.isna().any() or any(not str(label).strip() for label in labels):
        raise ValueError("Input must contain nonempty material labels")
    labels = labels.astype(str)
    if labels.duplicated().any():
        raise ValueError("Input material labels must be unique; select one evaluation or assign unique labels")
    for formula, vector in zip(formulas, vectors):
        composition = Composition(formula)
        if not composition or any(float(n) <= 0 or not float(n).is_integer() for n in composition.values()):
            raise ValueError(f"Formula must specify positive integer atom counts: {formula}")
        if schema is not None:
            if len(vector) != len(schema["features"]):
                raise ValueError("PDM vector width differs from schema")
            if not set(composition.get_el_amt_dict()).issubset(schema["elements"]):
                raise ValueError(f"Formula contains elements outside the model schema: {formula}")
    return list(zip(labels.tolist(), formulas, vectors)), schema


def gpu_devices(environ=None):
    environ = os.environ if environ is None else environ
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [] if visible.strip() in {"", "-1"} else [s.strip() for s in visible.split(",")]
    return [str(i) for i in range(torch.cuda.device_count())]


def generate_job(job_id, label, formula, vector, model_dir, log_dir, output_dir,
                 batch_size, num_batches, num_evals, device=None,
                 keep_intermediate=True, root=ROOT):
    """Retain the generated tensor until all exported structures are verified and copied."""
    pt_file = model_dir / f"eval_gen_{job_id}.pt"
    if pt_file.exists():
        raise FileExistsError(f"Refusing stale generation output: {pt_file}")
    env = os.environ.copy()
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    command = [sys.executable, root / "cond-cdvae/scripts/evaluate.py", "--model_path", model_dir,
               "--tasks", "gen", f"--formula={formula}", "--pressure=0", f"--label={job_id}",
               "--element_values=" + ",".join(map(str, vector)), f"--batch_size={batch_size}",
               f"--num_batches_to_samples={num_batches}", f"--num_evals={num_evals}"]
    with (log_dir / f"{job_id}.log").open("w") as stdout, (log_dir / f"{job_id}.err").open("w") as stderr:
        run_command(command, cwd=model_dir, env=env, stdout=stdout, stderr=stderr)
        if not pt_file.is_file():
            raise RuntimeError(f"Generation returned without {pt_file.name}")
        run_command([sys.executable, root / "cond-cdvae/scripts/extract_gen.py", pt_file],
                    cwd=model_dir, env=env, stdout=stdout, stderr=stderr)
    gen_dir = pt_file.with_suffix("") / "gen"
    manifest = pd.read_csv(gen_dir / "structure_manifest.csv",
                           dtype={"filename": str, "material_id": str}, keep_default_na=False)
    required = {"filename", "material_id", "eval_index", "sample_index"}
    if not required.issubset(manifest) or len(manifest) != batch_size * num_batches * num_evals:
        raise ValueError(f"Incomplete extraction for {job_id}; retaining {pt_file.name}")
    if manifest.filename.duplicated().any():
        raise ValueError("Extraction manifest has duplicate filenames")
    rows = []
    for record in manifest.to_dict("records"):
        filename = record["filename"]
        if Path(filename).name != filename or not filename.endswith(".vasp"):
            raise ValueError("Invalid structure filename in extraction manifest")
        source = gen_dir / filename
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"Missing exported structure: {source}")
        destination = output_dir / f"{job_id}_{filename}"
        if destination.exists():
            raise FileExistsError(destination)
        rows.append({**record, "filename": destination.name, "material_id": label, "job_id": job_id})
    for record, output_record in zip(manifest.to_dict("records"), rows):
        shutil.copy2(gen_dir / record["filename"], output_dir / output_record["filename"])
    # Persist identity before deleting intermediates, even if another job or the
    # parent process stops before the combined manifest is written.
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(exist_ok=True)
    pending_manifest = manifests_dir / f"{job_id}.csv.tmp"
    pd.DataFrame(rows).to_csv(pending_manifest, index=False)
    pending_manifest.replace(manifests_dir / f"{job_id}.csv")
    if not keep_intermediate:
        shutil.rmtree(pt_file.with_suffix(""))
        pt_file.unlink()
    return rows


def run_batch(config, model_dir, root=ROOT):
    model_dir = resolved_path(model_dir)
    batch_size = positive_integer(config["batch_size"], "batch_size")
    num_batches = positive_integer(config["num_batches_to_samples"], "num_batches_to_samples")
    num_evals = positive_integer(config.get("num_evals", 1), "num_evals")
    parallel = positive_integer(config.get("max_parallel_submissions", 1), "max_parallel_submissions")
    keep = config.get("keep_intermediate", True)
    if not isinstance(keep, bool):
        raise ValueError("keep_intermediate must be true or false")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    log_dir = model_dir / "parallel_logs" / run_id
    log_dir.mkdir(parents=True)
    inputs, schema = prepare_inputs(config, model_dir, log_dir, root)
    output_dir = model_dir / "final_generated_structures" / run_id
    output_dir.mkdir(parents=True)
    if schema is not None:
        shutil.copy2(model_dir / "pdm_schema.json", output_dir / "pdm_schema.json")
    devices = gpu_devices()
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {}
        for index, (label, formula, vector) in enumerate(inputs):
            job_id = f"v2_{run_id}_{index:06d}"
            future = pool.submit(generate_job, job_id, label, formula, vector, model_dir,
                                 log_dir, output_dir, batch_size, num_batches, num_evals,
                                 devices[index % len(devices)] if devices else None, keep, root)
            futures[future] = {"job_id": job_id, "material_id": label}
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception as exc:
                failure = {**futures[future], "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                print(f"[FAILED] {failure}", file=sys.stderr)
    if rows:
        pd.DataFrame(rows).sort_values(["job_id", "eval_index", "sample_index"]).to_csv(
            output_dir / "structure_manifest.csv", index=False)
    summary = {"run_id": run_id, "jobs": len(inputs), "structures": len(rows),
               "failures": failures, "output_dir": str(output_dir), "log_dir": str(log_dir)}
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise RuntimeError(f"{len(failures)} generation jobs failed; results retained. See {log_dir}")
    print(f"[SUCCESS] {len(rows)} structures saved to {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "use_model/config.yaml")
    parser.add_argument("--model-path", type=Path, default=Path.cwd())
    args = parser.parse_args()
    with args.config.expanduser().open() as stream:
        config = yaml.safe_load(stream)
    try:
        run_batch(config, args.model_path)
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"[ERROR] {exc}\n")


if __name__ == "__main__":
    main()
