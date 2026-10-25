"""Shared paths and checked subprocess execution for PSO stages."""
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


def repository_root():
    return Path(__file__).resolve().parents[1]


def load_config():
    with (repository_root() / "PSO/config.yaml").open() as stream:
        return yaml.safe_load(stream)


def run_python(script, *arguments, cwd=None):
    command = [sys.executable, str(script), *map(str, arguments)]
    print(f"[RUN] {shlex.join(command)}", flush=True)
    return subprocess.run(command, cwd=cwd, check=True)


def pdm_arguments(config):
    """Use the V2 training schema for every generation."""
    params = config["pdm"]
    modes = params["mode"]
    modes = modes.split() if isinstance(modes, str) else list(modes)
    args = ["--cutoff", str(params["cutoff"]), "--n_jobs", str(params["n_jobs"]),
            "--mode", *modes, "--starts_with", "POSCAR", "--ends_with", "optdone"]
    schema = Path(config.get("pdm_schema_path", Path(config["scaler_path"]).expanduser().parent / "pdm_schema.json")).expanduser().resolve()
    if not schema.is_file():
        raise FileNotFoundError(f"V2 PSO requires the training PDM schema: {schema}")
    args += ["--schema", str(schema)]
    return args


def validate_training_artifacts(config, model_dir=None):
    """Never recompute V2 geometry for an unverified legacy checkpoint/scaler."""
    import numpy as np
    from apollox_common.pdm import load_schema
    from prepare_dataset.standardize import load_scaler_stats
    model_dir = Path.cwd() if model_dir is None else Path(model_dir)
    scaler = Path(config['scaler_path']).expanduser().resolve()
    schema_path = scaler.parent / 'pdm_schema.json'
    schema = load_schema(schema_path)
    if load_schema(model_dir / 'pdm_schema.json') != schema:
        raise ValueError('PSO scaler schema differs from the trained model schema')
    model_stats = load_scaler_stats(model_dir / 'scaler_stats.txt')
    requested_stats = load_scaler_stats(scaler)
    if any(not np.array_equal(expected, actual) for expected, actual in zip(model_stats, requested_stats)):
        raise ValueError('PSO scaler statistics differ from the trained model statistics')
    if len(requested_stats[0]) != len(schema['features']):
        raise ValueError('PSO scaler dimension differs from the training schema')
    params = config['pdm']
    modes = params['mode'].split() if isinstance(params['mode'], str) else params['mode']
    if params['cutoff'] != schema['cutoff'] or set(modes) != set(schema['modes']):
        raise ValueError('PSO cutoff/modes must match the trained model schema')
    if 'pdm_schema_path' in config and load_schema(config['pdm_schema_path']) != schema:
        raise ValueError('Explicit PSO schema differs from the scaler schema')
    return schema


def require_new_paths(paths):
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError("Refusing to reuse previous run outputs: " + ", ".join(existing))
