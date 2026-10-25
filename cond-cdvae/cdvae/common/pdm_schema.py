"""Persist the PDM feature and normalization contract beside model checkpoints."""

import hashlib
import json
import math
import re
from itertools import combinations_with_replacement
from pathlib import Path
import shutil
import warnings


GEOMETRY = "periodic-minimum-image-clique-v2"


def read_pdm_schema(path, expected_dim):
    if not isinstance(expected_dim, int) or isinstance(expected_dim, bool) or expected_dim < 1:
        raise ValueError("V2 PDM requires an explicit positive data.pdm_dim")
    path = Path(path)
    with path.open() as source:
        schema = json.load(source)
    features = schema.get("features")
    if (
        schema.get("schema_version") != 2
        or schema.get("geometry") != GEOMETRY
        or not isinstance(features, list)
        or not features
        or any(not isinstance(name, str) or not name for name in features)
        or len(set(features)) != len(features)
    ):
        raise ValueError(f"Invalid V2 PDM schema: {path}")
    if len(features) != expected_dim:
        raise ValueError(
            f"PDM schema has {len(features)} features, but data.pdm_dim/embedding n_in is {expected_dim}"
        )
    for key in ("elements", "modes"):
        if not isinstance(schema.get(key), list) or not schema[key]:
            raise ValueError(f"PDM schema needs a non-empty {key} list: {path}")
    cutoff = schema.get("cutoff")
    if not isinstance(cutoff, (float, int)) or isinstance(cutoff, bool) or not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError(f"PDM schema cutoff must be finite and positive: {path}")
    from pymatgen.core import Element
    sizes = {"pair": 2, "triple": 3, "quadruple": 4}
    try:
        elements = sorted(set(schema["elements"]))
        for symbol in elements:
            Element(symbol)
        if set(schema["modes"]) - sizes.keys():
            raise ValueError("unknown PDM mode")
        modes = [mode for mode in sizes if mode in schema["modes"]]
        canonical = {"schema_version": 2, "geometry": GEOMETRY,
                     "features": sorted(''.join(group) for mode in modes
                         for group in combinations_with_replacement(elements, sizes[mode])),
                     "elements": elements, "modes": modes, "cutoff": float(cutoff)}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid PDM elements/modes in {path}") from exc
    if schema != canonical:
        raise ValueError(f"PDM schema {path} does not contain the canonical complete feature order")
    return schema


def read_external_stats(path, dimension):
    """Validate the persisted external standardization before training/loading."""
    text = Path(path).read_text()
    vectors = []
    for key in ('Mean', 'Std'):
        match = re.search(r'^Train ' + key + r':[ \t]*([^\r\n]+)', text, re.MULTILINE)
        try:
            values = [float(value) for value in match.group(1).split(',')] if match else []
        except ValueError as exc:
            raise ValueError(f'Invalid Train {key} in {path}') from exc
        if len(values) != dimension or not all(math.isfinite(value) for value in values):
            raise ValueError(f'Train {key} must contain {dimension} finite values in {path}')
        if key == 'Std' and any(value <= 0 for value in values):
            raise ValueError(f'Train Std must be strictly positive in {path}')
        vectors.append(values)
    return vectors

def preserve_training_pdm_contract(cfg, model_dir):
    """Require V2 schema/stats for new PDM training and reject incompatible resumes."""
    if "element_values" not in cfg.model.conditions.cond_keys:
        return None
    source_dir, model_dir = Path(cfg.data.root_path), Path(model_dir)
    schema_path = source_dir / "pdm_schema.json"
    stats_path = source_dir / "scaler_stats.txt"
    if not schema_path.is_file() or not stats_path.is_file():
        raise ValueError(
            "V2 PDM training requires pdm_schema.json and scaler_stats.txt in "
            f"data.root_path ({source_dir}). Prepare V2 data before training."
        )
    schema = read_pdm_schema(schema_path, cfg.data.pdm_dim)
    stats = read_external_stats(stats_path, cfg.data.pdm_dim)
    # Hydra permits each split path to be overridden independently of root_path.
    # Every split must use the same feature order and external normalization.
    datamodule = cfg.data.get('datamodule')
    if datamodule is not None:
        datasets = datamodule.get('datasets', {})
        for split in ('train', 'val', 'test'):
            configured = datasets.get(split)
            if configured is None:
                continue
            for dataset_cfg in ([configured] if split == 'train' else configured):
                try:
                    validate_inference_dataset_contract(source_dir, dataset_cfg.path, cfg.data.pdm_dim)
                except ValueError as exc:
                    raise ValueError(
                        f'Training {split} dataset {dataset_cfg.path} conflicts with data.root_path: {exc}'
                    ) from exc
    saved_schema = model_dir / "pdm_schema.json"
    saved_stats = model_dir / "scaler_stats.txt"
    if list(model_dir.glob("*.ckpt")) and (not saved_schema.is_file() or not saved_stats.is_file()):
        raise ValueError(
            "Cannot resume a legacy PDM checkpoint without its V2 schema and normalization "
            "contract. Use a new training directory; legacy checkpoints remain loadable for inference."
        )
    if saved_schema.exists() and read_pdm_schema(saved_schema, cfg.data.pdm_dim) != schema:
        raise ValueError("Training PDM schema differs from the model directory's saved schema")
    if saved_stats.exists() and read_external_stats(saved_stats, cfg.data.pdm_dim) != stats:
        raise ValueError("Training PDM normalization statistics differ from the model directory")
    model_dir.mkdir(parents=True, exist_ok=True)
    if schema_path.resolve() != saved_schema.resolve():
        shutil.copyfile(schema_path, saved_schema)
    if stats_path.resolve() != saved_stats.resolve():
        shutil.copyfile(stats_path, saved_stats)
    return schema


def attach_inference_pdm_contract(model, model_dir):
    from cdvae.common.condition_utils import pdm_input_dim

    dimension = pdm_input_dim(model)
    if dimension is None:
        return
    model_dir = Path(model_dir)
    schema_path = model_dir / "pdm_schema.json"
    if not schema_path.exists():
        warnings.warn(
            "Legacy PDM checkpoint has no pdm_schema.json. Its original feature order, "
            "geometry and external normalization cannot be verified; use its original "
            "preprocessing. It is not a V2 PDM model.",
            UserWarning,
        )
        model.pdm_schema = None
        return
    model.pdm_schema = read_pdm_schema(schema_path, dimension)
    if not (model_dir / "scaler_stats.txt").is_file():
        raise ValueError("V2 PDM model is missing its saved external scaler_stats.txt")
    read_external_stats(model_dir / "scaler_stats.txt", dimension)


def cache_contract(source_path, schema):
    """Fingerprint both the feature contract and the exact source table bytes."""
    schema_bytes = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    source_hash = hashlib.sha256()
    with Path(source_path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            source_hash.update(block)
    return {"cache_version": 2, "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
            "source_sha256": source_hash.hexdigest()}


def validate_inference_dataset_contract(model_dir, source_path, dimension):
    """Reject differently standardized/reordered data at reconstruction entrypoints."""
    model_dir, source_dir = Path(model_dir), Path(source_path).parent
    model_schema, data_schema = model_dir / 'pdm_schema.json', source_dir / 'pdm_schema.json'
    if not model_schema.exists() and not data_schema.exists():
        return  # Explicit legacy-to-legacy compatibility; loader emits its warning.
    if not model_schema.exists() or not data_schema.exists():
        raise ValueError('Cannot mix a V2 PDM model/dataset with a legacy dataset/model lacking its schema')
    if read_pdm_schema(model_schema, dimension) != read_pdm_schema(data_schema, dimension):
        raise ValueError('Reconstruction dataset PDM schema differs from the saved model schema')
    model_stats, data_stats = model_dir / 'scaler_stats.txt', source_dir / 'scaler_stats.txt'
    if not model_stats.is_file() or not data_stats.is_file():
        raise ValueError('V2 reconstruction requires scaler_stats.txt beside both model and dataset')
    if read_external_stats(model_stats, dimension) != read_external_stats(data_stats, dimension):
        raise ValueError('Reconstruction dataset normalization statistics differ from the saved model')
