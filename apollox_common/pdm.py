"""Versioned PDM feature order and periodic geometry shared by all workflows."""

import json
import random
from itertools import combinations, combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymatgen.core import Element, Lattice, Structure
from pymatgen.io.vasp import Poscar

SCHEMA_VERSION = 2
GEOMETRY_VERSION = "periodic-minimum-image-clique-v2"
MODE_SIZES = {"pair": 2, "triple": 3, "quadruple": 4}
METADATA_COLUMNS = {
    "label", "formula", "material_id", "cif_file", "Energy", "eval_index", "sample_index", "job_id"
}


def build_schema(elements, modes, cutoff):
    elements = sorted(set(elements))
    if not elements:
        raise ValueError("PDM schema needs at least one element")
    for symbol in elements:
        Element(symbol)
    if not modes or set(modes) - MODE_SIZES.keys():
        raise ValueError("PDM modes must contain pair, triple, or quadruple")
    modes = [mode for mode in MODE_SIZES if mode in modes]
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("PDM cutoff must be finite and positive")
    features = sorted(
        ''.join(group)
        for mode in modes
        for group in combinations_with_replacement(elements, MODE_SIZES[mode])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "features": features,
        "elements": elements,
        "modes": modes,
        "cutoff": float(cutoff),
        "geometry": GEOMETRY_VERSION,
    }


def load_schema(path):
    with Path(path).expanduser().open() as handle:
        schema = json.load(handle)
    if schema.get("schema_version") != SCHEMA_VERSION or schema.get("geometry") != GEOMETRY_VERSION:
        raise ValueError(f"Unsupported PDM schema/geometry in {path}; regenerate V2 data")
    expected = build_schema(schema["elements"], schema["modes"], schema["cutoff"])
    if schema != expected:
        raise ValueError(f"PDM schema {path} does not contain the canonical complete feature order")
    return schema


def save_schema(path, schema):
    path = Path(path).expanduser()
    if path.exists() and load_schema(path) != schema:
        raise ValueError(f"Refusing to overwrite a different PDM schema: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2) + "\n")


def align_features(data, schema, allow_missing=True):
    """Align raw descriptor columns by name; absent interactions are zero counts."""
    columns = [column for column in data.columns if column not in METADATA_COLUMNS]
    unknown = sorted(set(columns) - set(schema["features"]))
    if unknown:
        raise ValueError(f"PDM columns are absent from the training schema: {unknown}")
    missing = [name for name in schema["features"] if name not in columns]
    if missing and not allow_missing:
        raise ValueError(f"PDM columns missing from dataset: {missing}")
    features = data.reindex(columns=schema["features"], fill_value=0).apply(pd.to_numeric, errors="raise")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("PDM features must contain only finite numeric values")
    return features


def read_poscar(path):
    """Read valid POSCAR variants, including Cartesian scaling and selective flags."""
    structure = Poscar.from_file(path, check_for_potcar=False).structure
    if not structure.is_ordered or len(structure) == 0:
        raise ValueError(f"Expected a non-empty ordered structure: {path}")
    if (not np.isfinite(structure.lattice.matrix).all()
            or not np.isfinite(structure.cart_coords).all()
            or not np.isfinite(structure.volume) or structure.volume <= 0):
        raise ValueError(f"Expected finite coordinates and a finite positive-volume cell: {path}")
    atom_types = [site.specie.symbol for site in structure]
    elements = list(dict.fromkeys(atom_types))
    counts = [atom_types.count(element) for element in elements]
    return elements, counts, structure.cart_coords, atom_types, structure.lattice.matrix


def calculate_distances(positions, lattice_vectors):
    positions, lattice_vectors = np.asarray(positions), np.asarray(lattice_vectors)
    if (positions.ndim != 2 or positions.shape[1] != 3 or lattice_vectors.shape != (3, 3)
            or not np.isfinite(positions).all() or not np.isfinite(lattice_vectors).all()
            or abs(np.linalg.det(lattice_vectors)) <= 0):
        raise ValueError("Distance calculation requires finite positions and a positive-volume cell")
    lattice = Lattice(lattice_vectors)
    frac = lattice.get_fractional_coords(positions)
    return lattice.get_all_distances(frac, frac)


def shuffled_poscar(source, counts=None):
    """Shuffle only atomic sites, keeping each site's flags and MD data together."""
    structure = source.structure
    symbols = list(dict.fromkeys(site.specie.symbol for site in structure))
    if counts is None:
        counts = [sum(site.specie.symbol == symbol for site in structure) for symbol in symbols]
    if len(counts) != len(symbols) or any(count <= 0 for count in counts) or sum(counts) != len(structure):
        raise ValueError("Positive species counts must preserve the original atom total")
    permutation = list(range(len(structure)))
    random.shuffle(permutation)
    properties = {key: [values[index] for index in permutation] for key, values in structure.site_properties.items()}
    shuffled = Structure(
        structure.lattice,
        [symbol for symbol, count in zip(symbols, counts) for _ in range(count)],
        structure.frac_coords[permutation], site_properties=properties,
    )
    return Poscar(
        shuffled, comment=source.comment,
        selective_dynamics=properties.get('selective_dynamics'), velocities=properties.get('velocities'),
        predictor_corrector=properties.get('predictor_corrector'),
        predictor_corrector_preamble=source.predictor_corrector_preamble,
        lattice_velocities=source.lattice_velocities,
    )


def calculate_sro(distances, atom_types, cutoff, modes):
    """Count unordered site cliques using exact nearest periodic pair distances."""
    modes = build_schema(set(atom_types), modes, cutoff)['modes']
    adjacent = np.asarray(distances) < cutoff
    descriptors = {}
    for i in range(len(atom_types)):
        later_neighbors = np.flatnonzero(adjacent[i, i + 1:]) + i + 1
        for mode in modes:
            for other_sites in combinations(later_neighbors, MODE_SIZES[mode] - 1):
                if all(adjacent[j, k] for j, k in combinations(other_sites, 2)):
                    key = ''.join(sorted(atom_types[j] for j in (i, *other_sites)))
                    descriptors[key] = descriptors.get(key, 0) + 1
    return descriptors


def _compute_record(path, cutoff, modes):
    try:
        elements, counts, positions, atom_types, lattice = read_poscar(path)
        values = calculate_sro(calculate_distances(positions, lattice), atom_types, cutoff, modes)
        formula = ''.join(f"{element}{count}" for element, count in zip(elements, counts))
        return path.name, formula, elements, values
    except Exception as exc:
        raise ValueError(f"Failed to compute PDM for {path}: {exc}") from exc


def compute_folder(input_dir, output_csv, cutoff=None, modes=None, n_jobs=4,
                   starts_with='', ends_with='', schema_path=None, formula_metadata=False):
    input_dir, output_csv = Path(input_dir).expanduser(), Path(output_csv).expanduser()
    schema = load_schema(schema_path) if schema_path else None
    if schema:
        if cutoff is not None and cutoff != schema['cutoff']:
            raise ValueError("Requested cutoff differs from the training PDM schema")
        if modes is not None and set(modes) != set(schema['modes']):
            raise ValueError("Requested modes differ from the training PDM schema")
        cutoff, modes = schema['cutoff'], schema['modes']
    else:
        cutoff, modes = 5.0 if cutoff is None else cutoff, ['pair'] if modes is None else modes
    paths = sorted(path for path in input_dir.iterdir()
                   if path.is_file() and path.name not in {'structure_manifest.csv', 'pdm_schema.json'}
                   and not path.name.startswith('.') and path.resolve() != output_csv.resolve()
                   and path.name.startswith(starts_with) and path.name.endswith(ends_with))
    if not paths:
        raise ValueError(f"No matching POSCAR files in {input_dir}")
    records = Parallel(n_jobs=n_jobs)(delayed(_compute_record)(path, cutoff, modes) for path in paths)
    elements = set(element for _, _, symbols, _ in records for element in symbols)
    if schema is None:
        schema = build_schema(elements, modes, cutoff)
    elif elements - set(schema['elements']):
        raise ValueError(f"Input elements absent from training schema: {sorted(elements - set(schema['elements']))}")
    manifest_path = input_dir / 'structure_manifest.csv'
    manifest = None
    if formula_metadata and manifest_path.exists():
        manifest = pd.read_csv(manifest_path, dtype={'filename': str, 'material_id': str}, keep_default_na=False)
        if not {'filename', 'material_id', 'eval_index', 'sample_index'} <= set(manifest):
            raise ValueError(f"Incomplete structure manifest: {manifest_path}")
        if (manifest['filename'].duplicated().any() or manifest['filename'].str.strip().eq('').any()
                or manifest['material_id'].str.strip().eq('').any()):
            raise ValueError(f"Invalid or duplicate manifest entries: {manifest_path}")
        for key in ('eval_index', 'sample_index'):
            indices = pd.to_numeric(manifest[key], errors='raise')
            if not np.isfinite(indices).all() or (indices < 0).any() or (indices != indices.astype(int)).any():
                raise ValueError(f"Manifest {key} must contain non-negative integers")
        manifest = manifest.set_index('filename')
    rows = []
    for name, formula, _, values in records:
        if formula_metadata:
            row = {'label': name, 'formula': formula}
            if manifest is not None:
                if name not in manifest.index:
                    raise ValueError(f"No manifest entry for {name}")
                source = manifest.loc[name]
                row.update(label=source['material_id'], eval_index=int(source['eval_index']),
                           sample_index=int(source['sample_index']))
        else:
            row = {'material_id': name, 'cif_file': name + '.cif'}
        row.update({feature: values.get(feature, 0) for feature in schema['features']})
        rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    save_schema(output_csv.parent / 'pdm_schema.json', schema)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_csv, index=False)
    return frame, schema
