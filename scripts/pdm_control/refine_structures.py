#!/usr/bin/env python3
"""Refine exported POSCARs against a fixed PDM; write structures and final PDM only.

This is explicit coordinate postprocessing, not a claim of learned model control.
Targets follow the model's schema feature order. Standardized targets refer only
to the external train mean/std, never the model's secondary property scaler.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / 'cond-cdvae'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        require(fields and len(set(fields)) == len(fields), f'Missing or duplicate CSV columns: {path}')
        records = list(reader)
    require(all(None not in row and all(value is not None for value in row.values()) for row in records),
            f'Malformed CSV rows: {path}')
    return fields, records


def filename(value):
    require(isinstance(value, str) and value and value not in ('.', '..') and
            '/' not in value and '\\' not in value and Path(value).name == value,
            f'Expected an exact POSCAR basename, got {value!r}')
    return value


def vector(value, width):
    require(isinstance(value, list) and len(value) == width,
            f'Target must be a one-dimensional JSON array of {width} values')
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value),
            'Target values must be finite numbers (not strings, booleans, or nested arrays)')
    return [float(x) for x in value]


def targets_for_files(names, target, targets, width, mean, std, space):
    if target:
        values = vector(json.loads(target.read_text()), width)
        mapping = {name: values for name in names}
    else:
        fields, records = read_csv(targets)
        require({'filename', 'element_values'} <= set(fields), 'Targets CSV needs filename and element_values columns')
        mapping = {}
        for row in records:
            name = filename(row['filename'])
            require(name not in mapping, f'Duplicate target filename: {name}')
            mapping[name] = vector(json.loads(row['element_values']), width)
        require(set(mapping) == set(names), 'Targets CSV must map every input POSCAR exactly once, with no extra rows')
    if space == 'standardized':
        mapping = {name: [z * s + m for z, s, m in zip(values, std, mean)] for name, values in mapping.items()}
    require(all(math.isfinite(v) for values in mapping.values() for v in values), 'Inverse-standardized targets are nonfinite')
    return mapping


def hard_pdm(structure, schema):
    """Independent hard counts from pymatgen minimum-image distances."""
    import numpy as np
    require(structure.is_ordered and len(structure) > 0, 'Expected a nonempty ordered POSCAR')
    require(np.isfinite(structure.frac_coords).all() and np.isfinite(structure.lattice.matrix).all()
            and np.linalg.det(structure.lattice.matrix) > 0, 'Coordinates/cell must be finite with positive determinant')
    symbols = [site.specie.symbol for site in structure]
    require(set(symbols) <= set(schema['elements']), 'Structure contains elements absent from the model PDM schema')
    distances = structure.lattice.get_all_distances(structure.frac_coords, structure.frac_coords)
    values = dict.fromkeys(schema['features'], 0)
    for i in range(len(structure)):
        for j in range(i + 1, len(structure)):
            if distances[i, j] < schema['cutoff']:
                values[''.join(sorted((symbols[i], symbols[j])))] += 1
    return values


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--input-dir', type=Path, required=True, help='extract_gen gen directory containing *.vasp')
    result.add_argument('--output-dir', type=Path, required=True, help='New directory; existing paths are never overwritten')
    result.add_argument('--model-path', type=Path, required=True, help='Directory containing pdm_schema.json and scaler_stats.txt')
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument('--target', type=Path, help='JSON file containing one PDM array for every input structure')
    source.add_argument('--targets', type=Path, help='CSV with filename and JSON element_values for each structure')
    result.add_argument('--target-space', choices=('raw', 'standardized'), default='raw')
    result.add_argument('--config', type=Path, default=Path(__file__).with_name('guidance_config.json'))
    result.add_argument('--device', default='cpu')
    return result


def run(args):
    import numpy as np
    import torch
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar
    from cdvae.common.pdm_guidance import refine_pdm, _config
    from cdvae.common.differentiable_pdm import validate_schema
    from cdvae.common.pdm_schema import read_pdm_schema, read_external_stats

    source = args.input_dir.expanduser().resolve()
    model = args.model_path.expanduser().resolve()
    output = args.output_dir.expanduser().absolute()
    require(source.is_dir() and model.is_dir(), 'Input and model paths must be directories')
    require(not os.path.lexists(output), f'Output path already exists; refusing overwrite: {output}')
    destination = output.resolve()
    require(not destination.is_relative_to(source) and not source.is_relative_to(destination),
            'Input and output directories must not contain one another')
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() == '.vasp')
    require(paths, 'No .vasp structures found in input directory')
    require(all(p.is_file() and not p.is_symlink() for p in paths), 'Input POSCARs must be regular files, not links/directories')
    names = [filename(p.name) for p in paths]
    schema_path, stats_path = model / 'pdm_schema.json', model / 'scaler_stats.txt'
    schema_json = json.loads(schema_path.read_text())
    schema = read_pdm_schema(schema_path, len(schema_json.get('features', [])))
    validate_schema(schema)  # The existing refinement algorithm supports pair mode only.
    mean, std = read_external_stats(stats_path, len(schema['features']))
    config = _config(json.loads(args.config.expanduser().read_text()))
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda'), 'Device must be cpu or cuda[:index]')
    torch.empty(0, device=device)  # Reject unavailable devices before creating output.
    target_file = (args.target or args.targets).expanduser().resolve()
    targets = targets_for_files(names, target_file if args.target else None,
                                target_file if args.targets else None, len(schema['features']), mean, std, args.target_space)
    identities, manifest_fields = {}, None
    manifest = source / 'structure_manifest.csv'
    if manifest.exists():
        manifest_fields, records = read_csv(manifest)
        require('filename' in manifest_fields, 'Identity manifest needs a filename column')
        for row in records:
            name = filename(row['filename'])
            require(name not in identities, f'Duplicate identity filename: {name}')
            identities[name] = row
        require(set(identities) == set(names), 'Identity manifest must match input POSCAR filenames exactly')

    output.mkdir(parents=True, exist_ok=False)
    generated = output / 'gen'
    generated.mkdir()
    metadata = {'status': 'RUNNING', 'planned_files': names, 'target_space': args.target_space,
                'target_file_sha256': sha(target_file), 'schema': schema, 'config': config,
                'model_contract_sha256': {'pdm_schema.json': sha(schema_path), 'scaler_stats.txt': sha(stats_path)},
                'method': 'fixed_cell_target_pdm_refinement_v1',
                'note': 'Explicit coordinate postprocessing; completion does not guarantee PDM accuracy or physical stability.'}
    write_json(output / 'run.json', metadata)
    identity_columns = [key for key in ('material_id', 'eval_index', 'sample_index', 'job_id')
                        if manifest_fields and key in manifest_fields]
    fields = ['label', 'formula', *identity_columns, *schema['features']]
    good, failures = 0, 0
    # Each structure and optimizer history are released before reading the next.
    with (output / 'pdm.csv').open('x', newline='') as pdm_handle, (output / 'status.csv').open('x', newline='') as status_handle:
        pdm_writer = csv.DictWriter(pdm_handle, fieldnames=fields)
        status_writer = csv.DictWriter(status_handle, fieldnames=['filename', 'status', 'optimizer_status', 'error'])
        pdm_writer.writeheader(); status_writer.writeheader()
        manifest_handle = (generated / 'structure_manifest.csv').open('x', newline='') if manifest_fields else None
        try:
            manifest_writer = csv.DictWriter(manifest_handle, fieldnames=manifest_fields) if manifest_handle else None
            if manifest_writer:
                manifest_writer.writeheader()
            for path in paths:
                state = {'filename': path.name, 'status': 'error', 'optimizer_status': '', 'error': ''}
                temp = generated / ('.' + path.name + '.tmp')
                try:
                    structure = Poscar.from_file(path, check_for_potcar=False).structure
                    hard_pdm(structure, schema)  # Validate geometry/elements before calling the core.
                    cell = torch.tensor(np.asarray(structure.lattice.matrix), dtype=torch.float64, device=device)
                    result = refine_pdm(
                        torch.tensor(np.asarray(structure.frac_coords), dtype=torch.float64, device=device),
                        torch.tensor([site.specie.Z for site in structure], dtype=torch.long, device=device),
                        torch.tensor([len(structure)], dtype=torch.long, device=device), cell[None],
                        torch.tensor([targets[path.name]], dtype=torch.float64, device=device),
                        torch.tensor(std, dtype=torch.float64, device=device), schema, config)
                    state['optimizer_status'] = result['reports'][0]['status']
                    require(state['optimizer_status'] in ('steps_completed', 'stationary', 'line_search_stalled'),
                            'Refinement failed: ' + state['optimizer_status'])
                    require(torch.equal(result['cell_matrices'], cell[None]), 'Refinement unexpectedly changed the cell')
                    coords = result['frac_coords'].detach().cpu().numpy()
                    require(coords.shape == (len(structure), 3) and np.isfinite(coords).all(), 'Refinement returned invalid coordinates')
                    refined = Structure(structure.lattice, structure.species, coords, site_properties=structure.site_properties)
                    Poscar(refined, comment='PDM-refined ' + path.name).write_file(temp, significant_figures=16)
                    delivered = Poscar.from_file(temp, check_for_potcar=False).structure
                    require(np.allclose(delivered.lattice.matrix, structure.lattice.matrix, rtol=0, atol=1e-10)
                            and delivered.species == structure.species, 'Export changed lattice or atomic identities')
                    values = hard_pdm(delivered, schema)  # The actual delivered POSCAR is authoritative.
                    os.replace(temp, generated / path.name)
                    symbols = [site.specie.symbol for site in delivered]
                    row = {'label': path.name, 'formula': ''.join(f'{s}{symbols.count(s)}' for s in sorted(set(symbols))),
                           **{key: identities[path.name][key] for key in identity_columns}, **values}
                    pdm_writer.writerow(row); pdm_handle.flush()
                    if manifest_writer:
                        manifest_writer.writerow(identities[path.name]); manifest_handle.flush()
                    state['status'] = 'ok'; good += 1
                except Exception as exc:
                    state['error'] = f'{type(exc).__name__}: {exc}'
                    failures += 1
                    if temp.exists():
                        temp.unlink()
                status_writer.writerow(state); status_handle.flush()
        finally:
            if manifest_handle:
                manifest_handle.close()
    metadata.update(status='COMPLETE' if not failures else 'COMPLETE_WITH_ERRORS', exported=good, failed=failures)
    write_json(output / 'run.json', metadata)
    print(json.dumps({'output_dir': str(output), 'planned': len(paths), 'exported': good, 'failed': failures}))
    return 1 if failures else 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f'PDM refinement error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
