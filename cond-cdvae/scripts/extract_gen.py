"""Export every evaluation/sample with stable identity and transactional output."""

import csv
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import click
import numpy as np
import torch
from ase import Atoms
from ase.io import read, write
from joblib import Parallel, delayed


FIELDS = ('frac_coords', 'atom_types', 'lengths', 'angles', 'num_atoms')
MANIFEST_FIELDS = ('filename', 'material_id', 'eval_index', 'sample_index')


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _field(batch, name):
    return batch[name] if isinstance(batch, dict) else getattr(batch, name)


def atoms_from_batch(batch):
    """Split a single flattened graph batch without truncating mismatched arrays."""
    coords, types, lengths, angles, counts = [_array(_field(batch, key)) for key in FIELDS]
    if counts.ndim != 1 or not np.isfinite(counts).all() or (counts < 1).any() or not np.equal(counts, counts.astype(int)).all():
        raise ValueError('num_atoms must be a vector of positive integer counts')
    n_samples, n_atoms = len(counts), int(counts.sum())
    if (coords.shape != (n_atoms, 3) or types.shape != (n_atoms,)
            or lengths.shape != (n_samples, 3) or angles.shape != (n_samples, 3)):
        raise ValueError('Evaluation arrays do not share the declared sample/atom axes')
    if (not all(np.isfinite(value).all() for value in (coords, types, lengths, angles))
            or (lengths <= 0).any() or (angles <= 0).any() or (angles >= 180).any()
            or (types < 1).any() or (types > 118).any() or not np.equal(types, types.astype(int)).all()):
        raise ValueError('Evaluation contains invalid atom types, coordinates or cell parameters')
    result, offset = [], 0
    for index, count in enumerate(counts.astype(int)):
        atoms = Atoms(numbers=types[offset:offset + count].astype(int),
                      scaled_positions=coords[offset:offset + count],
                      cell=np.concatenate((lengths[index], angles[index])), pbc=True)
        if not np.isfinite(atoms.cell.array).all() or not np.isfinite(atoms.get_volume()) or atoms.get_volume() <= 1e-8:
            raise ValueError(f'Invalid cell at sample {index}')
        result.append(atoms)
        offset += count
    return result


def evaluation_records(data):
    """Use eval-major order while IDs always index the independent sample axis."""
    arrays = {key: _array(data[key]) for key in FIELDS}
    if arrays['num_atoms'].ndim != 2:
        raise ValueError('num_atoms must have shape [num_evals, num_samples]')
    n_evals, n_samples = arrays['num_atoms'].shape
    if not n_evals or not n_samples or any(len(value) != n_evals for value in arrays.values()):
        raise ValueError('Evaluation arrays must have the same nonempty eval axis')
    if 'input_data_batch' in data:
        ids = list(_field(data['input_data_batch'], 'mp_id'))
    else:
        ids = data.get('material_id_list')
        if ids is None or len(ids) == 0:
            ids = [f'gen_{index}' for index in range(n_samples)]
        else:
            ids = list(ids)
    if len(ids) != n_samples:
        raise ValueError(
            f'Expected {n_samples} IDs on the sample axis, got {len(ids)}. '
            'Regenerate legacy target outputs whose IDs were repeated on the eval axis.'
        )
    if any(value is None or str(value).strip() == '' or (isinstance(value, float) and not np.isfinite(value)) for value in ids):
        raise ValueError('Every exported sample must have a material ID')
    records = []
    for eval_index in range(n_evals):
        atoms_list = atoms_from_batch({key: value[eval_index] for key, value in arrays.items()})
        for sample_index, atoms in enumerate(atoms_list):
            records.append({'filename': f'{len(records)}.vasp',
                            'material_id': str(ids[sample_index]),
                            'eval_index': eval_index, 'sample_index': sample_index,
                            'atoms': atoms})
    return records


def extract_pt(gen_pt_file: Path):
    """Replace an export only after every VASP and the manifest are verified."""
    gen_pt_file = Path(gen_pt_file)
    click.echo(f'{gen_pt_file} processing...')
    data = torch.load(gen_pt_file, map_location='cpu', weights_only=False)
    records = evaluation_records(data)
    directory_name = gen_pt_file.stem if gen_pt_file.suffix == ".pt" else gen_pt_file.name.replace(".", "_")
    extract_dir = gen_pt_file.with_name(directory_name)
    extract_dir.mkdir(exist_ok=True)
    save_path = extract_dir / 'gen'
    pkl_path = extract_dir / (gen_pt_file.name + '.pkl')
    if save_path.is_symlink() or (save_path.exists() and not save_path.is_dir()):
        raise ValueError(f'Export path must be a regular directory: {save_path}')
    with tempfile.TemporaryDirectory(prefix='.export-', dir=extract_dir) as staging:
        staging = Path(staging)
        new_gen, backup = staging / 'gen', staging / 'previous-gen'
        new_gen.mkdir()
        for record in records:
            output = new_gen / record['filename']
            write(output, record['atoms'], format='vasp', direct=True)
            verified = read(output, format='vasp')
            if len(verified) != len(record['atoms']) or not np.array_equal(verified.numbers, record['atoms'].numbers):
                raise ValueError(f'Export verification failed: {output}')
        with (new_gen / 'structure_manifest.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows({key: record[key] for key in MANIFEST_FIELDS} for record in records)
        new_pickle = staging / 'structures.pkl'
        with new_pickle.open('wb') as stream:
            pickle.dump([record['atoms'] for record in records], stream)
        had_previous = save_path.exists()
        if had_previous:
            save_path.rename(backup)
        try:
            new_gen.rename(save_path)
            os.replace(new_pickle, pkl_path)
        except Exception:
            if save_path.exists():
                shutil.rmtree(save_path)
            if had_previous:
                backup.rename(save_path)
            raise
    return records


@click.command(help='Export all evaluations to eval_*/gen with structure_manifest.csv. Reruns replace the complete export deterministically.')
@click.argument('gen_pt_files', nargs=-1, required=True)
@click.option('-j', '--njobs', default=1, type=int, help='Parallel jobs for distinct input files.')
def save_gen_structure(gen_pt_files, njobs):
    paths = list(dict.fromkeys(Path(path).resolve() for path in gen_pt_files))
    Parallel(njobs)(delayed(extract_pt)(path) for path in paths)


if __name__ == '__main__':
    save_gen_structure()
