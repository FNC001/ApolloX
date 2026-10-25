"""Behavior regressions for the V2 data and descriptor contract."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apollox_common.pdm import (
    align_features, build_schema, calculate_distances, calculate_sro, compute_folder,
    load_schema, read_poscar, save_schema, shuffled_poscar,
)


def load(relative):
    spec = importlib.util.spec_from_file_location('v2_' + Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_structure(path, xcoords, species=('H', 'H', 'He', 'He')):
    structure = Structure(Lattice.cubic(30), species, [[x, 0, 0] for x in xcoords], coords_are_cartesian=True)
    Poscar(structure).write_file(path)


def test_schema_stabilizes_same_composition_with_different_observed_pairs(tmp_path):
    train, target = tmp_path / 'train', tmp_path / 'target'
    train.mkdir(); target.mkdir()
    write_structure(train / 'POSCAR-a', [0, .5, 1, 10])
    write_structure(target / 'POSCAR-b', [0, 10, 10.5, 11])
    frame, schema = compute_folder(train, train / 'pairs.csv', cutoff=.75, n_jobs=1, starts_with='POSCAR')
    other, _ = compute_folder(target, target / 'pairs.csv', schema_path=train / 'pdm_schema.json',
                              n_jobs=1, starts_with='POSCAR', formula_metadata=True)
    assert schema['features'] == ['HH', 'HHe', 'HeHe']
    assert frame[schema['features']].values.tolist() == [[1, 1, 0]]
    assert other[schema['features']].values.tolist() == [[0, 1, 1]]


def test_nonorthogonal_minimum_image_matches_pymatgen():
    lattice = Lattice.from_parameters(3, 3, 10, 90, 90, 60)
    structure = Structure(lattice, ['H', 'He'], [[0, 0, 0], [.49, .49, 0]])
    distances = calculate_distances(structure.cart_coords, lattice.matrix)
    assert distances[0, 1] == pytest.approx(1.50089973016)
    assert calculate_sro(distances, ['H', 'He'], 2., ['pair']) == {'HHe': 1}


@pytest.mark.parametrize('selective', [False, True])
def test_poscar_cartesian_scale_and_selective_flags(tmp_path, selective):
    path = tmp_path / 'POSCAR'
    header = 'Selective dynamics\n' if selective else ''
    flags = ' T F T' if selective else ''
    path.write_text(f'test\n2\n10 0 0\n0 10 0\n0 0 10\nH He\n1 1\n{header}Cartesian\n0 0 0{flags}\n1 0 0{flags}\n')
    _, _, positions, _, lattice = read_poscar(path)
    np.testing.assert_allclose(positions, [[0, 0, 0], [2, 0, 0]])
    np.testing.assert_allclose(lattice, np.eye(3) * 20)


def test_bad_structure_aborts_without_zero_descriptor_output(tmp_path):
    (tmp_path / 'POSCAR-bad').write_text('not a POSCAR')
    with pytest.raises(ValueError, match='Failed to compute PDM'):
        compute_folder(tmp_path, tmp_path / 'output.csv', n_jobs=1, starts_with='POSCAR')
    assert not (tmp_path / 'output.csv').exists()
    assert not (tmp_path / 'pdm_schema.json').exists()


@pytest.mark.parametrize('position', ['nan 0 0', 'inf 0 0'])
def test_nonfinite_poscar_coordinates_are_rejected(tmp_path, position):
    path = tmp_path / 'POSCAR'
    path.write_text(f'test\n1\n10 0 0\n0 10 0\n0 0 10\nH\n1\nDirect\n{position}\n')
    with pytest.raises(ValueError, match='finite'):
        read_poscar(path)


def test_schema_complete_multimode_and_rejects_conflicts(tmp_path):
    schema = build_schema(['He', 'H'], ['triple', 'pair'], 5)
    assert len(schema['features']) == 3 + 4
    save_schema(tmp_path / 'pdm_schema.json', schema)
    assert load_schema(tmp_path / 'pdm_schema.json') == schema
    with pytest.raises(ValueError, match='overwrite'):
        save_schema(tmp_path / 'pdm_schema.json', build_schema(['H'], ['pair'], 5))
    with pytest.raises(ValueError, match='cutoff differs'):
        compute_folder(tmp_path, tmp_path / 'output.csv', cutoff=4, schema_path=tmp_path / 'pdm_schema.json')


def test_descriptor_manifest_preserves_original_ids_and_eval_indices(tmp_path):
    write_structure(tmp_path / '0.vasp', [0, .5, 1, 10])
    write_structure(tmp_path / '1.vasp', [0, 10, 10.5, 11])
    pd.DataFrame({'filename': ['0.vasp', '1.vasp'], 'material_id': ['source-a', 'source-a'],
                  'eval_index': [0, 1], 'sample_index': [0, 0]}).to_csv(tmp_path / 'structure_manifest.csv', index=False)
    frame, _ = compute_folder(tmp_path, tmp_path / 'out.csv', cutoff=.75, n_jobs=1, ends_with='.vasp', formula_metadata=True)
    assert frame.label.tolist() == ['source-a', 'source-a']
    assert frame.eval_index.tolist() == [0, 1]
    assert frame.sample_index.tolist() == [0, 0]


def test_standardize_aligns_by_schema_and_preserves_manifest_metadata(tmp_path):
    standardize = load('prepare_dataset/standardize.py')
    save_schema(tmp_path / 'pdm_schema.json', build_schema(['H', 'He'], ['pair'], 5))
    scaler = tmp_path / 'scaler_stats.txt'
    scaler.write_text('Train Mean: 10,20,30\nTrain Std: 1,2,5\n')
    source = tmp_path / 'raw.csv'
    pd.DataFrame({'label': ['a'], 'formula': ['H2He2'], 'eval_index': [1], 'sample_index': [0],
                  'HeHe': [35], 'HHe': [24]}).to_csv(source, index=False)
    standardize.main(source, scaler, tmp_path / 'scaled.csv')
    result = pd.read_csv(tmp_path / 'scaled.csv')
    assert list(result.columns) == ['label', 'formula', 'eval_index', 'sample_index', 'HH', 'HHe', 'HeHe']
    assert result[['HH', 'HHe', 'HeHe']].values.tolist() == [[-10, 2, 1]]
    assert result.eval_index.tolist() == [1]
    with pytest.raises(ValueError, match='absent from the training schema'):
        align_features(pd.DataFrame({'LiLi': [1]}), load_schema(tmp_path / 'pdm_schema.json'))


def test_legacy_scaler_requires_explicit_opt_in(tmp_path):
    standardize = load('prepare_dataset/standardize.py')
    source, scaler = tmp_path / 'raw.csv', tmp_path / 'scaler_stats.txt'
    pd.DataFrame({'label': ['a'], 'formula': ['H2'], 'HH': [3]}).to_csv(source, index=False)
    scaler.write_text('Train Mean: 1\nTrain Std: 2\n')
    with pytest.raises(ValueError, match='schema not found'):
        standardize.main(source, scaler, tmp_path / 'scaled.csv')
    standardize.main(source, scaler, tmp_path / 'scaled.csv', allow_legacy_order=True)
    assert pd.read_csv(tmp_path / 'scaled.csv').HH.tolist() == [1]


def test_standardize_preserves_numeric_and_na_like_material_ids(tmp_path):
    standardize = load('prepare_dataset/standardize.py')
    save_schema(tmp_path / 'pdm_schema.json', build_schema(['H'], ['pair'], 5))
    scaler = tmp_path / 'scaler_stats.txt'
    scaler.write_text('Train Mean: 0\nTrain Std: 1\n')
    source = tmp_path / 'raw.csv'
    labels = ['001', '1', 'NA']
    pd.DataFrame({'label': labels, 'formula': ['H2'] * 3, 'HH': [1, 2, 3]}).to_csv(source, index=False)
    standardize.main(source, scaler, tmp_path / 'scaled.csv')
    output = pd.read_csv(tmp_path / 'scaled.csv', dtype={'label': str}, keep_default_na=False)
    assert output.label.tolist() == labels


def test_split_saves_train_only_scaler_and_full_precision(tmp_path):
    split = load('prepare_dataset/split_dataset.py')
    schema = build_schema(['H'], ['pair'], 5)
    save_schema(tmp_path / 'pdm_schema.json', schema)
    raw = pd.DataFrame({'material_id': [f'a{i}' for i in range(10)],
                        'cif_file': [f'a{i}.cif' for i in range(10)],
                        'HH': np.linspace(1e-8, 1e-7, 10)})
    source = tmp_path / 'raw.csv'; raw.to_csv(source, index=False)
    scaler = split.split_dataset(source, tmp_path / 'split')
    expected = raw.sample(frac=1, random_state=42).iloc[:8].HH
    assert scaler.mean_[0] == pytest.approx(expected.mean())
    assert scaler.scale_[0] == pytest.approx(expected.std(ddof=0))
    standardize = load('prepare_dataset/standardize.py')
    _, std = standardize.load_scaler_stats(tmp_path / 'split/scaler_stats.txt')
    assert std[0] > 0
    assert std[0] == scaler.scale_[0]
    assert load_schema(tmp_path / 'split/pdm_schema.json') == schema


def test_shuffle_preserves_site_metadata_without_shuffling_velocity_lines(monkeypatch):
    structure = Structure(Lattice.cubic(10), ['H', 'He'], [[.1, 0, 0], [.2, 0, 0]])
    source = Poscar(structure, selective_dynamics=[[True, False, True], [False, True, False]],
                    velocities=[[1, 2, 3], [4, 5, 6]])
    monkeypatch.setattr('apollox_common.pdm.random.shuffle', lambda values: values.reverse())
    result = shuffled_poscar(source)
    np.testing.assert_allclose(result.structure.frac_coords, [[.2, 0, 0], [.1, 0, 0]])
    assert result.structure[0].specie.symbol == 'H'
    np.testing.assert_equal(result.selective_dynamics, [[False, True, False], [True, False, True]])
    np.testing.assert_allclose(result.velocities, [[4, 5, 6], [1, 2, 3]])
    roundtrip = Poscar.from_str(result.get_str())
    np.testing.assert_allclose(roundtrip.velocities, result.velocities)


def test_variable_composition_has_positive_counts_without_rejection_loop():
    generator = load('generate_structure/bulk/generate_variable_component.py')
    assert generator.generate_random_numbers(4, 4) == [1, 1, 1, 1]
    assert generator.generate_random_numbers(4, 1) == [4]
    with pytest.raises(ValueError):
        generator.generate_random_numbers(2, 3)


def test_dataset_orchestrator_stops_at_failed_stage(tmp_path, monkeypatch):
    module = load('prepare_dataset/generate_dataset_main.py')
    config = yaml.safe_load((ROOT / 'prepare_dataset/config.yaml').read_text())
    config['dataset_path'] = str(tmp_path / 'with spaces')
    path = tmp_path / 'config.yaml'; path.write_text(yaml.safe_dump(config))
    calls = []
    def fail(command, check):
        calls.append(command)
        assert check
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(module.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.generate_dataset(path)
    assert len(calls) == 1
    assert str(tmp_path / 'with spaces') in calls[0]


def test_due_neighbor_merges_actual_worker_filename(tmp_path):
    merge = load('due_neighbor/merg_neighbor_csv.py')
    schema = build_schema(['H'], ['pair'], 5)
    for index in [1, 2]:
        folder = tmp_path / f'split_{index}'; folder.mkdir()
        save_schema(folder / 'pdm_schema.json', schema)
        pd.DataFrame({'material_id': [f'a{index}'], 'cif_file': [f'a{index}.cif'], 'HH': [index]}).to_csv(folder / 'all_structures_summary.csv', index=False)
    result = merge.collect_and_merge_csvs(tmp_path, tmp_path / 'combined.csv')
    assert pd.read_csv(result).material_id.tolist() == ['a1', 'a2']


def test_neighbor_scheduler_waits_for_foreground_processes(tmp_path, monkeypatch):
    scheduler = load('due_neighbor/submit_neigubor.py')
    (tmp_path / 'split_1').mkdir()
    monkeypatch.chdir(tmp_path)
    calls = []
    def run(command, **kwargs):
        assert kwargs['check'] is True
        assert 'shell' not in kwargs
        assert '&' not in command
        calls.append(command)
    monkeypatch.setattr(scheduler.subprocess, 'run', run)
    scheduler.submit_jobs(1, 1, max_cores=1)
    assert len(calls) == 1


def test_cif_extension_matches_material_id_and_missing_cif_fails(tmp_path):
    convert = load('prepare_dataset/transfer_POSCAR_to_cif.py')
    append = load('prepare_dataset/append_cif.py')
    path = tmp_path / 'POSCAR-a.vasp'; write_structure(path, [0, .5, 1, 10])
    convert.convert_poscar_to_cif(path, tmp_path)
    assert (tmp_path / 'POSCAR-a.vasp.cif').exists()
    assert '_cell_length_a' in append.read_cif_content('POSCAR-a.vasp', tmp_path)
    with pytest.raises(FileNotFoundError, match='CIF missing'):
        append.read_cif_content('absent', tmp_path)
