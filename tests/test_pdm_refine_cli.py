"""Ordinary refinement CLI: identity, unit conversion, failures and delivered PDM."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from apollox_common.pdm import build_schema
from scripts.pdm_control import refine_structures as cli


def read_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_rows(path, fields, records):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)


def case(tmp_path, names=('0.vasp',)):
    source, model = tmp_path / 'input', tmp_path / 'model'
    source.mkdir(); model.mkdir()
    schema = build_schema(['H'], ['pair'], 5.)
    (model / 'pdm_schema.json').write_text(json.dumps(schema))
    (model / 'scaler_stats.txt').write_text('Train Mean: 0.25\nTrain Std: 2\n')
    structure = Structure(Lattice.cubic(10), ['H', 'H'], [[0, 0, 0], [.49, 0, 0]])
    for name in names:
        Poscar(structure).write_file(source / name)
    records = [{'filename': name, 'material_id': '00017', 'eval_index': str(i), 'sample_index': '03'}
               for i, name in enumerate(names)]
    write_rows(source / 'structure_manifest.csv', list(records[0]), records)
    target, config = tmp_path / 'target.json', tmp_path / 'config.json'
    target.write_text('[1]'); config.write_text('{"steps": 1}')
    out = tmp_path / 'output'
    args = ['--input-dir', str(source), '--output-dir', str(out), '--model-path', str(model),
            '--target', str(target), '--config', str(config)]
    return source, model, target, out, args


def stub(status='stationary', coordinates=None, seen=None):
    def refine(frac, types, counts, cells, target, std, schema, config):
        if seen is not None:
            seen.append((target.detach().cpu().tolist(), std.detach().cpu().tolist()))
        return {'frac_coords': frac.clone() if coordinates is None else coordinates.to(frac),
                'cell_matrices': cells.clone(), 'reports': [{'status': status}],
                # Deliberately incorrect: CLI must recalculate from the exported file.
                'initial_hard_raw': torch.tensor([[999.]]), 'final_hard_raw': torch.tensor([[999.]]),
                'history': ['must not be exported']}
    return refine


def test_real_cpu_refinement_exports_only_final_counts_and_preserves_identity(tmp_path):
    source, model, target, out, args = case(tmp_path, ('0.vasp', '1.vasp'))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    assert cli.main(args) == 0
    assert {p.name for p in out.iterdir()} == {'gen', 'pdm.csv', 'status.csv', 'run.json'}
    assert read_rows(out / 'gen/structure_manifest.csv') == read_rows(source / 'structure_manifest.csv')
    values = read_rows(out / 'pdm.csv')
    assert len(values) == 2 and all(row['material_id'] == '00017' for row in values)
    for row in values:
        delivered = Poscar.from_file(out / 'gen' / row['label'], check_for_potcar=False).structure
        assert int(row['HH']) == cli.hard_pdm(delivered, json.loads((model / 'pdm_schema.json').read_text()))['HH']
        assert np.allclose(delivered.lattice.matrix, np.eye(3) * 10)
    assert all(row['status'] == 'ok' for row in read_rows(out / 'status.csv'))
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    combined = ''.join(p.read_text() for p in [out / 'pdm.csv', out / 'status.csv', out / 'run.json'])
    assert not any(key in combined for key in ('initial_hard', 'final_hard', 'history', 'mae', 'mre'))


def test_standardized_target_is_inverted_exactly_once_without_checkpoint_or_prop_scaler(tmp_path, monkeypatch):
    source, model, target, out, args = case(tmp_path)
    target.write_text('[0.375]')  # 0.375 * 2 + 0.25 == 1.
    seen = []
    monkeypatch.setattr('cdvae.common.pdm_guidance.refine_pdm', stub(seen=seen))
    assert cli.main(args + ['--target-space', 'standardized']) == 0
    assert seen == [([[1.]], [2.])]
    assert not list(model.glob('*.ckpt')) and not (model / 'prop_scalers.pt').exists()


def test_filename_mapping_not_csv_order_or_repeated_material_id(tmp_path, monkeypatch):
    source, model, target, out, args = case(tmp_path, ('00.vasp', '01.vasp'))
    csv_path = tmp_path / 'targets.csv'
    write_rows(csv_path, ['filename', 'element_values'],
               [{'filename': '01.vasp', 'element_values': '[0]'}, {'filename': '00.vasp', 'element_values': '[1]'}])
    seen = []
    monkeypatch.setattr('cdvae.common.pdm_guidance.refine_pdm', stub(seen=seen))
    pos = args.index('--target'); args[pos:pos + 2] = ['--targets', str(csv_path)]
    assert cli.main(args) == 0
    assert [entry[0] for entry in seen] == [[[1.]], [[0.]]]
    assert [row['label'] for row in read_rows(out / 'pdm.csv')] == ['00.vasp', '01.vasp']


@pytest.mark.parametrize('rows', [
    [{'filename': '0.vasp', 'element_values': '[1]'}] * 2,
    [], [{'filename': 'extra.vasp', 'element_values': '[1]'}],
    [{'filename': '../0.vasp', 'element_values': '[1]'}],
])
def test_bad_mapping_rejected_before_output(tmp_path, rows):
    source, model, target, out, args = case(tmp_path)
    csv_path = tmp_path / 'targets.csv'
    write_rows(csv_path, ['filename', 'element_values'], rows)
    pos = args.index('--target'); args[pos:pos + 2] = ['--targets', str(csv_path)]
    assert cli.main(args) == 1 and not out.exists()


@pytest.mark.parametrize('text', ['[NaN]', '[true]', '["1"]', '[[1]]', '[]', '[1,2]'])
def test_bad_target_values_rejected_before_output(tmp_path, text):
    source, model, target, out, args = case(tmp_path)
    target.write_text(text)
    assert cli.main(args) == 1 and not out.exists()


@pytest.mark.parametrize('std', ['0', 'nan', '-1'])
def test_invalid_training_std_rejected_before_output(tmp_path, std):
    source, model, target, out, args = case(tmp_path)
    (model / 'scaler_stats.txt').write_text('Train Mean: 0\nTrain Std: ' + std + '\n')
    assert cli.main(args) == 1 and not out.exists()


def test_existing_or_nested_output_cannot_change_input(tmp_path):
    source, model, target, out, args = case(tmp_path)
    out.mkdir(); (out / 'sentinel').write_text('keep')
    assert cli.main(args) == 1 and (out / 'sentinel').read_text() == 'keep'
    pos = args.index('--output-dir')
    for forbidden in (source, source / 'nested'):
        args[pos + 1] = str(forbidden)
        assert cli.main(args) == 1
    assert not (source / 'nested').exists()


def test_new_output_inside_model_is_allowed_without_overwriting_model(tmp_path, monkeypatch):
    source, model, target, out, args = case(tmp_path)
    before = {p.name: p.read_bytes() for p in model.iterdir()}
    args[args.index('--output-dir') + 1] = str(model / 'refined_structures')
    monkeypatch.setattr('cdvae.common.pdm_guidance.refine_pdm', stub())
    assert cli.main(args) == 0
    assert (model / 'refined_structures/gen/0.vasp').is_file()
    assert all((model / name).read_bytes() == contents for name, contents in before.items())


def test_bad_poscar_does_not_drop_other_attempts_or_fake_success(tmp_path):
    source, model, target, out, args = case(tmp_path, ('0.vasp', '1.vasp'))
    (source / '0.vasp').write_text('broken POSCAR')
    assert cli.main(args) == 1
    states = read_rows(out / 'status.csv')
    assert [r['status'] for r in states] == ['error', 'ok']
    assert states[0]['error'] and len(states) == 2
    assert [r['label'] for r in read_rows(out / 'pdm.csv')] == ['1.vasp']
    assert [r['filename'] for r in read_rows(out / 'gen/structure_manifest.csv')] == ['1.vasp']
    assert json.loads((out / 'run.json').read_text())['status'] == 'COMPLETE_WITH_ERRORS'


def test_known_impossible_target_rejected_by_unchanged_core(tmp_path):
    source, model, target, out, args = case(tmp_path)
    target.write_text('[2]')  # Two H atoms have at most one unordered site pair.
    assert cli.main(args) == 1
    states = read_rows(out / 'status.csv')
    assert 'pair-count bounds' in states[0]['error']
    assert not list((out / 'gen').glob('*.vasp')) and not read_rows(out / 'pdm.csv')


@pytest.mark.parametrize('status,exit_code', [('line_search_stalled', 0), ('nonfinite_gradient', 1),
                                            ('stationary_with_collisions', 1)])
def test_optimizer_termination_is_retained_without_precision_claim(tmp_path, monkeypatch, status, exit_code):
    source, model, target, out, args = case(tmp_path)
    monkeypatch.setattr('cdvae.common.pdm_guidance.refine_pdm', stub(status=status))
    assert cli.main(args) == exit_code
    assert read_rows(out / 'status.csv')[0]['optimizer_status'] == status


def test_final_file_hard_counts_are_authoritative_at_cutoff(tmp_path, monkeypatch):
    source, model, target, out, args = case(tmp_path)
    # A coordinate immediately below 0.5 rounds at POSCAR precision. Regardless
    # of that rounding, the CSV must describe the delivered file, not core output.
    coords = torch.tensor([[0., 0., 0.], [np.nextafter(.5, 0.), 0., 0.]], dtype=torch.float64)
    monkeypatch.setattr('cdvae.common.pdm_guidance.refine_pdm', stub(coordinates=coords))
    assert cli.main(args) == 0
    structure = Poscar.from_file(out / 'gen/0.vasp', check_for_potcar=False).structure
    expected = int(structure.lattice.get_all_distances(structure.frac_coords, structure.frac_coords)[0, 1] < 5.)
    assert int(read_rows(out / 'pdm.csv')[0]['HH']) == expected
    assert int(read_rows(out / 'pdm.csv')[0]['HH']) != 999


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'escape'])
def test_manifest_mismatch_rejected_before_output(tmp_path, mutation):
    source, model, target, out, args = case(tmp_path)
    records = read_rows(source / 'structure_manifest.csv')
    if mutation == 'duplicate': records *= 2
    if mutation == 'missing': records = []
    if mutation == 'escape': records[0]['filename'] = '../0.vasp'
    write_rows(source / 'structure_manifest.csv', ['filename', 'material_id', 'eval_index', 'sample_index'], records)
    assert cli.main(args) == 1 and not out.exists()
