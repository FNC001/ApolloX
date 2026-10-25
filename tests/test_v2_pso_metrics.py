"""Behavior tests for V2 PSO failure handling and scientific metric boundaries."""
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apollox_common.pdm import build_schema, save_schema
from PSO.pso_optimizer import EnergyInterpolator, PSO
from PSO import runtime, submit_opt, submit_one_gen
from evaluation_metrics import coverage_analysis, mean_relative_error, novelty_analysis, uniqueness_analysis, feather_to_csv, sample_feather_pkl


def artifacts(directory, schema=None, means='0,0,0'):
    directory.mkdir(parents=True, exist_ok=True)
    schema = schema or build_schema(['Co', 'Ni'], ['pair'], 5)
    save_schema(directory / 'pdm_schema.json', schema)
    (directory / 'scaler_stats.txt').write_text(f'Train Mean: {means}\nTrain Std: 1,1,1\n')
    return schema


def tables(tmp_path):
    truth = pd.DataFrame({'label': ['A', 'B'], 'formula': ['CoNi', 'CoNi'],
                          'CoCo': [10., 100.], 'CoNi': [20., 200.], 'NiNi': [30., 300.]})
    predicted = truth.iloc[::-1][['label', 'formula', 'NiNi', 'CoCo', 'CoNi']].copy()
    predicted['eval_index'] = 0
    predicted['sample_index'] = [1, 0]
    predicted['job_id'] = ['generation-1', 'generation-0']
    a, b = tmp_path / 'truth.csv', tmp_path / 'predicted.csv'
    truth.to_csv(a, index=False)
    predicted.to_csv(b, index=False)
    return a, b


def test_duplicate_pdm_uses_mean_energy_and_single_point_is_constant():
    surrogate = EnergyInterpolator([[1, 2], [1, 2], [2, 3]], [-4, -2, -5])
    np.testing.assert_allclose(surrogate.predict([[1, 2], [2, 3]]), [-3, -5])
    constant = EnergyInterpolator([[1, 2], [1, 2]], [-4, -2])
    np.testing.assert_allclose(constant.predict([[0, 0], [10, 20]]), [-3, -3])


@pytest.mark.parametrize('points,energies', [([[1, np.nan]], [-3]), ([[1, 2]], [np.inf]), ([], [])])
def test_surrogate_rejects_invalid_data(points, energies):
    with pytest.raises(ValueError):
        EnergyInterpolator(points, energies)


@pytest.mark.parametrize('start,velocity,expected', [(5, 4, 5), (5, -4, 1)])
def test_pso_evaluates_final_move_and_returns_best_position(start, velocity, expected):
    surrogate = EnergyInterpolator([[0], [10]], [0, 10])
    optimizer = PSO(1, [0], [10], surrogate, pop_size=1, max_iter=1, w=1, c1=0, c2=0)
    optimizer.X = np.array([[float(start)]])
    optimizer.V = np.array([[float(velocity)]])
    optimizer.pbest = optimizer.X.copy()
    optimizer.pbest_fitness = surrogate.predict(optimizer.X)
    optimizer._update_best()
    result = optimizer.optimize()
    assert result.tolist() == [[expected]]
    assert optimizer.gbest_fitness == expected
    assert np.issubdtype(result.dtype, np.integer)


def test_pso_training_artifacts_reject_wrong_scaler_and_old_model(tmp_path):
    training, model = tmp_path / 'training', tmp_path / 'model'
    artifacts(training)
    artifacts(model)
    config = {'scaler_path': str(training / 'scaler_stats.txt'), 'pdm': {'mode': 'pair', 'cutoff': 5}}
    runtime.validate_training_artifacts(config, model)
    (model / 'scaler_stats.txt').write_text('Train Mean: 0,1,0\nTrain Std: 1,1,1\n')
    with pytest.raises(ValueError, match='statistics differ'):
        runtime.validate_training_artifacts(config, model)
    (model / 'pdm_schema.json').unlink()
    with pytest.raises(FileNotFoundError):
        runtime.validate_training_artifacts(config, model)


def prepare_pso(tmp_path, monkeypatch):
    model, repo, training = tmp_path / 'model', tmp_path / 'repo', tmp_path / 'training'
    artifacts(model)
    artifacts(training)
    (repo / 'opt').mkdir(parents=True)
    (repo / 'opt/chgnet_cpu.py').write_text('# external process fixture')
    (repo / 'PSO').mkdir()
    config = yaml.safe_load((ROOT / 'PSO/config.yaml').read_text())
    config['scaler_path'] = str(training / 'scaler_stats.txt')
    config['opt']['max_workers'] = 1
    config['opt']['min_free_mem_gb'] = 0
    monkeypatch.chdir(model)
    monkeypatch.setattr(submit_opt, 'repository_root', lambda: repo)
    monkeypatch.setattr(submit_opt, 'load_config', lambda: config)
    return model, repo, config


def test_cpu_configuration_is_honored_and_failed_job_aborts_generation(tmp_path, monkeypatch):
    model, repo, config = prepare_pso(tmp_path, monkeypatch)
    gen = model / 'temp/pt_files_1/eval_gen_POSCAR-job/gen'
    gen.mkdir(parents=True)
    (gen / '0.vasp').write_text('external optimizer input fixture')
    called = []
    def fail(script, *args, **kwargs):
        called.append(script)
        raise subprocess.CalledProcessError(7, [str(script)])
    monkeypatch.setattr(submit_opt, 'run_python', fail)
    with pytest.raises(subprocess.CalledProcessError):
        submit_opt.run_chgnet_and_pdm(1)
    assert called == [repo / 'opt/chgnet_cpu.py']
    assert not (model / 'pdm_and_energy/updated_all_structures_summary_batch_2.csv').exists()


def test_stale_energy_cannot_be_reused(tmp_path, monkeypatch):
    (tmp_path / 'sorted_energies.csv').write_text('name,energy\nold,-10\n')
    def forbidden(*args, **kwargs):
        pytest.fail('Old outputs must be rejected before launching a job')
    monkeypatch.setattr(submit_opt, 'run_python', forbidden)
    with pytest.raises(FileExistsError, match='previous run outputs'):
        submit_opt.run_task(tmp_path, 'POSCAR-new.vasp', Path('cpu.py'), {}, Path('compute.py'),
                            {'output_csv': 'all_structures_summary.csv'}, ROOT, ['--schema', 'schema.json'])


def test_generation_requires_exact_output_not_unrelated_pt(tmp_path, monkeypatch):
    model, repo, config = prepare_pso(tmp_path, monkeypatch)
    monkeypatch.setattr(submit_one_gen, 'repository_root', lambda: ROOT)
    monkeypatch.setattr(submit_one_gen, 'load_config', lambda: config)
    poscars = model / 'poscars/generation1'
    poscars.mkdir(parents=True)
    (poscars / 'POSCAR-source').write_text('fixture\n1\n10 0 0\n0 10 0\n0 0 10\nCo Ni\n1 1\n')
    unrelated = model / 'old_checkpoint.pt'
    unrelated.write_bytes(b'old artifact')
    def preprocess(script, *args, **kwargs):
        if Path(script).name == 'pso_optimizer.py':
            Path('temp').mkdir(exist_ok=True)
        elif Path(script).name == 'standardize.py':
            pd.DataFrame([['particle-0', 'p.cif', 0, 0, 0]],
                         columns=['material_id', 'cif_file', 'CoCo', 'CoNi', 'NiNi']).to_csv(
                             'temp/standardized_optimized_particles_distribution1.csv', index=False)
        else:
            pytest.fail('Extraction must not run when the expected generation output is missing')
    monkeypatch.setattr(submit_one_gen, 'run_python', preprocess)
    monkeypatch.setattr(submit_one_gen.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0))
    with pytest.raises(RuntimeError, match='expected output'):
        submit_one_gen.generate_one_generation(1)
    assert unrelated.read_bytes() == b'old artifact'


def test_mre_aligns_ids_and_feature_names_and_ignores_metadata(tmp_path):
    true, predicted = tables(tmp_path)
    result = mean_relative_error.compute_average_relative_error(true, predicted, tmp_path / 'errors.csv')
    assert result.label.tolist() == ['A', 'B']
    assert result.mean_relative_error.tolist() == [0, 0]


def test_mre_preserves_na_and_numeric_string_ids_in_real_csv(tmp_path):
    truth, predicted = tmp_path / 'truth.csv', tmp_path / 'predicted.csv'
    truth.write_text('material_id,formula,HH\nNA,H2,1\n001,H2,2\n1,H2,3\n')
    predicted.write_text('label,formula,HH\n1,H2,3\nNA,H2,1\n001,H2,2\n')
    result = mean_relative_error.compute_average_relative_error(
        truth, predicted, tmp_path / 'errors.csv')
    assert result.label.tolist() == ['NA', '001', '1']
    assert result.mean_relative_error.tolist() == [0, 0, 0]


@pytest.mark.parametrize('mutation', ['duplicate', 'missing_id', 'wrong_id', 'nan', 'wrong_feature'])
def test_mre_rejects_invalid_alignment(tmp_path, mutation):
    true, predicted = tables(tmp_path)
    frame = pd.read_csv(predicted)
    if mutation == 'duplicate': frame.loc[1, 'label'] = frame.loc[0, 'label']
    if mutation == 'missing_id': frame.loc[1, 'label'] = None
    if mutation == 'wrong_id': frame.loc[1, 'label'] = 'other'
    if mutation == 'nan': frame.loc[1, 'CoNi'] = np.nan
    if mutation == 'wrong_feature': frame.rename(columns={'CoNi': 'FeFe'}, inplace=True)
    frame.to_csv(predicted, index=False)
    with pytest.raises(ValueError):
        mean_relative_error.compute_average_relative_error(true, predicted, tmp_path / 'errors.csv')


def test_mre_requires_selection_for_multiple_evaluations(tmp_path):
    true, predicted = tables(tmp_path)
    frame = pd.read_csv(predicted)
    second = frame.copy()
    second['eval_index'] = 1
    pd.concat([frame, second]).to_csv(predicted, index=False)
    with pytest.raises(ValueError, match='select --eval-index'):
        mean_relative_error.compute_average_relative_error(true, predicted, tmp_path / 'errors.csv')
    result = mean_relative_error.compute_average_relative_error(true, predicted, tmp_path / 'errors.csv', eval_index=1)
    assert result.mean_relative_error.tolist() == [0, 0]


def test_coverage_cli_and_novelty_align_named_features(tmp_path, monkeypatch):
    true, predicted = tables(tmp_path)
    command = [sys.executable, str(ROOT / 'evaluation_metrics/coverage_analysis.py'), '--known', str(true), '--new', str(predicted)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    coverage = pd.read_csv(tmp_path / 'coverage_results.csv')
    assert coverage[['recall', 'precision']].to_numpy().tolist() == [[100, 100]]
    monkeypatch.chdir(tmp_path)
    novelty = novelty_analysis.calculate_novelty(true, predicted, 0.2, 1)
    assert novelty.is_novel.tolist() == [0, 0]
    bad = pd.read_csv(predicted).rename(columns={'CoNi': 'FeFe'})
    bad.to_csv(predicted, index=False)
    with pytest.raises(ValueError, match='feature names differ'):
        coverage_analysis.calculate_coverage(true, predicted, 0.2, [1])
    with pytest.raises(ValueError, match='feature names differ'):
        novelty_analysis.calculate_novelty(true, predicted, 0.2, 1)


def test_uniqueness_ignores_non_numeric_manifest_metadata(tmp_path):
    _, predicted = tables(tmp_path)
    frame = pd.read_csv(predicted)
    frame.loc[:, ['CoCo', 'CoNi', 'NiNi']] = [1., 2., 3.]
    frame.to_csv(predicted, index=False)
    uniqueness_analysis.analyze_and_save(predicted, 0.2, tmp_path / 'unique.csv', str(tmp_path / 'unique.json'))
    summary = json.loads((tmp_path / 'unique.json').read_text())
    assert summary['analysis_summary']['unique_clusters_found'] == 1


def test_feather_native_array_restores_using_schema(tmp_path):
    artifacts(tmp_path, means='10,20,30')
    feather = tmp_path / 'data.feather'
    pd.DataFrame({'material_id': ['POSCAR-CoNi-0'], 'element_values': [[1., 2., 3.]]}).to_feather(feather)
    result = feather_to_csv.main(feather, None, tmp_path / 'scaler_stats.txt', tmp_path / 'data.csv')
    assert result[['CoCo', 'CoNi', 'NiNi']].to_numpy().tolist() == [[11, 22, 33]]
    pd.DataFrame({'material_id': ['POSCAR-CoNi-0'], 'element_values': [[1., 2.]]}).to_feather(feather)
    with pytest.raises(ValueError, match='expected 3'):
        feather_to_csv.main(feather, None, tmp_path / 'scaler_stats.txt', tmp_path / 'data.csv')


def test_sampler_preserves_alignment_and_updates_cache_provenance(tmp_path):
    schema = artifacts(tmp_path)
    frame = pd.DataFrame({'material_id': ['a', 'b', 'c'], 'element_values': [[1, 2, 3]] * 3})
    source, cache = tmp_path / 'source.feather', tmp_path / 'source.pkl'
    frame.to_feather(source)
    cache.write_bytes(pickle.dumps([{'material_id': value} for value in frame.material_id]))
    Path(str(cache) + '.meta.json').write_text(json.dumps(sample_feather_pkl.contract(source, schema)))
    target, target_cache = tmp_path / 'sample.feather', tmp_path / 'sample.pkl'
    sample_feather_pkl.sample(source, cache, 2, target, target_cache)
    sampled = pd.read_feather(target)
    records = pickle.loads(target_cache.read_bytes())
    assert sampled.material_id.tolist() == [record['material_id'] for record in records]
    metadata = json.loads(Path(str(target_cache) + '.meta.json').read_text())
    assert all(metadata[key] == value for key, value in sample_feather_pkl.contract(target, schema).items())


@pytest.mark.parametrize('entry', ['sample_feather_pkl', 'sample_feather_pkl.py'])
def test_sampler_compatibility_cli(entry):
    result = subprocess.run([sys.executable, str(ROOT / 'evaluation_metrics' / entry), '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--feather_out' in result.stdout


def test_successful_generation_consumes_only_local_unused_reserve(tmp_path, monkeypatch):
    model, repo, config = prepare_pso(tmp_path, monkeypatch)
    base = model / 'temp/pt_files_1'
    columns = ['material_id', 'cif_file', 'CoCo', 'CoNi', 'NiNi', 'Energy']
    merged_files = []
    for index in range(3):
        folder = base / f'eval_gen_POSCAR-job{index}/gen'
        folder.mkdir(parents=True)
        identifier = f'POSCAR-job{index}.vasp.optdone'
        (folder / identifier).write_text('fresh optimized structure')
        merged = folder / 'Merged_all_structures_with_energy.csv'
        pd.DataFrame([[identifier, identifier + '.cif', 1, 2, 3, -10]], columns=columns).to_csv(merged, index=False)
        merged_files.append(merged)
    initial = model / 'pso_initial/poscar'
    initial.mkdir(parents=True)
    reserve_rows = []
    for index in range(3):
        identifier = f'POSCAR-reserve{index}.optdone'
        (initial / identifier).write_text('unused initial structure')
        reserve_rows.append([identifier, identifier + '.cif', 3, 2, 1, -9])
    reserve_path = model / 'pso_initial/reserve.csv'
    pd.DataFrame(reserve_rows, columns=columns).to_csv(reserve_path, index=False)
    monkeypatch.setattr(submit_opt, 'run_chgnet_parallel', lambda *args, **kwargs: merged_files)
    submit_opt.run_chgnet_and_pdm(1)
    assert len(pd.read_csv(model / 'pdm_and_energy/updated_all_structures_summary_batch_2.csv')) == 5
    assert len(list((model / 'poscars/generation2').iterdir())) == 5
    assert pd.read_csv(reserve_path).material_id.tolist() == ['POSCAR-reserve2.optdone']
    assert not (repo / 'PSO/poscar').exists()
    assert not (repo / 'PSO/Merged_all_structures_with_energy.csv').exists()


def test_single_export_manifest_must_match_every_structure(tmp_path):
    gen = tmp_path / 'gen'
    gen.mkdir()
    (gen / '0.vasp').write_text('verified exporter output fixture')
    pd.DataFrame([['0.vasp', 'sample-0', 0, 0]],
                 columns=['filename', 'material_id', 'eval_index', 'sample_index']).to_csv(gen / 'structure_manifest.csv', index=False)
    submit_one_gen.validate_single_export(tmp_path)
    (gen / '1.vasp').write_text('unexpected stale structure')
    with pytest.raises(ValueError, match='do not match'):
        submit_one_gen.validate_single_export(tmp_path)


def test_feather_cannot_be_decoded_with_another_training_scaler(tmp_path):
    source, wrong_training = tmp_path / 'source', tmp_path / 'wrong_training'
    artifacts(source, means='10,20,30')
    artifacts(wrong_training, means='100,200,300')
    feather = source / 'data.feather'
    pd.DataFrame({'material_id': ['sample-0'], 'element_values': [[1., 2., 3.]]}).to_feather(feather)
    with pytest.raises(ValueError, match='source scaler statistics differ'):
        feather_to_csv.main(feather, None, wrong_training / 'scaler_stats.txt', tmp_path / 'wrong.csv')
    assert not (tmp_path / 'wrong.csv').exists()
