"""V2 model, persistence and export regressions using real CPU tensors/files."""

import importlib.util
import json
import logging
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data

from cdvae.common.condition_utils import prepare_scalar_values
from cdvae.common.data_utils import StandardScalerTorch, get_scaler_from_data_list
from cdvae.common.pdm_schema import (
    GEOMETRY, attach_inference_pdm_contract, cache_contract,
    preserve_training_pdm_contract, read_pdm_schema,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'cond-cdvae/scripts'
SCHEMA = {'schema_version': 2, 'features': ['FeFe', 'FeNi', 'NiNi'],
          'elements': ['Fe', 'Ni'], 'modes': ['pair'], 'cutoff': 5.0, 'geometry': GEOMETRY}


def script(name, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location('v2_' + name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contract_files(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'pdm_schema.json').write_text(json.dumps(SCHEMA))
    (directory / 'scaler_stats.txt').write_text('Train Mean: 0,0,0\nTrain Std: 1,1,1\n')


def test_training_contract_copies_schema_stats_and_rejects_changes(tmp_path):
    source, model_dir = tmp_path / 'data', tmp_path / 'model'
    contract_files(source)
    cfg = OmegaConf.create({'data': {'root_path': str(source), 'pdm_dim': 3},
                            'model': {'conditions': {'cond_keys': ['composition', 'element_values']}}})
    preserve_training_pdm_contract(cfg, model_dir)
    assert json.loads((model_dir / 'pdm_schema.json').read_text()) == SCHEMA
    assert (model_dir / 'scaler_stats.txt').read_bytes() == (source / 'scaler_stats.txt').read_bytes()
    (source / 'scaler_stats.txt').write_text('Train Mean: 1,0,0\nTrain Std: 1,1,1\n')
    with pytest.raises(ValueError, match='normalization statistics differ'):
        preserve_training_pdm_contract(cfg, model_dir)
    (source / 'scaler_stats.txt').unlink()
    with pytest.raises(ValueError, match='requires pdm_schema.json and scaler_stats.txt'):
        preserve_training_pdm_contract(cfg, model_dir)


def test_legacy_checkpoint_cannot_be_resumed_as_v2(tmp_path):
    source, model_dir = tmp_path / 'data', tmp_path / 'model'
    contract_files(source)
    model_dir.mkdir()
    (model_dir / 'epoch=1.ckpt').touch()
    cfg = OmegaConf.create({'data': {'root_path': str(source), 'pdm_dim': 3},
                            'model': {'conditions': {'cond_keys': ['element_values']}}})
    with pytest.raises(ValueError, match='Cannot resume a legacy'):
        preserve_training_pdm_contract(cfg, model_dir)
    model = SimpleNamespace(multiemb=SimpleNamespace(cond_keys=['element_values'],
                             sub_emb_list=[SimpleNamespace(n_in=3)]))
    with pytest.warns(UserWarning, match='not a V2 PDM model'):
        attach_inference_pdm_contract(model, model_dir)
    assert model.pdm_schema is None


def test_schema_dimension_and_duplicate_feature_validation(tmp_path):
    contract_files(tmp_path)
    with pytest.raises(ValueError, match='3 features'):
        read_pdm_schema(tmp_path / 'pdm_schema.json', 2)
    bad = dict(SCHEMA, features=['FeFe', 'FeFe', 'NiNi'])
    (tmp_path / 'pdm_schema.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='Invalid V2 PDM schema'):
        read_pdm_schema(tmp_path / 'pdm_schema.json', 3)


def test_spg_scaler_and_legacy_float_stats_are_copyable():
    scaler = get_scaler_from_data_list([{'spgno': 1}, {'spgno': 230}], 'spgno').copy()
    assert scaler.means.item() == 115.5
    scaler.means, scaler.stds = 115.5, 115.0  # Old checkpoint pickle representation.
    copied = scaler.copy()
    actual = prepare_scalar_values('spgno', [1, 230], ['spgno'], [copied])
    torch.testing.assert_close(actual, torch.tensor([[-114.5 / 115], [114.5 / 115]]))


class ScalarDecoder:
    def __init__(self):
        self.device = torch.device('cpu')
        self.hparams = SimpleNamespace(latent_dim=2, data=SimpleNamespace(prop=['pressure']))
        self.prop_scalers = [StandardScalerTorch([10.], [2.])]
        self.multiemb = SimpleNamespace(cond_keys=['composition', 'pressure'])
        self.seen = []
        outer = self
        class Recorder:
            cond_keys = ['composition', 'pressure']
            def __call__(self, conditions):
                outer.seen.append(conditions)
                return conditions
        self.multiemb = Recorder()
    def zgivenc(self, z, c):
        return z
    def langevin_dynamics(self, z, kwargs, num_atoms, atom_types):
        return {'frac_coords': torch.zeros(int(num_atoms.sum()), 3), 'num_atoms': num_atoms,
                'atom_types': atom_types, 'lengths': torch.ones(len(num_atoms), 3),
                'angles': torch.full((len(num_atoms), 3), 90.)}


def test_target_scales_scalars_and_preserves_id_axis_across_batches(tmp_path, monkeypatch):
    module, model = script('evaluate', monkeypatch), ScalarDecoder()
    target = tmp_path / 'targets.csv'
    pd.DataFrame({'formula': ['Fe2', 'Ni2', 'FeNi'], 'pressure': [14, 18, 20],
                  'material_id': ['001', 'NA', 'C']}).to_csv(target, index=False)
    outputs = module.target_generation(model, SimpleNamespace(save_traj=False), 2, 2,
                                       str(target), batch_size=2)
    assert outputs[1].shape == (2, 6)
    assert outputs[-1] == ['001', 'NA', 'C', '001', 'NA', 'C']
    torch.testing.assert_close(model.seen[0]['pressure'], torch.tensor([[2.], [4.]]))
    torch.testing.assert_close(model.seen[1]['pressure'], torch.tensor([[5.]]))
    pd.DataFrame({'formula': ['Fe'], 'pressure': [float('nan')]}).to_csv(target, index=False)
    with pytest.raises(ValueError, match='scalar'):
        module.target_generation(model, SimpleNamespace(save_traj=False), 1, 1, str(target))


def eval_payload(reconstruction=False):
    payload = {'num_atoms': torch.tensor([[1, 2], [1, 2]]),
               'atom_types': torch.tensor([[26, 28, 28], [26, 28, 28]]),
               'frac_coords': torch.tensor([[[0., 0, 0], [.1, .1, .1], [.5, .5, .5]],
                                             [[.2, 0, 0], [.1, .1, .1], [.6, .6, .6]]]),
               'lengths': torch.full((2, 2, 3), 4.), 'angles': torch.full((2, 2, 3), 90.)}
    if reconstruction:
        payload['input_data_batch'] = dict(mp_id=['same_id', 'same_id'],
            **{key: val[0] for key, val in payload.items()})
    else:
        payload['material_id_list'] = ['A', 'B']
    return payload


@pytest.mark.parametrize('reconstruction', [False, True])
def test_export_all_evals_manifest_ids_and_idempotence(tmp_path, monkeypatch, reconstruction):
    module = script('extract_gen', monkeypatch)
    path = tmp_path / 'eval_recon.pt'
    torch.save(eval_payload(reconstruction), path)
    module.extract_pt(path)
    directory = tmp_path / 'eval_recon/gen'
    manifest = pd.read_csv(directory / 'structure_manifest.csv')
    assert len(list(directory.glob('*.vasp'))) == 4
    assert manifest.eval_index.tolist() == [0, 0, 1, 1]
    assert manifest.sample_index.tolist() == [0, 1, 0, 1]
    assert manifest.material_id.tolist() == (['same_id'] * 4 if reconstruction else ['A', 'B', 'A', 'B'])
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    module.extract_pt(path)
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}
    assert len(pickle.loads((tmp_path / 'eval_recon/eval_recon.pt.pkl').read_bytes())) == 4


def test_failed_export_never_marks_complete_or_destroys_previous(tmp_path, monkeypatch):
    module = script('extract_gen', monkeypatch)
    path = tmp_path / 'eval_gen.pt'
    torch.save(eval_payload(), path)
    def fail(*args, **kwargs):
        raise OSError('simulated full disk')
    real_write = module.write
    monkeypatch.setattr(module, 'write', fail)
    with pytest.raises(OSError, match='full disk'):
        module.extract_pt(path)
    assert not (tmp_path / 'eval_gen/eval_gen.pt.pkl').exists()
    assert not (tmp_path / 'eval_gen/gen').exists()
    monkeypatch.setattr(module, 'write', real_write)
    module.extract_pt(path)
    old_pickle = (tmp_path / 'eval_gen/eval_gen.pt.pkl').read_bytes()
    old_manifest = (tmp_path / 'eval_gen/gen/structure_manifest.csv').read_bytes()
    monkeypatch.setattr(module, 'write', fail)
    with pytest.raises(OSError):
        module.extract_pt(path)
    assert (tmp_path / 'eval_gen/eval_gen.pt.pkl').read_bytes() == old_pickle
    assert (tmp_path / 'eval_gen/gen/structure_manifest.csv').read_bytes() == old_manifest


def test_ambiguous_legacy_target_id_order_rejected(monkeypatch):
    module = script('extract_gen', monkeypatch)
    data = eval_payload()
    data['material_id_list'] *= 2
    with pytest.raises(ValueError, match='sample axis'):
        module.evaluation_records(data)


@pytest.mark.parametrize('lengths,angles', [([float('nan'), 3, 3], [90, 90, 90]),
    ([3, 3, 3], [0, 90, 90]), ([3, 3, 3], [10, 10, 170]), ([-1, 3, 3], [90, 90, 90])])
def test_invalid_cells_count_as_invalid_without_crashing(monkeypatch, lengths, angles):
    module = script('compute_metrics', monkeypatch)
    crystal = module.Crystal({'frac_coords': np.zeros((1, 3)), 'atom_types': np.array([26]),
                             'lengths': np.array(lengths), 'angles': np.array(angles)})
    assert not crystal.constructed and not crystal.valid
    assert crystal.struct_fp is None


def test_v2_cache_requires_matching_schema_and_source(tmp_path, monkeypatch):
    import cdvae.pl_data.dataset as datasets
    contract_files(tmp_path)
    source = tmp_path / 'train.feather'
    pd.DataFrame({'material_id': ['A'], 'element_values': [[1., 2., 3.]]}).to_feather(source)
    graph = (np.array([[0., 0., 0.]]), np.array([26]), np.array([3., 3., 3.]),
             np.array([90., 90., 90.]), np.zeros((0, 2), dtype=int), np.zeros((0, 3), dtype=int), 1)
    record = {'mp_id': 'A', 'element_values': [1., 2., 3.], 'graph_arrays': graph}
    monkeypatch.setattr(datasets, 'preprocess', lambda *args, **kwargs: [dict(record)])
    monkeypatch.setattr(hydra.utils, 'log', logging.getLogger(__name__), raising=False)
    options = dict(name='v2', path=str(source), force_process=False, prop=['element_values'],
                   pdm_dim=3, niggli=False, primitive=False, graph_method='crystalnn',
                   preprocess_workers=1, lattice_scale_method='scale_length')
    datasets.CrystDataset(**options)
    meta = Path(str(source.with_suffix('.pkl')) + '.meta.json')
    assert json.loads(meta.read_text()) == cache_contract(source, SCHEMA)
    datasets.CrystDataset(**options)
    # Same dimensionality, changed schema: shape checks alone cannot catch it.
    (tmp_path / 'pdm_schema.json').write_text(json.dumps(dict(SCHEMA, cutoff=6.0)))
    with pytest.raises(ValueError, match='force_process=true'):
        datasets.CrystDataset(**options)
    (tmp_path / 'pdm_schema.json').write_text(json.dumps(SCHEMA))
    pd.DataFrame({'material_id': ['B'], 'element_values': [[1., 2., 3.]]}).to_feather(source)
    with pytest.raises(ValueError, match='force_process=true'):
        datasets.CrystDataset(**options)
    options['force_process'] = True
    datasets.CrystDataset(**options)
    assert json.loads(meta.read_text()) == cache_contract(source, SCHEMA)


def small_config(predict_property=None):
    overrides = ['data=apollox/mp_apollox', 'data.pdm_dim=3', 'model.latent_dim=8',
        'model.encoder.hidden_channels=16', 'model.encoder.int_emb_size=8',
        'model.encoder.out_emb_channels=16', 'model.encoder.num_blocks=1',
        'model.encoder.num_spherical=3', 'model.encoder.num_radial=3',
        'model.decoder.hidden_dim=16', 'model.decoder.num_blocks=1',
        'model.radius=3', 'model.max_neighbors=8', 'model.num_noise_level=2',
        'model.zgivenc.no_mlp=False']
    if predict_property is not None:
        overrides.append(f'model.predict_property={str(predict_property).lower()}')
    with initialize_config_dir(str(ROOT / 'cond-cdvae/conf'), version_base='1.1'):
        return compose('default', overrides=overrides)


def small_batch():
    edge = torch.tensor([(i, j) for i in range(4) for j in range(4) if i != j]).T
    return Batch.from_data_list([Data(
        frac_coords=torch.tensor([[.1,.1,.1], [.4,.1,.1], [.1,.4,.1], [.1,.1,.4]]),
        atom_types=torch.tensor([26,28,26,28]), lengths=torch.full((1,3),4.),
        angles=torch.full((1,3),90.), edge_index=edge, to_jimages=torch.zeros((12,3), dtype=torch.long),
        num_atoms=4, num_bonds=12, num_nodes=4, mp_id=f'a{k}', pressure=torch.tensor([[float(k)]]),
        element_values=torch.tensor([[1.,2.,3.]]) + k) for k in range(2)])


def test_default_apollox_auxiliary_prediction_is_off():
    assert small_config().model.predict_property is False
    with initialize_config_dir(str(ROOT / 'cond-cdvae/conf'), version_base='1.1'):
        cfg = compose('default', overrides=['data=carbon'])
    assert cfg.model.predict_property is True


def test_real_optional_heads_backward_checkpoint_and_short_generation(tmp_path, monkeypatch):
    import pytorch_lightning as pl
    torch.manual_seed(13)
    cfg = small_config(predict_property=True)
    model = hydra.utils.instantiate(cfg.model, optim=cfg.optim, data=cfg.data,
                                    logging=cfg.logging, _recursive_=False)
    batch = small_batch()
    model.lattice_scaler = StandardScalerTorch([4 / 4**(1/3)]*3 + [90.]*3, [.1]*3 + [1.]*3)
    model.prop_scalers = [StandardScalerTorch([0.], [1.]), StandardScalerTorch([0.,0.,0.], [1.,1.,1.])]
    before_inputs, after_inputs = [], []
    model.prop_recall_model_before_cond.register_forward_pre_hook(lambda _, args: before_inputs.append(args[0]))
    model.prop_recall_model_after_cond.register_forward_pre_hook(lambda _, args: after_inputs.append(args[0]))
    outputs = model(batch, teacher_forcing=True)
    _, loss = model.compute_stats(batch, outputs, 'train')
    loss.backward()
    assert before_inputs[0].shape == (2, 8)
    assert after_inputs[0].shape == (2, 64)
    for head in (model.prop_recall_model_before_cond, model.prop_recall_model_after_cond):
        gradients = [parameter.grad for parameter in head.parameters()]
        assert all(value is not None and torch.isfinite(value).all() for value in gradients)
        assert sum(value.abs().sum() for value in gradients) > 0
    contract_files(tmp_path)
    OmegaConf.save(cfg, tmp_path / 'hparams.yaml')
    torch.save({'state_dict': model.state_dict(), 'hyper_parameters': dict(model.hparams),
                'pytorch-lightning_version': pl.__version__}, tmp_path / 'epoch=0-step=0.ckpt')
    torch.save(model.lattice_scaler, tmp_path / 'lattice_scaler.pt')
    torch.save(model.prop_scalers, tmp_path / 'prop_scalers.pt')
    loaded, _, _ = script('eval_utils', monkeypatch).load_model(tmp_path)
    assert loaded.training is False
    assert loaded.pdm_schema == SCHEMA
    assert all(not layer.training for layer in loaded.modules() if isinstance(layer, torch.nn.Dropout))
    conditions = loaded.multiemb(loaded.build_conditions(batch))
    cond_z = loaded.zgivenc(torch.randn(2, 8), conditions)
    samples = loaded.langevin_dynamics(cond_z, SimpleNamespace(
        n_step_each=1, step_lr=1e-5, min_sigma=0, save_traj=False, disable_bar=True),
        batch.num_atoms, batch.atom_types)
    assert samples['frac_coords'].shape == (8, 3)
    assert torch.isfinite(samples['frac_coords']).all()


def test_model_schema_validation_matches_canonical_order(tmp_path):
    contract_files(tmp_path)
    assert read_pdm_schema(tmp_path / 'pdm_schema.json', 3) == SCHEMA
    (tmp_path / 'pdm_schema.json').write_text(json.dumps(dict(SCHEMA, features=list(reversed(SCHEMA['features'])))))
    with pytest.raises(ValueError, match='canonical complete feature order'):
        read_pdm_schema(tmp_path / 'pdm_schema.json', 3)


def test_repeated_evaluation_paths_keep_distinct_pt_stems(tmp_path, monkeypatch):
    module = script('evaluate', monkeypatch)
    first = module.evaluation_output_path(tmp_path, 'eval_gen.pt')
    first.touch()
    second = module.evaluation_output_path(tmp_path, 'eval_gen.pt')
    second.touch()
    third = module.evaluation_output_path(tmp_path, 'eval_gen.pt')
    assert [path.name for path in (first, second, third)] == ['eval_gen.pt', 'eval_gen_1.pt', 'eval_gen_2.pt']


def test_reconstruction_input_contract_must_match_model(tmp_path):
    from cdvae.common.pdm_schema import validate_inference_dataset_contract
    model, data = tmp_path / 'model', tmp_path / 'data'
    contract_files(model)
    contract_files(data)
    validate_inference_dataset_contract(model, data / 'test.feather', 3)
    (data / 'scaler_stats.txt').write_text('Train Mean: 1,0,0\nTrain Std: 1,1,1\n')
    with pytest.raises(ValueError, match='normalization statistics differ'):
        validate_inference_dataset_contract(model, data / 'test.feather', 3)
    contract_files(data)
    (data / 'pdm_schema.json').write_text(json.dumps(dict(SCHEMA, cutoff=6.)))
    with pytest.raises(ValueError, match='schema differs'):
        validate_inference_dataset_contract(model, data / 'test.feather', 3)
    (data / 'pdm_schema.json').unlink()
    with pytest.raises(ValueError, match='Cannot mix'):
        validate_inference_dataset_contract(model, data / 'test.feather', 3)
    (model / 'pdm_schema.json').unlink()
    validate_inference_dataset_contract(model, data / 'test.feather', None)



@pytest.mark.parametrize('split', ['train', 'val', 'test'])
def test_training_split_override_must_match_root_contract(tmp_path, split):
    root, alternate, model = tmp_path / 'root', tmp_path / 'alternate', tmp_path / 'model'
    contract_files(root)
    contract_files(alternate)
    datasets = {'train': {'path': str(root / 'train.feather')},
                'val': [{'path': str(root / 'val.feather')}],
                'test': [{'path': str(root / 'test.feather')}]}
    entry = {'path': str(alternate / 'custom.feather')}
    datasets[split] = entry if split == 'train' else [entry]
    cfg = OmegaConf.create({'data': {'root_path': str(root), 'pdm_dim': 3,
                                    'datamodule': {'datasets': datasets}},
                            'model': {'conditions': {'cond_keys': ['element_values']}}})
    preserve_training_pdm_contract(cfg, model)
    (alternate / 'scaler_stats.txt').write_text('Train Mean: 3,2,1\nTrain Std: 1,1,1\n')
    with pytest.raises(ValueError, match=f'Training {split} dataset.*conflicts'):
        preserve_training_pdm_contract(cfg, model)
    contract_files(alternate)
    (alternate / 'pdm_schema.json').write_text(json.dumps(dict(SCHEMA, cutoff=7.)))
    with pytest.raises(ValueError, match=f'Training {split} dataset.*conflicts'):
        preserve_training_pdm_contract(cfg, model)
