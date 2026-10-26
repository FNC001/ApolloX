#!/usr/bin/env python3
"""Explicit fresh-optimizer PDM-control pilot. No test set or proxy selection.

A model snapshot is an input to an independent prior-generation validator, not
an accuracy claim. All scientific inputs must be frozen before a pilot launch.
"""
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def append_json(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')
        f.flush()


def validate_spec(spec, max_steps=None):
    required = ('repo', 'data', 'out', 'warmstart_checkpoint',
                'warmstart_checkpoint_sha256', 'warmstart_hparams',
                'model_overlay', 'model_target', 'frozen_files')
    for key in required:
        if not spec.get(key):
            raise ValueError('Missing spec field: ' + key)
    if spec.get('mode') not in ('pilot', 'smoke'):
        raise ValueError('mode must be explicit pilot or smoke')
    if spec.get('mode') == 'pilot':
        if max_steps is not None or spec.get('expected_counts') != {'train': 10000, 'val': 1000}:
            raise ValueError('Formal pilot requires all 10000 train / 1000 val and no max-steps')
        if spec.get('budget_seconds') != 1800 or spec.get('midpoint_seconds') != 900:
            raise ValueError('Formal pilot uses the predeclared 0/900/1800-second snapshots')
    elif max_steps is None or max_steps < 1 or 'smoke' not in Path(spec['out']).name.lower():
        raise ValueError('Smoke needs positive --max-steps and a separate smoke-named output')
    if any(k in spec for k in ('test', 'test_path', 'resume', 'ckpt_path')):
        raise ValueError('This runner does not accept test or resume inputs')
    if spec.get('cost_pdm') not in (1, 10, 100):
        raise ValueError('cost_pdm must be in the predeclared 1/10/100 grid')
    for key, expected in [('seed', 20261026), ('lr', 1e-4)]:
        if spec.get(key) != expected:
            raise ValueError(f'{key} must be {expected}')
    if spec.get('batch_size') not in (16, 32) or not 0 <= spec.get('num_workers', -1) <= 4:
        raise ValueError('batch_size must be 16/32; workers must be 0..4')
    if spec.get('gradient_audit_every', 25) < 1:
        raise ValueError('gradient_audit_every must be positive')
    if not 0 < spec['midpoint_seconds'] < spec['budget_seconds']:
        raise ValueError('Invalid checkpoint times')
    if not isinstance(spec['frozen_files'], dict):
        raise ValueError('frozen_files must map absolute paths to SHA256')
    return spec


def verify_frozen(spec):
    for name, expected in spec['frozen_files'].items():
        if not Path(name).is_absolute() or sha256(name) != expected:
            raise ValueError('Frozen input mismatch: ' + name)
    if sha256(spec['warmstart_checkpoint']) != spec['warmstart_checkpoint_sha256']:
        raise ValueError('Warmstart checkpoint SHA mismatch')
    required = [spec['warmstart_checkpoint'], spec['warmstart_hparams'], spec['model_overlay']]
    data = Path(spec['data'])
    required += [str(data / name) for name in (
        'pdm_schema.json', 'scaler_stats.txt', 'train.feather', 'val.feather',
        'train.pkl', 'val.pkl', 'train.pkl.meta.json', 'val.pkl.meta.json')]
    if any(str(Path(p).resolve()) not in spec['frozen_files'] for p in required):
        raise ValueError('frozen_files must bind all train/val/model inputs, including caches')


class WeightedProxy:
    """Keep graph and eligible-graph denominators separate."""
    def __init__(self):
        self.n = 0
        self.prior = 0
        self.sums = {}
        self.denominators = {}
        self.ids = []

    def add(self, values, graph_count, prior_count, ids=()):
        self.n += graph_count
        self.prior += prior_count
        self.ids.extend(map(str, ids))
        for key, value in values.items():
            if not math.isfinite(float(value)):
                raise ValueError('Nonfinite diagnostic: ' + key)
        weights = {key: graph_count for key in ('loss', 'base_loss', 'lattice_loss', 'coord_loss', 'kld_loss')}
        for key, count in (
            ('pdm_aux_loss', 'pdm_eligible_graphs'),
            ('pdm_prior_aux_loss', 'pdm_prior_eligible_graphs'),
            ('pdm_posterior_aux_loss', 'pdm_posterior_eligible_graphs'),
        ):
            weights[key] = int(values[count])
        for key, weight in weights.items():
            self.sums[key] = self.sums.get(key, 0.) + float(values[key]) * weight
            self.denominators[key] = self.denominators.get(key, 0) + weight

    def result(self):
        return {'graphs': self.n, 'prior_graphs': self.prior,
                'prior_fraction': self.prior / self.n if self.n else None,
                'means': {k: self.sums[k] / n if n else None for k, n in self.denominators.items()},
                'denominators': self.denominators.copy()}


class FitClock:
    def __init__(self, budget, midpoint, clock=time.monotonic):
        self.budget, self.midpoint, self.clock = budget, midpoint, clock
        self.started = None
        self.mid_saved = False

    def start(self):
        self.started = self.clock()

    def elapsed(self):
        return 0. if self.started is None else self.clock() - self.started

    def midpoint_due(self):
        return not self.mid_saved and self.elapsed() >= self.midpoint

    def expired(self):
        return self.elapsed() >= self.budget


@contextlib.contextmanager
def isolated_rng(seed):
    import numpy as np
    import torch
    states = (random.getstate(), np.random.get_state(), torch.random.get_rng_state(),
              torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.random.set_rng_state(states[2])
        if states[3] is not None:
            torch.cuda.set_rng_state_all(states[3])


def load_weight_state(model, checkpoint):
    """Only state_dict is consumed; old step, optimizer and scheduler are evidence."""
    import torch
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(ckpt.get('state_dict'), dict):
        raise ValueError('Missing warmstart state_dict')
    for name, value in ckpt['state_dict'].items():
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite warmstart tensor: ' + name)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    return {'old_global_step': ckpt.get('global_step'), 'old_epoch': ckpt.get('epoch'),
            'optimizer_states_not_loaded': True, 'scheduler_states_not_loaded': True}


def create_config(spec):
    from omegaconf import OmegaConf
    from cdvae.common.pdm_schema import read_pdm_schema, read_external_stats
    cfg = OmegaConf.load(spec['warmstart_hparams'])
    OmegaConf.set_struct(cfg, False)
    overlay = json.loads(Path(spec['model_overlay']).read_text())
    cfg.model = OmegaConf.merge(cfg.model, overlay)
    cfg.model._target_ = spec['model_target']
    cfg.model.teacher_forcing_lattice = False
    cfg.model.predict_property = False
    cfg.model.pdm_control.cost = spec['cost_pdm']
    cfg.model.pdm_control.temperature = .1
    cfg.model.pdm_control.max_sigma = .5
    cfg.model.pdm_control.prior_probability = .5
    cfg.data.root_path = str(Path(spec['data']).resolve())
    cfg.data.datamodule.scaler_path = None  # Fit using this train split only.
    cfg.data.datamodule.datasets.test = []  # No old test is configured or loaded.
    for split in ('train', 'val'):
        configs = [cfg.data.datamodule.datasets.train] if split == 'train' else cfg.data.datamodule.datasets.val
        if len(configs) != 1:
            raise ValueError('Exactly one train and one validation split required')
        configs[0].path = str(Path(spec['data']) / (split + '.feather'))
        configs[0].save_path = str(Path(spec['data']) / (split + '.pkl'))
        configs[0].force_process = False
        cfg.data.datamodule.batch_size[split] = spec['batch_size']
        cfg.data.datamodule.num_workers[split] = spec['num_workers']
    cfg.optim.optimizer = {'_target_': 'torch.optim.Adam', 'lr': spec['lr'],
                           'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0., 'amsgrad': False}
    cfg.optim.use_lr_scheduler = False
    cfg.optim.pop('lr_scheduler', None)
    cfg.train.pop('early_stopping', None)
    cfg.train.pop('model_checkpoints', None)
    cfg.train.pl_trainer = {'max_epochs': -1, 'precision': '32-true', 'gradient_clip_val': .5,
                            'num_sanity_val_steps': 0, 'check_val_every_n_epoch': 1,
                            'limit_train_batches': 1., 'limit_val_batches': 1.,
                            'enable_checkpointing': False}
    cfg.pdm_control_run = {'mode': spec['mode'], 'budget_seconds': spec['budget_seconds'],
        'midpoint_seconds': spec['midpoint_seconds'], 'weights_only_warmstart': True,
        'warmstart_sha256': spec['warmstart_checkpoint_sha256'], 'fresh_optimizer': True,
        'test_used': False, 'proxy_selects_checkpoint': False,
        'selection': 'independent prior-generation validation only'}
    cfg.train.random_seed = spec['seed']
    cfg.train.deterministic = False
    cfg.logging.val_check_interval = 1
    cfg.logging.wandb.mode = 'disabled'
    schema = read_pdm_schema(Path(spec['data']) / 'pdm_schema.json', cfg.data.pdm_dim)
    mean, std = read_external_stats(Path(spec['data']) / 'scaler_stats.txt', cfg.data.pdm_dim)
    if OmegaConf.to_container(cfg.model.pdm_control.schema) != schema:
        raise ValueError('Auxiliary descriptor schema differs from data schema')
    if list(cfg.model.pdm_control.external_mean) != mean or list(cfg.model.pdm_control.external_std) != std:
        raise ValueError('Auxiliary normalization differs from train external scalers')
    if cfg.model._target_ not in ('cdvae.pl_modules.pdm_control_model.PDMControlCDVAE',
                                  'pdm_control_model.PDMControlCDVAE'):
        raise ValueError('Explicit versioned PDMControlCDVAE class required')
    return cfg


def split_ids(spec):
    import pandas as pd
    result = {}
    for split in ('train', 'val'):
        ids = pd.read_feather(Path(spec['data']) / (split + '.feather'), columns=['material_id'])['material_id']
        if ids.isna().any() or ids.astype(str).duplicated().any() or len(ids) != spec['expected_counts'][split]:
            raise ValueError('Incorrect or duplicate split IDs: ' + split)
        result[split] = ids.astype(str).tolist()
    if set(result['train']) & set(result['val']):
        raise ValueError('Train/validation ID overlap')
    return result


def make_callback(spec, out, cfg, expected_val_ids, max_steps):
    import torch
    from pytorch_lightning import Callback

    class Audit(Callback):
        def __init__(self):
            self.clock = FitClock(spec['budget_seconds'], spec['midpoint_seconds'])
            self.last = None
            self.snapshots = []
            self.grad_record = None
            self.validation_rng = None
            self.cancelled = None
            self.last_train_step = 0
            self.aux_gradient_checks = 0
            self.train_totals = WeightedProxy()
            self._old_handlers = {}

        def capture(self, module, args, result):
            self.last = result

        def values(self, model):
            x = self.last
            def scalar(key):
                value = x[key]
                return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            v = {k: scalar(k) for k in ('lattice_loss', 'coord_loss', 'kld_loss', 'pdm_aux_loss',
                   'pdm_prior_aux_loss', 'pdm_posterior_aux_loss', 'pdm_eligible_graphs',
                   'pdm_prior_eligible_graphs', 'pdm_posterior_eligible_graphs')}
            v['base_loss'] = float(model.hparams.cost_lattice)*v['lattice_loss'] + float(model.hparams.cost_coord)*v['coord_loss'] + float(model.hparams.beta)*v['kld_loss']
            v['loss'] = v['base_loss'] + spec['cost_pdm'] * v['pdm_aux_loss']
            return v

        def snapshot(self, trainer, model, name):
            folder = out / 'checkpoints' / name
            if folder.exists():
                raise ValueError('Snapshot already exists: ' + str(folder))
            folder.mkdir(parents=True)
            for file in ('hparams.yaml', 'pdm_schema.json', 'scaler_stats.txt', 'prop_scalers.pt', 'lattice_scaler.pt'):
                shutil.copyfile(out / file, folder / file)
            file = folder / f'epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt'
            trainer.save_checkpoint(file)
            item = {'name': name, 'path': str(folder), 'checkpoint': str(file),
                    'sha256': sha256(file), 'bytes': file.stat().st_size,
                    'global_step': trainer.global_step, 'epoch': trainer.current_epoch,
                    'elapsed_fit_seconds': self.clock.elapsed(), 'created_utc': utc(),
                    'selection': 'unselected_candidate_for_independent_prior_validation',
                    'model_contract_sha256': {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}}
            self.snapshots.append(item)
            write_json(out / 'checkpoint_manifest.json', {'snapshots': self.snapshots,
                'selection_rule': 'external prior generation OFF/ON only; proxy is diagnostic',
                'test_set_used': False})

        def on_fit_start(self, trainer, model):
            self.hook = model.register_forward_hook(self.capture)
            self.clock.start()
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._old_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, lambda number, frame: setattr(self, 'cancelled', number))

        def on_train_start(self, trainer, model):
            if trainer.global_step != 0 or len(trainer.optimizers) != 1 or trainer.optimizers[0].state:
                raise ValueError('Fresh optimizer and new global step 0 required')
            self.snapshot(trainer, model, 'initial')

        def on_before_backward(self, trainer, model, loss):
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite training loss')
            step = trainer.global_step
            self.grad_record = None
            if step < 3 or step % spec.get('gradient_audit_every', 25) == 0:
                pdm_index = list(model.multiemb.cond_keys).index('element_values')
                groups = {
                    'decoder': [p for p in model.decoder.parameters() if p.requires_grad],
                    'pdm_embedding': [p for p in model.multiemb.sub_emb_list[pdm_index].parameters() if p.requires_grad],
                }
                if not all(groups.values()):
                    raise ValueError('Could not identify decoder / active PDM embedding parameters')
                params = groups['decoder'] + groups['pdm_embedding']
                gradients = torch.autograd.grad(spec['cost_pdm'] * self.last['pdm_aux_loss'], params,
                                                retain_graph=True, allow_unused=True)
                self.grad_record = {}
                offset = 0
                for name, values in groups.items():
                    gs = gradients[offset:offset + len(values)]
                    offset += len(values)
                    if any(g is not None and not bool(torch.isfinite(g).all()) for g in gs):
                        raise ValueError('Nonfinite auxiliary gradient: ' + name)
                    norm = math.sqrt(sum(float(g.detach().double().square().sum().cpu()) for g in gs if g is not None))
                    self.grad_record[name + '_aux_gradient_norm'] = norm
                    self.grad_record[name + '_aux_gradient_tensors'] = sum(g is not None for g in gs)
                self.aux_gradient_checks += 1

        def on_after_backward(self, trainer, model):
            for name, param in model.named_parameters():
                if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                    raise ValueError('Nonfinite total gradient: ' + name)

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            v = self.values(model)
            n = int(batch.num_graphs)
            prior = int(self.last['prior_mask'].sum().detach().cpu())
            self.train_totals.add(v, n, prior)
            record = {'step': trainer.global_step, 'epoch': trainer.current_epoch, 'graphs': n,
                      'prior_graphs': prior, 'elapsed_fit_seconds': self.clock.elapsed(),
                      'values': v, 'aux_gradient_audit': self.grad_record,
                      'lr': trainer.optimizers[0].param_groups[0]['lr'],
                      'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                      'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0}
            append_json(out / 'training_steps.jsonl', record)
            self.last_train_step = trainer.global_step
            # Native epoch summary must not retain all training graphs.
            model.training_step_outputs[-1] = model.training_step_outputs[-1].detach()
            self.last = None
            if self.clock.midpoint_due():
                self.snapshot(trainer, model, 'mid15min')
                self.clock.mid_saved = True
            if self.clock.expired() or self.cancelled is not None:
                trainer.should_stop = True
            write_json(out / 'status.json', {'state': 'training', 'step': trainer.global_step,
                'epoch': trainer.current_epoch, 'elapsed_fit_seconds': self.clock.elapsed(), 'utc': utc()})

        def on_validation_start(self, trainer, model):
            if self.validation_rng is not None:
                raise RuntimeError('Nested validation RNG context')
            self.validation_rng = isolated_rng(spec['seed'] + 1)
            self.validation_rng.__enter__()
            self.val = WeightedProxy()

        def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
            self.val.add(self.values(model), int(batch.num_graphs), int(self.last['prior_mask'].sum().cpu()), batch.mp_id)
            model.validation_step_outputs[-1] = model.validation_step_outputs[-1].detach()
            self.last = None

        def on_validation_end(self, trainer, model):
            try:
                if self.val.ids != expected_val_ids:
                    raise ValueError('Validation was not the complete ordered declared validation split')
                append_json(out / 'validation_proxy.jsonl', dict(self.val.result(),
                    epoch=trainer.current_epoch, step=trainer.global_step,
                    elapsed_fit_seconds=self.clock.elapsed(), diagnostic_only=True,
                    training_rng_restored=True, selection_performed=False))
            finally:
                self.validation_rng.__exit__(None, None, None)
                self.validation_rng = None
            if self.clock.midpoint_due():
                self.snapshot(trainer, model, 'mid15min')
                self.clock.mid_saved = True
            if self.clock.expired() or self.cancelled is not None:
                trainer.should_stop = True

        def on_exception(self, trainer, model, exception):
            if self.validation_rng is not None:
                self.validation_rng.__exit__(None, None, None)
                self.validation_rng = None
            try:
                self.snapshot(trainer, model, 'interrupted')
            except Exception as saving_error:
                append_json(out / 'errors.jsonl', {'emergency_save_error': repr(saving_error)})

        def teardown(self, trainer, model, stage):
            if hasattr(self, 'hook'):
                self.hook.remove()
            for sig, old in self._old_handlers.items():
                signal.signal(sig, old)

    return Audit()


def run(spec, spec_path, max_steps=None, check_only=False):
    validate_spec(spec, max_steps)
    verify_frozen(spec)
    original_spec_sha = sha256(spec_path)
    repo = Path(spec['repo']).resolve()
    os.environ['PROJECT_ROOT'] = str(repo / 'cond-cdvae')
    os.environ['WANDB_MODE'] = 'disabled'
    sys.path.insert(0, str(repo / 'cond-cdvae'))
    sys.path.insert(0, str(repo))
    if spec.get('model_module_dir'):
        sys.path.insert(0, spec['model_module_dir'])
    import torch
    import hydra
    import pytorch_lightning as pl
    from omegaconf import OmegaConf
    from pytorch_lightning.loggers import CSVLogger
    from cdvae.common.pdm_schema import preserve_training_pdm_contract
    torch.set_num_threads(min(4, spec.get('cpu_threads', 2)))
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(spec['seed'], workers=True)
    ids = split_ids(spec)
    cfg = create_config(spec)
    dm = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
    dm.setup('fit')
    # Lightning creates validation iterators before on_validation_start. A
    # separate loader generator prevents that base-seed draw perturbing train RNG.
    original_val_dataloader = dm.val_dataloader
    def isolated_val_dataloader():
        loaders = original_val_dataloader()
        for loader in loaders:
            loader.generator = torch.Generator().manual_seed(spec['seed'] + 1)
        return loaders
    dm.val_dataloader = isolated_val_dataloader
    if len(dm.train_dataset) != len(ids['train']) or len(dm.val_datasets) != 1 or len(dm.val_datasets[0]) != len(ids['val']):
        raise ValueError('Loaded dataset count differs from declared split')
    model = hydra.utils.instantiate(cfg.model, optim=cfg.optim, data=cfg.data, logging=cfg.logging, _recursive_=False)
    model.lattice_scaler = dm.lattice_scaler.copy()
    model.prop_scalers = [x.copy() for x in dm.prop_scalers]
    warm = load_weight_state(model, spec['warmstart_checkpoint'])
    if spec['mode'] == 'pilot' and (warm['old_global_step'] != 8764 or warm['old_epoch'] != 27):
        raise ValueError('Pilot must warmstart from declared 2h validation-best epoch27/step8764')
    receipt = {'state': 'constructor_preflight_pass', 'spec_sha256': original_spec_sha,
        'warmstart': warm, 'warmstart_sha256': spec['warmstart_checkpoint_sha256'],
        'model_class': cfg.model._target_, 'cost_pdm': spec['cost_pdm'],
        'counts': {k: len(v) for k, v in ids.items()}, 'test_configured': False,
        'optimizer': 'fresh Adam, fixed learning rate; old optimizer/scheduler not loaded',
        'source_identity': spec.get('source_identity'),
        'frozen_files': spec['frozen_files'], 'created_utc': utc()}
    if check_only:
        print(json.dumps(receipt, indent=2))
        return 0
    out = Path(spec['out']).resolve()
    out.mkdir(parents=True, exist_ok=False)
    lock = (out / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(out / 'frozen_spec.json', spec)
    write_json(out / 'provenance.json', dict(receipt, runner_sha256=sha256(__file__),
        configuration='new PDM-control version; weights-only warmstart, no legacy resume',
        fit_budget_includes='training, validation and checkpoint IO after on_fit_start',
        thread_environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')},
        torch_version=torch.__version__, lightning_version=pl.__version__))
    preserve_training_pdm_contract(cfg, out)
    (out / 'hparams.yaml').write_text(OmegaConf.to_yaml(cfg))
    torch.save(dm.lattice_scaler, out / 'lattice_scaler.pt')
    torch.save(dm.prop_scalers, out / 'prop_scalers.pt')
    callback = make_callback(spec, out, cfg, ids['val'], max_steps)
    trainer = pl.Trainer(default_root_dir=str(out), accelerator=spec.get('accelerator', 'gpu'),
        devices=1, precision='32-true', max_epochs=-1, max_steps=max_steps or -1,
        callbacks=[callback], logger=CSVLogger(str(out), name='proxy_logs'),
        enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False,
        num_sanity_val_steps=0, check_val_every_n_epoch=1, limit_train_batches=1.,
        limit_val_batches=1., gradient_clip_val=.5, deterministic=False,
        benchmark=False, log_every_n_steps=1)
    try:
        trainer.fit(model, datamodule=dm, ckpt_path=None)
        if callback.cancelled is not None:
            callback.snapshot(trainer, model, 'cancelled')
            raise InterruptedError('Signal requested stop: ' + str(callback.cancelled))
        if spec['mode'] == 'pilot' and not callback.clock.expired():
            raise RuntimeError('Pilot stopped before its frozen time budget')
        if callback.clock.midpoint_due():
            callback.snapshot(trainer, model, 'mid15min')
            callback.clock.mid_saved = True
        callback.snapshot(trainer, model, 'final')
        verify_frozen(spec)
        if sha256(spec_path) != original_spec_sha:
            raise ValueError('Input spec mutated during training')
        result = {'state': 'training_complete_awaiting_prior_validation', 'global_step': trainer.global_step,
                  'epoch': trainer.current_epoch, 'elapsed_fit_seconds': callback.clock.elapsed(),
                  'reason': 'smoke_max_steps' if max_steps else 'budget_exhausted',
                  'converged': False, 'test_used': False, 'selected_checkpoint': None,
                  'training_aggregate': callback.train_totals.result(),
                  'aux_gradient_checks': callback.aux_gradient_checks, 'completed_utc': utc()}
        write_json(out / 'status.json', result)
        return 0
    except BaseException as exc:
        write_json(out / 'status.json', {'state': 'failed', 'error': repr(exc),
            'global_step': trainer.global_step, 'utc': utc(), 'automatic_retry': False})
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--check-only', action='store_true', help='Read-only data/model constructor check; no optimizer step')
    args = p.parse_args()
    args.spec = args.spec.resolve()
    spec = json.loads(args.spec.read_text())
    return run(spec, args.spec, args.max_steps, args.check_only)


if __name__ == '__main__':
    try:
        result = main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    else:
        sys.exit(result)
