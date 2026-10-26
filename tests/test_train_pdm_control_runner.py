import copy
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE.parent / 'scripts' / 'train_pdm_control.py'
spec = importlib.util.spec_from_file_location('control_train', RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def specimen():
    return {'mode': 'pilot', 'repo': '/repo', 'data': '/data', 'out': '/pilot',
        'warmstart_checkpoint': '/weights', 'warmstart_checkpoint_sha256': 'a'*64,
        'warmstart_hparams': '/hparams', 'model_overlay': '/overlay',
        'model_target': 'cdvae.pl_modules.pdm_control_model.PDMControlCDVAE',
        'frozen_files': {'/weights': 'a'*64}, 'cost_pdm': 10, 'seed': 20261026,
        'lr': 1e-4, 'batch_size': 16, 'num_workers': 0, 'budget_seconds': 1800,
        'midpoint_seconds': 900, 'expected_counts': {'train': 10000, 'val': 1000}}


class RunnerTests(unittest.TestCase):
    def test_formal_cannot_be_maxsteps_or_test_or_resume(self):
        for patch in ({'test_path': '/oldtest'}, {'resume': True}, {'expected_counts': {'train': 2, 'val': 1}}, {'budget_seconds': 1200}):
            with self.assertRaises(ValueError): runner.validate_spec(dict(specimen(), **patch))
        with self.assertRaises(ValueError): runner.validate_spec(specimen(), 50)
        self.assertEqual(runner.validate_spec(specimen())['cost_pdm'], 10)

    def test_only_predeclared_grid(self):
        for cost in (1, 10, 100): runner.validate_spec(dict(specimen(), cost_pdm=cost))
        for cost in (0, 2, float('nan')):
            with self.assertRaises(ValueError): runner.validate_spec(dict(specimen(), cost_pdm=cost))

    def test_smoke_requires_separate_explicit_output(self):
        s = dict(specimen(), mode='smoke')
        with self.assertRaises(ValueError): runner.validate_spec(s, 50)
        s['out'] = '/separate-smoke'
        runner.validate_spec(s, 50)
        with self.assertRaises(ValueError): runner.validate_spec(s)

    def test_aux_aggregation_does_not_count_ineligible_graphs(self):
        agg = runner.WeightedProxy()
        def values(aux, eligible):
            return dict(loss=4., base_loss=1., lattice_loss=1., coord_loss=1., kld_loss=1.,
                pdm_aux_loss=aux, pdm_prior_aux_loss=aux, pdm_posterior_aux_loss=0.,
                pdm_eligible_graphs=eligible, pdm_prior_eligible_graphs=eligible,
                pdm_posterior_eligible_graphs=0)
        agg.add(values(3., 2), 16, 8)
        agg.add(values(0., 0), 16, 8)
        r = agg.result()
        self.assertEqual(r['means']['pdm_aux_loss'], 3.)
        self.assertEqual(r['denominators']['pdm_aux_loss'], 2)
        self.assertIsNone(r['means']['pdm_posterior_aux_loss'])
        self.assertEqual(r['graphs'], 32)

    def test_rng_restored_even_on_validation_failure(self):
        random.seed(4); np.random.seed(4); torch.manual_seed(4)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        with self.assertRaises(RuntimeError):
            with runner.isolated_rng(99):
                random.random(); np.random.rand(); torch.rand(5)
                raise RuntimeError('validation failed')
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))

    def test_weights_only_does_not_restore_optimizer_or_training_step(self):
        with tempfile.TemporaryDirectory() as d:
            old = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(old.parameters(), lr=.05)
            old(torch.ones(1, 2)).sum().backward(); optimizer.step()
            ckpt = Path(d) / 'model.ckpt'
            torch.save({'state_dict': old.state_dict(), 'optimizer_states': [optimizer.state_dict()],
                        'global_step': 8764, 'epoch': 27}, ckpt)
            new = torch.nn.Linear(2, 1)
            fresh = torch.optim.Adam(new.parameters(), lr=1e-4)
            result = runner.load_weight_state(new, ckpt)
            self.assertEqual(fresh.state, {})
            self.assertEqual(fresh.param_groups[0]['lr'], 1e-4)
            self.assertTrue(result['optimizer_states_not_loaded'])
            for a, b in zip(old.parameters(), new.parameters()): self.assertTrue(torch.equal(a, b))
            with self.assertRaises(RuntimeError): runner.load_weight_state(torch.nn.Linear(3, 1), ckpt)

    def test_deadlines_do_not_depend_on_validation_loss(self):
        now = [100.]
        clock = runner.FitClock(1800, 900, clock=lambda: now[0]); clock.start()
        now[0] = 999.; self.assertFalse(clock.midpoint_due())
        now[0] = 1000.; self.assertTrue(clock.midpoint_due()); self.assertFalse(clock.expired())
        clock.mid_saved = True
        now[0] = 1900.; self.assertFalse(clock.midpoint_due()); self.assertTrue(clock.expired())

    def test_cli_invalid_spec_is_nonzero_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as d:
            s = specimen(); s['out'] = str(Path(d)/'never-created'); s['cost_pdm'] = 2
            p = Path(d)/'spec.json'; p.write_text(json.dumps(s))
            r = subprocess.run([sys.executable, str(RUNNER_PATH), '--spec', str(p)], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('predeclared', r.stderr)
            self.assertFalse(Path(s['out']).exists())

    def test_frozen_input_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'weights'; p.write_bytes(b'original')
            s = specimen(); s['frozen_files'] = {str(p): runner.sha256(p)}
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Frozen input mismatch'): runner.verify_frozen(s)


# Real Lightning callback integration uses tiny differentiable modules, not a
# copied callback implementation; no native graph network or GPU is run.
class LightningIntegration(unittest.TestCase):
    def test_callbacks_save_initial_audit_gradients_and_full_validation(self):
        import pytorch_lightning as pl
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        class Module(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.save_hyperparameters({'cost_lattice':1.,'cost_coord':1.,'beta':.01})
                self.decoder=torch.nn.Linear(1,1)
                self.multiemb=torch.nn.Module()
                self.multiemb.cond_keys=['element_values']
                self.multiemb.sub_emb_list=torch.nn.ModuleList([torch.nn.Linear(1,1)])
                self.training_step_outputs=[];self.validation_step_outputs=[]
            def forward(self,batch):
                x=self.decoder(self.multiemb.sub_emb_list[0](batch.x))
                aux=x.square().mean()+.01
                zero=aux*0
                return {'lattice_loss':aux,'coord_loss':aux,'kld_loss':zero,
                    'pdm_aux_loss':aux,'pdm_prior_aux_loss':aux,'pdm_posterior_aux_loss':aux,
                    'pdm_eligible_graphs':batch.num_graphs,'pdm_prior_eligible_graphs':1,
                    'pdm_posterior_eligible_graphs':1,'prior_mask':torch.tensor([True,False])}
            def training_step(self,b,i):
                x=self(b);loss=12*x['pdm_aux_loss'];self.training_step_outputs.append(loss);return loss
            def validation_step(self,b,i):
                x=self(b);loss=12*x['pdm_aux_loss'];self.validation_step_outputs.append(loss);return loss
            def configure_optimizers(self): return torch.optim.Adam(self.parameters(),lr=1e-4)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            for name in ('hparams.yaml','pdm_schema.json','scaler_stats.txt','prop_scalers.pt','lattice_scaler.pt'):
                (out/name).write_text('test fixture')
            samples=[Data(x=torch.ones(1,1),mp_id='a'),Data(x=torch.ones(1,1)*2,mp_id='b')]
            cb=runner.make_callback(specimen(),out,None,['a','b'],1)
            model=Module()
            trainer=pl.Trainer(accelerator='cpu',devices=1,max_epochs=1,max_steps=1,
                callbacks=[cb],logger=False,enable_checkpointing=False,enable_model_summary=False,
                enable_progress_bar=False,num_sanity_val_steps=0,limit_val_batches=1.)
            trainer.fit(model,train_dataloaders=DataLoader(samples,batch_size=2),val_dataloaders=DataLoader(samples,batch_size=2))
            manifest=json.loads((out/'checkpoint_manifest.json').read_text())
            self.assertEqual(manifest['snapshots'][0]['global_step'],0)
            record=json.loads((out/'training_steps.jsonl').read_text().splitlines()[0])
            self.assertGreater(record['aux_gradient_audit']['decoder_aux_gradient_norm'],0)
            self.assertGreater(record['aux_gradient_audit']['pdm_embedding_aux_gradient_norm'],0)
            val=json.loads((out/'validation_proxy.jsonl').read_text().splitlines()[0])
            self.assertEqual(val['graphs'],2)
            self.assertFalse(val['selection_performed'])
            self.assertEqual(trainer.global_step,1)

if __name__ == '__main__': unittest.main(verbosity=2)
