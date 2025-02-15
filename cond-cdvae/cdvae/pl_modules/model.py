import warnings
from typing import Any, Dict

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from cdvae.common.data_utils import (
    EPSILON,
    StandardScalerTorch,
    cart_to_frac_coords,
    frac_to_cart_coords,
    lengths_angles_to_volume,
    mard,
    min_distance_sqr_pbc,
)
from cdvae.common.utils import PROJECT_ROOT
from cdvae.pl_modules.basic_blocks import build_mlp
from cdvae.pl_modules.conditioning import MultiEmbedding, ZGivenC
from cdvae.pl_modules.decoder import GemNetTDecoder
from cdvae.pl_modules.embeddings import KHOT_EMBEDDINGS, MAX_ATOMIC_NUM
from cdvae.pl_modules.gnn import DimeNetPlusPlusWrap
from cdvae.pl_modules.recall_head import PropRecall


def detact_overflow(x: torch.Tensor, threshold, batch, label: str):
    if x.dim() == 1:
        overflow = x > threshold
    elif x.dim() == 2:
        overflow = torch.any(x > threshold, dim=1)
    else:
        warnings.warn(f"{label} dimension not 1 or 2, skip")
        return
    idx = torch.nonzero(overflow, as_tuple=True)[0]  # overflow index
    if idx.size(0) > 0:
        warnings.warn(f"{label} exceed {threshold}: {[batch.mp_id[i] for i in idx]}")


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer,
            params=self.parameters(),
            _convert_="partial",
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 5,
                "monitor": "val_loss",
            },
        }


class CDVAE(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        hydra.utils.log.info("Initializing encoder ...")
        # =================== Embedding multi-conditions ==============
        self.multiemb: MultiEmbedding = hydra.utils.instantiate(
            self.hparams.conditions,
            _recursive_=False,
        )
        self.agg_c: ZGivenC = hydra.utils.instantiate(self.hparams.agg_c)
        self.zgivenc: ZGivenC = hydra.utils.instantiate(self.hparams.zgivenc)
        # ======================= Encoder =============================
        self.encoder: DimeNetPlusPlusWrap = hydra.utils.instantiate(
            self.hparams.encoder,
            num_targets=self.hparams.latent_dim,
        )
        # ==================== mu & std ==========================
        self.fc_mu = nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim)
        hydra.utils.log.info("Initializing encoder done")
        # ======================= Decoder =======================
        hydra.utils.log.info("Initializing decoder ...")
        self.decoder: GemNetTDecoder = hydra.utils.instantiate(
            self.hparams.decoder, _recursive_=False
        )
        # dynamic z dim
        # ============ Lattice and other Recall head ===============
        self.fc_lattice = build_mlp(**self.hparams.lattice_recall)
        if self.hparams.predict_property:
            self.prop_recall_model_before_cond: PropRecall = hydra.utils.instantiate(
                self.hparams.prop_recall, _recursive_=False
            )
            self.prop_recall_model_after_cond: PropRecall = hydra.utils.instantiate(
                self.hparams.prop_recall, _recursive_=False
            )
        # ===== split lattice lengths and angles =====
        # self.fc_lengths = build_mlp(
        #     self.hparams.latent_dim,
        #     self.hparams.hidden_dim,
        #     self.hparams.fc_num_layers,
        #     3,
        # )
        # self.fc_angles = build_mlp(
        #     self.hparams.latent_dim,
        #     self.hparams.hidden_dim,
        #     self.hparams.fc_num_layers,
        #     3,
        # )
        # ============================================
        hydra.utils.log.info("Initializing decoder done")

        sigmas = torch.Tensor(
            np.exp(
                np.linspace(
                    np.log(self.hparams.sigma_begin),
                    np.log(self.hparams.sigma_end),
                    self.hparams.num_noise_level,
                )
            ),
        )

        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        type_sigmas = torch.Tensor(
            np.exp(
                np.linspace(
                    np.log(self.hparams.type_sigma_begin),
                    np.log(self.hparams.type_sigma_end),
                    self.hparams.num_noise_level,
                )
            ),
        )

        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)

        # obtain from datamodule.
        self.lattice_scaler = StandardScalerTorch(0, 1)
        self.prop_scalers = []

    def build_conditions(self, batch):
        conditions = {}
        for cond_key in self.hparams.conditions.cond_keys:
            if cond_key == 'composition':
                conditions[cond_key] = (batch.atom_types, batch.num_atoms)
            else:
                conditions[cond_key] = batch[cond_key]
        return conditions

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        # assert torch.isfinite(logvar).all()
        # assert torch.isfinite(std).all()
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, batch, cond_vec):
        """
        encode crystal structures to latents.
        """
        hidden = self.encoder(batch, cond_vec)

        # debug
        detact_overflow(hidden, 100, batch, "hidden")

        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)

        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    def decode_stats(
        self,
        z,
        gt_num_atoms,
        gt_lengths=None,
        gt_angles=None,
        teacher_forcing=False,
    ):
        """
        decode key stats from latent embeddings.
        batch is input during training for teach-forcing.
        """
        lengths_and_angles, lengths, angles = self.predict_lattice(z, gt_num_atoms)
        # Train stage
        if self.hparams.teacher_forcing_lattice and teacher_forcing:
            lengths = gt_lengths
            angles = gt_angles
        # Generate stage, i.e. langevin dynamics
        else:
            pass
        return (
            lengths_and_angles,
            lengths,
            angles,
        )

    @torch.no_grad()
    def langevin_dynamics(self, cond_z, ld_kwargs, gt_num_atoms, gt_atom_types):
        """
        decode crystral structure from latent embeddings.
        ld_kwargs: args for doing annealed langevin dynamics sampling:
            n_step_each:  number of steps for each sigma level.
            step_lr:      step size param.
            min_sigma:    minimum sigma to use in annealed langevin dynamics.
            save_traj:    if <True>, save the entire LD trajectory.
            disable_bar:  disable the progress bar of langevin dynamics.
        gt_num_atoms: if not <None>, use the ground truth number of atoms.
        gt_atom_types: if not <None>, use the ground truth atom types.
        """
        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []

        # obtain key stats.
        _, lengths, angles = self.decode_stats(cond_z, gt_num_atoms)
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms

        # obtain atom types.
        # composition_per_atom = F.softmax(composition_per_atom, dim=-1)
        # if gt_atom_types is None:
        #     cur_atom_types = self.sample_composition(
        #         composition_per_atom, num_atoms
        #     )
        # else:
        #     cur_atom_types = gt_atom_types
        cur_atom_types = gt_atom_types

        # init coords.
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=cond_z.device)

        # annealed langevin dynamics.
        for sigma in tqdm(
            self.sigmas,
            total=self.sigmas.size(0),
            disable=ld_kwargs.disable_bar,
            mininterval=10,
            ncols=79,
        ):
            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            for step in range(ld_kwargs.n_step_each):
                noise_cart = torch.randn_like(cur_frac_coords) * torch.sqrt(
                    step_size * 2
                )
                pred_cart_coord_diff, pred_atom_types = self.decoder(
                    cond_z,
                    cur_frac_coords,
                    cur_atom_types,
                    num_atoms,
                    lengths,
                    angles,
                )
                cur_cart_coords = frac_to_cart_coords(
                    cur_frac_coords, lengths, angles, num_atoms
                )
                pred_cart_coord_diff = pred_cart_coord_diff / sigma
                cur_cart_coords = (
                    cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart
                )
                cur_frac_coords = cart_to_frac_coords(
                    cur_cart_coords, lengths, angles, num_atoms
                )

                if gt_atom_types is None:  # never used
                    cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 1

                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    all_pred_cart_coord_diff.append(step_size * pred_cart_coord_diff)
                    all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)

        output_dict = {
            'num_atoms': num_atoms,
            'lengths': lengths,
            'angles': angles,
            'frac_coords': cur_frac_coords,
            'atom_types': cur_atom_types,
            'is_traj': False,
        }

        if ld_kwargs.save_traj:
            output_dict.update(
                dict(
                    all_frac_coords=torch.stack(all_frac_coords, dim=0),
                    all_atom_types=torch.stack(all_atom_types, dim=0),
                    all_pred_cart_coord_diff=torch.stack(
                        all_pred_cart_coord_diff, dim=0
                    ),
                    all_noise_cart=torch.stack(all_noise_cart, dim=0),
                    is_traj=True,
                )
            )

        return output_dict

    def sample(self, conditions, ld_kwargs):
        pass
        # """sample

        # Args:
        #     conditions (dict):
        #         {'composition': (atom_types, num_atoms), cond_key: cond_val, ...}
        #     ld_kwargs (dict): langevin-dynamics args dict

        # Returns:
        #     dict: arrays of all generated samples
        # """
        # gt_atom_types, gt_num_atoms = conditions['composition']
        # num_samples = gt_num_atoms.shape[0]
        # z = torch.randn(num_samples, self.hparams.hidden_dim, device=self.device)
        # # cond z
        # cond_vec = self.multiemb(conditions)
        # cond_z = self.agg_cond(cond_vec, z)
        # samples = self.langevin_dynamics(cond_z, ld_kwargs, gt_num_atoms, gt_atom_types)
        # return samples

    def forward(self, batch, teacher_forcing=False, training=True):
        # 1. Adding c to data X
        conditions = self.build_conditions(batch)
        c_dict = self.multiemb(conditions)  # dict of each condition vector
        cond_vec = self.agg_c(z=None, c=c_dict)  # c
        mu, log_var, z = self.encode(batch, cond_vec)
        # 2. No adding c to data X
        # hacky way to resolve the NaN issue. Will need more careful debugging later.
        # mu, log_var, z = self.encode(batch, None)

        assert torch.isfinite(z).all()

        cond_z = self.zgivenc(z, c_dict)  # z (B, *)

        if self.hparams.predict_property:
            pred_property_before_cond = self.prop_recall_model_before_cond(batch)
            pred_property_after_cond = self.prop_recall_model_after_cond(batch)
            prop_loss_before_cond = self.property_loss(pred_property_before_cond, batch)
            prop_loss_after_cond = self.property_loss(pred_property_after_cond, batch)
        else:
            prop_loss_before_cond = 0.0
            prop_loss_after_cond = 0.0

        if (self.global_rank == 0) and (self.global_step % 20 == 1):
            log_cond_dict = {
                "cond/z_mean": wandb.Histogram(z.detach().cpu().mean(0)),
                "cond/z_std": wandb.Histogram(z.detach().cpu().std(0)),
                "cond/cond_z_mean": wandb.Histogram(cond_z.detach().cpu().mean(0)),
                "cond/cond_z_std": wandb.Histogram(cond_z.detach().cpu().std(0)),
            }
            wandb.log(
                log_cond_dict,
                step=self.global_step,
            )

        # pred lattice from cond_z
        # (B, 6)                 (B, 3)        (B, 3)
        pred_lengths_and_angles, pred_lengths, pred_angles = self.decode_stats(
            cond_z,
            batch.num_atoms,
            batch.lengths,
            batch.angles,
            teacher_forcing,
        )

        # debug
        detact_overflow(pred_lengths_and_angles, 10000, batch, "lattice")

        # sample noise levels. noise on each atom
        noise_level = torch.randint(
            0,
            self.sigmas.size(0),
            (batch.num_atoms.size(0),),
            device=self.device,
        )
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(
            batch.num_atoms, dim=0
        )

        # add noise to the cart coords
        cart_noises_per_atom = (
            torch.randn_like(batch.frac_coords) * used_sigmas_per_atom[:, None]
        )
        cart_coords = frac_to_cart_coords(
            batch.frac_coords, pred_lengths, pred_angles, batch.num_atoms
        )
        # N(X, σX^2 I) =    X    +    σX * N(0, 1)
        cart_coords = cart_coords + cart_noises_per_atom
        noisy_frac_coords = cart_to_frac_coords(  # ~X
            cart_coords, pred_lengths, pred_angles, batch.num_atoms
        )

        pred_cart_coord_diff, _ = self.decoder(
            cond_z,
            noisy_frac_coords,
            batch.atom_types,
            batch.num_atoms,
            pred_lengths,
            pred_angles,
        )

        # compute loss.
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch
        )

        kld_loss = self.kld_loss(mu, log_var)

        return {
            'lattice_loss': lattice_loss,
            'coord_loss': coord_loss,
            'kld_loss': kld_loss,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': noisy_frac_coords,
            'z': z,
            "prop_loss_before_cond": prop_loss_before_cond,
            "prop_loss_after_cond": prop_loss_after_cond,
        }

    def predict_lattice(self, z, num_atoms):
        self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        # ===== split lattice lengths and angles =====
        # lengths = self.fc_lengths(z)
        # angles = self.fc_angles(z)
        # pred_lengths_and_angles = torch.concat([lengths, angles], dim=1)
        # ============================================
        scaled_preds = self.lattice_scaler.inverse_transform(pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        if self.hparams.data.lattice_scale_method == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float() ** (1 / 3)
        # <pred_lengths_and_angles> is scaled.
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def lattice_loss(self, pred_lengths_and_angles, batch):
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        if self.hparams.data.lattice_scale_method == 'scale_length':
            target_lengths = batch.lengths / batch.num_atoms.view(-1, 1).float() ** (
                1 / 3
            )
        target_angles = batch.angles
        # target_angles = batch.angles / 180 * np.pi
        target_lengths_and_angles = torch.cat([target_lengths, target_angles], dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles
        )
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def property_loss(self, pred_property: dict, batch):
        return torch.sum(
            torch.Tensor(
                [
                    F.mse_loss(pred_val, batch[prop_key])
                    for prop_key, pred_val in pred_property.items()
                ]
            )
        )

    def coord_loss(
        self,
        pred_cart_coord_diff,
        noisy_frac_coords,
        used_sigmas_per_atom,
        batch,
    ):
        noisy_cart_coords = frac_to_cart_coords(
            noisy_frac_coords, batch.lengths, batch.angles, batch.num_atoms
        )
        target_cart_coords = frac_to_cart_coords(
            batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms
        )
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords,
            noisy_cart_coords,
            batch.lengths,
            batch.angles,
            batch.num_atoms,
            self.device,
            return_vector=True,
        )

        target_cart_coord_diff = (
            target_cart_coord_diff / used_sigmas_per_atom[:, None] ** 2
        )
        pred_cart_coord_diff = pred_cart_coord_diff / used_sigmas_per_atom[:, None]

        loss_per_atom = torch.sum(
            (target_cart_coord_diff - pred_cart_coord_diff) ** 2, dim=1
        )

        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
            dim=0,
        )
        return kld_loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        teacher_forcing = self.current_epoch <= self.hparams.teacher_forcing_max_epoch
        outputs = self(batch, teacher_forcing, training=True)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        B = batch.num_graphs
        prog_key = ["train_loss", "val_loss", "test_loss"]
        prog_dict = {key: log_dict.pop(key) for key in prog_key if key in log_dict}
        log_kwargs = {
            "on_epoch": True, "batch_size": B, "prog_bar": True, "sync_dist": True
        }
        self.log_dict(prog_dict, **log_kwargs)
        self.log_dict(log_dict, **log_kwargs)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        B = batch.num_graphs
        prog_key = ["train_loss", "val_loss", "test_loss"]
        prog_dict = {key: log_dict.pop(key) for key in prog_key if key in log_dict}
        log_kwargs = {
            "on_epoch": True, "batch_size": B, "prog_bar": True, "sync_dist": True
        }
        self.log_dict(prog_dict, **log_kwargs)
        self.log_dict(log_dict, **log_kwargs)
        self.validation_step_outputs.append(loss)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        B = batch.num_graphs
        prog_key = ["train_loss", "val_loss", "test_loss"]
        prog_dict = {key: log_dict.pop(key) for key in prog_key if key in log_dict}
        log_kwargs = {
            "on_epoch": True, "batch_size": B, "prog_bar": True, "sync_dist": True
        }
        self.log_dict(prog_dict, **log_kwargs)
        self.log_dict(log_dict, **log_kwargs)
        return loss

    def compute_stats(self, batch, outputs, prefix):
        lattice_loss = outputs['lattice_loss']
        coord_loss = outputs['coord_loss']
        kld_loss = outputs['kld_loss']
        prop_loss_before_cond = outputs["prop_loss_before_cond"]
        prop_loss_after_cond = outputs["prop_loss_after_cond"]

        loss = (
            +self.hparams.cost_lattice * lattice_loss
            + self.hparams.cost_coord * coord_loss
            + self.hparams.beta * kld_loss
            + self.hparams.cost_property * prop_loss_before_cond
            + self.hparams.cost_property * prop_loss_after_cond
        )
        assert torch.isfinite(lattice_loss)
        assert torch.isfinite(coord_loss)
        assert torch.isfinite(kld_loss)

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_kld_loss': kld_loss,
            f"{prefix}_prop_loss_before_cond": prop_loss_before_cond,
            f"{prefix}_prop_loss_after_cond": prop_loss_after_cond,
        }

        if prefix != 'train':
            # validation/test loss only has coord and type
            # loss = self.hparams.cost_coord * coord_loss
            # use all weighted loss

            # evalute lattice prediction.
            pred_lengths_and_angles = outputs['pred_lengths_and_angles']
            scaled_preds = self.lattice_scaler.inverse_transform(
                pred_lengths_and_angles
            )
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]

            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * batch.num_atoms.view(-1, 1).float() ** (
                    1 / 3
                )
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)

            log_dict.update(
                {
                    f'{prefix}_loss': loss,
                    f'{prefix}_lengths_mard': lengths_mard,
                    f'{prefix}_angles_mae': angles_mae,
                    f'{prefix}_volumes_mard': volumes_mard,
                }
            )

        return log_dict, loss

    def on_train_epoch_end(self):
        # do something with all training_step outputs, for example:
        trn_loss_epoch_mean = torch.stack(self.training_step_outputs).mean().item()
        val_loss_epoch_mean = (
            torch.stack(self.validation_step_outputs).mean().item()
            if len(self.validation_step_outputs) != 0
            else None
        )
        print(
            f"{self.current_epoch=} {self.global_step=} "
            f"{trn_loss_epoch_mean=} {val_loss_epoch_mean=}"
        )
        # free up the memory
        self.training_step_outputs.clear()
        self.validation_step_outputs.clear()


class CrystGNN_Supervise(BaseModule):
    """
    GNN model for fitting the supervised objectives for crystals.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder: DimeNetPlusPlusWrap = hydra.utils.instantiate(
            self.hparams.encoder,
            num_targets=sum(self.hparams.num_targets),
            supervise=True,
        )

    def forward(self, batch):
        preds = self.encoder(batch)
        preds = torch.split(preds, self.hparms.num_targets)
        return preds

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def compute_stats(self, batch, preds, prefix):
        pass
