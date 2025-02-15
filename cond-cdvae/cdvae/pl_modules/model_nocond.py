import warnings
from typing import Any, Dict

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
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
from cdvae.pl_modules.decoder import GemNetTDecoder
from cdvae.pl_modules.embeddings import KHOT_EMBEDDINGS, MAX_ATOMIC_NUM
from cdvae.pl_modules.gnn import DimeNetPlusPlusWrap


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
        self.encoder: DimeNetPlusPlusWrap = hydra.utils.instantiate(
            self.hparams.encoder,
            num_targets=self.hparams.latent_dim,
        )
        self.fc_mu = nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim)
        self.fc_var = nn.Linear(
            self.hparams.latent_dim,
            self.hparams.latent_dim,
        )
        hydra.utils.log.info("Initializing encoder done")

        hydra.utils.log.info("Initializing decoder ...")
        self.decoder: GemNetTDecoder = hydra.utils.instantiate(
            self.hparams.decoder,
            _recursive_=False,
        )
        self.fc_num_atoms = build_mlp(
            self.hparams.latent_dim,
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            self.hparams.max_atoms + 1,
        )
        self.fc_composition = build_mlp(
            self.hparams.latent_dim,
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            MAX_ATOMIC_NUM,
        )
        self.fc_lattice = build_mlp(
            self.hparams.latent_dim,
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers,
            6,
            self.hparams.lattice_dropout,
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
        self.lattice_scaler = StandardScalerTorch(
            torch.tensor(0, dtype=torch.get_default_dtype()),
            torch.tensor(1, dtype=torch.get_default_dtype()),
        )
        self.prop_scalers = []

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, batch, cond_vec=None):
        """
        encode crystal structures to latents.
        """
        hidden = self.encoder(batch, cond_vec=None)

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
        # Train stage
        if gt_num_atoms is not None:
            lengths_and_angles, lengths, angles = self.predict_lattice(z, gt_num_atoms)
            num_atoms = self.predict_num_atoms(z)  # prob
            composition_per_atom = self.predict_composition(z, gt_num_atoms)
            if self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms = self.predict_num_atoms(z).argmax(dim=-1)  # index
            lengths_and_angles, lengths, angles = self.predict_lattice(z, num_atoms)
            composition_per_atom = self.predict_composition(z, num_atoms)
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom

    @torch.no_grad()
    def langevin_dynamics(self, z, ld_kwargs, gt_num_atoms=None, gt_atom_types=None):
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
        num_atoms, _, lengths, angles, composition_per_atom = self.decode_stats(
            z, gt_num_atoms
        )
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms

        # obtain atom types.
        composition_per_atom = F.softmax(composition_per_atom, dim=-1)
        if gt_atom_types is None:
            cur_atom_types = self.sample_composition(composition_per_atom, num_atoms)
        else:
            cur_atom_types = gt_atom_types

        # init coords.
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=self.device)

        # annealed langevin dynamics.
        for sigma in tqdm(
            self.sigmas,
            total=self.sigmas.size(0),
            disable=ld_kwargs.disable_bar,
        ):
            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            for step in range(ld_kwargs.n_step_each):
                noise_cart = torch.randn_like(cur_frac_coords) * torch.sqrt(
                    step_size * 2
                )
                pred_cart_coord_diff, pred_atom_types = self.decoder(
                    z,
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

    def sample(self, num_samples, ld_kwargs):
        """sample

        Args:
            num_samples (int): number of samples
            ld_kwargs (dict): langevin-dynamics args dict

        Returns:
            dict: arrays of all generated samples
        """
        z = torch.randn(num_samples, self.hparams.hidden_dim, device=self.device)
        samples = self.langevin_dynamics(z, ld_kwargs)
        return samples

    def forward(self, batch, teacher_forcing=False, training=True):
        # hacky way to resolve the NaN issue. Will need more careful debugging later.
        mu, log_var, z = self.encode(batch)
        assert torch.isfinite(z).all()

        # z (B, lattent_dim)

        # pred lattice from cond_z
        (
            pred_num_atoms,  # (B,)
            pred_lengths_and_angles,  # (B, 6)
            pred_lengths,  # (B, 3)
            pred_angles,  # (B, 3)
            pred_composition_per_atom,
        ) = self.decode_stats(
            z,
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

        # add noise to atom types and sample atom types
        type_noise_level = torch.randint(
            0, self.type_sigmas.size(0), (batch.num_atoms.size(0),), device=self.device
        )
        used_type_sigmas_per_atom = self.type_sigmas[
            type_noise_level
        ].repeat_interleave(batch.num_atoms, dim=0)
        # A ~ p(A)
        pred_composition_probs = F.softmax(pred_composition_per_atom.detach(), dim=-1)
        atom_type_probs = (
            F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM)
            + pred_composition_probs * used_type_sigmas_per_atom[:, None]
        )
        rand_atom_types = (
            torch.multinomial(atom_type_probs, num_samples=1).squeeze(1) + 1
        )

        pred_cart_coord_diff, pred_atom_types = self.decoder(
            z,
            noisy_frac_coords,
            rand_atom_types,
            batch.num_atoms,
            pred_lengths,
            pred_angles,
        )

        # compute loss.
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch
        )
        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
        composition_loss = self.composition_loss(
            pred_composition_per_atom, batch.atom_types, batch
        )
        type_loss = self.type_loss(
            pred_atom_types, batch.atom_types, used_type_sigmas_per_atom, batch
        )

        kld_loss = self.kld_loss(mu, log_var)

        return {
            'lattice_loss': lattice_loss,
            'coord_loss': coord_loss,
            'kld_loss': kld_loss,
            #
            "num_atom_loss": num_atom_loss,
            "composition_loss": composition_loss,
            "type_loss": type_loss,
            "pred_num_atoms": pred_num_atoms,
            "pred_atom_types": pred_atom_types,
            "pred_composition_per_atom": pred_composition_per_atom,
            #
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': noisy_frac_coords,
            'z': z,
        }

    def generate_rand_init(
        self, pred_composition_per_atom, pred_lengths, pred_angles, num_atoms, batch
    ):
        rand_frac_coords = torch.rand(num_atoms.sum(), 3, device=num_atoms.device)
        pred_composition_per_atom = F.softmax(pred_composition_per_atom, dim=-1)
        rand_atom_types = self.sample_composition(pred_composition_per_atom, num_atoms)
        return rand_frac_coords, rand_atom_types

    def sample_composition(self, composition_prob, num_atoms):
        """
        Samples composition such that it exactly satisfies composition_prob
        """
        batch = torch.arange(len(num_atoms), device=num_atoms.device).repeat_interleave(
            num_atoms
        )
        assert composition_prob.size(0) == num_atoms.sum() == batch.size(0)
        composition_prob = scatter(composition_prob, index=batch, dim=0, reduce='mean')

        all_sampled_comp = []

        for comp_prob, num_atom in zip(list(composition_prob), list(num_atoms)):
            comp_num = torch.round(comp_prob * num_atom)
            atom_type = torch.nonzero(comp_num, as_tuple=True)[0] + 1
            atom_num = comp_num[atom_type - 1].long()

            sampled_comp = atom_type.repeat_interleave(atom_num, dim=0)

            # if the rounded composition gives less atoms, sample the rest
            if sampled_comp.size(0) < num_atom:
                left_atom_num = num_atom - sampled_comp.size(0)

                left_comp_prob = comp_prob - comp_num.float() / num_atom

                left_comp_prob[left_comp_prob < 0.0] = 0.0
                left_comp = torch.multinomial(
                    left_comp_prob, num_samples=left_atom_num, replacement=True
                )
                # convert to atomic number
                left_comp = left_comp + 1
                sampled_comp = torch.cat([sampled_comp, left_comp], dim=0)

            sampled_comp = sampled_comp[torch.randperm(sampled_comp.size(0))]
            sampled_comp = sampled_comp[:num_atom]
            all_sampled_comp.append(sampled_comp)

        all_sampled_comp = torch.cat(all_sampled_comp, dim=0)
        assert all_sampled_comp.size(0) == num_atoms.sum()
        return all_sampled_comp

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

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)

    def predict_composition(self, z, num_atoms):
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom

    def num_atom_loss(self, pred_num_atoms, batch):
        return F.cross_entropy(pred_num_atoms, batch.num_atoms)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(
            pred_composition_per_atom, target_atom_types, reduction='none'
        )
        return scatter(loss, batch.batch, reduce='mean').mean()

    def type_loss(
        self, pred_atom_types, target_atom_types, used_type_sigmas_per_atom, batch
    ):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_atom_types, target_atom_types, reduction='none')
        # rescale loss according to noise
        loss = loss / used_type_sigmas_per_atom
        return scatter(loss, batch.batch, reduce='mean').mean()

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
        self.log_dict(prog_dict, on_epoch=True, batch_size=B, prog_bar=True)
        self.log_dict(log_dict, on_epoch=True, batch_size=B, prog_bar=False)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        B = batch.num_graphs
        prog_key = ["train_loss", "val_loss", "test_loss"]
        prog_dict = {key: log_dict.pop(key) for key in prog_key if key in log_dict}
        self.log_dict(prog_dict, on_epoch=True, batch_size=B, prog_bar=True)
        self.log_dict(log_dict, on_epoch=True, batch_size=B, prog_bar=False)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        B = batch.num_graphs
        prog_key = ["train_loss", "val_loss", "test_loss"]
        prog_dict = {key: log_dict.pop(key) for key in prog_key if key in log_dict}
        self.log_dict(prog_dict, on_epoch=True, batch_size=B, prog_bar=True)
        self.log_dict(log_dict, on_epoch=True, batch_size=B, prog_bar=False)
        return loss

    def compute_stats(self, batch, outputs, prefix):
        num_atom_loss = outputs['num_atom_loss']
        type_loss = outputs['type_loss']
        composition_loss = outputs['composition_loss']
        #
        lattice_loss = outputs['lattice_loss']
        coord_loss = outputs['coord_loss']
        kld_loss = outputs['kld_loss']

        loss = (
            +self.hparams.cost_lattice * lattice_loss
            + self.hparams.cost_coord * coord_loss
            + self.hparams.beta * kld_loss
            #
            + self.hparams.cost_natom * num_atom_loss
            + self.hparams.cost_type * type_loss
            + self.hparams.cost_composition * composition_loss
        )
        assert torch.isfinite(lattice_loss)
        assert torch.isfinite(coord_loss)
        assert torch.isfinite(kld_loss)

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_kld_loss': kld_loss,
            #
            f'{prefix}_natom_loss': num_atom_loss,
            f'{prefix}_type_loss': type_loss,
            f'{prefix}_composition_loss': composition_loss,
        }

        if prefix != 'train':
            # validation/test loss only has coord and type
            loss = (
                self.hparams.cost_coord * coord_loss
                + self.hparams.cost_type * type_loss
            )

            # evaluate num_atom prediction.
            pred_num_atoms = outputs['pred_num_atoms'].argmax(dim=-1)
            num_atom_accuracy = (
                pred_num_atoms == batch.num_atoms
            ).sum() / batch.num_graphs

            # evaluate atom type prediction.
            pred_atom_types = outputs['pred_atom_types']
            target_atom_types = outputs['target_atom_types']
            type_accuracy = pred_atom_types.argmax(dim=-1) == (target_atom_types - 1)
            type_accuracy = scatter(
                type_accuracy.float(), batch.batch, dim=0, reduce='mean'
            ).mean()

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
                    #
                    f'{prefix}_natom_accuracy': num_atom_accuracy,
                    f'{prefix}_type_accuracy': type_accuracy,
                }
            )

        return log_dict, loss
