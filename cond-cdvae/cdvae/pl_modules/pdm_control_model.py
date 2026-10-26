"""Explicit PDM-control model version with structure-space supervision."""
import math
import torch
from omegaconf import OmegaConf
from cdvae.pl_modules.model import CDVAE
from cdvae.common.data_utils import frac_to_cart_coords, cart_to_frac_coords, lattice_params_to_matrix_torch
from cdvae.common.differentiable_pdm import PairPDMDescriptor, standardized_pdm, raw_pdm_from_model_condition, denoised_pdm_auxiliary_loss


class PDMControlCDVAE(CDVAE):
    """Adds a structure-space objective and an explicit prior training branch.

    Version 1 uses a soft pair descriptor; actual conditional-generation quality
    must be established by independent hard-descriptor validation. No new
    trainable head is added, and old weight names are retained. A new checkpoint must
    identify this class and pdm_control.version=1. It must not be silently loaded
    as a legacy CDVAE training model, whose loss/latent semantics are different.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.hparams.get('pdm_control')
        if OmegaConf.is_config(config):
            config = OmegaConf.to_container(config, resolve=True)
        if not isinstance(config, dict) or config.get('version') != 1:
            raise ValueError('PDMControlCDVAE requires explicit pdm_control.version=1')
        if self.hparams.teacher_forcing_lattice:
            raise ValueError('PDM-control training requires teacher_forcing_lattice=false')
        if self.hparams.predict_property:
            raise ValueError('PDM-control training requires predict_property=false; no latent property head')
        if 'element_values' not in self.hparams.data.conditions or self.hparams.data.prop.count('element_values') != 1:
            raise ValueError('element_values must be an active condition and unique scaled property')
        self.pdm_control_config = config
        self.pdm_descriptor = PairPDMDescriptor(config['schema'], max_image_range=config.get('max_image_range',8), pair_chunk_size=config.get('pair_chunk_size',512))
        if len(self.pdm_descriptor.features) != self.hparams.data.pdm_dim:
            raise ValueError('PDM schema and model input dimension differ')
        standardized_pdm(torch.zeros(1, len(self.pdm_descriptor.features)),config['external_mean'],config['external_std'])
        for key in ('cost','temperature','max_sigma'):
            if not math.isfinite(float(config[key])) or config[key] <= 0:
                raise ValueError(f'pdm_control.{key} must be positive and finite')
        if not 0. < config['prior_probability'] < 1.:
            raise ValueError('prior_probability must lie strictly between zero and one')

    @staticmethod
    def mixed_latents(posterior_z, probability):
        prior_mask = torch.rand(len(posterior_z), device=posterior_z.device) < probability
        prior_z = torch.randn_like(posterior_z)
        return torch.where(prior_mask[:,None], prior_z, posterior_z), prior_mask

    @staticmethod
    def posterior_kl(mu, log_var, prior_mask):
        per_graph = -.5 * (1 + log_var - mu.square() - log_var.exp()).sum(-1)
        mask = ~prior_mask
        return per_graph[mask].mean() if bool(mask.any()) else mu.sum() * 0.

    def forward(self, batch, teacher_forcing=False, training=True):
        cfg = self.pdm_control_config
        conditions = self.build_conditions(batch)
        c_dict = self.multiemb(conditions)
        cond_vec = self.agg_c(z=None,c=c_dict)
        mu,log_var,posterior_z = self.encode(batch,cond_vec)
        # Validation uses the same predeclared mixture; its runner fixes RNG.
        # Generation remains the native prior-z entry point.
        z,prior_mask = self.mixed_latents(posterior_z,cfg['prior_probability'])
        if not bool(torch.isfinite(z).all()):
            raise ValueError('Non-finite latent')
        cond_z = self.zgivenc(z,c_dict)
        pred_la,lengths,angles = self.decode_stats(cond_z,batch.num_atoms,batch.lengths,batch.angles,False)
        levels = torch.randint(0,len(self.sigmas),(len(batch.num_atoms),),device=z.device)
        graph_sigmas = self.sigmas[levels]
        sigmas = graph_sigmas.repeat_interleave(batch.num_atoms)
        cart = frac_to_cart_coords(batch.frac_coords,lengths,angles,batch.num_atoms)
        noisy_cart = cart + torch.randn_like(cart)*sigmas[:,None]
        noisy_frac = cart_to_frac_coords(noisy_cart,lengths,angles,batch.num_atoms)
        prediction,_ = self.decoder(cond_z,noisy_frac,batch.atom_types,batch.num_atoms,lengths,angles)
        cells = lattice_params_to_matrix_torch(lengths,angles)
        property_index = list(self.hparams.data.prop).index('element_values')
        scaler = self.prop_scalers[property_index]
        raw_target = raw_pdm_from_model_condition(batch.element_values,scaler.means,scaler.stds,cfg['external_mean'],cfg['external_std'])
        aux = denoised_pdm_auxiliary_loss(self.pdm_descriptor,noisy_frac,prediction,sigmas,batch.atom_types,batch.num_atoms,cells,raw_target,cfg['external_mean'],cfg['external_std'],temperature=cfg['temperature'],max_sigma=cfg['max_sigma'])
        def branch_mean(mask):
            selected = mask & aux['mask']
            return aux['per_graph_loss'][selected].mean() if bool(selected.any()) else prediction.sum()*0.
        zero = prediction.sum()*0.
        return {
            'lattice_loss': self.lattice_loss(pred_la,batch),
            'coord_loss': self.coord_loss(prediction,noisy_frac,sigmas,batch,lengths,angles),
            'kld_loss': self.posterior_kl(mu,log_var,prior_mask),
            'pred_lengths_and_angles': pred_la,'pred_lengths': lengths,'pred_angles': angles,
            'pred_cart_coord_diff': prediction,'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,'rand_frac_coords': noisy_frac,'z': z,
            'prop_loss_before_cond': zero,'prop_loss_after_cond': zero,
            'pdm_aux_loss': aux['loss'],'pdm_prior_aux_loss': branch_mean(prior_mask),
            'pdm_posterior_aux_loss': branch_mean(~prior_mask),
            'pdm_eligible_graphs': aux['eligible_graphs'],
            'pdm_prior_eligible_graphs': int((aux['mask'] & prior_mask).sum()),
            'pdm_posterior_eligible_graphs': int((aux['mask'] & ~prior_mask).sum()),
            'prior_mask': prior_mask,'graph_sigmas': graph_sigmas,
        }

    def compute_stats(self,batch,outputs,prefix):
        logs,base_loss = super().compute_stats(batch,outputs,prefix)
        loss = base_loss + self.pdm_control_config['cost']*outputs['pdm_aux_loss']
        if not bool(torch.isfinite(loss)):
            raise ValueError('Non-finite PDM-control loss')
        logs[f'{prefix}_base_loss'] = base_loss
        logs[f'{prefix}_loss'] = loss
        for key in ('pdm_aux_loss','pdm_prior_aux_loss','pdm_posterior_aux_loss','pdm_eligible_graphs','pdm_prior_eligible_graphs','pdm_posterior_eligible_graphs'):
            logs[f'{prefix}_{key}'] = outputs[key]
        logs[f'{prefix}_prior_fraction'] = outputs['prior_mask'].float().mean()
        return logs,loss
