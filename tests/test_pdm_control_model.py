import os,sys
from pathlib import Path
import pytest
import torch
from cdvae.pl_modules.pdm_control_model import PDMControlCDVAE


def test_prior_branch_really_removes_posterior_z_and_gradient():
    torch.manual_seed(17)
    posterior=torch.ones(128,4,requires_grad=True)
    z,mask=PDMControlCDVAE.mixed_latents(posterior,.5)
    assert mask.any() and (~mask).any()
    assert torch.equal(z[~mask],posterior[~mask])
    assert not torch.equal(z[mask],posterior[mask])
    z.sum().backward()
    assert torch.equal(posterior.grad[mask],torch.zeros_like(posterior.grad[mask]))
    assert torch.equal(posterior.grad[~mask],torch.ones_like(posterior.grad[~mask]))


def test_kl_applies_only_to_posterior_examples():
    mu=torch.tensor([[20.,20.],[1.,2.]],requires_grad=True)
    logvar=torch.zeros_like(mu)
    loss=PDMControlCDVAE.posterior_kl(mu,logvar,torch.tensor([True,False]))
    assert loss.item()==pytest.approx(2.5)
    loss.backward()
    assert mu.grad[0].abs().sum()==0
    assert torch.equal(mu.grad[1],mu.detach()[1])


def test_all_prior_kl_is_differentiable_zero():
    mu=torch.ones(2,3,requires_grad=True)
    loss=PDMControlCDVAE.posterior_kl(mu,torch.zeros_like(mu),torch.ones(2,dtype=torch.bool))
    loss.backward()
    assert loss.item()==0 and torch.equal(mu.grad,torch.zeros_like(mu))


def test_coordinate_loss_uses_explicit_decoder_cell_and_preserves_old_signature():
    from types import SimpleNamespace
    from cdvae.pl_modules.model import CDVAE
    from cdvae.common.data_utils import frac_to_cart_coords, cart_to_frac_coords
    batch=SimpleNamespace(frac_coords=torch.tensor([[.1,.2,.3],[.4,.5,.6]]),
        lengths=torch.full((1,3),10.),angles=torch.full((1,3),90.),
        num_atoms=torch.tensor([2]),batch=torch.tensor([0,0]))
    lengths=torch.full((1,3),11.);angles=batch.angles
    noise=torch.tensor([[.2,0,0],[.2,0,0]])
    noisy=cart_to_frac_coords(frac_to_cart_coords(batch.frac_coords,lengths,angles,batch.num_atoms)+noise,lengths,angles,batch.num_atoms)
    obj=SimpleNamespace(device=torch.device('cpu'))
    sigmas=torch.ones(2)
    wrong_frame=CDVAE.coord_loss(obj,-noise,noisy,sigmas,batch)
    correct_frame=CDVAE.coord_loss(obj,-noise,noisy,sigmas,batch,lengths=lengths,angles=angles)
    assert wrong_frame>1e-5 and correct_frame<1e-9
    with pytest.raises(ValueError,match='both'):
        CDVAE.coord_loss(obj,-noise,noisy,sigmas,batch,lengths=lengths)
    same_cell_noise=cart_to_frac_coords(frac_to_cart_coords(batch.frac_coords,batch.lengths,angles,batch.num_atoms)+noise,batch.lengths,angles,batch.num_atoms)
    assert CDVAE.coord_loss(obj,-noise,same_cell_noise,sigmas,batch)<1e-9


def test_pdm_control_config_composes_and_requires_explicit_contract():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    repo=Path(__file__).resolve().parents[1]
    with initialize_config_dir(str(repo/'cond-cdvae/conf'),version_base='1.1'):
        cfg=compose(config_name='default',overrides=['model=pdm_control','data=apollox/mp_apollox','data.pdm_dim=21'])
    assert cfg.model._target_=='cdvae.pl_modules.pdm_control_model.PDMControlCDVAE'
    assert cfg.model.latent_dim==256 and cfg.model.encoder.num_blocks==4
    assert cfg.model.teacher_forcing_lattice is False and cfg.model.predict_property is False
    assert OmegaConf.is_missing(cfg.model.pdm_control,'schema')
    assert OmegaConf.is_missing(cfg.model.pdm_control,'external_mean')
    assert OmegaConf.is_missing(cfg.model.pdm_control,'external_std')
    assert cfg.model.pdm_control.prior_probability==.5
