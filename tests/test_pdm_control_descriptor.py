import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from pymatgen.core import Lattice
from apollox_common.pdm import build_schema, calculate_sro
from cdvae.common.differentiable_pdm import PairPDMDescriptor, soft_pair_pdm, nearest_image_displacements, standardized_pdm, raw_pdm_from_model_condition, denoised_fractional_coordinates, denoised_pdm_auxiliary_loss


def setup_two(distance=4.9):
    schema = build_schema(['B','O'], ['pair'], 5.)
    frac = torch.tensor([[.1,.1,.1],[.1 + distance / 20.,.1,.1]], dtype=torch.float64)
    return schema, frac, torch.tensor([5,8]), torch.tensor([2]), torch.eye(3, dtype=torch.float64).unsqueeze(0)*20


def test_hard_matches_native_pymatgen_and_feature_order():
    schema = build_schema(['B','Co','Fe','Mo','Ni','O'], ['pair'], 5.)
    assert len(schema['features']) == 21
    rng = np.random.default_rng(17)
    frac = rng.random((30,3))
    symbols = (['B','Co','Fe','Mo','Ni','O']*5)
    atomic = torch.tensor([5,27,26,42,28,8]*5)
    cell = np.array([[9.,0,0],[7.9,3.,0],[3.2,1.5,10.]])
    expected = calculate_sro(Lattice(cell).get_all_distances(frac,frac), symbols, 5., ['pair'])
    got = PairPDMDescriptor(schema)(torch.tensor(frac), atomic, torch.tensor([30]), torch.tensor(cell).unsqueeze(0), hard=True)
    assert got.tolist()[0] == [expected.get(f,0) for f in schema['features']]


def test_periodic_translation_permutation_and_rotation():
    schema, frac, types, num, cells = setup_two()
    base = soft_pair_pdm(frac,types,num,cells,schema)
    shifted = frac + torch.tensor([[3.,-2.,1.],[-4.,7.,-2.]])
    assert torch.allclose(base,soft_pair_pdm(shifted,types,num,cells,schema),atol=1e-12)
    assert torch.allclose(base,soft_pair_pdm(frac.flip(0),types.flip(0),num,cells,schema),atol=1e-12)
    rotation = torch.tensor([[0.,-1,0],[1.,0,0],[0,0,1.]],dtype=torch.float64)
    assert torch.allclose(base,soft_pair_pdm(frac,types,num,cells@rotation,schema),atol=1e-12)


def test_unordered_pair_count_once_no_self_or_cross_graph():
    schema,frac,types,num,cells=setup_two(4.)
    doubled=PairPDMDescriptor(schema)(torch.cat((frac,frac)),torch.cat((types,types)),torch.tensor([2,2]),cells.repeat(2,1,1),hard=True)
    assert doubled.tolist()==[[0.,1.,0.],[0.,1.,0.]]


def test_hard_strict_cutoff_and_soft_limit():
    schema,frac,types,num,cells=setup_two(5.)
    descriptor=PairPDMDescriptor(schema)
    frac = torch.tensor([[.125, .125, .125], [.375, .125, .125]], dtype=torch.float64)
    assert descriptor(frac,types,num,cells,hard=True).sum()==0
    assert descriptor(frac,types,num,cells,temperature=.1).sum().item()==pytest.approx(.5)
    for distance in (4.8,5.2):
        _,frac,types,num,cells=setup_two(distance)
        assert torch.allclose(descriptor(frac,types,num,cells,temperature=.001),descriptor(frac,types,num,cells,hard=True),atol=1e-10)


def test_nearest_image_adaptive_search_not_naive_rounding():
    cell=torch.tensor([[10.,0,0],[9.8,.6,0],[0,0,10.]],dtype=torch.float64)
    deltas=torch.tensor([[.49,.49,.2],[-.49,.1,.1],[.3,-.4,.2]],dtype=torch.float64)
    vectors=nearest_image_displacements(deltas,cell,max_image_range=32)
    expected=[Lattice(cell.numpy()).get_distance_and_image(np.zeros(3),d.numpy())[0] for d in deltas]
    assert np.allclose(torch.linalg.vector_norm(vectors,dim=-1).numpy(),expected,atol=1e-10)
    assert torch.linalg.vector_norm(vectors[0]) < torch.linalg.vector_norm(deltas[0]@cell)
    with pytest.raises(ValueError,match='image search'):
        nearest_image_displacements(deltas,cell,max_image_range=1)


def test_coordinate_and_cell_gradients_finite_and_correct():
    schema,frac,types,num,cells=setup_two()
    frac.requires_grad_();cells.requires_grad_()
    fn=lambda f,c: soft_pair_pdm(f,types,num,c,schema,temperature=.2)
    assert torch.autograd.gradcheck(fn,(frac,cells),eps=1e-6,atol=1e-5)
    value=fn(frac,cells).sum();value.backward()
    assert torch.isfinite(frac.grad).all() and frac.grad.abs().sum()>0
    assert torch.isfinite(cells.grad).all() and cells.grad.abs().sum()>0


def test_collision_gradients_finite():
    schema,frac,types,num,cells=setup_two(0.)
    frac.requires_grad_()
    soft_pair_pdm(frac,types,num,cells,schema).sum().backward()
    assert torch.isfinite(frac.grad).all() # It is zero at a collision: PDM is not a collision penalty.


def test_scaler_roundtrip_uses_both_transforms():
    raw=torch.tensor([[2.,7.,4.]],dtype=torch.float64)
    mean=torch.tensor([1.,3.,5.]);std=torch.tensor([2.,4.,6.])
    model_mean=torch.tensor([.2,-.1,.3]);model_std=torch.tensor([1.1,.9,1.2])
    external=standardized_pdm(raw,mean,std)
    internal=(external-model_mean)/model_std
    assert torch.allclose(raw_pdm_from_model_condition(internal,model_mean,model_std,mean,std),raw)


def test_x0_uses_sigma_not_sigma_squared():
    schema,clean,types,num,cells=setup_two()
    sigma=torch.tensor([.2,.2],dtype=torch.float64)
    displacement=torch.tensor([[.03,-.02,0.],[-.01,.02,.01]],dtype=torch.float64)
    noisy=clean+displacement@torch.linalg.inv(cells[0])
    x0=denoised_fractional_coordinates(noisy,-displacement/sigma[:,None],sigma,num,cells)
    assert torch.allclose(x0,clean,atol=1e-12)


def test_aux_loss_reaches_decoder_and_has_clear_sigma_mask():
    schema,frac,types,num,cells=setup_two(5.1)
    pred=torch.zeros_like(frac,requires_grad=True)
    target=torch.tensor([[0.,1.,0.]],dtype=torch.float64)
    result=denoised_pdm_auxiliary_loss(PairPDMDescriptor(schema),frac,pred,torch.tensor([.2,.2]),types,num,cells,target,[0,0,0],[1,2,1],temperature=.2,max_sigma=.5)
    result['loss'].backward()
    assert result['eligible_graphs']==1
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum()>0
    # Gradient descent must pull the BO pair closer when the desired count is one.
    assert pred.grad[1,0]>0 and pred.grad[0,0]<0
    masked=denoised_pdm_auxiliary_loss(PairPDMDescriptor(schema),frac,pred,torch.tensor([2.,2.]),types,num,cells,target,[0,0,0],[1,2,1],max_sigma=.5)
    assert masked['eligible_graphs']==0 and masked['loss'].item()==0
    masked['loss'].backward()


def test_validation_rejects_malformed_contracts():
    schema,frac,types,num,cells=setup_two()
    with pytest.raises(ValueError,match='absent'):
        soft_pair_pdm(frac,torch.tensor([1,8]),num,cells,schema)
    with pytest.raises(ValueError,match='positive determinant'):
        soft_pair_pdm(frac,types,num,cells*0,schema)
    with pytest.raises(ValueError,match='positive'):
        standardized_pdm(torch.zeros(1,3),[0,0,0],[1,0,1])
