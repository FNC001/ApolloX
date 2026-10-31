"""Explicit local topology, interval gradients, periodicity and coverage audits."""
import json
import math
import random

from ase import Atoms
import numpy as np
import pytest
import torch

from cdvae.common.local_geometry import LocalGeometryError, local_geometry_penalty


def triad(angle=70., first=1.5, second=1.7, dtype=torch.float64):
    radians=math.radians(angle)
    coords=torch.tensor([[first,0.,0.],[0.,0.,0.],[second*math.cos(radians),second*math.sin(radians),0.]],dtype=dtype)/10+.3
    cells=torch.eye(3,dtype=dtype)[None]*10
    counts=torch.tensor([3])
    bonds=torch.tensor([[0,1],[1,2]])
    angles=torch.tensor([[0,1,2]])
    return coords,counts,cells,bonds,torch.tensor([[1.4,1.8],[1.4,1.8]],dtype=dtype),angles,torch.tensor([[80.,100.]],dtype=dtype)


def total(*args):
    result=local_geometry_penalty(*args)
    return result['bond_loss']+result['angle_loss']


def test_inside_reference_intervals_gives_zero_loss_and_finite_zero_gradient():
    args=list(triad(angle=90.));args[0].requires_grad_()
    result=local_geometry_penalty(*args)
    assert result['bond_loss']==0 and result['angle_loss']==0
    assert result['audit']['interval_constraints_satisfied'] is True
    assert result['audit']['angle_degrees']==pytest.approx([90.])
    total(*args).backward()
    assert torch.equal(args[0].grad,torch.zeros_like(args[0]))
    json.dumps(result['audit'],allow_nan=False)


def test_bond_stretch_gradient_moves_toward_the_reference_interval():
    args=list(triad(angle=90.,first=2.3));args[0].requires_grad_()
    result=local_geometry_penalty(*args)
    assert result['bond_loss']>0 and result['angle_loss']==0
    result['bond_loss'].backward()
    assert args[0].grad[0,0]>0 and args[0].grad[1,0]<0
    corrected=args[0].detach()-.0005*args[0].grad
    after=local_geometry_penalty(corrected,*args[1:])
    assert after['bond_loss']<result['bond_loss']
    assert result['audit']['bond_violation_indices']==[0]


@pytest.mark.parametrize('initial_angle',[55.,135.])
def test_wrong_angle_gradient_moves_angle_toward_interval(initial_angle):
    args=list(triad(angle=initial_angle));args[0].requires_grad_()
    result=local_geometry_penalty(*args)
    assert result['angle_loss']>0
    result['angle_loss'].backward()
    corrected=args[0].detach()-.0005*args[0].grad
    after=local_geometry_penalty(corrected,*args[1:])
    assert after['angle_loss']<result['angle_loss']
    if initial_angle<80:
        assert after['audit']['angle_degrees'][0]>result['audit']['angle_degrees'][0]
    else:
        assert after['audit']['angle_degrees'][0]<result['audit']['angle_degrees'][0]
    assert result['audit']['angle_violation_indices']==[0]


def test_gradcheck_away_from_mic_and_interval_boundaries():
    args=list(triad(angle=60.,first=2.2,second=1.8));args[0].requires_grad_()
    assert torch.autograd.gradcheck(lambda x:total(x,*args[1:]),(args[0],),eps=1e-6,atol=1e-5,rtol=1e-4)


def test_periodic_translations_of_individual_atoms_preserve_losses_and_gradients():
    args=list(triad(angle=60.,first=2.2));args[0].requires_grad_()
    a=local_geometry_penalty(*args);ga=torch.autograd.grad(a['bond_loss']+a['angle_loss'],args[0])[0]
    shifted=(args[0].detach()+torch.tensor([[2.,-1,0],[-2.,1.,3.],[1.,1.,-2.]])).requires_grad_()
    b=local_geometry_penalty(shifted,*args[1:]);gb=torch.autograd.grad(b['bond_loss']+b['angle_loss'],shifted)[0]
    torch.testing.assert_close(a['bond_loss'],b['bond_loss'],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(a['angle_loss'],b['angle_loss'],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(ga,gb,rtol=1e-10,atol=1e-10)


def test_cartesian_rotation_of_cell_preserves_local_geometry():
    args=list(triad(angle=60.,first=2.2));a=local_geometry_penalty(*args)
    t=.43;rotation=torch.tensor([[math.cos(t),-math.sin(t),0],[math.sin(t),math.cos(t),0],[0,0,1.]],dtype=torch.float64)
    args[2]=args[2]@rotation
    b=local_geometry_penalty(*args)
    torch.testing.assert_close(a['bond_loss'],b['bond_loss'],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(a['angle_loss'],b['angle_loss'],rtol=1e-12,atol=1e-12)
    assert a['audit']['angle_degrees']==pytest.approx(b['audit']['angle_degrees'],abs=1e-12)


def test_skew_cell_mic_matches_independent_ase_bonds_and_center_vectors():
    cell=torch.tensor([[4.,0,0],[3.9,.8,0],[0,0,6.]],dtype=torch.float64)
    coords=torch.tensor([[.49,.49,.03],[0.,0.,0.],[.1,.1,.2]],dtype=torch.float64)
    result=local_geometry_penalty(coords,torch.tensor([3]),cell[None],[[0,1],[1,2]],[[.2,3.],[.2,3.]],[[0,1,2]],[[10.,160.]])
    atoms=Atoms('CCC',scaled_positions=coords.numpy(),cell=cell.numpy(),pbc=True)
    assert result['audit']['bond_distances_A']==pytest.approx([atoms.get_distance(0,1,mic=True),atoms.get_distance(1,2,mic=True)],abs=1e-12)
    assert result['audit']['angle_degrees'][0]==pytest.approx(atoms.get_angle(0,1,2,mic=True),abs=1e-10)


def test_graph_balancing_uses_each_actual_cell_and_not_constraint_count():
    # Graph0: one bond length 2, midpoint1 => loss1. Graph1: two length3 bonds => loss4.
    coords=torch.tensor([[.1,0,0],[.3,0,0],[.1,.1,0],[.25,.1,0],[.25,.25,0]],dtype=torch.float64)
    cells=torch.stack([torch.eye(3)*10,torch.eye(3)*20]).double()
    result=local_geometry_penalty(coords,torch.tensor([2,3]),cells,[[0,1],[2,3],[3,4]],[[1.,1.]]*3,[],[])
    torch.testing.assert_close(result['per_graph_bond_loss'],torch.tensor([1.,4.],dtype=torch.float64))
    assert float(result['bond_loss'])==pytest.approx(2.5)
    assert result['audit']['coverage']['bond_constraint_counts']==[1,2]
    assert result['audit']['coverage']['angle_constraint_counts']==[0,0]


def test_missing_graph_constraints_are_reported_and_do_not_pass_empty_coverage():
    coords=torch.tensor([[0.,0,0],[.1,0,0],[.2,.2,.2]],dtype=torch.float64,requires_grad=True)
    result=local_geometry_penalty(coords,torch.tensor([2,1]),torch.eye(3,dtype=torch.float64)[None].repeat(2,1,1)*10,
                                  [[0,1]],[[.9,1.1]],[],[])
    assert result['bond_loss']==0 and result['angle_loss']==0
    audit=result['audit']
    assert audit['has_constraints'] and audit['coverage']['unconstrained_graph_indices']==[1]
    assert audit['all_declared_constraints_satisfied'] is True
    assert audit['interval_constraints_satisfied'] is None
    assert audit['graphs'][1]['has_constraints'] is False
    assert audit['graphs'][1]['interval_constraints_satisfied'] is None
    assert not audit['coverage']['profile_completeness_checked']


def test_absent_constraints_contribute_zero_to_explicit_all_graph_mean():
    coords=torch.tensor([[0.,0,0],[.2,0,0],[.2,.2,.2]],dtype=torch.float64)
    result=local_geometry_penalty(coords,torch.tensor([2,1]),torch.eye(3,dtype=torch.float64)[None].repeat(2,1,1)*10,
                                  [[0,1]],[[1.,1.]],[],[])
    torch.testing.assert_close(result['per_graph_bond_loss'],torch.tensor([1.,0.],dtype=torch.float64))
    assert result['bond_loss']==.5


def test_fully_empty_topology_has_zero_differentiable_loss_but_no_satisfaction_claim():
    coords=torch.tensor([[.1,.2,.3]],dtype=torch.float64,requires_grad=True)
    result=local_geometry_penalty(coords,torch.tensor([1]),torch.eye(3,dtype=torch.float64)[None],[],[],[],[])
    (result['bond_loss']+result['angle_loss']).backward()
    assert torch.equal(coords.grad,torch.zeros_like(coords))
    assert result['audit']['has_constraints'] is False
    assert result['audit']['interval_constraints_satisfied'] is None
    assert result['audit']['all_declared_constraints_satisfied'] is None


def test_fixed_inputs_detached_unmodified_and_random_streams_unchanged():
    args=list(triad(dtype=torch.float32));args[0].requires_grad_();args[2].requires_grad_()
    args[4].requires_grad_();args[6].requires_grad_()
    originals=[v.detach().clone() for v in args]
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    result=local_geometry_penalty(*args)
    (result['bond_loss']+result['angle_loss']).backward()
    assert args[0].grad is not None and torch.isfinite(args[0].grad).all()
    assert all(args[i].grad is None for i in (2,4,6))
    assert all(torch.equal(v,old) for v,old in zip(args,originals))
    assert result['bond_loss'].dtype==torch.float64
    assert random.getstate()==rng[0] and torch.equal(torch.get_rng_state(),rng[2])
    now=np.random.get_state()
    assert now[0]==rng[1][0] and np.array_equal(now[1],rng[1][1]) and now[2:]==rng[1][2:]
    json.dumps(result['audit'],allow_nan=False)


@pytest.mark.parametrize('angle',[0.,180.])
def test_collinear_defined_angles_have_finite_cosine_gradients(angle):
    args=list(triad(angle=angle));args[0].requires_grad_()
    result=local_geometry_penalty(*args)
    result['angle_loss'].backward()
    assert torch.isfinite(args[0].grad).all()
    assert result['audit']['angle_degrees'][0]==pytest.approx(angle)


def test_coincident_angle_arm_fails_explicitly_with_partial_audit():
    args=list(triad());args[0][0]=args[0][1]
    with pytest.raises(LocalGeometryError,match='undefined') as caught:local_geometry_penalty(*args)
    audit=caught.value.audit
    assert audit['undefined_angle_indices']==[0]
    assert audit['coincident_bond_indices']==[0]
    assert audit['interval_constraints_satisfied'] is False
    json.dumps(audit,allow_nan=False)


@pytest.mark.parametrize('bad',['self_bond','self_angle','cross_graph','duplicate_bond','reverse_duplicate_bond',
                              'duplicate_angle','reverse_duplicate_angle','missing_arm','out_of_range','negative_index',
                              'float_indices','bool_indices','wrong_shape'])
def test_invalid_topology_is_rejected_without_dropping_constraints(bad):
    args=list(triad())
    if bad=='self_bond':args[3]=torch.tensor([[0,0],[1,2]])
    elif bad=='self_angle':args[5]=torch.tensor([[0,1,1]])
    elif bad=='cross_graph':args[1]=torch.tensor([1,2]);args[2]=args[2].repeat(2,1,1)
    elif bad in ('duplicate_bond','reverse_duplicate_bond'):
        args[3]=torch.cat([args[3],torch.tensor([[0,1] if bad=='duplicate_bond' else [1,0]])]);args[4]=args[4][:1].repeat(3,1)
    elif bad in ('duplicate_angle','reverse_duplicate_angle'):
        args[5]=torch.tensor([[0,1,2],[0,1,2] if bad=='duplicate_angle' else [2,1,0]]);args[6]=args[6].repeat(2,1)
    elif bad=='missing_arm':args[3]=args[3][:1];args[4]=args[4][:1]
    elif bad=='out_of_range':args[3][0,0]=3
    elif bad=='negative_index':args[3][0,0]=-1
    elif bad=='float_indices':args[3]=args[3].double()
    elif bad=='bool_indices':args[3]=args[3].bool()
    else:args[3]=args[3][:,:1]
    with pytest.raises(ValueError):local_geometry_penalty(*args)


@pytest.mark.parametrize('bad',['bond_zero','bond_reverse','bond_nan','angle_negative','angle_large','angle_reverse',
                              'angle_nan','bounds_count','cell_singular','nonfinite_coords','bad_counts','empty_graphs'])
def test_invalid_ranges_and_geometry_are_rejected(bad):
    args=list(triad())
    if bad=='bond_zero':args[4][0,0]=0
    elif bad=='bond_reverse':args[4][0]=torch.tensor([2.,1.])
    elif bad=='bond_nan':args[4][0,0]=float('nan')
    elif bad=='angle_negative':args[6][0,0]=-1
    elif bad=='angle_large':args[6][0,1]=181
    elif bad=='angle_reverse':args[6][0]=torch.tensor([100.,80.])
    elif bad=='angle_nan':args[6][0,0]=float('nan')
    elif bad=='bounds_count':args[4]=args[4][:1]
    elif bad=='cell_singular':args[2][0,0]=0
    elif bad=='nonfinite_coords':args[0][0,0]=float('inf')
    elif bad=='bad_counts':args[1]=torch.tensor([2])
    else:args[1]=torch.tensor([],dtype=torch.long);args[2]=args[2][:0]
    with pytest.raises(ValueError):local_geometry_penalty(*args)


def test_python_reference_bounds_keep_their_double_precision():
    args=list(triad())
    args[4]=[[1.1234567890123,1.9876543210987]]*2
    result=local_geometry_penalty(*args)
    assert result['audit']['bond_bounds_A']==args[4]


@pytest.mark.parametrize('bad',[torch.tensor([[1.+1j,2.+0j],[1.+0j,2.+0j]]),[[True,2.],[1.,2.]],
                               torch.tensor([[True,True],[True,True]])])
def test_complex_or_boolean_reference_bounds_are_not_silently_converted(bad):
    args=list(triad());args[4]=bad
    with pytest.raises(ValueError):local_geometry_penalty(*args)


def test_compressed_bond_gradient_increases_length():
    args=list(triad(angle=90.,first=.8));args[0].requires_grad_()
    result=local_geometry_penalty(*args)
    result['bond_loss'].backward()
    assert args[0].grad[0,0]<0 and args[0].grad[1,0]>0
    changed=local_geometry_penalty(args[0].detach()-.0005*args[0].grad,*args[1:])
    assert changed['bond_loss']<result['bond_loss']


@pytest.mark.parametrize('skew',[False,True])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_equal_nearest_periodic_angle_images_fail_with_audit(skew,dtype):
    cell=torch.tensor([[10.,0,0],[1.,10.,0],[.2,.3,10.]],dtype=dtype) if skew else torch.eye(3,dtype=dtype)*10
    coords=torch.tensor([[.5,0,0],[0.,0,0],[.25,.25,0]],dtype=dtype)
    original=coords.clone()
    with pytest.raises(LocalGeometryError,match='distinct angle-arm directions') as caught:
        local_geometry_penalty(coords,torch.tensor([3]),cell[None],[[0,1],[1,2]],[[3.,6.]]*2,[[0,1,2]],[[0.,180.]])
    audit=caught.value.audit
    assert audit['failure']=='ambiguous_periodic_angle_image'
    assert audit['interval_constraints_satisfied'] is False
    assert audit['undefined_angle_indices']==[0]
    arms=audit['angle_image_uniqueness']['ambiguous_arms']
    assert len(arms)==1 and arms[0]['arm']=='first' and arms[0]['angle_index']==0
    assert arms[0]['candidate_vectors_A']==[[-5.,0.,0.],[5.,0.,0.]]
    assert arms[0]['distance_tie_tolerance_A2']>0
    assert torch.equal(coords,original)
    json.dumps(audit,allow_nan=False)


def test_periodic_angle_ambiguity_is_invariant_to_individual_wrapping():
    coords=torch.tensor([[.5,0,0],[0.,0,0],[.25,.25,0]],dtype=torch.float64)
    shifts=torch.tensor([[2.,-1,3],[-3.,2,1],[1.,0,-2]],dtype=torch.float64)
    audits=[]
    for source in (coords,coords+shifts):
        with pytest.raises(LocalGeometryError) as caught:
            local_geometry_penalty(source,torch.tensor([3]),torch.eye(3,dtype=torch.float64)[None]*10,
                [[0,1],[1,2]],[[3.,6.]]*2,[[0,1,2]],[[130.,140.]])
        audits.append(caught.value.audit)
    assert audits[0]['angle_image_uniqueness']==audits[1]['angle_image_uniqueness']


def test_equal_nearest_bond_distance_without_angle_is_not_ambiguous():
    coords=torch.tensor([[.5,0,0],[0.,0,0]],dtype=torch.float64,requires_grad=True)
    result=local_geometry_penalty(coords,torch.tensor([2]),torch.eye(3,dtype=torch.float64)[None]*10,
        [[0,1]],[[4.9,5.1]],[],[])
    assert result['audit']['interval_constraints_satisfied'] is True
    assert result['audit']['bond_distances_A']==[5.]
    assert result['audit']['angle_image_uniqueness']['ambiguous_arms']==[]
    result['bond_loss'].backward()
    assert torch.isfinite(coords.grad).all()


def test_repeated_equal_direction_candidates_are_not_ambiguous(monkeypatch):
    import cdvae.common.local_geometry as module
    monkeypatch.setattr(module,'product',lambda *args:iter([(0,0,0),(0,0,0)]))
    rows=module._ambiguous_angle_images(torch.tensor([[.1,.1,0.]],dtype=torch.float64),torch.eye(3,dtype=torch.float64)*10)
    assert rows==[]


def test_nearby_but_distinct_minimum_image_retains_geometry_and_gradient():
    coords=torch.tensor([[.5-1e-8,0,0],[0.,0,0],[.25,.25,0]],dtype=torch.float64,requires_grad=True)
    result=local_geometry_penalty(coords,torch.tensor([3]),torch.eye(3,dtype=torch.float64)[None]*10,
        [[0,1],[1,2]],[[3.,6.]]*2,[[0,1,2]],[[130.,140.]])
    assert result['audit']['angle_degrees']==pytest.approx([45.])
    assert result['audit']['angle_image_uniqueness']['ambiguous_arms']==[]
    result['angle_loss'].backward()
    assert torch.isfinite(coords.grad).all() and bool((coords.grad!=0).any())


@pytest.mark.parametrize('position',[3,5])
@pytest.mark.parametrize('value',[False,np.bool_(False),torch.tensor(False)])
def test_nested_mixed_boolean_topology_is_rejected_before_coercion(position,value):
    args=list(triad())
    args[position]=[[value,1],[1,2]] if position==3 else [[value,1,2]]
    with pytest.raises(ValueError,match='boolean indices'):local_geometry_penalty(*args)


@pytest.mark.parametrize('value',[np.array([[False,True],[True,True]]),np.array([[False,1],[1,2]],dtype=object),torch.tensor([[False,True],[True,True]])])
def test_boolean_arrays_and_object_topology_are_rejected(value):
    args=list(triad());args[3]=value
    with pytest.raises(ValueError,match='boolean indices'):local_geometry_penalty(*args)
