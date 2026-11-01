"""All-site reference lower bounds, periodic gradients and explicit coverage."""
import copy
import json
import random

import numpy as np
import pytest
import torch
from ase import Atoms

from cdvae.common.profile_pair_contacts import ProfileContactError, profile_pair_lower_bound_penalty
from cdvae.common.reference_geometry_profile import POLICY


def profile(groups=None):
    groups = groups or {'B|Mo': [2.1269607747370674, 2.848364067944148]}
    return {'version': 1, 'source': 'md', 'fit_split': 'train', 'policy': copy.deepcopy(POLICY),
        'bond_groups': {key: {'covered': True, 'bounds': bounds, 'reasons': [],
            'structure_count': 48, 'observation_count': 300} for key, bounds in groups.items()},
        'angle_groups': {}}


def two(distance=1.756626892967297, dtype=torch.float64, species=(5, 42)):
    frac = torch.tensor([[.1, .2, .3], [.1+distance/10, .2, .3]], dtype=dtype, requires_grad=True)
    types = torch.tensor(species, dtype=torch.long)
    counts = torch.tensor([2], dtype=torch.long)
    cells = (torch.eye(3, dtype=dtype)*10)[None]
    return frac, types, counts, cells


def call(args, reference=None):
    return profile_pair_lower_bound_penalty(*args, profile() if reference is None else reference, source='md')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_observed_new_B_Mo_contact_is_penalized_and_gradient_separates(dtype):
    args = two(dtype=dtype)
    result = call(args)
    pair = result['audit']['pairs'][0]
    assert pair['pair'] == [0, 1] and pair['group_key'] == 'B|Mo'
    assert pair['distance_A'] == pytest.approx(1.756626892967297, abs=3e-7)
    assert pair['lower_bound_A'] == 2.1269607747370674 and pair['violation']
    expected = ((pair['lower_bound_A']-pair['distance_A'])/pair['lower_bound_A'])**2/2
    assert float(result['loss']) == pytest.approx(expected, rel=1e-10)
    gradient, = torch.autograd.grad(result['loss'], args[0])
    moved = args[0]-.01*gradient
    after = call((moved, *args[1:]))
    assert after['audit']['pairs'][0]['distance_A'] > pair['distance_A']
    assert after['loss'] < result['loss'] and torch.isfinite(gradient).all()
    assert gradient.sum(0) == pytest.approx([0., 0., 0.], abs=1e-7)


def test_no_upper_bound_force_on_a_far_distinct_site_pair():
    args = two(distance=4.)
    result = call(args)
    assert result['audit']['pairs'][0]['distance_A'] > profile()['bond_groups']['B|Mo']['bounds'][1]
    assert float(result['loss']) == 0. and result['audit']['lower_bound_constraints_satisfied'] is True
    assert result['audit']['upper_distance_penalty'] is False
    gradient, = torch.autograd.grad(result['loss'], args[0])
    assert torch.equal(gradient, torch.zeros_like(gradient))
    changed = profile(); changed['bond_groups']['B|Mo']['bounds'][1] = 100.
    assert torch.equal(call(args, changed)['loss'], result['loss'])


def test_same_species_distinct_sites_are_included():
    result = call(two(distance=1.5, species=(5, 5)), profile({'B|B': [2., 3.]}))
    assert result['audit']['distinct_site_pair_count'] == 1
    assert result['audit']['pairs'][0]['group_key'] == 'B|B'
    assert result['audit']['violation_pair_count'] == 1


def test_skew_periodic_distances_match_independent_ASE():
    frac = torch.tensor([[.03,.11,.17], [.95,.07,.21], [.4,.42,.43]], dtype=torch.float64, requires_grad=True)
    cells = torch.tensor([[[4.1,0,0],[3.6,2.3,0],[.3,.5,4.2]]], dtype=torch.float64)
    numbers = torch.tensor([5,42,5]);counts = torch.tensor([3])
    reference = profile({'B|Mo':[2.,3.], 'B|B':[1.2,2.]})
    result = call((frac,numbers,counts,cells), reference)
    atoms = Atoms(numbers=numbers.tolist(), scaled_positions=frac.detach().numpy(), cell=cells[0].numpy(), pbc=True)
    distances = atoms.get_all_distances(mic=True)
    for row in result['audit']['pairs']:
        assert row['distance_A'] == pytest.approx(distances[tuple(row['pair'])], abs=1e-12)
    gradient, = torch.autograd.grad(result['loss'], frac)
    assert torch.isfinite(gradient).all()


def test_gradcheck_away_from_image_and_lower_bound_boundaries():
    args = two(distance=1.75)
    assert torch.autograd.gradcheck(lambda coords: call((coords,*args[1:]))['loss'], (args[0],), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_individual_wrapping_and_cartesian_rotation_preserve_loss_and_gradients():
    args = two();first = call(args)
    g1, = torch.autograd.grad(first['loss'],args[0])
    shift = torch.tensor([[1.,-2.,0.],[-3.,0.,2.]],dtype=torch.float64)
    moved = (args[0].detach()+shift).requires_grad_()
    rotated = args[3] @ torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=torch.float64)
    second = call((moved,args[1],args[2],rotated))
    g2, = torch.autograd.grad(second['loss'],moved)
    assert float(second['loss']) == pytest.approx(float(first['loss']), abs=1e-14)
    assert torch.allclose(g1,g2,atol=1e-13,rtol=0)


def test_normalization_uses_atom_counts_and_equal_graph_mean_not_all_pair_mean():
    positions = torch.tensor([[0.,0,0],[1.,0,0], [0,0,0],[1.,0,0],[5.,5.,5.]],dtype=torch.float64)
    cells = torch.stack([torch.eye(3,dtype=torch.float64)*10,torch.eye(3,dtype=torch.float64)*20])
    frac = torch.cat([positions[:2]/10,positions[2:]/20]).requires_grad_()
    numbers = torch.tensor([5,42,5,42,5]);counts = torch.tensor([2,3])
    reference = profile({'B|Mo':[2.,3.], 'B|B':[2.,3.]})
    result = call((frac,numbers,counts,cells), reference)
    assert result['per_graph_loss'].tolist() == pytest.approx([.25/2, .25/3])
    assert float(result['loss']) == pytest.approx((.25/2+.25/3)/2)
    assert [r['distinct_site_pair_count'] for r in result['audit']['graphs']] == [1,3]
    assert all((r['pair'][0]<2)==(r['pair'][1]<2) for r in result['audit']['pairs'])


def test_missing_groups_are_retained_without_fallback_and_partial_never_becomes_full():
    frac = torch.tensor([[0.,0,0],[.4,0,0],[0,.4,0]],dtype=torch.float64,requires_grad=True)
    args = (frac,torch.tensor([5,42,8]),torch.tensor([3]),torch.eye(3,dtype=torch.float64)[None]*10)
    result = call(args)
    audit = result['audit']
    assert audit['covered_pair_count']==1 and audit['uncovered_pair_count']==2
    assert audit['declared_lower_bounds_satisfied'] is True and audit['lower_bound_constraints_satisfied'] is None
    assert audit['status']=='partial_reference_coverage' and not audit['coverage_complete']
    missing = [r for r in audit['pairs'] if not r['reference_covered']]
    assert {r['group_key'] for r in missing}=={'B|O','Mo|O'}
    assert all(r['lower_bound_A'] is None and r['violation'] is None for r in missing)
    assert audit['graphs'][0]['missing_group_pair_counts']=={'B|O':1,'Mo|O':1}
    json.dumps(audit,allow_nan=False)


def test_profile_group_insufficient_support_remains_uncovered():
    reference=profile();reference['bond_groups']['B|Mo'].update(covered=False,reasons=['insufficient_structures'],bounds=None)
    result=call(two(),reference)
    assert float(result['loss'])==0. and result['audit']['status']=='unavailable_reference_coverage'
    assert result['audit']['pairs'][0]['missing_reasons']==['insufficient_structures']
    assert result['audit']['lower_bound_constraints_satisfied'] is None


def test_single_site_and_uncovered_pairs_give_differentiable_zero_with_unavailable_status():
    frac=torch.tensor([[.1,.2,.3]],dtype=torch.float64,requires_grad=True)
    args=(frac,torch.tensor([5]),torch.tensor([1]),torch.eye(3,dtype=torch.float64)[None]*10)
    result=call(args)
    assert float(result['loss'])==0. and result['audit']['graphs'][0]['status']=='no_distinct_site_pairs'
    assert result['audit']['lower_bound_constraints_satisfied'] is None and not result['audit']['coverage_complete']
    assert torch.equal(torch.autograd.grad(result['loss'],frac)[0],torch.zeros_like(frac))
    args=two(species=(8,8));result=call(args)
    assert result['audit']['graphs'][0]['status']=='missing_reference_lower_bounds'
    assert result['audit']['lower_bound_constraints_satisfied'] is None
    assert torch.equal(torch.autograd.grad(result['loss'],args[0])[0],torch.zeros_like(args[0]))


@pytest.mark.parametrize('species',[(5,42),(8,8)])
def test_exact_coincident_distinct_sites_fail_with_complete_current_pair_evidence(species):
    args=two(distance=0.,species=species)
    with pytest.raises(ProfileContactError,match='undefined separating direction') as caught:
        call(args)
    audit=caught.value.audit
    assert audit['status']=='undefined_distance_direction' and audit['undefined_pair_indices']==[0]
    assert audit['pairs'][0]['distance_A']==0. and audit['pairs'][0]['pair']==[0,1]
    assert audit['lower_bound_constraints_satisfied'] is False
    assert audit['planned_distinct_site_pair_count']==audit['measured_pair_count']==1
    assert audit['has_constraints']==(species==(5,42))
    json.dumps(audit,allow_nan=False)


def test_empty_graph_coverage_has_explicit_denominator_in_a_mixed_batch():
    first=two(distance=1.)
    frac=torch.cat([first[0],torch.tensor([[.3,.3,.3]],dtype=torch.float64)]).requires_grad_()
    args=(frac,torch.tensor([5,42,8]),torch.tensor([2,1]),first[3].repeat(2,1,1))
    result=call(args);single=call(first)
    assert float(result['loss'])==pytest.approx(float(single['loss'])/2)
    assert result['audit']['graphs'][1]['status']=='no_distinct_site_pairs'
    assert result['audit']['graphs'][1]['lower_bound_constraints_satisfied'] is None
    assert not result['audit']['coverage_complete']


def test_failure_in_later_graph_retains_prior_measurements_and_planned_counts():
    first=two();second=two(distance=0.)
    args=(torch.cat([first[0],second[0]]),torch.cat([first[1],second[1]]),
          torch.tensor([2,2]),torch.cat([first[3],second[3]]))
    with pytest.raises(ProfileContactError) as caught:call(args)
    audit=caught.value.audit
    assert audit['planned_graph_count']==2 and audit['planned_distinct_site_pair_count']==2
    assert audit['measured_pair_count']==2 and audit['failed_graph']==1
    assert audit['undefined_pair_indices']==[1] and audit['violation_pair_indices']==[0,1]
    assert audit['pairs'][0]['distance_A']>0 and audit['pairs'][1]['distance_A']==0


def test_fixed_profile_cell_species_and_random_streams_remain_unchanged():
    args=list(two());args[3].requires_grad_();reference=profile()
    before=[x.detach().clone() for x in args];old_profile=copy.deepcopy(reference)
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    result=call(tuple(args),reference);result['loss'].backward()
    assert args[0].grad is not None and args[3].grad is None
    assert all(torch.equal(old,x.detach()) for old,x in zip(before,args)) and reference==old_profile
    result['audit']['pairs'][0]['lower_bound_A']=123.
    assert reference==old_profile
    assert random.getstate()==rng[0] and np.array_equal(np.random.get_state()[1],rng[1][1])
    assert torch.equal(torch.get_rng_state(),rng[2])


@pytest.mark.parametrize('bad',['bool_coords','bool_species','mixed_bool_species','bool_counts','mixed_bool_counts',
                               'bool_cell','zero_count','bad_sum','float_species','bad_species','bad_cell','nan_coords',
                               'source','bool_lower','nan_lower','unsupported_profile'])
def test_invalid_inputs_are_rejected_before_geometry(bad):
    args=list(two());reference=profile();source='md'
    if bad=='bool_coords':args[0]=args[0].bool()
    elif bad=='bool_species':args[1]=args[1].bool()
    elif bad=='mixed_bool_species':args[1]=[5,True]
    elif bad=='bool_counts':args[2]=args[2].bool()
    elif bad=='mixed_bool_counts':args[2]=[1,True]
    elif bad=='bool_cell':args[3]=args[3].bool()
    elif bad=='zero_count':args[2]=torch.tensor([0])
    elif bad=='bad_sum':args[2]=torch.tensor([3])
    elif bad=='float_species':args[1]=args[1].float()
    elif bad=='bad_species':args[1]=torch.tensor([5,119])
    elif bad=='bad_cell':args[3]=torch.zeros_like(args[3])
    elif bad=='nan_coords':args[0]=args[0].detach().clone();args[0][0,0]=float('nan')
    elif bad=='source':source='other'
    elif bad=='bool_lower':reference['bond_groups']['B|Mo']['bounds'][0]=True
    elif bad=='nan_lower':reference['bond_groups']['B|Mo']['bounds'][0]=float('nan')
    else:reference['version']=True
    with pytest.raises(ValueError):
        profile_pair_lower_bound_penalty(*args,reference,source=source)
