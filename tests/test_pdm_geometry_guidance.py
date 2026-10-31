"""Declared PDM and local intervals jointly guide fixed-cell refinement."""
import copy
import json
import math
import random

import numpy as np
import pytest
import torch

from cdvae.common.local_geometry import local_geometry_penalty
from cdvae.common.pdm_geometry_guidance import refine_pdm_with_local_geometry


SCHEMA={'schema_version':2,'geometry':'periodic-minimum-image-clique-v2',
        'modes':['pair'],'elements':['C'],'features':['CC'],'cutoff':3.}


def example(angle=55.,first=2.2,second=1.5,dtype=torch.float64):
    a=math.radians(angle)
    frac=torch.tensor([[first,0,0],[0,0,0],[second*math.cos(a),second*math.sin(a),0]],dtype=dtype)/10+.3
    cell=torch.eye(3,dtype=dtype)[None]*10
    constraints={'bond_pairs':[[0,1],[1,2]],'bond_bounds_A':[[1.4,1.6],[1.4,1.6]],
                 'angle_triples':[[0,1,2]],'angle_bounds_deg':[[80.,100.]]}
    return frac,torch.tensor([6,6,6]),torch.tensor([3]),cell,constraints


def run_case(args,target=3.,std=1.,steps=80,**kwargs):
    frac,types,counts,cells,constraints=args
    return refine_pdm_with_local_geometry(frac,types,counts,cells,[[target]]*len(counts),[std],SCHEMA,
        {'steps':steps,'line_search_steps':6,'max_step_A':.04,'temperature':.1},
        local_constraints=constraints,bond_weight=kwargs.pop('bond_weight',5.),angle_weight=kwargs.pop('angle_weight',5.),**kwargs)


def assert_objective_and_trace(result):
    cfg=result['config']
    for report in result['reports']:
        for terms in (report['initial_components'],report['final_components']):
            if terms is None:continue
            assert terms['loss']==pytest.approx(terms['soft_pdm_loss']+cfg['geometry_weight']*terms['repulsion_loss']
                +result['bond_weight']*terms['bond_loss']+result['angle_weight']*terms['angle_loss'])
        assert report['iterations']<=cfg['steps']
        for row in report['history']:
            assert len(row['proposals'])<=cfg['line_search_steps']
            for proposal in row['proposals']:
                assert proposal['max_cartesian_step_A']<=cfg['max_step_A']
                if 'loss' in proposal and proposal['loss'] is not None:
                    assert proposal['loss']==pytest.approx(proposal['soft_pdm_loss']+proposal['weighted_repulsion_loss']
                        +proposal['weighted_bond_loss']+proposal['weighted_angle_loss'])
                if proposal['accepted']:
                    assert proposal['loss']<=row['loss']
                    assert proposal['reason']=='combined_objective_nonincreasing'
            if row['accepted']:
                assert row['accepted_loss']<=row['loss']
                assert sum(p['accepted'] for p in row['proposals'])==1
    json.dumps(result['reports'],allow_nan=False)
    json.dumps(result['local_geometry_audits'],allow_nan=False)


def test_declared_bond_and_angle_improve_with_fixed_three_pair_pdm_target():
    args=example()
    before=local_geometry_penalty(args[0],args[2],args[3],**args[4])
    result=run_case(args)
    after=local_geometry_penalty(result['frac_coords'],args[2],args[3],**args[4])
    assert after['bond_loss']<before['bond_loss']
    assert after['angle_loss']<before['angle_loss']
    assert after['audit']['bond_distances_A'][0]<before['audit']['bond_distances_A'][0]
    assert after['audit']['angle_degrees'][0]>before['audit']['angle_degrees'][0]
    assert result['final_hard_raw'].tolist()==[[3.]]
    assert result['reports'][0]['final_components']['loss']<result['reports'][0]['initial_components']['loss']
    assert_objective_and_trace(result)


def test_pdm_target_remains_in_objective_and_changes_result_with_same_local_bounds():
    # Broad explicit intervals leave both requested PDM counts geometrically possible.
    args=list(example(angle=90.,first=2.9,second=2.9))
    args[4]={'bond_pairs':[[0,1],[1,2]],'bond_bounds_A':[[1.,4.],[1.,4.]],
             'angle_triples':[[0,1,2]],'angle_bounds_deg':[[10.,170.]]}
    low=run_case(args,target=0.,steps=50,bond_weight=.1,angle_weight=.1)
    high=run_case(args,target=3.,steps=50,bond_weight=.1,angle_weight=.1)
    assert not torch.equal(low['frac_coords'],high['frac_coords'])
    assert float(low['final_hard_raw'][0,0])<float(high['final_hard_raw'][0,0])
    for result in (low,high):
        assert result['reports'][0]['final_components']['soft_pdm_loss']<result['reports'][0]['initial_components']['soft_pdm_loss']
        assert_objective_and_trace(result)


def test_competing_intervals_can_lower_total_while_pdm_loss_increases():
    args=list(example(angle=90.,first=3.5,second=3.5))
    result=run_case(args,target=0.,steps=70,bond_weight=20.,angle_weight=20.)
    report=result['reports'][0]
    assert report['final_components']['loss']<report['initial_components']['loss']
    assert report['final_components']['bond_loss']<report['initial_components']['bond_loss']
    assert report['final_components']['soft_pdm_loss']>report['initial_components']['soft_pdm_loss']
    assert_objective_and_trace(result)


def test_batch_global_constraints_rebase_correctly_with_different_cells():
    a=example();b=example(angle=120.,first=1.8,second=2.3)
    b=list(b);b[3]=b[3]*1.3;b[0]=(b[0]-.3)/1.3+.3
    constraints={'bond_pairs':[[0,1],[1,2],[3,4],[4,5]],'bond_bounds_A':a[4]['bond_bounds_A']+b[4]['bond_bounds_A'],
                 'angle_triples':[[0,1,2],[3,4,5]],'angle_bounds_deg':a[4]['angle_bounds_deg']+b[4]['angle_bounds_deg']}
    batch=(torch.cat([a[0],b[0]]),torch.cat([a[1],b[1]]),torch.tensor([3,3]),torch.cat([a[3],b[3]]),constraints)
    combined=run_case(batch,steps=20);left=run_case(a,steps=20);right=run_case(b,steps=20)
    assert torch.equal(combined['frac_coords'][:3],left['frac_coords'])
    assert torch.equal(combined['frac_coords'][3:],right['frac_coords'])
    assert combined['reports'][1]['global_atom_offset']==3
    assert combined['reports'][1]['local_constraints']['bond_pairs']==[[0,1],[1,2]]
    assert combined['reports'][1]['local_constraints']['angle_triples']==[[0,1,2]]
    assert combined['reports'][1]['global_bond_constraint_indices']==[2,3]
    assert combined['reports'][1]['global_angle_constraint_indices']==[1]
    assert_objective_and_trace(combined)


def test_undefined_angle_retains_original_graph_and_processes_other_graph():
    a=list(example());a[0][0]=a[0][1];b=example()
    constraints={'bond_pairs':[[0,1],[1,2],[3,4],[4,5]],'bond_bounds_A':a[4]['bond_bounds_A']+b[4]['bond_bounds_A'],
                 'angle_triples':[[0,1,2],[3,4,5]],'angle_bounds_deg':a[4]['angle_bounds_deg']+b[4]['angle_bounds_deg']}
    batch=(torch.cat([a[0],b[0]]),torch.cat([a[1],b[1]]),torch.tensor([3,3]),torch.cat([a[3],b[3]]),constraints)
    result=run_case(batch,steps=5)
    assert len(result['reports'])==len(result['local_geometry_audits'])==2
    assert result['reports'][0]['status']=='undefined_local_geometry'
    assert result['local_geometry_audits'][0]['undefined_angle_indices']==[0]
    assert torch.equal(result['frac_coords'][:3],a[0])
    assert torch.equal(result['input_frac_coords'],batch[0])
    assert result['reports'][1]['iterations']==5
    assert not torch.equal(result['frac_coords'][3:],b[0])
    assert result['final_hard_raw'].shape==(2,1)
    assert_objective_and_trace(result)


def test_empty_local_constraints_return_original_with_missing_coverage_status():
    args=list(example());args[4]={'bond_pairs':[],'bond_bounds_A':[],'angle_triples':[],'angle_bounds_deg':[]}
    result=run_case(args,steps=5)
    report=result['reports'][0]
    assert report['status']=='missing_local_constraints'
    assert report['iterations']==0 and not report['has_local_constraints']
    assert report['interval_constraints_satisfied'] is None
    assert not report['optimizer_stationary']
    assert torch.equal(result['frac_coords'],args[0])
    assert result['local_geometry_audits'][0]['has_constraints'] is False
    assert_objective_and_trace(result)


def test_fixed_inputs_and_random_streams_unchanged_with_detached_outputs():
    args=list(example(dtype=torch.float32))
    args[0].requires_grad_();args[3].requires_grad_()
    before=[value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value) for value in args]
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    result=run_case(args,steps=3)
    assert all(torch.equal(a,b) if torch.is_tensor(a) else a==b for a,b in zip(args,before))
    assert result['frac_coords'].dtype==args[0].dtype and not result['frac_coords'].requires_grad
    assert torch.equal(result['cell_matrices'],args[3]) and torch.equal(result['atom_types'],args[1])
    assert args[0].grad is None and args[3].grad is None
    assert random.getstate()==rng[0] and torch.equal(torch.get_rng_state(),rng[2])
    now=np.random.get_state();assert now[0]==rng[1][0] and np.array_equal(now[1],rng[1][1]) and now[2:]==rng[1][2:]
    assert_objective_and_trace(result)


def test_skew_cell_periodic_translation_preserves_refinement():
    args=list(example());args[3]=torch.tensor([[[10.,0,0],[2.,9.,0],[.5,.3,10.]]],dtype=torch.float64)
    shifted=list(args);shifted[0]=args[0]+torch.tensor([[1.,-1,0],[-2,1,2],[0,2,-1]],dtype=torch.float64)
    plain=run_case(args,steps=3);translated=run_case(shifted,steps=3)
    torch.testing.assert_close(plain['frac_coords'],translated['frac_coords'],atol=1e-12,rtol=1e-12)
    assert torch.equal(plain['final_hard_raw'],translated['final_hard_raw'])
    assert plain['reports'][0]['final_components']['loss']==pytest.approx(translated['reports'][0]['final_components']['loss'],abs=1e-12)


def test_one_step_budget_is_reported_without_claiming_optimizer_convergence():
    result=run_case(example(),steps=1)
    report=result['reports'][0]
    assert report['status']=='steps_completed' and report['iterations']==1
    assert not report['optimizer_stationary']
    assert not report['interval_constraints_satisfied']
    assert not report['profile_completeness_checked']
    assert_objective_and_trace(result)


@pytest.mark.parametrize('name',['bond_weight','angle_weight'])
@pytest.mark.parametrize('value',[True,0.,-1.,float('nan'),float('inf'),'1'])
def test_weights_require_explicit_positive_finite_numbers(name,value):
    args=example();kwargs={'bond_weight':1.,'angle_weight':1.};kwargs[name]=value
    with pytest.raises(ValueError):run_case(args,steps=1,**kwargs)


@pytest.mark.parametrize('bad',['missing_key','extra_key','crossgraph','missing_arm','duplicate','wrong_target'])
def test_invalid_constraints_or_targets_reject_before_refinement(bad):
    args=list(example())
    target=3.
    if bad=='missing_key':del args[4]['bond_bounds_A']
    elif bad=='extra_key':args[4]['unexpected']=1
    elif bad=='crossgraph':args[2]=torch.tensor([1,2]);args[3]=args[3].repeat(2,1,1)
    elif bad=='missing_arm':args[4]['bond_pairs']=[[0,1]];args[4]['bond_bounds_A']=[[1.,2.]]
    elif bad=='duplicate':args[4]['angle_triples']*=2;args[4]['angle_bounds_deg']*=2
    else:target=4.
    original=args[0].clone()
    with pytest.raises(ValueError):run_case(args,target=target,steps=1)
    assert torch.equal(args[0],original)


def test_finite_float32_large_gradient_uses_finite_norm_and_nonzero_step():
    args=example(dtype=torch.float32)
    result=run_case(args,steps=2,bond_weight=1e30,angle_weight=1.)
    report=result['reports'][0]
    assert report['status']=='steps_completed'
    assert not torch.equal(result['frac_coords'],args[0])
    assert result['frac_coords'].dtype==torch.float32
    assert report['final_components']['loss']<report['initial_components']['loss']
    for row in report['history']:
        assert math.isfinite(row['gradient_rms']) and row['gradient_rms']>0
        assert row['accepted']
    assert_objective_and_trace(result)


def test_nonfinite_source_dtype_gradient_preserves_original_and_failure():
    args=example(dtype=torch.float32)
    result=run_case(args,steps=2,bond_weight=1e300,angle_weight=1.)
    report=result['reports'][0]
    assert report['status']=='nonfinite_gradient'
    assert report['iterations']==1 and not report['history'][0]['accepted']
    assert report['history'][0]['proposals']==[]
    assert torch.equal(result['frac_coords'],args[0])
    assert torch.equal(result['input_frac_coords'],args[0])
    assert_objective_and_trace(result)


def test_ambiguous_periodic_angle_retains_original_graph_and_other_graph_result():
    a=list(example());a[0]=torch.tensor([[.5,0,0],[0.,0,0],[.25,.25,0]],dtype=torch.float64)
    a[4]['bond_bounds_A']=[[3.,6.]]*2;a[4]['angle_bounds_deg']=[[130.,140.]]
    b=example()
    constraints={'bond_pairs':[[0,1],[1,2],[3,4],[4,5]],'bond_bounds_A':a[4]['bond_bounds_A']+b[4]['bond_bounds_A'],
        'angle_triples':[[0,1,2],[3,4,5]],'angle_bounds_deg':a[4]['angle_bounds_deg']+b[4]['angle_bounds_deg']}
    batch=(torch.cat([a[0],b[0]]),torch.cat([a[1],b[1]]),torch.tensor([3,3]),torch.cat([a[3],b[3]]),constraints)
    result=run_case(batch,steps=3)
    assert len(result['reports'])==2
    assert result['reports'][0]['status']=='undefined_local_geometry'
    assert result['local_geometry_audits'][0]['failure']=='ambiguous_periodic_angle_image'
    assert torch.equal(result['frac_coords'][:3],a[0])
    assert torch.equal(result['input_frac_coords'],batch[0])
    assert not torch.equal(result['frac_coords'][3:],b[0])
    assert result['reports'][1]['iterations']==3
    assert_objective_and_trace(result)
