"""Staged topology refresh, fixed budgets and retained failure coordinates."""
import copy
import random

import numpy as np
import pytest
import torch

import cdvae.common.pdm_topology_refresh as module
from cdvae.common.pdm_geometry_guidance import refine_pdm_with_local_geometry
from cdvae.common.reference_geometry_profile import POLICY, build_local_constraints


def case():
    frac=torch.tensor([[.1,.1,.1],[.25,.1,.1],[.1,.29,.1]],dtype=torch.float64)
    types=torch.tensor([6,8,8]);counts=torch.tensor([3]);cells=torch.eye(3,dtype=torch.float64)[None]*10
    target=torch.tensor([[0.,2.,1.]],dtype=torch.float64);std=torch.ones(3,dtype=torch.float64)
    schema={'schema_version':2,'geometry':'periodic-minimum-image-clique-v2','modes':['pair'],
            'elements':['C','O'],'features':['CC','CO','OO'],'cutoff':3.}
    def group(bounds):return {'covered':True,'bounds':bounds,'reasons':[], 'structure_count':5,'observation_count':30}
    profile={'version':1,'source':'declared-toy','fit_split':'train','policy':copy.deepcopy(POLICY),
             'bond_groups':{'C|O':group([1.2,1.7]),'O|O':group([1.,1.5])},
             'angle_groups':{'C|O|O|CN=2':group([60.,120.])}}
    config={'steps':4,'temperature':.1,'max_step_A':.01,'min_distance_A':1.,'geometry_weight':10.,'line_search_steps':2}
    kwargs={'profile':profile,'source':'declared-toy','structure_id':'toy','stage_steps':[2,2],
            'bond_weight':1.,'angle_weight':1.}
    return (frac,types,counts,cells,target,std,schema,config),kwargs


def fake_result(frac,types,cells,*,status='steps_completed',iterations=2):
    return {'frac_coords':frac.detach().clone(),'atom_types':types.detach().clone(),
            'cell_matrices':cells.detach().clone(),
            'reports':[{'status':status,'iterations':iterations,'error':None,'history':[]}],
            'final_hard_raw':torch.tensor([[0.,2.,1.]],dtype=frac.dtype)}


def test_each_stage_refreshes_neighbors_and_center_CN_using_same_profile(monkeypatch):
    args,kwargs=case();calls=[]
    def fake(frac,types,counts,cells,target,std,schema,config,**options):
        calls.append({'config':copy.deepcopy(config),'constraints':copy.deepcopy(options['local_constraints'])})
        updated=frac.clone()
        if len(calls)==1:updated[2,1]=.25
        return fake_result(updated,types,cells,iterations=config['steps'])
    monkeypatch.setattr(module,'refine_pdm_with_local_geometry',fake)
    result=module.refine_pdm_with_topology_refresh(*args,**kwargs)
    assert calls[0]['constraints']['bond_pairs']==[[0,1]] and calls[0]['constraints']['angle_triples']==[]
    assert calls[1]['constraints']['bond_pairs']==[[0,1],[0,2]] and calls[1]['constraints']['angle_triples']==[[1,0,2]]
    assert result['stages'][0]['binding']['frozen_topology']['coordination_numbers'][0]==1
    assert result['stages'][1]['binding']['frozen_topology']['coordination_numbers'][0]==2
    assert torch.equal(result['stages'][0]['end_frac_coords'],result['stages'][1]['start_frac_coords'])
    assert result['total_iterations']==4 and result['budget_respected'] and result['completed_stage_count']==2
    assert result['final_reference']['coverage_complete']
    assert result['final_local_geometry_audit']['interval_constraints_satisfied'] is True
    assert result['topology_changes_from_initial']['added_bonds']==[[0,2]]
    assert result['profile_refit_performed'] is False and result['cross_stage_objective_comparison'] is False


@pytest.mark.parametrize('status',['stationary','line_search_stalled'])
def test_early_normal_stage_stops_do_not_transfer_unused_iterations(monkeypatch,status):
    args,kwargs=case();args=list(args);args[-1]['steps']=10;kwargs['stage_steps']=[4,6];seen=[]
    def fake(frac,types,counts,cells,target,std,schema,config,**options):
        seen.append(config['steps'])
        return fake_result(frac,types,cells,status=status,iterations=1)
    monkeypatch.setattr(module,'refine_pdm_with_local_geometry',fake)
    result=module.refine_pdm_with_topology_refresh(*args,**kwargs)
    assert seen==[4,6] and result['total_iterations']==2 and result['unused_planned_iterations']==8
    assert [stage['unused_allocated_steps'] for stage in result['stages']]==[3,5]
    assert result['status']=='stages_completed' and result['budget_respected']


@pytest.mark.parametrize('status',['undefined_local_geometry','undefined_pair_contacts','nonfinite_objective',
                                  'nonfinite_gradient','evaluation_failed'])
def test_failed_stage_retains_returned_last_coordinates_and_never_runs_next(monkeypatch,status):
    args,kwargs=case();seen=[]
    def fake(frac,types,counts,cells,target,std,schema,config,**options):
        seen.append(True);current=frac.clone();current[2,1]=.26
        return fake_result(current,types,cells,status=status,iterations=1)
    monkeypatch.setattr(module,'refine_pdm_with_local_geometry',fake)
    result=module.refine_pdm_with_topology_refresh(*args,**kwargs)
    assert len(seen)==len(result['stages'])==1 and result['status']=='stopped'
    assert result['frac_coords'][2,1]==.26 and result['total_iterations']==1
    assert torch.equal(result['frac_coords'],result['stages'][0]['result']['frac_coords'])
    assert result['stages'][0]['status']==status and result['completed_stage_count']==0


def test_second_stage_failure_keeps_its_last_iterate_not_a_previous_stage_best(monkeypatch):
    args,kwargs=case();seen=[]
    def fake(frac,types,counts,cells,target,std,schema,config,**options):
        seen.append(True);current=frac.clone();current[2,1]=.25 if len(seen)==1 else .24
        return fake_result(current,types,cells,status='steps_completed' if len(seen)==1 else 'evaluation_failed',iterations=1)
    monkeypatch.setattr(module,'refine_pdm_with_local_geometry',fake)
    result=module.refine_pdm_with_topology_refresh(*args,**kwargs)
    assert result['frac_coords'][2,1]==.24 and result['stages'][0]['end_frac_coords'][2,1]==.25
    assert result['status']=='stopped' and result['total_iterations']==2


def test_unsupported_periodic_topology_stops_before_optimizer(monkeypatch):
    args,kwargs=case();args=list(args)
    args[0]=torch.tensor([[0.,0,0],[.5,0,0],[0,.5,0]],dtype=torch.float64)
    args[3]=torch.eye(3,dtype=torch.float64)[None]
    def forbidden(*a,**k):raise AssertionError('Optimizer must not run')
    monkeypatch.setattr(module,'refine_pdm_with_local_geometry',forbidden)
    result=module.refine_pdm_with_topology_refresh(*args,**kwargs)
    assert result['status']=='stopped' and result['stages'][0]['status']=='unsupported_periodic_topology'
    assert torch.equal(result['frac_coords'],args[0]) and result['total_iterations']==0
    assert result['final_local_geometry_audit']['interval_constraints_satisfied'] is None


def test_missing_coverage_retains_structure_without_a_pass(monkeypatch):
    args,kwargs=case();kwargs['profile']['bond_groups']={};kwargs['profile']['angle_groups']={};seen=[]
    def fake(frac,types,counts,cells,target,std,schema,config,**options):
        seen.append(True);return fake_result(frac,types,cells,status='missing_local_constraints',iterations=0)
    monkeypatch.setattr(module,'refine_pdm_with_local_geometry',fake)
    result=module.refine_pdm_with_topology_refresh(*args,**kwargs)
    assert len(seen)==1 and result['status']=='stopped' and not result['final_reference']['has_constraints']
    assert result['final_local_geometry_audit']['interval_constraints_satisfied'] is None
    assert torch.equal(result['frac_coords'],args[0])


@pytest.mark.parametrize('bad',[[],[True,3],[1,0,3],[1,-1,4],[1.,3],[1,2],'2,2'])
def test_invalid_stage_partition_rejected_before_any_optimizer(monkeypatch,bad):
    args,kwargs=case();kwargs['stage_steps']=bad
    def forbidden(*a,**k):raise AssertionError('Optimizer must not run')
    monkeypatch.setattr(module,'refine_pdm_with_local_geometry',forbidden)
    with pytest.raises(ValueError,match='stage_steps'):
        module.refine_pdm_with_topology_refresh(*args,**kwargs)


@pytest.mark.parametrize('bad',['multiple_graphs','source','bool_contact','zero_contact','nan_bond','bool_target','bad_std'])
def test_invalid_fixed_inputs_rejected(monkeypatch,bad):
    args,kwargs=case();args=list(args)
    if bad=='multiple_graphs':args[2]=torch.tensor([1,2]);args[3]=args[3].repeat(2,1,1)
    elif bad=='source':kwargs['source']='other'
    elif bad=='bool_contact':kwargs['contact_weight']=True
    elif bad=='zero_contact':kwargs['contact_weight']=0.
    elif bad=='nan_bond':kwargs['bond_weight']=float('nan')
    elif bad=='bool_target':args[4]=[[False,2.,1.]]
    else:args[5]=torch.tensor([1.,0.,1.],dtype=torch.float64)
    def forbidden(*a,**k):raise AssertionError('Optimizer must not run')
    monkeypatch.setattr(module,'refine_pdm_with_local_geometry',forbidden)
    with pytest.raises(ValueError):module.refine_pdm_with_topology_refresh(*args,**kwargs)


@pytest.mark.parametrize('failure',['exception','nonfinite_return','changed_cell','over_budget'])
def test_core_contract_failures_keep_last_available_coordinates_and_evidence(monkeypatch,failure):
    args,kwargs=case();seen=[]
    def fake(frac,types,counts,cells,target,std,schema,config,**options):
        seen.append(True)
        if failure=='exception':raise RuntimeError('recorded failure')
        result=fake_result(frac,types,cells,iterations=999 if failure=='over_budget' else 1)
        if failure=='nonfinite_return':result['frac_coords'][0,0]=float('nan')
        if failure=='changed_cell':result['cell_matrices'][0,0,0]=11.
        return result
    monkeypatch.setattr(module,'refine_pdm_with_local_geometry',fake)
    result=module.refine_pdm_with_topology_refresh(*args,**kwargs)
    assert len(seen)==1 and result['status']=='stopped' and torch.equal(result['frac_coords'],args[0])
    assert result['stages'][0]['status']=='stage_evaluation_failed'
    if failure in ('exception','over_budget'):
        assert not result['iteration_accounting_complete'] and not result['budget_respected']
    else:
        assert result['stages'][0]['result'] is not None and result['total_iterations']==1


def test_inputs_and_all_random_streams_unchanged_with_explicit_contact_option(monkeypatch):
    args,kwargs=case();kwargs['contact_weight']=2.
    original=[x.clone() if torch.is_tensor(x) else copy.deepcopy(x) for x in args];before=copy.deepcopy(kwargs);seen=[]
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    def fake(frac,types,counts,cells,target,std,schema,config,**options):
        seen.append(options)
        assert options['contact_source']=='declared-toy' and options['contact_weight']==2.
        assert options['contact_profile']==before['profile']
        return fake_result(frac,types,cells,iterations=config['steps'])
    monkeypatch.setattr(module,'refine_pdm_with_local_geometry',fake)
    result=module.refine_pdm_with_topology_refresh(*args,**kwargs)
    assert len(seen)==2 and result['final_contact_audit'] is not None
    assert kwargs==before and all(torch.equal(a,b) if torch.is_tensor(a) else a==b for a,b in zip(args,original))
    assert random.getstate()==rng[0] and np.array_equal(np.random.get_state()[1],rng[1][1])
    assert torch.equal(torch.get_rng_state(),rng[2])


def test_single_200_step_partition_matches_one_direct_refinement_exactly():
    args,kwargs=case();args=list(args);args[-1]['steps']=200;kwargs['stage_steps']=[200];kwargs['contact_weight']=1.
    binding=build_local_constraints(module._atoms(args[0],args[1],args[3][0]),kwargs['profile'],source=kwargs['source'],structure_id='direct')
    direct=refine_pdm_with_local_geometry(*args,local_constraints=binding['constraints'],
        bond_weight=kwargs['bond_weight'],angle_weight=kwargs['angle_weight'],
        contact_profile=kwargs['profile'],contact_source=kwargs['source'],contact_weight=1.)
    wrapped=module.refine_pdm_with_topology_refresh(*args,**kwargs)
    assert torch.equal(wrapped['frac_coords'],direct['frac_coords'])
    assert torch.equal(wrapped['final_hard_raw'],direct['final_hard_raw'])
    assert wrapped['stages'][0]['result']['reports']==direct['reports']
    assert wrapped['total_iterations']==direct['reports'][0]['iterations']<=200


def test_second_stage_binding_exception_preserves_first_stage_coordinates_and_ledger(monkeypatch):
    args,kwargs=case();original_builder=module.build_local_constraints;optimizer_calls=[]
    def build(atoms,profile,**options):
        if options['structure_id'].endswith('-stage1'):
            raise RuntimeError('injected second-stage binding failure')
        return original_builder(atoms,profile,**options)
    def refine(frac,types,counts,cells,target,std,schema,config,**options):
        optimizer_calls.append(True);current=frac.clone();current[2,1]=.25
        return fake_result(current,types,cells,iterations=config['steps'])
    monkeypatch.setattr(module,'build_local_constraints',build)
    monkeypatch.setattr(module,'refine_pdm_with_local_geometry',refine)
    result=module.refine_pdm_with_topology_refresh(*args,**kwargs)
    assert len(optimizer_calls)==1 and len(result['stages'])==2
    assert result['status']=='stopped' and result['stages'][1]['status']=='stage_binding_failed'
    assert 'injected second-stage binding failure' in result['error']
    assert result['frac_coords'][2,1]==.25
    assert torch.equal(result['frac_coords'],result['stages'][0]['result']['frac_coords'])
    assert torch.equal(result['frac_coords'],result['stages'][1]['start_frac_coords'])
    assert result['total_iterations']==2 and result['stages'][1]['iterations']==0
    assert result['budget_respected'] and result['final_diagnostics']['complete']


@pytest.mark.parametrize('scope',['final_atoms','final_reference_binding','final_local_geometry',
                                  'final_topology_changes','final_pair_contacts','final_hard_pdm'])
def test_final_diagnostic_exception_preserves_completed_stage_results_and_coordinates(monkeypatch,scope):
    args,kwargs=case();optimizer_calls=[]
    def refine(frac,types,counts,cells,target,std,schema,config,**options):
        optimizer_calls.append(True);current=frac.clone();current[2,1]=.25
        return fake_result(current,types,cells,iterations=config['steps'])
    def fail(*a,**k):raise RuntimeError('injected '+scope)
    monkeypatch.setattr(module,'refine_pdm_with_local_geometry',refine)
    if scope=='final_atoms':
        original=module._atoms;calls=[]
        def atoms(*a,**k):
            calls.append(True)
            if len(calls)==3:fail()
            return original(*a,**k)
        monkeypatch.setattr(module,'_atoms',atoms)
    elif scope=='final_reference_binding':
        original=module.build_local_constraints
        def build(atoms,profile,**options):
            if options['structure_id'].endswith('-final'):fail()
            return original(atoms,profile,**options)
        monkeypatch.setattr(module,'build_local_constraints',build)
    elif scope=='final_local_geometry':
        monkeypatch.setattr(module,'_independent_intervals',fail)
    elif scope=='final_topology_changes':
        monkeypatch.setattr(module,'audit_frozen_topology',fail)
    elif scope=='final_pair_contacts':
        kwargs['contact_weight']=1.
        monkeypatch.setattr(module,'profile_pair_lower_bound_penalty',fail)
    else:
        original=module.PairPDMDescriptor.forward;calls=[]
        def forward(self,*a,**k):
            calls.append(True)
            if len(calls)==2:fail()
            return original(self,*a,**k)
        monkeypatch.setattr(module.PairPDMDescriptor,'forward',forward)
    result=module.refine_pdm_with_topology_refresh(*args,**kwargs)
    assert len(optimizer_calls)==len(result['stages'])==result['completed_stage_count']==2
    assert result['status']=='final_diagnostics_failed' and result['total_iterations']==4
    assert result['budget_respected'] and not result['final_diagnostics']['complete']
    assert [row['scope'] for row in result['final_diagnostics']['errors']]==[scope]
    assert 'injected '+scope in result['error']
    assert result['frac_coords'][2,1]==.25
    assert torch.equal(result['frac_coords'],result['stages'][-1]['result']['frac_coords'])
    if scope=='final_reference_binding':assert result['final_reference'] is None
    if scope=='final_hard_pdm':assert result['final_hard_raw'] is None
