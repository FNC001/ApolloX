"""Explicit contact and topology-stage CLI paths; run numerical checks remotely."""
import json

import numpy as np
import pytest
import torch
from pymatgen.io.vasp import Poscar

from cdvae.common.profile_pair_contacts import ProfileContactError
from scripts.pdm_control import refine_geometry_structures as cli
from test_pdm_geometry_refine_cli import (
    assert_all_rows, rows, setup_case, stub, update_profile,
)


def stage_stub(*, status='stages_completed', diagnostics=True, shifted=False,
               incomplete_stage=False, seen=None, marker='stage-failure'):
    def refine(frac,types,counts,cells,target,std,schema,config,**kwargs):
        if seen is not None:seen.append((target.detach().clone(), config.copy(), kwargs))
        coords=frac.clone()+(.01 if shifted else 0.)
        stages=[]
        for index,steps in enumerate(kwargs['stage_steps']):
            stages.append({'stage':index,'allocated_steps':steps,'iterations':1,'unused_allocated_steps':steps-1,
                'iteration_accounting_complete':True,'status':'stationary','error':None,
                'binding':{'coverage_complete':not (incomplete_stage and index==0)},
                'start_frac_coords':frac.clone(),'end_frac_coords':coords.clone(),
                'result':{'initial_hard_raw':[11,22,33],'history':[{'iteration':0,'value':2.}]}})
        if status=='stopped':
            stages[-1].update(status='stage_evaluation_failed',error=marker)
        return {'status':status,'error':marker if status!='stages_completed' else None,
            'frac_coords':coords,'atom_types':types.clone(),'num_atoms':counts.clone(),'cell_matrices':cells.clone(),
            'stages':stages,'stage_steps':list(kwargs['stage_steps']),'planned_total_steps':config['steps'],
            'total_iterations':len(stages),'budget_respected':True,'iteration_accounting_complete':True,
            'completed_stage_count':len(stages)-(status=='stopped'),
            'final_diagnostics':{'complete':diagnostics,'errors':[] if diagnostics else
                [{'scope':'final_local_geometry','error':marker,'partial_audit':{'raw':[1,2,3]}}]},
            'final_hard_raw':torch.full_like(target,999.)}
    return refine


def assert_final_pdm(out,model):
    schema=json.loads((model/'pdm_schema.json').read_text())
    for row in rows(out/'pdm.csv'):
        structure=Poscar.from_file(out/'gen'/row['label'],check_for_potcar=False).structure
        assert [int(row[key]) for key in schema['features']]==list(cli.hard_pdm(structure,schema).values())


def test_default_options_leave_legacy_fields_and_call_contract_unchanged(tmp_path,monkeypatch):
    source,model,out,profile,args=setup_case(tmp_path)
    parsed=cli.parser().parse_args(args)
    assert parsed.contact_weight is None and parsed.topology_stage_steps is None
    seen=[]
    def core(*inputs,**kwargs):
        seen.append(kwargs)
        return stub(status='stationary')(*inputs,**kwargs)
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry',core)
    monkeypatch.setattr('cdvae.common.pdm_topology_refresh.refine_pdm_with_topology_refresh',
        lambda *a,**kw:pytest.fail('default path must not call topology wrapper'))
    assert cli.main(args)==0
    assert set(seen[0])=={'local_constraints','bond_weight','angle_weight'}
    for filename in ('geometry.csv','status.csv'):
        assert not any(key.startswith('contact_') or key.startswith('topology_stage_')
                       for key in rows(out/filename)[0])
    metadata=json.loads((out/'run.json').read_text())
    assert 'contact_weight' not in metadata and 'topology_stage_steps' not in metadata
    assert metadata['method']=='fixed_cell_pdm_local_geometry_refinement_v1'
    assert {p.name for p in out.iterdir()}=={'gen','pdm.csv','geometry.csv','status.csv','run.json'}


@pytest.mark.parametrize('extra', [
    ['--contact-weight','0'],['--contact-weight','-1'],['--contact-weight','nan'],['--contact-weight','inf'],
    ['--topology-stage-steps','0','2'],['--topology-stage-steps','-1','3'],
    ['--topology-stage-steps','1'],['--topology-stage-steps','1','1','1'],
])
def test_invalid_optin_configuration_precedes_output_creation(tmp_path,extra):
    source,model,out,profile,args=setup_case(tmp_path)
    assert cli.main(args+extra)==1
    assert not out.exists()


@pytest.mark.parametrize('field,value',[('contact_weight',True),('topology_stage_steps',[True,1]),
                                        ('topology_stage_steps',[])])
def test_programmatic_boolean_or_empty_options_rejected(tmp_path,field,value):
    source,model,out,profile,args=setup_case(tmp_path)
    parsed=cli.parser().parse_args(args);setattr(parsed,field,value)
    with pytest.raises(ValueError):cli.run(parsed)
    assert not out.exists()


def test_contacts_forward_checked_profile_and_report_partial_final_pair_coverage(tmp_path,monkeypatch):
    source,model,out,profile,args=setup_case(tmp_path)
    seen=[]
    def core(*inputs,**kwargs):
        seen.append(kwargs)
        return stub(status='stationary')(*inputs,**kwargs)
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry',core)
    assert cli.main(args+['--contact-weight','2'])==1
    assert seen[0]['contact_profile']==json.loads(profile.read_text())
    assert seen[0]['contact_source']=='md' and seen[0]['contact_weight']==2
    row=rows(out/'geometry.csv')[0]
    assert row['contact_distinct_site_pair_count']=='21'
    assert row['contact_measured_pair_count']=='21'
    assert row['contact_covered_pair_count']=='6' and row['contact_uncovered_pair_count']=='15'
    assert row['contact_coverage_complete']=='False'
    assert row['contact_lower_bound_constraints_satisfied']==''
    assert rows(out/'status.csv')[0]['status']=='partial_contact_reference_coverage'
    assert_final_pdm(out,model)


def test_topology_wrapper_receives_fixed_budget_and_keeps_trace_separate(tmp_path,monkeypatch):
    source,model,out,profile,args=setup_case(tmp_path)
    seen=[];before=profile.read_bytes()
    monkeypatch.setattr('cdvae.common.pdm_topology_refresh.refine_pdm_with_topology_refresh',stage_stub(seen=seen))
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry',
        lambda *a,**kw:pytest.fail('wrapper path must not invoke legacy CLI branch'))
    audit=tmp_path/'audit'
    assert cli.main(args+['--topology-stage-steps','1','1','--audit-dir',str(audit)])==0
    assert len(seen)==1
    assert seen[0][1]['steps']==2 and seen[0][2]['stage_steps']==[1,1]
    assert seen[0][2]['profile']==json.loads(profile.read_text())
    assert seen[0][2]['contact_weight'] is None
    assert profile.read_bytes()==before
    state=rows(out/'status.csv')[0];ledger=json.loads(state['topology_stage_ledger'])
    assert state['topology_planned_steps']=='2' and state['topology_iterations']=='2'
    assert [item['allocated_steps'] for item in ledger['stages']]==[1,1]
    assert ledger['final_diagnostics']['complete'] is True
    final_text=''.join((out/name).read_text() for name in ('status.csv','geometry.csv','pdm.csv','run.json'))
    assert all(key not in final_text for key in ('initial_hard_raw','history','start_frac_coords','end_frac_coords'))
    full=json.loads((audit/'0.vasp.json').read_text())
    assert full['refinement']['stages'][0]['result']['history']
    assert_final_pdm(out,model)


@pytest.mark.parametrize('status,diagnostics', [('stopped',True),('final_diagnostics_failed',False),('stages_completed',False)])
def test_wrapper_failure_retains_last_coordinates_ledger_and_error_even_when_csv_recomputes(tmp_path,monkeypatch,status,diagnostics):
    source,model,out,profile,args=setup_case(tmp_path)
    initial=Poscar.from_file(source/'0.vasp',check_for_potcar=False).structure
    monkeypatch.setattr('cdvae.common.pdm_topology_refresh.refine_pdm_with_topology_refresh',
        stage_stub(status=status,diagnostics=diagnostics,shifted=True))
    assert cli.main(args+['--topology-stage-steps','1','1'])==1
    assert_all_rows(out,('0.vasp',))
    state=rows(out/'status.csv')[0]
    assert state['status']=='error' and state['coordinate_state']=='last_iterate'
    assert state['optimizer_status']==status and state['error']
    assert len(json.loads(state['topology_stage_ledger'])['stages'])==2
    delivered=Poscar.from_file(out/'gen/0.vasp',check_for_potcar=False).structure
    np.testing.assert_allclose(delivered.frac_coords,initial.frac_coords+.01,atol=1e-14)
    assert rows(out/'pdm.csv')[0]['pdm_status']=='computed'
    assert_final_pdm(out,model)


def test_partial_intermediate_stage_cannot_become_full_coverage_from_final_recheck(tmp_path,monkeypatch):
    source,model,out,profile,args=setup_case(tmp_path)
    monkeypatch.setattr('cdvae.common.pdm_topology_refresh.refine_pdm_with_topology_refresh',
        stage_stub(incomplete_stage=True))
    assert cli.main(args+['--topology-stage-steps','1','1'])==1
    assert rows(out/'geometry.csv')[0]['final_topology_coverage_complete']=='True'
    assert rows(out/'status.csv')[0]['status']=='partial_reference_coverage'


def test_missing_local_reference_can_reach_contact_core_but_cannot_claim_full_coverage(tmp_path,monkeypatch):
    source,model,out,profile,args=setup_case(tmp_path)
    update_profile(profile,args,lambda item:item.update(bond_groups={},angle_groups={}))
    seen=[]
    def core(*inputs,**kwargs):
        seen.append(kwargs)
        return stub(status='stationary')(*inputs,**kwargs)
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry',core)
    assert cli.main(args+['--contact-weight','1'])==1
    assert len(seen)==1 and not seen[0]['local_constraints']['bond_pairs']
    geometry=rows(out/'geometry.csv')[0]
    assert geometry['contact_covered_pair_count']=='0'
    assert geometry['contact_lower_bound_constraints_satisfied']==''
    assert rows(out/'status.csv')[0]['status']!='refined'


def test_failed_first_staged_input_does_not_remove_remaining_denominators(tmp_path,monkeypatch):
    source,model,out,profile,args=setup_case(tmp_path,('0.vasp','1.vasp'))
    calls=[]
    def refine(*inputs,**kwargs):
        calls.append(kwargs['structure_id'])
        return stage_stub(status='stopped' if len(calls)==1 else 'stages_completed')(*inputs,**kwargs)
    monkeypatch.setattr('cdvae.common.pdm_topology_refresh.refine_pdm_with_topology_refresh',refine)
    assert cli.main(args+['--topology-stage-steps','1','1'])==1
    assert calls==['0.vasp','1.vasp']
    assert_all_rows(out,('0.vasp','1.vasp'))
    assert [row['status'] for row in rows(out/'status.csv')]==['error','refined']


def test_final_contact_failure_preserves_partial_audit_and_never_reports_zero_unknown_violations(tmp_path,monkeypatch):
    source,model,out,profile,args=setup_case(tmp_path)
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry',stub(status='stationary'))
    audit={'status':'undefined_periodic_distance','planned_graph_count':1,'planned_distinct_site_pair_count':21,
        'measured_pair_count':0,'coverage_complete':False,'graphs':[{'covered_pair_count':6,
        'missing_group_pair_counts':{'H|H':15}}],'violation_pair_indices':[], 'undefined_pair_indices':[]}
    def fail(*a,**kw):raise ProfileContactError('distance-evaluation-failed',audit)
    monkeypatch.setattr('cdvae.common.profile_pair_contacts.profile_pair_lower_bound_penalty',fail)
    internal=tmp_path/'audit'
    assert cli.main(args+['--contact-weight','1','--audit-dir',str(internal)])==1
    geometry=rows(out/'geometry.csv')[0]
    assert geometry['contact_measured_pair_count']=='0'
    assert geometry['contact_violation_pair_count']=='' and geometry['contact_undefined_pair_count']==''
    assert geometry['contact_status']=='undefined_periodic_distance'
    assert rows(out/'status.csv')[0]['status']=='error'
    assert json.loads((internal/'0.vasp.json').read_text())['delivered_pair_contacts']==audit
    assert_final_pdm(out,model)


def test_contacts_and_two_stages_real_core_fixed_total_budget(tmp_path):
    source,model,out,profile,args=setup_case(tmp_path)
    audit=tmp_path/'audit'
    code=cli.main(args+['--contact-weight','1','--topology-stage-steps','1','1','--audit-dir',str(audit)])
    assert code==1  # H-H lower bounds are not covered by this CH reference profile.
    assert_all_rows(out,('0.vasp',))
    evidence=json.loads((audit/'0.vasp.json').read_text())
    result=evidence['refinement']
    assert result['planned_total_steps']==2 and result['stage_steps']==[1,1]
    assert result['budget_respected'] is True and result['total_iterations']<=2
    assert len(result['stages'])==2
    assert result['contact_weight']==1 and result['source']=='md'
    assert rows(out/'geometry.csv')[0]['contact_coverage_complete']=='False'
    assert rows(out/'status.csv')[0]['status']!='refined'
    assert_final_pdm(out,model)


@pytest.mark.parametrize('change', ['excess_iterations','wrong_total','wrong_partition','diagnostic_errors'])
def test_inconsistent_wrapper_accounting_cannot_report_success(tmp_path,monkeypatch,change):
    source,model,out,profile,args=setup_case(tmp_path)
    def refine(*inputs,**kwargs):
        result=stage_stub(shifted=True)(*inputs,**kwargs)
        if change=='excess_iterations':result['total_iterations']=3
        elif change=='wrong_total':result['planned_total_steps']=3
        elif change=='wrong_partition':result['stage_steps']=[2]
        else:result['final_diagnostics']['errors']=[{'scope':'final_hard_pdm','error':'not-finite'}]
        return result
    monkeypatch.setattr('cdvae.common.pdm_topology_refresh.refine_pdm_with_topology_refresh',refine)
    assert cli.main(args+['--topology-stage-steps','1','1'])==1
    state=rows(out/'status.csv')[0]
    assert state['status']=='error' and state['coordinate_state']=='last_iterate'
    assert state['optimizer_status']=='stages_completed'
    assert_final_pdm(out,model)


def test_contact_violations_are_measured_on_delivered_coordinates(tmp_path,monkeypatch):
    source,model,out,profile,args=setup_case(tmp_path)
    structure=Poscar.from_file(source/'0.vasp',check_for_potcar=False).structure
    coords=torch.tensor(structure.frac_coords)
    coords[1]=coords[0]+torch.tensor([.004,.002,.001],dtype=coords.dtype)
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry',stub(status='stationary',coords=coords))
    assert cli.main(args+['--contact-weight','1'])==1
    geometry=rows(out/'geometry.csv')[0]
    assert int(geometry['contact_violation_pair_count'])>=1
    assert geometry['contact_lower_bound_constraints_satisfied']=='False'
    assert geometry['minimum_separation_satisfied']=='False'
    assert_final_pdm(out,model)
