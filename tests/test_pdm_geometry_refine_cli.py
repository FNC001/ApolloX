"""Final-file geometry/PDM CLI, explicit profiles and retained attempt identities."""
import csv
import hashlib
import json

from ase import Atoms
import numpy as np
import pytest
import torch
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from apollox_common.pdm import build_schema
from cdvae.common.reference_geometry_profile import fit_reference_profile
from scripts.pdm_control import refine_geometry_structures as cli


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def put_rows(path,records):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def star(index=0):
    arms=np.array([[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0],[0,0,1.],[0,0,-1.]])
    offsets=np.array([[0,.8,-.4],[0,-.3,.2],[.4,0,-.7],[-.2,0,.6],[.7,-.5,0],[-.4,.3,0]])
    arms=arms*(1.+index*.01)+offsets*(index+1)*.006
    return Atoms(numbers=[6]+[1]*6,positions=np.concatenate([[[0.,0,0]],arms])+10.,cell=np.eye(3)*30,pbc=True)


def setup_case(tmp_path,names=('0.vasp',)):
    source,model,out=tmp_path/'input',tmp_path/'model',tmp_path/'final';source.mkdir();model.mkdir()
    profile=fit_reference_profile([{'id':str(i),'source':'md','split':'train','atoms':star(i)} for i in range(5)],source='md')
    profile_path=tmp_path/'profile.json';profile_path.write_text(json.dumps(profile))
    schema=build_schema(['C','H'],['pair'],3.)
    (model/'pdm_schema.json').write_text(json.dumps(schema))
    (model/'scaler_stats.txt').write_text('Train Mean: 0,0,0\nTrain Std: 1,1,1\n')
    atoms=star(2);structure=Structure(atoms.cell.array,atoms.get_chemical_symbols(),atoms.get_scaled_positions())
    for name in names:Poscar(structure).write_file(source/name)
    put_rows(source/'structure_manifest.csv',[{'filename':name,'material_id':'00017','source':'md','sample_index':'02'} for name in names])
    target=tmp_path/'target.json';target.write_text(json.dumps([cli.hard_pdm(structure,schema)[key] for key in schema['features']]))
    config=tmp_path/'config.json';config.write_text('{"steps":2}')
    args=['--input-dir',str(source),'--output-dir',str(out),'--model-path',str(model),'--target',str(target),
          '--config',str(config),'--profile',str(profile_path),'--profile-sha256',cli.sha(profile_path),
          '--source','md','--bond-weight','5','--angle-weight','5']
    return source,model,out,profile_path,args


def update_profile(path,args,change):
    profile=json.loads(path.read_text());change(profile);path.write_text(json.dumps(profile))
    args[args.index('--profile-sha256')+1]=cli.sha(path)


def assert_all_rows(out,names):
    assert [r['label'] for r in rows(out/'pdm.csv')]==list(names)
    assert [r['filename'] for r in rows(out/'geometry.csv')]==list(names)
    assert [r['filename'] for r in rows(out/'status.csv')]==list(names)
    assert [r['filename'] for r in rows(out/'gen/structure_manifest.csv')]==list(names)
    assert {p.name for p in (out/'gen').glob('*.vasp')}==set(names)


def test_real_cli_exports_final_geometry_pdm_and_all_identities(tmp_path):
    source,model,out,profile,args=setup_case(tmp_path,('0.vasp','1.vasp'))
    before={p.name:p.read_bytes() for p in source.iterdir()}
    assert cli.main(args)==0
    assert {p.name for p in out.iterdir()}=={'gen','pdm.csv','geometry.csv','status.csv','run.json'}
    assert_all_rows(out,('0.vasp','1.vasp'))
    schema=json.loads((model/'pdm_schema.json').read_text())
    for pdm,state,geometry in zip(rows(out/'pdm.csv'),rows(out/'status.csv'),rows(out/'geometry.csv')):
        structure=Poscar.from_file(out/'gen'/pdm['label'],check_for_potcar=False).structure
        assert [int(pdm[k]) for k in schema['features']]==list(cli.hard_pdm(structure,schema).values())
        assert state['status']=='refined' and pdm['pdm_status']=='computed'
        assert geometry['reference_coverage_complete']=='True'
        assert pdm['material_id']==geometry['material_id']==state['material_id']=='00017'
        assert state['input_sha256']==hashlib.sha256(before[pdm['label']]).hexdigest()
        assert state['output_sha256']==cli.sha(out/'gen'/pdm['label'])
    assert before=={p.name:p.read_bytes() for p in source.iterdir()}
    content=''.join((out/name).read_text() for name in ('pdm.csv','geometry.csv','status.csv','run.json'))
    assert all(key not in content for key in ('initial_components','history','input_frac_coords','initial_hard'))


@pytest.mark.parametrize('bad',['sha','source','nan_weight','zero_weight','missing_sha','audit_nested','existing_output'])
def test_invalid_global_config_rejected_before_outputs(tmp_path,bad):
    source,model,out,profile,args=setup_case(tmp_path)
    if bad=='sha':args[args.index('--profile-sha256')+1]='0'*64
    elif bad=='source':args[args.index('--source')+1]='legacy'
    elif bad=='nan_weight':args[args.index('--bond-weight')+1]='nan'
    elif bad=='zero_weight':args[args.index('--angle-weight')+1]='0'
    elif bad=='missing_sha':args[args.index('--profile-sha256')+1]=''
    elif bad=='audit_nested':args+=['--audit-dir',str(out/'audit')]
    else:out.mkdir();(out/'sentinel').write_text('keep')
    assert cli.main(args)==1
    if bad=='existing_output':assert (out/'sentinel').read_text()=='keep'
    else:assert not out.exists()


def test_bad_file_retained_with_empty_metric_rows_and_other_input_processed(tmp_path):
    source,model,out,profile,args=setup_case(tmp_path,('0.vasp','1.vasp'))
    (source/'0.vasp').write_text('malformed POSCAR')
    assert cli.main(args)==1
    assert_all_rows(out,('0.vasp','1.vasp'))
    assert (out/'gen/0.vasp').read_bytes()==(source/'0.vasp').read_bytes()
    assert [r['status'] for r in rows(out/'status.csv')]==['error','refined']
    assert rows(out/'pdm.csv')[0]['pdm_status']=='unavailable'
    assert rows(out/'pdm.csv')[0]['CC']==''
    assert rows(out/'geometry.csv')[0]['declared_intervals_satisfied']==''
    assert json.loads((out/'run.json').read_text())['retained_files']==2


def test_missing_reference_retains_original_and_never_claims_geometry_pass(tmp_path):
    source,model,out,profile,args=setup_case(tmp_path)
    update_profile(profile,args,lambda p:p.update(bond_groups={},angle_groups={}))
    assert cli.main(args)==1
    assert_all_rows(out,('0.vasp',))
    assert (out/'gen/0.vasp').read_bytes()==(source/'0.vasp').read_bytes()
    state=rows(out/'status.csv')[0];geometry=rows(out/'geometry.csv')[0]
    assert state['status']=='missing_local_constraints' and state['coordinate_state']=='input_retained'
    assert geometry['reference_coverage_complete']=='False' and geometry['covered_bonds']=='0'
    assert geometry['declared_intervals_satisfied']=='' and geometry['geometry_status']=='no_declared_constraints'


def test_partial_angle_coverage_optimizes_declared_subset_without_claiming_complete(tmp_path):
    source,model,out,profile,args=setup_case(tmp_path)
    update_profile(profile,args,lambda p:p.update(angle_groups={}))
    assert cli.main(args)==1
    state=rows(out/'status.csv')[0];geometry=rows(out/'geometry.csv')[0]
    assert state['status']=='partial_reference_coverage' and state['coordinate_state']=='optimized'
    assert geometry['reference_coverage_complete']=='False' and geometry['covered_angles']=='0'
    assert len(json.loads(geometry['omitted_angles']))==15
    assert int(geometry['covered_bonds'])==6


def test_unsupported_periodic_topology_retains_original(tmp_path):
    source,model,out,profile,args=setup_case(tmp_path)
    Poscar(Structure(np.eye(3)*.5,['C'],[[0,0,0]])).write_file(source/'0.vasp')
    assert cli.main(args)==1
    assert (out/'gen/0.vasp').read_bytes()==(source/'0.vasp').read_bytes()
    assert rows(out/'status.csv')[0]['status']=='unsupported_periodic_topology'
    geometry=rows(out/'geometry.csv')[0]
    assert geometry['reference_coverage_complete']=='False'
    assert geometry['total_bonds']=='' and geometry['declared_intervals_satisfied']==''
    assert geometry['minimum_distance_A']=='' and geometry['minimum_separation_satisfied']==''


def stub(status='nonfinite_gradient',coords=None,seen=None):
    def refine(frac,types,counts,cells,target,std,schema,config,**kwargs):
        if seen is not None:seen.append(target.tolist())
        return {'frac_coords':frac.clone() if coords is None else coords.to(frac),'atom_types':types.clone(),
                'cell_matrices':cells.clone(),'reports':[{'status':status,'error':None,'history':[{'iteration':0}]}],
                'final_hard_raw':torch.tensor([[999.,999.,999.]])}
    return refine


def test_failed_core_last_iterate_is_exported_and_metrics_recomputed(tmp_path,monkeypatch):
    source,model,out,profile,args=setup_case(tmp_path)
    original=Poscar.from_file(source/'0.vasp',check_for_potcar=False).structure
    coords=torch.tensor(original.frac_coords)+.01
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry',stub(coords=coords))
    assert cli.main(args)==1
    assert_all_rows(out,('0.vasp',))
    state=rows(out/'status.csv')[0]
    assert state['status']=='error' and state['coordinate_state']=='last_iterate' and state['optimizer_status']=='nonfinite_gradient'
    delivered=Poscar.from_file(out/'gen/0.vasp',check_for_potcar=False).structure
    np.testing.assert_allclose(delivered.frac_coords,coords.numpy(),atol=1e-14)
    assert all(rows(out/'pdm.csv')[0][k]!='999' for k in ('CC','CH','HH'))


def test_invalid_core_output_retains_input_and_full_separate_audit(tmp_path,monkeypatch):
    source,model,out,profile,args=setup_case(tmp_path)
    coords=torch.full((7,3),float('nan'))
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry',stub(coords=coords))
    audit=tmp_path/'audit'
    assert cli.main(args+['--audit-dir',str(audit)])==1
    assert (out/'gen/0.vasp').read_bytes()==(source/'0.vasp').read_bytes()
    evidence=json.loads((audit/'0.vasp.json').read_text())
    assert evidence['refinement']['frac_coords'][0][0]=={'nonfinite':'nan'}
    assert evidence['status']['status']=='error' and 'input_frac_coords' in evidence
    assert evidence['refinement']['reports'][0]['history']


def test_standardized_target_transformed_once_and_profile_fixed(tmp_path,monkeypatch):
    source,model,out,profile,args=setup_case(tmp_path)
    target=tmp_path/'target.json';target.write_text('[0,3,7.5]')
    (model/'scaler_stats.txt').write_text('Train Mean: 0,0,0\nTrain Std: 2,2,2\n')
    seen=[];before=profile.read_bytes()
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry',stub(status='stationary',seen=seen))
    assert cli.main(args+['--target-space','standardized'])==0
    assert seen==[[[0.,6.,15.]]] and profile.read_bytes()==before


@pytest.mark.parametrize('change',['extra_bond','isolated_atom'])
def test_final_new_neighbor_coverage_and_violations_are_diagnosed_without_reoptimization(tmp_path,monkeypatch,change):
    source,model,out,profile,args=setup_case(tmp_path)
    structure=Poscar.from_file(source/'0.vasp',check_for_potcar=False).structure
    coords=torch.tensor(structure.frac_coords)
    if change=='extra_bond':coords[2]=coords[1]+torch.tensor([0.,.005,0.])
    else:coords[2]=torch.tensor([.9,.9,.9])
    seen=[]
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry',stub(status='stationary',coords=coords,seen=seen))
    original_profile=profile.read_bytes()
    assert cli.main(args)==1
    assert len(seen)==1 and profile.read_bytes()==original_profile
    row=rows(out/'geometry.csv')[0]
    assert row['reference_coverage_complete']=='True'
    assert row['final_topology_coverage_complete']=='False'
    assert row['topology_status']=='changed_site_topology'
    assert row['covered_bonds']=='6'
    assert rows(out/'status.csv')[0]['status']=='partial_final_reference_coverage'
    if change=='extra_bond':
        assert row['final_topology_total_bonds']=='7'
        assert row['final_topology_covered_bonds']=='6'
        assert any(item['key']=='H|H' for item in json.loads(row['final_topology_missing_bonds']))
        assert int(row['angle_violation_count'])>0
        assert row['minimum_separation_satisfied']=='False'
    else:
        assert row['final_topology_total_bonds']=='5'
        assert row['final_topology_zero_coordination_count']=='1'
        assert int(row['bond_violation_count'])>0


@pytest.mark.parametrize('explicit',[None,False,True])
def test_new_cli_defaults_to_preserve_feasible_geometry_but_records_explicit_setting(tmp_path,explicit):
    source,model,out,profile,args=setup_case(tmp_path)
    config={'steps':1}
    if explicit is not None:config['preserve_feasible_geometry']=explicit
    (tmp_path/'config.json').write_text(json.dumps(config))
    assert cli.main(args)==0
    assert json.loads((out/'run.json').read_text())['config']['preserve_feasible_geometry']==(True if explicit is None else explicit)


def test_per_input_copy_failure_keeps_status_denominator_and_processes_remaining(tmp_path,monkeypatch):
    source,model,out,profile,args=setup_case(tmp_path,('0.vasp','1.vasp'))
    original_copy=cli.shutil.copyfile
    def copy(src,dst,*a,**kw):
        if str(src)==str(source/'0.vasp'):raise OSError('simulated unreadable input')
        return original_copy(src,dst,*a,**kw)
    monkeypatch.setattr(cli.shutil,'copyfile',copy)
    assert cli.main(args)==1
    assert [r['label'] for r in rows(out/'pdm.csv')]==['0.vasp','1.vasp']
    states=rows(out/'status.csv')
    assert [r['status'] for r in states]==['error','refined']
    assert states[0]['coordinate_state']=='unavailable' and states[0]['output_sha256']==''
    assert 'simulated unreadable input' in states[0]['error']
    assert len(rows(out/'geometry.csv'))==2
    meta=json.loads((out/'run.json').read_text())
    assert meta['planned']==2 and meta['retained_files']==1 and meta['errors']==1


def test_output_checksum_read_failure_preserves_all_attempt_rows(tmp_path,monkeypatch):
    source,model,out,profile,args=setup_case(tmp_path,('0.vasp','1.vasp'))
    original_sha=cli.sha
    def checksum(path):
        if str(path)==str(out/'gen/0.vasp'):raise OSError('simulated output checksum read failure')
        return original_sha(path)
    monkeypatch.setattr(cli,'sha',checksum)
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry',stub(status='stationary'))
    assert cli.main(args)==1
    assert_all_rows(out,('0.vasp','1.vasp'))
    states=rows(out/'status.csv')
    assert [row['status'] for row in states]==['error','refined']
    assert states[0]['output_sha256']=='' and 'output_sha256_error' in states[0]['error']
    assert states[1]['output_sha256']


def test_audit_write_failure_reported_without_losing_remaining_attempt_rows(tmp_path,monkeypatch):
    source,model,out,profile,args=setup_case(tmp_path,('0.vasp','1.vasp'))
    audit=tmp_path/'audit';original_write=cli.write_json
    def write(path,value):
        if path==audit/'0.vasp.json':raise OSError('simulated audit write failure')
        return original_write(path,value)
    monkeypatch.setattr(cli,'write_json',write)
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry',stub(status='stationary'))
    assert cli.main(args+['--audit-dir',str(audit)])==1
    assert_all_rows(out,('0.vasp','1.vasp'))
    states=rows(out/'status.csv')
    assert [row['status'] for row in states]==['error','refined']
    assert [row['audit_status'] for row in states]==['error','written']
    assert 'simulated audit write failure' in states[0]['audit_error']
    assert json.loads((audit/'1.vasp.json').read_text())['status']['audit_status']=='written'


def test_failed_export_and_cleanup_keep_original_and_record_temporary_evidence(tmp_path,monkeypatch):
    from pathlib import Path
    source,model,out,profile,args=setup_case(tmp_path,('0.vasp','1.vasp'))
    original_replace,original_unlink=cli.os.replace,Path.unlink
    def replace(src,dst):
        if dst==out/'gen/0.vasp':raise OSError('simulated export failure')
        return original_replace(src,dst)
    def unlink(path,*a,**kw):
        if path==out/'gen/.0.vasp.tmp':raise OSError('simulated cleanup failure')
        return original_unlink(path,*a,**kw)
    monkeypatch.setattr(cli.os,'replace',replace);monkeypatch.setattr(Path,'unlink',unlink)
    monkeypatch.setattr('cdvae.common.pdm_geometry_guidance.refine_pdm_with_local_geometry',stub(status='stationary'))
    assert cli.main(args)==1
    assert_all_rows(out,('0.vasp','1.vasp'))
    assert (out/'gen/0.vasp').read_bytes()==(source/'0.vasp').read_bytes()
    assert (out/'gen/.0.vasp.tmp').exists()
    states=rows(out/'status.csv')
    assert [row['status'] for row in states]==['error','refined']
    assert 'temporary_file_retained' in states[0]['error']


def test_single_site_without_reference_neighbors_has_unknown_pair_separation(tmp_path):
    source,model,out,profile,args=setup_case(tmp_path)
    Poscar(Structure(np.eye(3)*30,['C'],[[0,0,0]])).write_file(source/'0.vasp')
    assert cli.main(args)==1
    geometry=rows(out/'geometry.csv')[0]
    assert geometry['minimum_distance_A']=='' and geometry['minimum_separation_satisfied']==''
    assert geometry['declared_intervals_satisfied']==''
    assert geometry['final_topology_intervals_satisfied']==''
    assert geometry['final_topology_coverage_complete']=='False'
    assert json.loads((out/'run.json').read_text())['minimum_distance_scope']=='distinct_site_minimum_image_pairs'
