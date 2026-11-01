"""Train-only profile fitting, explicit coverage and representable topology."""
import copy
import hashlib
import io
import json

from ase import Atoms
from ase.data import covalent_radii
from ase.io import read, write
import numpy as np
import pytest

from cdvae.common.reference_geometry_profile import (
    POLICY, fit_reference_profile, build_local_constraints, audit_frozen_topology,
    load_reference_profile,
)


def star(index=0,coordination=6):
    arms=np.array([[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0],[0,0,1.],[0,0,-1.]])[:coordination]
    t=(index+1)*.006
    offsets=np.array([[0,.8,-.4],[0,-.3,.2],[.4,0,-.7],[-.2,0,.6],[.7,-.5,0],[-.4,.3,0]])[:coordination]
    arms=arms*(1.+index*.01)+offsets*t
    positions=np.concatenate([[[0.,0.,0.]],arms])+10.
    return Atoms(numbers=[6]+[1]*coordination,positions=positions,cell=np.eye(3)*30,pbc=True)


def references(n=5,source='md',coordination=6,identical=False):
    return [{'id':f'train-{i}','source':source,'split':'train','atoms':star(0 if identical else i,coordination)} for i in range(n)]


def profile():return fit_reference_profile(references(),source='md')


def test_fixed_quantiles_support_counts_and_reference_provenance():
    refs=references();result=fit_reference_profile(refs,source='md')
    assert result['reference_count']==result['usable_reference_count']==5
    assert result['unsupported_reference_count']==0
    bond=result['bond_groups']['C|H'];angle=result['angle_groups']['C|H|H|CN=6']
    assert bond['covered'] and bond['observation_count']==30 and bond['structure_count']==5
    assert angle['covered'] and angle['observation_count']==75 and angle['structure_count']==5
    expected=[]
    for record in refs:
        expected.extend(record['atoms'].get_distances(0,range(1,7),mic=True).tolist())
    assert bond['bounds']==pytest.approx(np.quantile(expected,[.01,.99],method='linear'))
    assert result['policy']==POLICY
    assert result['reference_ids']==[r['id'] for r in refs]
    assert all(r['status']=='used' for r in result['references'])
    json.dumps(result,allow_nan=False)


def test_profile_policy_return_does_not_alias_module_constants():
    a=profile();other=profile();before=copy.deepcopy(other)
    a['policy']['quantiles'][0]=.3
    assert POLICY['quantiles']==[.01,.99]
    assert profile()['policy']['quantiles']==[.01,.99]
    assert other==before


def test_profile_receipts_and_groups_are_isolated_from_inputs_and_other_profiles():
    refs=references();positions=[r['atoms'].positions.copy() for r in refs]
    first=fit_reference_profile(refs,source='md');second=fit_reference_profile(refs,source='md')
    before=copy.deepcopy(second)
    first['references'][0]['coordination_numbers'][0]=999
    first['references'][0]['zero_coordination_atoms'].append(999)
    first['reference_ids'][0]='changed'
    group=first['bond_groups']['C|H'];quantiles=list(group['observed_quantiles'])
    group['structure_ids'][0]='changed';group['bounds'][0]=.9
    assert group['observed_quantiles']==quantiles
    assert second==before and fit_reference_profile(refs,source='md')==before
    assert refs[0]['id']=='train-0'
    assert all(np.array_equal(r['atoms'].positions,p) for r,p in zip(refs,positions))


def test_constraint_and_topology_audit_outputs_do_not_alias_caller_profile_or_topology():
    fitted=profile();original=copy.deepcopy(fitted)
    atoms=star();built=build_local_constraints(atoms,fitted,source='md',structure_id='g')
    built['constraints']['bond_bounds_A'][0][0]=.7
    built['constraints']['angle_bounds_deg'][0][0]=70.
    assert fitted==original
    frozen=copy.deepcopy(built['frozen_topology']);before=copy.deepcopy(frozen)
    audit=audit_frozen_topology(atoms,frozen)
    audit['original_coordination_numbers'][0]=999
    audit['original_zero_coordination_atoms'].append(999)
    assert frozen==before


@pytest.mark.parametrize('bad',['validation','test','other_source','duplicate_id','missing_source','missing_split'])
def test_nontraining_or_ambiguous_records_are_rejected(bad):
    refs=references()
    if bad in ('validation','test'):refs[0]['split']=bad
    elif bad=='other_source':refs[0]['source']='other'
    elif bad=='duplicate_id':refs[1]['id']=refs[0]['id']
    elif bad=='missing_source':del refs[0]['source']
    else:del refs[0]['split']
    with pytest.raises(ValueError):fit_reference_profile(refs,source='md')


def test_insufficient_groups_remain_uncovered_without_fallback():
    few=fit_reference_profile(references(4),source='md')
    assert not few['bond_groups']['C|H']['covered']
    assert 'insufficient_structures' in few['bond_groups']['C|H']['reasons']
    sparse=fit_reference_profile(references(5,coordination=3),source='md')
    assert sparse['bond_groups']['C|H']['structure_count']==5
    assert sparse['bond_groups']['C|H']['observation_count']==15
    assert sparse['bond_groups']['C|H']['reasons']==['insufficient_observations']
    assert sparse['bond_groups']['C|H']['bounds'] is None
    built=build_local_constraints(star(coordination=3),sparse,source='md',structure_id='g')
    assert built['status']=='missing_local_constraints' and not built['has_constraints']
    assert built['coverage']['total_bonds']==3 and built['coverage']['total_angles']==3
    assert len(built['missing_bonds'])==3 and len(built['omitted_angles'])==3


def test_degenerate_reference_group_is_not_converted_to_an_ideal_interval():
    refs=[]
    for i in range(5):
        atoms=star(0)
        positions=atoms.positions.copy();vectors=positions[1:]-positions[0]
        positions[1:]=positions[0]+vectors/np.linalg.norm(vectors,axis=1)[:,None]
        # Explicit axis-aligned coordinates give exactly repeated lengths.
        positions[1:]=positions[0]+np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
        atoms.positions=positions;refs.append({'id':str(i),'source':'md','split':'train','atoms':atoms})
    fitted=fit_reference_profile(refs,source='md')
    assert fitted['bond_groups']['C|H']['observed_quantiles']==[1.,1.]
    assert not fitted['bond_groups']['C|H']['covered']
    assert fitted['bond_groups']['C|H']['bounds'] is None
    assert 'degenerate_interval' in fitted['bond_groups']['C|H']['reasons']


def test_complete_profile_builds_all_bonds_angles_and_freezes_initial_topology():
    atoms=star(2);fitted=profile();before=atoms.copy()
    built=build_local_constraints(atoms,fitted,source='md',structure_id='generated-1')
    assert built['status']=='reference_coverage_complete' and built['coverage_complete']
    assert built['coverage']=={'total_bonds':6,'covered_bonds':6,'total_angles':15,'covered_angles':15,
                              'site_count':7,'zero_coordination_count':0}
    assert built['frozen_topology']['coordination_numbers']==[6,1,1,1,1,1,1]
    assert len(built['constraints']['bond_pairs'])==6 and len(built['constraints']['angle_triples'])==15
    assert all(t[1]==0 for t in built['constraints']['angle_triples'])
    assert np.array_equal(atoms.positions,before.positions) and np.array_equal(atoms.cell,before.cell)
    json.dumps(built,allow_nan=False)


def test_coordination_specific_missing_angles_keep_supported_bonds():
    built=build_local_constraints(star(2,coordination=5),profile(),source='md',structure_id='different-cn')
    assert built['status']=='partial_reference_coverage' and not built['coverage_complete']
    assert built['coverage']['covered_bonds']==5 and built['coverage']['total_angles']==10
    assert built['coverage']['covered_angles']==0
    assert all(row['key']=='C|H|H|CN=5' for row in built['omitted_angles'])
    assert all(row['reasons']==['missing_reference_group'] for row in built['omitted_angles'])


def test_angle_group_is_omitted_when_either_bond_arm_lacks_reference_bounds():
    fitted=profile();fitted['bond_groups']={}
    built=build_local_constraints(star(),fitted,source='md',structure_id='no-arms')
    assert built['coverage']['total_angles']==15 and built['coverage']['covered_angles']==0
    assert all(row['omitted_angle_reason']=='unsupported_bond_arm' for row in built['omitted_angles'])
    assert all(row['reasons']==['unsupported_bond_arm'] for row in built['omitted_angles'])
    assert all(len(row['unsupported_bond_arms'])==2 for row in built['omitted_angles'])
    assert built['status']=='missing_local_constraints'


def test_zero_coordination_is_reported_even_with_other_covered_constraints():
    atoms=star();atoms+=Atoms('He',positions=[[20,20,20]])
    built=build_local_constraints(atoms,profile(),source='md',structure_id='isolated')
    assert built['status']=='partial_reference_coverage'
    assert built['zero_coordination_atoms']==[7]
    assert built['coverage']['zero_coordination_count']==1
    assert built['coverage']['covered_bonds']==6


def test_self_periodic_image_and_multiple_site_images_are_explicitly_unsupported():
    fitted=profile()
    self_image=Atoms('C',positions=[[0,0,0]],cell=np.eye(3)*1.5,pbc=True)
    duplicate=Atoms('CC',positions=[[0,0,0],[1.5,0,0]],cell=np.eye(3)*3,pbc=True)
    for atoms,reason in [(self_image,'self_periodic_images'),(duplicate,'multiple_periodic_images_for_site_pair')]:
        built=build_local_constraints(atoms,fitted,source='md',structure_id=reason)
        assert built['status']=='unsupported_periodic_topology' and built['reason']==reason
        assert not built['has_constraints'] and built['frozen_topology'] is None
        assert built['coverage']['total_bonds'] is None
        assert built['constraints']=={'bond_pairs':[],'bond_bounds_A':[],'angle_triples':[],'angle_bounds_deg':[]}


def test_unsupported_and_isolated_fit_structures_retain_reference_denominators():
    refs=references()
    refs += [{'id':'periodic-self','source':'md','split':'train','atoms':Atoms('C',positions=[[0,0,0]],cell=np.eye(3)*1.5,pbc=True)},
             {'id':'alone','source':'md','split':'train','atoms':Atoms('He',positions=[[0,0,0]],cell=np.eye(3)*30,pbc=True)}]
    fitted=fit_reference_profile(refs,source='md')
    assert fitted['reference_count']==7 and fitted['usable_reference_count']==5
    assert fitted['unsupported_reference_count']==fitted['no_local_observation_reference_count']==1
    assert len(fitted['references'])==7
    assert fitted['references'][-2]['status']=='unsupported'
    assert fitted['references'][-1]['status']=='no_local_observations'
    assert fitted['references'][-1]['zero_coordination_atoms']==[0]


def test_cutoff_is_pair_sum_of_declared_ase_covalent_radii():
    fitted=profile();cutoff=1.2*(covalent_radii[6]+covalent_radii[1])
    atoms=Atoms('CH',positions=[[10,10,10],[10+cutoff-.001,10,10]],cell=np.eye(3)*30,pbc=True)
    inside=build_local_constraints(atoms,fitted,source='md',structure_id='inside')
    atoms.positions[1,0]=10+cutoff+.001
    outside=build_local_constraints(atoms,fitted,source='md',structure_id='outside')
    assert inside['coverage']['total_bonds']==1 and outside['coverage']['total_bonds']==0
    assert outside['zero_coordination_atoms']==[0,1]


def test_post_refinement_audit_reports_changes_without_replacing_constraints():
    atoms=star();built=build_local_constraints(atoms,profile(),source='md',structure_id='g')
    frozen=copy.deepcopy(built['frozen_topology']);constraints=copy.deepcopy(built['constraints'])
    unchanged=audit_frozen_topology(atoms,frozen)
    assert unchanged['status']=='unchanged_site_topology' and unchanged['fixed_inputs_match']
    atoms.positions[1]=[20,20,20]
    changed=audit_frozen_topology(atoms,frozen)
    assert changed['status']=='changed_site_topology'
    assert changed['removed_bonds']==[[0,1]] and len(changed['removed_angles'])==5
    assert changed['current_coordination_numbers'][0]==5
    assert not changed['constraints_replaced'] and frozen==built['frozen_topology'] and constraints==built['constraints']
    atoms.set_cell(np.eye(3)*31)
    assert audit_frozen_topology(atoms,frozen)['status']=='fixed_inputs_changed'


def test_periodic_wrapping_does_not_change_unique_site_topology():
    atoms=star();built=build_local_constraints(atoms,profile(),source='md',structure_id='g')
    atoms.positions[1]+=atoms.cell[0];atoms.positions[2]-=atoms.cell[1]
    audit=audit_frozen_topology(atoms,built['frozen_topology'])
    assert audit['status']=='unchanged_site_topology'
    assert audit['added_bonds']==audit['removed_bonds']==[]


def test_skew_vasp_roundtrip_records_exact_cell_difference_with_readback_tolerance():
    atoms=star(2)
    atoms.set_cell([[30.,1.2345678912345678e-6,0.],
                    [.3141592653589793,30.,1.777771234567891e-6],
                    [.2718,.14142,30.]])
    built=build_local_constraints(atoms,profile(),source='md',structure_id='skew-roundtrip')
    original=copy.deepcopy(built)
    stream=io.StringIO();write(stream,atoms,format='vasp',direct=True)
    stream.seek(0);restored=read(stream,format='vasp')
    audit=audit_frozen_topology(restored,built['frozen_topology'])
    assert audit['fixed_inputs_match'] and audit['status']=='unchanged_site_topology'
    assert not audit['cell_exact_match']
    assert 0.<audit['cell_max_abs_difference_A']<1e-10
    assert audit['cell_abs_tolerance_A']==1e-10 and audit['cell_relative_tolerance']==0.
    assert built==original
    restored.cell[0,1]+=2e-10
    changed=audit_frozen_topology(restored,built['frozen_topology'])
    assert not changed['fixed_inputs_match'] and changed['status']=='fixed_inputs_changed'
    assert changed['cell_max_abs_difference_A']>changed['cell_abs_tolerance_A']
    assert built==original


def test_skew_cell_periodic_vectors_preserve_reference_bond_lengths_and_angles():
    atoms=star(2)
    atoms.set_cell([[30.,0,0],[12.,27.,0],[5.,4.,29.]])
    built=build_local_constraints(atoms,profile(),source='md',structure_id='skew')
    assert built['coverage_complete']
    atoms.positions[1]+=atoms.cell[1]
    atoms.positions[2]-=atoms.cell[2]
    wrapped=build_local_constraints(atoms,profile(),source='md',structure_id='skew-wrapped')
    assert wrapped['coverage_complete']
    assert [x['length_A'] for x in wrapped['frozen_topology']['bonds']]==pytest.approx(
        [x['length_A'] for x in built['frozen_topology']['bonds']],abs=1e-12)
    assert [x['angle_deg'] for x in wrapped['frozen_topology']['angles']]==pytest.approx(
        [x['angle_deg'] for x in built['frozen_topology']['angles']],abs=1e-10)


def test_empty_reference_profile_preserves_generated_constraint_denominators():
    fitted=fit_reference_profile([],source='md')
    assert fitted['reference_count']==0 and fitted['bond_groups']==fitted['angle_groups']=={}
    built=build_local_constraints(star(),fitted,source='md',structure_id='uncovered')
    assert built['status']=='missing_local_constraints'
    assert built['coverage']['total_bonds']==6 and built['coverage']['total_angles']==15
    assert len(built['missing_bonds'])==6 and len(built['omitted_angles'])==15


def test_fit_and_binding_preserve_numpy_rng_and_reference_inputs():
    refs=references();before=[r['atoms'].positions.copy() for r in refs]
    rng=np.random.get_state()
    fitted=fit_reference_profile(refs,source='md')
    first=build_local_constraints(star(),fitted,source='md',structure_id='g')
    second=build_local_constraints(star(),fitted,source='md',structure_id='g')
    after=np.random.get_state()
    assert first==second and fitted==fit_reference_profile(refs,source='md')
    assert rng[0]==after[0] and np.array_equal(rng[1],after[1]) and rng[2:]==after[2:]
    assert all(np.array_equal(r['atoms'].positions,b) for r,b in zip(refs,before))


def test_loading_requires_file_digest_and_matching_source(tmp_path):
    fitted=profile();path=tmp_path/'profile.json'
    content=json.dumps(fitted,allow_nan=False,sort_keys=True).encode()
    path.write_bytes(content);digest=hashlib.sha256(content).hexdigest()
    assert load_reference_profile(path,source='md',expected_sha256=digest)==fitted
    with pytest.raises(ValueError,match='matching same-source'):
        load_reference_profile(path,source='other',expected_sha256=digest)
    path.write_bytes(content+b'\n')
    with pytest.raises(ValueError,match='SHA256 does not match'):
        load_reference_profile(path,source='md',expected_sha256=digest)


@pytest.mark.parametrize('digest',[None,True,'', '0'*63, 'G'*64,'F'*64])
def test_missing_or_invalid_frozen_digest_is_rejected_before_file_read(tmp_path,digest):
    with pytest.raises(ValueError,match='explicit lowercase SHA256'):
        load_reference_profile(tmp_path/'does-not-exist.json',source='md',expected_sha256=digest)


@pytest.mark.parametrize('bad',['source','nonpositive','degenerate','boolean','too_few_structures','policy',
                               'malformed_group','missing_reasons','contradictory_coverage'])
def test_invalid_profile_cannot_supply_bounds(bad):
    fitted=profile();source='md'
    group=fitted['bond_groups']['C|H']
    if bad=='source':source='other'
    elif bad=='nonpositive':group['bounds']=[0.,1.]
    elif bad=='degenerate':group['bounds']=[1.,1.]
    elif bad=='boolean':group['bounds']=[True,2.]
    elif bad=='too_few_structures':group['structure_count']=4
    elif bad=='policy':fitted['policy']['quantiles']=[.05,.95]
    elif bad=='malformed_group':fitted['bond_groups']['C|H']=None
    elif bad=='missing_reasons':del group['reasons']
    else:group['covered']=False
    with pytest.raises(ValueError):build_local_constraints(star(),fitted,source=source,structure_id='bad')
