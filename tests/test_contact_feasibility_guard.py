"""Covered-contact guard checks; numerical execution belongs on the remote host."""
import copy
import json

from ase import Atoms
import numpy as np
import pytest
import torch

import cdvae.common.pdm_geometry_guidance as core
import cdvae.common.pdm_topology_refresh as staged
from cdvae.common.profile_pair_contacts import profile_pair_lower_bound_penalty
from cdvae.common.reference_geometry_profile import POLICY
from test_pdm_geometry_contacts import contacts, reference
from test_pdm_geometry_guidance import example, run_case
from test_pdm_topology_refresh import case, fake_result


SCHEMA = {'schema_version': 2, 'geometry': 'periodic-minimum-image-clique-v2',
          'modes': ['pair'], 'elements': ['C'], 'features': ['CC'], 'cutoff': 3.}
EMPTY = dict(bond_pairs=[], bond_bounds_A=[], angle_triples=[], angle_bounds_deg=[])


def pair_case(distance=2., dtype=torch.float64):
    frac = torch.tensor([[0., 0., 0.], [distance/8, 0., 0.]], dtype=dtype)
    return frac, torch.tensor([6, 6]), torch.tensor([2]), torch.eye(3, dtype=dtype)[None]*8


def run_pair(args, *, guard=True, steps=2, max_step=.04, profile=None):
    frac, types, counts, cells = args
    return core.refine_pdm_with_local_geometry(frac, types, counts, cells, [[1.]], [1.], SCHEMA,
        dict(steps=steps, line_search_steps=6, max_step_A=max_step, temperature=.1),
        local_constraints=EMPTY, bond_weight=1., angle_weight=1.,
        contact_profile=profile or reference(), contact_source='md', contact_weight=1.,
        preserve_feasible_contacts=guard)


def drive_inward(monkeypatch):
    def objective(coords, *args):
        loss = coords[1, 0]
        return loss, loss, coords.sum()*0
    monkeypatch.setattr(core, '_objective', objective)


def zero_contact_loss_with_real_audit(monkeypatch):
    def helper(coords, *args, **kwargs):
        value = profile_pair_lower_bound_penalty(coords, *args, **kwargs)
        value['loss'] = coords.sum()*0
        return value
    monkeypatch.setattr(core, 'profile_pair_lower_bound_penalty', helper)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_strict_boundary_blocks_inward_motion_at_actual_return_dtype(monkeypatch, dtype):
    drive_inward(monkeypatch)
    args = pair_case(dtype=dtype)
    guarded = run_pair(args)
    plain = run_pair(args, guard=False)
    report = guarded['reports'][0]
    assert report['status'] == 'line_search_stalled'
    assert torch.equal(guarded['frac_coords'], args[0])
    assert guarded['frac_coords'].dtype == dtype
    assert not torch.equal(plain['frac_coords'], args[0])
    guard = report['contact_feasibility_guard']
    assert guard['initial_feasible'] is True and guard['final_feasible'] is True
    assert guard['first_feasible_iteration'] == -1
    assert guard['rejected_proposals'] == 6
    for proposal in report['history'][0]['proposals']:
        assert proposal['reason'] == 'covered_contact_feasibility_regression'
        evidence = proposal['contact_guard_evidence']
        reconstructed = torch.tensor(evidence['proposal_frac_coords'], dtype=dtype)
        audit = profile_pair_lower_bound_penalty(reconstructed, *args[1:], reference(), source='md')['audit']
        assert audit['declared_lower_bounds_satisfied'] is False
        assert evidence['violated_pairs'][0]['distance_A'] < 2.
    json.dumps(report, allow_nan=False)


def test_feasibility_uses_distances_even_when_soft_contact_loss_is_zero(monkeypatch):
    drive_inward(monkeypatch); zero_contact_loss_with_real_audit(monkeypatch)
    result = run_pair(pair_case())
    report = result['reports'][0]
    assert report['contact_feasibility_guard']['rejected_proposals'] == 6
    assert all(p['contact_lower_bound_loss'] == 0 for p in report['history'][0]['proposals'])
    assert report['final_contact_audit']['declared_lower_bounds_satisfied'] is True


def test_initial_infeasible_can_enter_feasible_region_then_cannot_leave(monkeypatch):
    zero_contact_loss_with_real_audit(monkeypatch)
    def objective(coords, *args):
        loss = (coords[1, 0]-.25).square()
        return loss, loss, coords.sum()*0
    monkeypatch.setattr(core, '_objective', objective)
    result = run_pair(pair_case(1.96875), steps=3, max_step=.0625)
    report = result['reports'][0]; guard = report['contact_feasibility_guard']
    assert guard['initial_feasible'] is False and guard['ever_feasible'] is True
    assert guard['first_feasible_iteration'] == 0 and guard['rejected_proposals'] >= 1
    assert report['history'][0]['accepted'] is True
    assert report['history'][1]['proposals'][0]['reason'] == 'covered_contact_feasibility_regression'
    assert guard['final_feasible'] is True


def test_periodic_skew_cell_guard_agrees_with_independent_ase_mic(monkeypatch):
    # The initial 2 A distance crosses the periodic x boundary of a triclinic cell.
    cells = torch.tensor([[[8., 0., 0.], [1., 8., 0.], [.5, .25, 8.]]], dtype=torch.float64)
    frac = torch.tensor([[.875, .2, .2], [.125, .2, .2]], dtype=torch.float64)
    args = frac, torch.tensor([6, 6]), torch.tensor([2]), cells
    drive_inward(monkeypatch)
    result = run_pair(args)
    atoms = Atoms(numbers=[6, 6], scaled_positions=result['frac_coords'].numpy(), cell=cells[0].numpy(), pbc=True)
    assert atoms.get_distance(0, 1, mic=True) >= 2.
    assert result['reports'][0]['contact_feasibility_guard']['rejected_proposals'] > 0


def test_partial_coverage_guard_does_not_claim_all_geometry_pass(monkeypatch):
    drive_inward(monkeypatch)
    frac, types, counts, cells = pair_case()
    frac = torch.cat([frac, torch.tensor([[.5, .5, .5]], dtype=frac.dtype)])
    types = torch.tensor([6, 6, 8]); counts = torch.tensor([3])
    schema = dict(SCHEMA, elements=['C', 'O'], features=['CC', 'CO', 'OO'])
    result = core.refine_pdm_with_local_geometry(frac, types, counts, cells, [[1., 0., 0.]], [1.]*3, schema,
        dict(steps=2, line_search_steps=6, max_step_A=.04), local_constraints=EMPTY,
        bond_weight=1., angle_weight=1., contact_profile=reference(), contact_source='md',
        contact_weight=1., preserve_feasible_contacts=True)
    report = result['reports'][0]; guard = report['contact_feasibility_guard']
    assert guard['final_feasible'] is True and guard['coverage_complete'] is False
    assert guard['full_physical_validity_claimed'] is False
    assert report['final_contact_audit']['uncovered_pair_count'] == 2
    assert report['final_contact_audit']['lower_bound_constraints_satisfied'] is None


def test_no_covered_bounds_remains_unknown_and_retains_missing_status():
    profile = reference(); profile['bond_groups'] = {}
    result = run_pair(pair_case(), profile=profile)
    report = result['reports'][0]; guard = report['contact_feasibility_guard']
    assert report['status'] == 'missing_local_constraints'
    assert guard['initial_feasible'] is None and guard['final_feasible'] is None
    assert not guard['ever_feasible'] and guard['rejected_proposals'] == 0


def test_default_and_explicit_false_preserve_original_fields_and_results():
    for kwargs in ({}, contacts()):
        a = run_case(example(), steps=2, **kwargs)
        b = run_case(example(), steps=2, preserve_feasible_contacts=False, **kwargs)
        assert a.keys() == b.keys() and a['reports'] == b['reports']
        assert torch.equal(a['frac_coords'], b['frac_coords'])
        assert 'preserve_feasible_contacts' not in b
        assert 'contact_feasibility_guard' not in b['reports'][0]


@pytest.mark.parametrize('bad', [None, 0, 1, 'true', [], np.bool_(True)])
def test_guard_must_be_explicit_boolean(bad):
    with pytest.raises(ValueError, match='preserve_feasible_contacts'):
        run_case(example(), steps=1, preserve_feasible_contacts=bad, **contacts())


def test_guard_requires_full_contact_configuration():
    with pytest.raises(ValueError, match='requires'):
        run_case(example(), steps=1, preserve_feasible_contacts=True)


def test_multigraph_guard_rebases_contacts_and_matches_separate_results(monkeypatch):
    drive_inward(monkeypatch)
    a = pair_case(); b = pair_case(2.1)
    args = torch.cat([a[0], b[0]]), torch.cat([a[1], b[1]]), torch.tensor([2, 2]), torch.cat([a[3], b[3]])
    combined = core.refine_pdm_with_local_geometry(*args, [[1.], [1.]], [1.], SCHEMA,
        dict(steps=2, line_search_steps=6, max_step_A=.04, temperature=.1),
        local_constraints=EMPTY, bond_weight=1., angle_weight=1., contact_profile=reference(),
        contact_source='md', contact_weight=1., preserve_feasible_contacts=True)
    separate = [run_pair(a), run_pair(b)]
    assert torch.equal(combined['frac_coords'], torch.cat([r['frac_coords'] for r in separate]))
    assert [r['global_atom_offset'] for r in combined['reports']] == [0, 2]
    assert all(r['contact_feasibility_guard']['final_feasible'] for r in combined['reports'])


def test_undefined_contact_retains_raw_failure_and_does_not_claim_feasible():
    args = list(pair_case()); args[0][1] = args[0][0]
    result = run_pair(args)
    report = result['reports'][0]
    assert report['status'] == 'undefined_pair_contacts'
    assert report['initial_contact_audit']['undefined_pair_indices'] == [0]
    assert not report['contact_feasibility_guard']['ever_feasible']
    assert torch.equal(result['frac_coords'], args[0])
    assert report['contact_guard_failure_evidence']['input_frac_coords'] == args[0].tolist()


def test_unexpected_proposal_error_preserves_last_and_attempted_coordinates(monkeypatch):
    drive_inward(monkeypatch)
    calls = []
    def faulty(coords, *args, **kwargs):
        calls.append(True)
        if len(calls) == 3:
            raise RuntimeError('injected contact evaluation failure')
        return profile_pair_lower_bound_penalty(coords, *args, **kwargs)
    monkeypatch.setattr(core, 'profile_pair_lower_bound_penalty', faulty)
    args = pair_case(); result = run_pair(args)
    report = result['reports'][0]
    assert report['status'] == 'evaluation_failed'
    evidence = report['contact_guard_failure_evidence']
    assert evidence['last_frac_coords'] == args[0].tolist()
    assert evidence['last_proposal_frac_coords'] is not None
    assert evidence['last_proposal_frac_coords'] != args[0].tolist()
    assert 'injected contact' in evidence['error']
    assert torch.equal(result['frac_coords'], args[0])


def test_wrapper_forwards_guard_without_changing_budgets_and_sums_evidence(monkeypatch):
    args, kwargs = case(); kwargs.update(contact_weight=1., preserve_feasible_contacts=True)
    seen = []
    def fake(frac, types, counts, cells, target, std, schema, config, **options):
        seen.append(options)
        result = fake_result(frac, types, cells, iterations=config['steps'])
        result['reports'][0]['contact_feasibility_guard'] = {
            'enabled': True, 'initial_feasible': True, 'ever_feasible': True,
            'final_feasible': True, 'rejected_proposals': 3}
        return result
    monkeypatch.setattr(staged, 'refine_pdm_with_local_geometry', fake)
    result = staged.refine_pdm_with_topology_refresh(*args, **kwargs)
    assert result['status'] == 'stages_completed' and result['total_iterations'] == 4
    assert all(item['preserve_feasible_contacts'] is True for item in seen)
    assert [stage['allocated_steps'] for stage in result['stages']] == [2, 2]
    assert result['contact_feasibility_guard']['rejected_proposals'] == 6
    assert result['contact_feasibility_guard']['history_complete'] is True


def test_wrapper_missing_guard_receipt_stops_with_last_coordinates(monkeypatch):
    args, kwargs = case(); kwargs.update(contact_weight=1., preserve_feasible_contacts=True)
    def fake(frac, types, counts, cells, target, std, schema, config, **options):
        return fake_result(frac+.001, types, cells, iterations=config['steps'])
    monkeypatch.setattr(staged, 'refine_pdm_with_local_geometry', fake)
    result = staged.refine_pdm_with_topology_refresh(*args, **kwargs)
    assert result['status'] == 'stopped' and len(result['stages']) == 1
    assert 'guard stage evidence' in result['error']
    assert torch.equal(result['frac_coords'], args[0]+.001)
    assert result['contact_feasibility_guard']['history_complete'] is False


def test_wrapper_catches_final_regression_even_if_core_claims_success(monkeypatch):
    args, kwargs = case(); kwargs.update(contact_weight=1., preserve_feasible_contacts=True)
    def fake(frac, types, counts, cells, target, std, schema, config, **options):
        changed = frac.clone(); changed[1, 0] = changed[0, 0]+.11
        result = fake_result(changed, types, cells, iterations=config['steps'])
        result['reports'][0]['contact_feasibility_guard'] = {
            'enabled': True, 'initial_feasible': True, 'ever_feasible': True,
            'final_feasible': True, 'rejected_proposals': 0}
        return result
    monkeypatch.setattr(staged, 'refine_pdm_with_local_geometry', fake)
    result = staged.refine_pdm_with_topology_refresh(*args, **kwargs)
    assert result['status'] == 'final_diagnostics_failed'
    assert result['final_diagnostics']['complete'] is False
    assert result['contact_feasibility_guard']['final_feasible'] is False
    assert result['stages'][0]['result']['frac_coords'][1, 0] == result['frac_coords'][1, 0]


@pytest.mark.parametrize('options', [{'preserve_feasible_contacts': True},
    {'preserve_feasible_contacts': 1, 'contact_weight': 1.}])
def test_wrapper_invalid_guard_configuration_is_rejected(options):
    args, kwargs = case(); kwargs.update(options)
    with pytest.raises(ValueError, match='preserve_feasible_contacts'):
        staged.refine_pdm_with_topology_refresh(*args, **kwargs)
