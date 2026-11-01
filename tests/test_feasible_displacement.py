"""Constrained directions and exact endpoint checks; run numerics remotely."""
import copy
import json

from ase import Atoms
import numpy as np
import pytest
import torch

import cdvae.common.pdm_geometry_guidance as core
import cdvae.common.pdm_topology_refresh as staged
from cdvae.common.feasible_displacement import FeasibleDisplacementError, project_feasible_displacement
from cdvae.common.profile_pair_contacts import profile_pair_lower_bound_penalty
from test_profile_pair_contacts import profile


EMPTY = dict(bond_pairs=[], bond_bounds_A=[], angle_triples=[], angle_bounds_deg=[])


def make(coords, *, dtype=torch.float64, cell=None, types=None, reference=None):
    cell = torch.eye(3, dtype=dtype)*8 if cell is None else torch.tensor(cell, dtype=dtype)
    frac = torch.tensor(coords, dtype=dtype)
    types = torch.tensor(types or [6]*len(coords))
    reference = reference or profile({'C|C': [2., 3.]})
    audit = profile_pair_lower_bound_penalty(frac, types, torch.tensor([len(coords)]), cell[None], reference, source='md')['audit']
    return frac, cell, types, reference, audit


def project(case, raw, *, contact=True, minimum=True, radius=.125, sweeps=32):
    frac, cell, types, reference, audit = case
    return project_feasible_displacement(frac, cell, torch.tensor(raw, dtype=frac.dtype), audit,
        contact_latched=contact, minimum_latched=minimum, min_distance_A=1., max_step_A=radius, max_sweeps=sweeps)


def endpoint(case, displacement):
    frac, cell, types, reference, audit = case
    moved = torch.remainder(frac+displacement@torch.linalg.inv(cell), 1.)
    atoms = Atoms(numbers=types.tolist(), scaled_positions=moved.double().numpy(), cell=cell.double().numpy(), pbc=True)
    return atoms, moved


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_analytic_tangent_projection_strict_endpoint_and_descent(dtype):
    case = make([[0, 0, 0], [.25, 0, 0]], dtype=dtype)
    raw = [[.03125, 0, 0], [-.03125, .015625, 0]]
    result = project(case, raw)
    expected = torch.tensor([[0., 0, 0], [0, .015625, 0]], dtype=dtype)
    assert torch.equal(torch.tensor(result['audit']['dykstra_cartesian_displacement'], dtype=dtype), expected)
    assert torch.equal(result['displacement'], expected*(1-64*torch.finfo(dtype).eps))
    assert float((-torch.tensor(raw, dtype=dtype)*expected).sum()) < 0
    atoms, moved = endpoint(case, result['displacement'])
    assert atoms.get_distance(0, 1, mic=True) >= 2.
    audit = result['audit']
    assert audit['usable'] and audit['returned_dtype_residual']['feasible']
    assert audit['coordinate_dtype'] == str(dtype)
    assert audit['nearest_projection_optimality_claimed'] is False
    assert audit['raw_cartesian_displacement'] == torch.tensor(raw, dtype=dtype).tolist()
    assert len(audit['sweeps']) <= 32
    json.dumps(audit, allow_nan=False)


def test_two_orthogonal_active_contacts_and_redundant_pair():
    case = make([[0, 0, 0], [.25, 0, 0], [0, .25, 0]])
    result = project(case, [[.03125, .03125, 0], [-.03125, 0, .015625], [0, -.03125, 0]])
    expected = torch.tensor([[0., 0, 0], [0, 0, .015625], [0, 0, 0]], dtype=torch.float64)
    assert torch.equal(torch.tensor(result['audit']['dykstra_cartesian_displacement'], dtype=torch.float64), expected)
    assert torch.equal(result['displacement'], expected*(1-64*torch.finfo(torch.float64).eps))
    assert len(result['audit']['iterated_constraint_indices']) == 2
    assert len(result['audit']['redundant_constraint_indices']) == 1
    assert result['audit']['constraint_count'] == 3
    atoms, _ = endpoint(case, result['displacement'])
    assert np.min(atoms.get_all_distances(mic=True)[np.triu_indices(3, 1)]) >= 2.


def test_atom_balls_participate_in_projection_not_clipped_after():
    case = make([[0, 0, 0], [.25, 0, 0]])
    result = project(case, [[1., 0, 0], [1., 0, 0]], radius=.125)
    assert torch.equal(torch.tensor(result['audit']['dykstra_cartesian_displacement'], dtype=torch.float64),
                       torch.tensor([[.125, 0, 0], [.125, 0, 0]], dtype=torch.float64))
    # Zero slack plus common nonzero translation has cancellation uncertainty;
    # the conservative returned construction may be stationary, not descent.
    assert torch.count_nonzero(result['displacement']) == 0
    assert result['audit']['status'] == 'contracted_stationary'
    assert result['audit']['descent_claimed'] is False
    assert result['audit']['returned_dtype_residual']['feasible']
    assert any(any(x != 0 for x in correction) for correction in result['audit']['sweeps'][0]['ball_corrections'])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_no_latched_guards_still_enforces_balls_without_inventing_contacts(dtype):
    case = make([[0, 0, 0], [.25, 0, 0]], dtype=dtype)
    result = project(case, [[1., 0, 0], [0, -1., 0]], contact=False, minimum=False, radius=.125)
    assert torch.linalg.vector_norm(result['displacement'].double(), dim=-1).max() <= .125
    assert result['audit']['no_latched_halfspaces'] is True
    assert result['audit']['constraint_count'] == 0
    assert result['audit']['returned_dtype_residual']['feasible']


def test_minimum_latch_constrains_uncovered_pairs_without_filling_profile():
    case = make([[0, 0, 0], [.125, 0, 0]], types=[6, 8], reference=profile({'C|C': [2., 3.]}))
    result = project(case, [[.03125, 0, 0], [-.03125, .015625, 0]], contact=False)
    assert result['audit']['coverage_complete'] is False
    assert result['audit']['covered_contact_constraint_count'] == 0
    assert result['audit']['minimum_separation_constraint_count'] == 1
    assert result['audit']['constraints'][0]['scopes'] == ['minimum_separation']
    assert endpoint(case, result['displacement'])[0].get_distance(0, 1, mic=True) >= 1.


def test_skew_cells_and_individual_periodic_translations_use_exact_normals():
    case = make([[.03, .11, .17], [.95, .07, .21], [.4, .42, .43]],
        cell=[[4.1, 0, 0], [3.6, 2.3, 0], [.3, .5, 4.2]], reference=profile({'C|C': [.1, 3.]}))
    raw = [[.015625, 0, 0], [0, .015625, 0], [0, 0, .015625]]
    result = project(case, raw, minimum=False)
    frac, cell, types, ref, audit = case
    atoms = Atoms(numbers=types.tolist(), scaled_positions=frac.numpy(), cell=cell.numpy(), pbc=True)
    distances = atoms.get_all_distances(mic=True)
    for row in result['audit']['constraints']:
        assert row['current_distance_A'] == pytest.approx(distances[tuple(row['pair'])], abs=1e-12)
    moved = frac+torch.tensor([[1, -2, 0], [-3, 0, 2], [0, 1, -1]])
    translated = (moved, cell, types, ref, profile_pair_lower_bound_penalty(moved, types, torch.tensor([3]), cell[None], ref, source='md')['audit'])
    other = project(translated, raw, minimum=False)
    assert torch.equal(other['displacement'], result['displacement'])


def test_finite_coupled_sweep_retains_raw_violation_and_explicit_zero_contraction():
    case = make([[0, 0, 0], [.25, 0, 0], [.5, 0, 0]])
    result = project(case, [[.0625, 0, 0], [0, 0, 0], [-.0625, 0, 0]], sweeps=1)
    audit = result['audit']
    assert audit['status'] == 'contracted_stationary' and audit['stationary'] is True
    assert audit['actual_sweeps'] == 1 and len(audit['sweeps']) == 1
    assert audit['dykstra_final_residual']['maximum_halfspace_violation_A'] > 0
    assert audit['returned_dtype_residual']['feasible'] is True
    assert audit['origin_contraction']['alpha_upper_bound'] == 0.
    assert torch.count_nonzero(result['displacement']) == 0
    assert len(audit['projected_candidate_frac_coords']) == 3
    assert len(audit['sweeps'][0]['halfspace_corrections']) == len(audit['iterated_constraint_indices'])
    assert audit['usable'] is True and audit['descent_claimed'] is False


def test_latched_infeasible_current_fails_without_recovery_guarantee():
    case = make([[0, 0, 0], [.2, 0, 0]])
    with pytest.raises(FeasibleDisplacementError) as caught:
        project(case, [[0, 0, 0], [.03125, 0, 0]])
    assert caught.value.audit['status'] == 'latched_current_infeasible'
    # Before the contact latch, its original outward recovery direction remains allowed.
    result = project(case, [[0, 0, 0], [.03125, 0, 0]], contact=False)
    assert result['audit']['covered_contact_constraint_count'] == 0
    assert torch.equal(result['displacement'], torch.tensor([[0., 0, 0], [.03125, 0, 0]], dtype=torch.float64)*(1-64*torch.finfo(torch.float64).eps))


def test_coincident_direction_and_unsupported_image_range_fail_with_raw_evidence():
    audit = {'has_constraints': True, 'planned_distinct_site_pair_count': 1, 'measured_pair_count': 1,
             'pairs': [{'pair': [0, 1], 'reference_covered': True, 'lower_bound_A': 1.}]}
    coords = torch.zeros((2, 3), dtype=torch.float64); raw = torch.ones_like(coords)*.01
    with pytest.raises(FeasibleDisplacementError) as caught:
        project_feasible_displacement(coords, torch.eye(3, dtype=torch.float64)*8, raw, audit,
            contact_latched=True, minimum_latched=False, min_distance_A=1., max_step_A=.04, max_sweeps=32)
    assert caught.value.audit['status'] == 'undefined_periodic_normal'
    assert caught.value.audit['input_frac_coords'] == coords.tolist()
    coords[1, 0] = .2
    with pytest.raises(FeasibleDisplacementError) as caught:
        project_feasible_displacement(coords, torch.tensor([[1., 0, 0], [100., .1, 0], [0, 0, 1.]], dtype=torch.float64), raw, audit,
            contact_latched=True, minimum_latched=False, min_distance_A=1., max_step_A=.04, max_sweeps=32)
    assert caught.value.audit['status'] == 'undefined_periodic_normal'


def chemistry_case():
    reference = profile({'C|N': [2., 3.], 'C|O': [1., 4.], 'N|O': [1., 4.]})
    case = make([[0, 0, 0], [.25, 0, 0], [.265625, .265625, 0]], types=[6, 7, 8], reference=reference)
    schema = {'schema_version': 2, 'geometry': 'periodic-minimum-image-clique-v2', 'modes': ['pair'],
        'elements': ['C', 'N', 'O'], 'features': ['CC', 'CN', 'CO', 'NN', 'NO', 'OO'], 'cutoff': 3.}
    return case, schema


def core_run(case, schema, **options):
    frac, cell, types, reference, audit = case
    return core.refine_pdm_with_local_geometry(frac, types, torch.tensor([len(frac)]), cell[None],
        [[0., 1., 1., 0., 1., 0.]], [1.]*6, schema,
        dict(steps=1, max_step_A=.0625, line_search_steps=6, preserve_feasible_geometry=True),
        local_constraints=EMPTY, bond_weight=1., angle_weight=1., contact_profile=reference,
        contact_source='md', contact_weight=1., preserve_feasible_contacts=True, **options)


def test_actual_soft_pdm_zero_slack_contact_has_honest_stationary_return():
    case, schema = chemistry_case()
    result = core_run(case, schema, contact_projection_sweeps=32)
    report = result['reports'][0]
    assert report['status'] == 'projected_direction_not_descent'
    direction = report['history'][0]['contact_projection']
    assert direction['origin_contraction']['alpha_upper_bound'] == 0.
    assert direction['gradient_dot_displacement'] == 0.
    assert report['history'][0]['proposals'] == []
    assert torch.equal(result['frac_coords'], case[0])
    assert report['contact_feasibility_guard']['final_feasible'] is True
    assert report['minimum_separation']['geometry_feasible'] is True


def test_actual_soft_pdm_descent_with_positive_slack_keeps_exact_endpoint():
    original, schema = chemistry_case()
    # Keep every profile bound and target fixed. This separately declared
    # configuration has0.25A C-N slack and tests a real nonzero descent step.
    case = make([[0, 0, 0], [.28125, 0, 0], [.265625, .265625, 0]],
                types=[6, 7, 8], reference=original[3])
    result = core_run(case, schema, contact_projection_sweeps=32)
    report = result['reports'][0]
    assert report['status'] == 'steps_completed'
    assert report['history'][0]['accepted']
    assert report['final_components']['soft_pdm_loss'] < report['initial_components']['soft_pdm_loss']
    assert report['contact_feasibility_guard']['final_feasible'] is True
    assert report['minimum_separation']['geometry_feasible'] is True
    assert report['history'][0]['contact_projection']['gradient_dot_displacement'] < 0
    assert report['contact_projection']['actual_sweeps'] <= 32


def test_no_feasible_descent_keeps_last_state_and_unused_budget():
    from test_contact_feasibility_guard import pair_case, reference
    args = pair_case()
    schema = {'schema_version': 2, 'geometry': 'periodic-minimum-image-clique-v2', 'modes': ['pair'],
              'elements': ['C'], 'features': ['CC'], 'cutoff': 2.}
    result = core.refine_pdm_with_local_geometry(*args, [[1.]], [1.], schema, dict(steps=3, max_step_A=.0625),
        local_constraints=EMPTY, bond_weight=1., angle_weight=1., contact_profile=reference(),
        contact_source='md', contact_weight=1., preserve_feasible_contacts=True, contact_projection_sweeps=32)
    report = result['reports'][0]
    assert report['status'] == 'projected_direction_not_descent'
    assert report['iterations'] == 1 and report['history'][0]['proposals'] == []
    assert torch.equal(result['frac_coords'], args[0])
    assert report['history'][0]['contact_projection']['gradient_dot_displacement'] == 0.


def test_default_none_does_not_invoke_solver_or_add_fields(monkeypatch):
    case, schema = chemistry_case()
    def forbidden(*args, **kwargs):
        raise AssertionError('Default path invoked projection')
    monkeypatch.setattr(core, 'project_feasible_displacement', forbidden)
    before = torch.get_rng_state().clone()
    first = core_run(case, schema)
    second = core_run(case, schema, contact_projection_sweeps=None)
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(first['frac_coords'], second['frac_coords'])
    assert first['reports'] == second['reports']
    assert 'contact_projection_sweeps' not in first and 'contact_projection' not in first['reports'][0]


def test_initial_recovery_does_not_fake_latch_and_later_projection_uses_it(monkeypatch):
    from test_contact_feasibility_guard import pair_case, reference, zero_contact_loss_with_real_audit
    zero_contact_loss_with_real_audit(monkeypatch)
    def objective(coords, *args):
        loss = (coords[1, 0]-.25).square()
        return loss, loss, coords.sum()*0
    monkeypatch.setattr(core, '_objective', objective)
    args = pair_case(1.96875)
    schema = {'schema_version': 2, 'geometry': 'periodic-minimum-image-clique-v2', 'modes': ['pair'],
              'elements': ['C'], 'features': ['CC'], 'cutoff': 3.}
    result = core.refine_pdm_with_local_geometry(*args, [[1.]], [1.], schema, dict(steps=3, max_step_A=.0625),
        local_constraints=EMPTY, bond_weight=1., angle_weight=1., contact_profile=reference(),
        contact_source='md', contact_weight=1., preserve_feasible_contacts=True, contact_projection_sweeps=32)
    report = result['reports'][0]
    assert report['contact_feasibility_guard']['initial_feasible'] is False
    assert report['contact_feasibility_guard']['ever_feasible'] is True
    assert report['contact_feasibility_guard']['final_feasible'] is True
    assert report['history'][0]['contact_projection']['contact_latched'] is False
    assert report['history'][1]['contact_projection']['contact_latched'] is True


def test_staged_projection_keeps_contact_latch_and_fixed_sweep_accounting(monkeypatch):
    from test_contact_feasibility_guard import pair_case, reference
    def objective(coords, *args):
        loss = -coords[1, 1]
        return loss, loss, coords.sum()*0
    monkeypatch.setattr(core, '_objective', objective)
    args = pair_case(); schema = {'schema_version': 2, 'geometry': 'periodic-minimum-image-clique-v2',
        'modes': ['pair'], 'elements': ['C'], 'features': ['CC'], 'cutoff': 3.}
    result = staged.refine_pdm_with_topology_refresh(*args, [[1.]], [1.], schema,
        dict(steps=2, max_step_A=.0625, preserve_feasible_geometry=True),
        profile=reference(), source='md', structure_id='latch', stage_steps=[1, 1],
        bond_weight=1., angle_weight=1., contact_weight=1., preserve_feasible_contacts=True, contact_projection_sweeps=32)
    assert result['total_iterations'] == 2
    assert result['budget_respected'] is True
    assert all(stage['contact_feasibility_guard']['initial_feasible'] is True for stage in result['stages'])
    assert all(stage['contact_projection']['actual_sweeps'] <= 32 for stage in result['stages'])
    assert result['contact_feasibility_guard']['final_feasible'] is True


def test_multigraph_projection_uses_local_pairs_and_matches_separate_results(monkeypatch):
    from test_contact_feasibility_guard import pair_case, reference
    def objective(coords, *args):
        loss = coords[1, 0]-coords[1, 1]
        return loss, loss, coords.sum()*0
    monkeypatch.setattr(core, '_objective', objective)
    a, b = pair_case(), pair_case(2.125)
    combined = torch.cat([a[0], b[0]]), torch.cat([a[1], b[1]]), torch.tensor([2, 2]), torch.cat([a[3], b[3]])
    schema = {'schema_version': 2, 'geometry': 'periodic-minimum-image-clique-v2',
        'modes': ['pair'], 'elements': ['C'], 'features': ['CC'], 'cutoff': 3.}
    def run(args):
        return core.refine_pdm_with_local_geometry(*args, [[1.]]*len(args[2]), [1.], schema,
            dict(steps=1, max_step_A=.0625), local_constraints=EMPTY,
            bond_weight=1., angle_weight=1., contact_profile=reference(), contact_source='md', contact_weight=1.,
            preserve_feasible_contacts=True, contact_projection_sweeps=32)
    result = run(combined); separate = [run(a), run(b)]
    assert torch.equal(result['frac_coords'], torch.cat([value['frac_coords'] for value in separate]))
    assert [row['global_atom_offset'] for row in result['reports']] == [0, 2]
    for row, expected in zip(result['reports'], separate):
        projection = row['history'][0]['contact_projection']
        assert projection == expected['reports'][0]['history'][0]['contact_projection']
        assert projection['constraints'][0]['pair'] == [0, 1]


@pytest.mark.parametrize('value', [0, -1, 33, True, 1.5])
def test_invalid_projection_budget_rejected(value):
    case, schema = chemistry_case()
    with pytest.raises(ValueError, match='contact_projection_sweeps'):
        core_run(case, schema, contact_projection_sweeps=value)


def test_invalid_contact_bound_error_keeps_projection_input():
    case = make([[0, 0, 0], [.25, 0, 0]])
    modified = copy.deepcopy(case[-1]); modified['pairs'][0]['lower_bound_A'] = True
    with pytest.raises(FeasibleDisplacementError) as caught:
        project((*case[:-1], modified), [[0, 0, 0], [.03125, 0, 0]])
    assert caught.value.audit['status'] == 'invalid_contact_lower_bound'
    assert len(caught.value.audit['raw_cartesian_displacement']) == 2


def test_unexpected_solver_error_retains_raw_audit(monkeypatch):
    case = make([[0, 0, 0], [.25, 0, 0]])
    def failed(*args, **kwargs):
        raise RuntimeError('injected determinant failure')
    monkeypatch.setattr(torch.linalg, 'det', failed)
    with pytest.raises(FeasibleDisplacementError) as caught:
        project(case, [[0, 0, 0], [.03125, 0, 0]])
    assert caught.value.audit['status'] == 'projection_evaluation_failed'
    assert caught.value.audit['actual_sweeps'] == 0
    assert caught.value.audit['input_frac_coords'] == case[0].tolist()
    assert 'injected determinant' in caught.value.audit['error']


def test_periodic_image_switch_still_requires_exact_endpoint_guard(monkeypatch):
    from test_contact_feasibility_guard import reference, zero_contact_loss_with_real_audit
    zero_contact_loss_with_real_audit(monkeypatch)
    case = make([[0, 0, 0], [.5, 0, 0]], cell=[[4., 0, 0], [0, 8., 0], [0, 0, 8.]])
    probe = project(case, [[0, 0, 0], [0, .03125, 0]])
    normal_x = probe['audit']['constraints'][0]['normal'][0]
    def objective(coords, *args):
        loss = -normal_x*coords[1, 0]
        return loss, loss, coords.sum()*0
    monkeypatch.setattr(core, '_objective', objective)
    frac, cell, types, ref, audit = case
    schema = {'schema_version': 2, 'geometry': 'periodic-minimum-image-clique-v2',
        'modes': ['pair'], 'elements': ['C'], 'features': ['CC'], 'cutoff': 3.}
    result = core.refine_pdm_with_local_geometry(frac, types, torch.tensor([2]), cell[None], [[1.]], [1.], schema,
        dict(steps=1, max_step_A=.0625, line_search_steps=6), local_constraints=EMPTY,
        bond_weight=1., angle_weight=1., contact_profile=reference(), contact_source='md', contact_weight=1.,
        preserve_feasible_contacts=True, contact_projection_sweeps=32)
    report = result['reports'][0]
    assert report['status'] == 'line_search_stalled'
    assert report['history'][0]['contact_projection']['returned_dtype_residual']['feasible'] is True
    assert report['contact_feasibility_guard']['rejected_proposals'] == 6
    assert all(row['reason'] == 'covered_contact_feasibility_regression' for row in report['history'][0]['proposals'])
    assert torch.equal(result['frac_coords'], frac)


def test_wrapper_forwards_projection_and_retains_partial_coverage(monkeypatch):
    case, schema = chemistry_case(); seen = []
    original = staged.refine_pdm_with_local_geometry
    def spy(*args, **kwargs):
        seen.append(kwargs['contact_projection_sweeps'])
        return original(*args, **kwargs)
    monkeypatch.setattr(staged, 'refine_pdm_with_local_geometry', spy)
    frac, cell, types, reference, audit = case
    result = staged.refine_pdm_with_topology_refresh(frac, types, torch.tensor([3]), cell[None],
        [[0., 1., 1., 0., 1., 0.]], [1.]*6, schema,
        dict(steps=1, max_step_A=.0625, preserve_feasible_geometry=True),
        profile=reference, source='md', structure_id='toy', stage_steps=[1],
        bond_weight=1., angle_weight=1., contact_weight=1., preserve_feasible_contacts=True, contact_projection_sweeps=32)
    assert seen == [32] and result['contact_projection_sweeps'] == 32
    assert result['stages'][0]['contact_projection']['max_sweeps_per_direction'] == 32
    assert result['final_reference']['coverage_complete'] is False
    assert result['contact_feasibility_guard']['full_physical_validity_claimed'] is False
