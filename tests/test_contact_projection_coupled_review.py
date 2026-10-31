"""Additional adversarial contracts; prepared locally, execute only remotely.

No asserted numerical outcome in this file is a recorded passing result.
The zero vector witnesses feasibility independently of solver convergence.
"""
import json

import pytest
import torch

from cdvae.common.feasible_displacement import FeasibleDisplacementError, project_feasible_displacement


def nonorthogonal_case():
    # Exact Cartesian positions (0,0), (2,0), (1.5,2). The first two
    # contact normals share atom0 and have nonzero dot product3/5.
    frac = torch.tensor([[0., 0., 0.], [.25, 0., 0.], [.1875, .25, 0.]], dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64)*8
    audit = {'has_constraints': True, 'coverage_complete': False,
        'planned_distinct_site_pair_count': 3, 'measured_pair_count': 3,
        'pairs': [
            {'pair': [0, 1], 'reference_covered': True, 'lower_bound_A': 2.},
            {'pair': [0, 2], 'reference_covered': True, 'lower_bound_A': 2.5},
            {'pair': [1, 2], 'reference_covered': False, 'lower_bound_A': None}]}
    return frac, cell, audit


def call(raw, *, sweeps=32):
    frac, cell, audit = nonorthogonal_case()
    return project_feasible_displacement(frac, cell, torch.tensor(raw, dtype=frac.dtype), audit,
        contact_latched=True, minimum_latched=False, min_distance_A=1.,
        max_step_A=.125, max_sweeps=sweeps)


def assert_returned_contract(audit):
    """Inspect actual returned values, never just the usable/status flag."""
    value = torch.tensor(audit['projected_cartesian_displacement'], dtype=torch.float64)
    assert bool(torch.isfinite(value).all())
    for row in audit['constraints']:
        i, j = row['pair']
        normal = torch.tensor(row['normal'], dtype=torch.float64)
        assert float((normal*(value[j]-value[i])).sum()) >= row['rhs_A']
    assert bool((torch.linalg.vector_norm(value, dim=-1) <= .125).all())
    assert audit['returned_dtype_residual']['feasible'] is True


def test_nonorthogonal_contact_intersection_has_explicit_feasible_boundary_point():
    # A common translation at the ball boundary satisfies both contact
    # equalities. This independently anchors feasibility of the hard problem.
    result = call([[.125, 0., 0.]]*3)
    audit = result['audit']
    assert audit['constraint_count'] == 2
    n0, n1 = (torch.tensor(row['normal'], dtype=torch.float64) for row in audit['constraints'])
    assert 0 < float(n0@n1) < 1
    assert all(row['rhs_A'] == 0 for row in audit['constraints'])
    assert torch.equal(torch.tensor(audit['dykstra_cartesian_displacement'], dtype=torch.float64),
                       torch.tensor([[.125, 0., 0.]]*3, dtype=torch.float64))
    assert torch.count_nonzero(result['displacement']) == 0
    assert audit['status'] == 'contracted_stationary' and audit['descent_claimed'] is False
    assert_returned_contract(audit)


def test_nonorthogonal_halfspaces_and_active_balls_preserve_finite32_contract():
    # Large asymmetric requested motion activates both halfspace corrections
    # and ball corrections. This is a fixed input, never selected by outcomes.
    # A feasible set does not imply32 Dykstra sweeps reach a numerical fixed
    # point. Either strictly feasible approximate output or an audited strict
    # residual rejection is valid; silently relaxing a bound is not.
    try:
        result = call([[1024., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    except FeasibleDisplacementError as error:
        audit = error.audit
        assert audit['status'] == 'projection_residual_infeasible'
        assert audit['usable'] is False
        residual = audit['returned_dtype_residual']
        assert not residual['feasible']
        assert residual['maximum_halfspace_violation_A'] > 0 or residual['maximum_ball_violation_A'] > 0
    else:
        audit = result['audit']
        assert audit['status'] in ('contracted_direction', 'contracted_stationary')
        assert audit['origin_contraction']['extra_solver_sweeps'] == 0
        assert audit['usable'] is True
        assert_returned_contract(audit)
    assert audit['actual_sweeps'] == len(audit['sweeps']) == 32
    assert audit['numerical_fixed_point'] is False
    assert audit['nearest_projection_optimality_claimed'] is False
    assert audit['feasibility_tolerance_A'] == 0.
    # Zero is feasible for every frozen halfspace and every positive-radius
    # atom ball. A rejection here cannot mean the feasible set was empty.
    assert all(row['rhs_A'] <= 0 for row in audit['constraints'])
    first = audit['sweeps'][0]
    assert len(first['halfspace_corrections']) == 2
    assert all(value != 0 for value in first['halfspace_corrections'])
    assert all(any(component != 0 for component in row) for row in first['ball_corrections'])
    assert any(row['maximum_primal_dual_change_A'] > audit['fixed_point_change_tolerance_A']
               for row in audit['sweeps'])
    assert len(audit['projected_candidate_frac_coords']) == 3
    json.dumps(audit, allow_nan=False)


def test_dtype_interiorization_keeps_float32_ball_below_original_strict_bound():
    # float32(0.1) is larger than the supplied float64 radius0.1. Axis-aligned
    # projection avoids an ambiguous oblique norm construction. Fixed dtype
    # interiorization changes the direction, never the original radius.
    def project(dtype):
        frac = torch.tensor([[0., 0., 0.], [.25, 0., 0.]], dtype=dtype)
        return project_feasible_displacement(frac, torch.eye(3, dtype=dtype)*8,
            torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=dtype), {},
            contact_latched=False, minimum_latched=False, min_distance_A=1.,
            max_step_A=.1, max_sweeps=32)

    reference = project(torch.float64)
    assert reference['audit']['returned_dtype_residual']['feasible']
    assert reference['audit']['constraint_count'] == 0
    result = project(torch.float32)
    audit = result['audit']
    assert audit['status'] == 'contracted_direction'
    assert audit['no_latched_halfspaces'] and audit['constraint_count'] == 0
    assert audit['sweeps'][-1]['feasible'] is True
    assert audit['sweeps'][-1]['maximum_ball_violation_A'] == 0.
    assert audit['returned_dtype_residual']['maximum_ball_violation_A'] == 0.
    assert audit['returned_dtype_residual']['violated_atom_balls'] == []
    assert audit['usable'] is True
    assert float(torch.tensor(.1, dtype=torch.float32).double()) > .1
    assert torch.linalg.vector_norm(result['displacement'].double(), dim=-1).max() < .1
    assert audit['origin_contraction']['output_dtype_epsilon'] == torch.finfo(torch.float32).eps
    json.dumps(audit, allow_nan=False)
