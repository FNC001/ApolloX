"""Strict finite-direction construction boundaries; numeric tests run remotely."""
import json

import numpy as np
import pytest
import torch

import cdvae.common.feasible_displacement as module
from test_feasible_displacement import make, project, endpoint


def contract(value, normals, rhs, radius, dtype=torch.float64):
    value = torch.tensor(value, dtype=torch.float64)
    edges = torch.tensor([[0, 1]]*len(rhs), dtype=torch.long).reshape(-1, 2)
    normals = torch.tensor(normals, dtype=torch.float64).reshape(-1, 3)
    rhs = torch.tensor(rhs, dtype=torch.float64)
    return module._contract_toward_origin(value, edges, normals, rhs, radius, dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_all_halfspace_and_ball_bounds_share_one_scale(dtype):
    raw = [[.6, .8, 0.], [-.6, -.8, 0.]]
    value, audit = contract(raw, [[1., 0., 0.], [0., 1., 0.]], [-.6, -.4], .5, dtype)
    # y halfspace permits only1/4, x permits1/2, and both atom balls1/2.
    assert 0 < audit['alpha_upper_bound'] < .25
    assert audit['limiting_halfspace_indices'] == [1]
    assert 0 < audit['halfspace_alpha_upper_bounds_capped_at_one'][0] < .5
    assert 0 < audit['halfspace_alpha_upper_bounds_capped_at_one'][1] < .25
    assert audit['atom_ball_alpha_upper_bounds_capped_at_one'] == [.5, .5]
    actual = value.double().numpy()
    assert np.all(np.linalg.norm(actual, axis=-1) <= .5)
    assert (actual[1]-actual[0])[0] >= -.6
    assert (actual[1]-actual[0])[1] >= -.4
    assert np.sum(-np.asarray(raw)*actual) < 0
    assert audit['fixed_epsilon_multiplier'] == 64
    assert audit['geometry_thresholds_relaxed'] is False
    assert audit['extra_solver_sweeps'] == 0
    assert audit['nearest_projection_optimality_claimed'] is False
    json.dumps(audit, allow_nan=False)


def test_zero_rhs_negative_dot_returns_explicit_stationary_origin():
    value, audit = contract([[.01, 0., 0.], [-.01, .01, 0.]], [[1., 0., 0.]], [0.], 1.)
    assert torch.count_nonzero(value) == 0
    assert audit['alpha_upper_bound'] == 0.
    assert audit['zero_direction'] is True
    assert audit['stationary_reason'] == 'zero_contracted_direction'
    assert audit['halfspace_alpha_upper_bounds_capped_at_one'] == [0.]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_nearzero_rhs_has_no_lower_bound_relaxation(dtype):
    value, audit = contract([[1., 0., 0.], [-1., 0., 0.]], [[1., 0., 0.]], [-1e-300], 1., dtype)
    actual = value.double().numpy()
    assert (actual[1]-actual[0])[0] >= -1e-300
    assert audit['finite_roundoff_guarantee_claimed'] is False
    assert 0 < audit['alpha_upper_bound'] < 5e-301
    if dtype == torch.float32:
        assert audit['zero_direction'] is True
    json.dumps(audit, allow_nan=False)


def test_uncapped_ratio_overflow_is_avoided_without_losing_inactive_constraint():
    value, audit = contract([[1e-300, 0., 0.], [0., 0., 0.]], [[1., 0., 0.]], [-1e100], 1.)
    assert audit['halfspace_alpha_upper_bounds_capped_at_one'] == [1.]
    assert audit['alpha_upper_bound'] == 1.
    assert torch.isfinite(value).all()
    json.dumps(audit, allow_nan=False)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_previously_feasible_dykstra_iterate_still_receives_fixed_interiorization(dtype):
    case = make([[0, 0, 0], [.25, 0, 0]], dtype=dtype)
    result = project(case, [[0., .03125, 0.], [0., .03125, 0.]])
    audit = result['audit']
    raw = torch.tensor(audit['dykstra_cartesian_displacement'], dtype=torch.float64)
    assert audit['dykstra_final_residual']['feasible'] is True
    assert audit['dykstra_iterate_status'] == 'numerical_fixed_point'
    assert audit['status'] == 'contracted_direction'
    assert audit['origin_contraction']['alpha_upper_bound'] == 1.
    assert not torch.equal(result['displacement'].double(), raw)
    atoms, _ = endpoint(case, result['displacement'])
    assert atoms.get_distance(0, 1, mic=True) >= 2.
    assert audit['returned_dtype_residual']['feasible'] is True


def test_actual_returned_value_fails_closed_if_contraction_still_violates(monkeypatch):
    case = make([[0, 0, 0], [.25, 0, 0]])
    def invalid(*args):
        return torch.tensor([[.01, 0, 0], [-.01, 0, 0]], dtype=torch.float64), {'zero_direction': False}
    monkeypatch.setattr(module, '_contract_toward_origin', invalid)
    with pytest.raises(module.FeasibleDisplacementError) as caught:
        project(case, [[.01, 0, 0], [-.01, 0, 0]])
    audit = caught.value.audit
    assert audit['usable'] is False and audit['status'] == 'projection_residual_infeasible'
    assert audit['returned_dtype_residual']['maximum_halfspace_violation_A'] > 0
    assert audit['feasibility_tolerance_A'] == 0.
    assert 'dykstra_cartesian_displacement' in audit and 'origin_contraction' in audit


def test_nonfinite_and_origin_infeasible_inputs_are_not_recovered():
    with pytest.raises(ValueError, match='origin'):
        contract([[0., 0, 0], [0., 0, 0]], [[1., 0, 0]], [.01], 1.)
    with pytest.raises(ValueError, match='finite'):
        contract([[float('inf'), 0, 0], [0., 0, 0]], [[1., 0, 0]], [-.01], 1.)


def test_one_ulp_negative_rhs_residual_is_removed_by_direction_change():
    # An unchanged finite solver iterate can be one representable float outside
    # a negative-RHS face. Retain the raw value; change the vector, not the bound.
    rhs = -.03125
    outside = np.nextafter(np.float64(rhs), -np.inf)
    raw = [[0., 0., 0.], [float(outside), .00390625, 0.]]
    raw_dot = np.asarray(raw, dtype=np.float64)[1, 0]
    assert raw_dot < rhs and 0. < rhs-raw_dot < 1e-16
    value, audit = contract(raw, [[1., 0., 0.]], [rhs], .125)
    actual = value.double().numpy()
    assert (actual[1]-actual[0])[0] >= rhs
    assert np.linalg.norm(actual, axis=-1).max() <= .125
    assert np.sum(-np.asarray(raw)*actual) < 0.
    assert 0. < audit['alpha_upper_bound'] < 1.
    assert np.max(np.abs(actual-np.asarray(raw))) < 1e-14
    assert audit['extra_solver_sweeps'] == 0 and audit['geometry_thresholds_relaxed'] is False
    assert audit['zero_direction'] is False
