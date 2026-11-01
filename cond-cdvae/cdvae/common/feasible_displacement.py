"""Finite Dykstra construction of a constrained Cartesian descent direction."""
import math

import torch

from cdvae.common.differentiable_pdm import nearest_image_displacements


class FeasibleDisplacementError(ValueError):
    def __init__(self, message, audit):
        self.audit = audit
        super().__init__(message)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name+' must be positive and finite')


def _contract_toward_origin(value, edges, normals, rhs, max_step_A, output_dtype):
    """Construct a finite interior direction without relaxing any original bound."""
    dots = (normals*(value[edges[:, 1]]-value[edges[:, 0]])).sum(-1)
    norms = torch.linalg.vector_norm(value, dim=-1)
    if not bool(torch.isfinite(value).all() and torch.isfinite(dots).all()
                and torch.isfinite(norms).all() and torch.isfinite(rhs).all()) or bool((rhs > 0).any()):
        raise ValueError('Contraction requires finite halfspaces containing the origin')
    # Cover cancellation using the magnitudes before subtraction, rather than
    # the magnitude of the already cancelled dot product or the face slack.
    absolute_sums = (normals.abs()*(value[edges[:, 1]].abs()+value[edges[:, 0]].abs())).sum(-1)
    allowance = 64*torch.finfo(output_dtype).eps*absolute_sums
    if not bool(torch.isfinite(absolute_sums).all() and torch.isfinite(allowance).all()):
        raise ValueError('Nonfinite roundoff-aware absolute product sum or allowance')
    allowance = torch.where(allowance > 0,
        torch.nextafter(allowance, torch.full_like(allowance, float('inf'))), allowance)
    conservative_dots = dots-allowance
    conservative_dots = torch.where(allowance > 0,
        torch.nextafter(conservative_dots, torch.full_like(conservative_dots, -float('inf'))), conservative_dots)
    if not bool(torch.isfinite(allowance).all() and torch.isfinite(conservative_dots).all()):
        raise ValueError('Nonfinite roundoff-aware conservative dot product')
    # Bounds are capped at the existing unit interval before division, avoiding
    # overflow from a tiny negative dot whose uncapped ratio is larger than one.
    half_bounds = torch.ones_like(rhs)
    limiting = (conservative_dots < 0) & (conservative_dots < rhs)
    half_bounds[limiting] = rhs[limiting]/conservative_dots[limiting]
    inward = limiting & (half_bounds > 0)
    half_bounds[inward] = torch.nextafter(half_bounds[inward], torch.zeros_like(half_bounds[inward]))
    ball_bounds = torch.ones_like(norms)
    outside = norms > max_step_A
    ball_bounds[outside] = max_step_A/norms[outside]
    alpha = min(1., float(half_bounds.min()) if len(half_bounds) else 1., float(ball_bounds.min()))
    if not math.isfinite(alpha) or not 0. <= alpha <= 1.:
        raise ValueError('Invalid origin-contraction coefficient')
    shrink = 1.-64*torch.finfo(output_dtype).eps
    scale = alpha*shrink
    returned = (value*scale).to(dtype=output_dtype)
    zero = bool((returned == 0).all())
    record = {'method': 'roundoff_aware_origin_contraction_after_finite_dykstra',
              'roundoff_epsilon_multiplier': 64,
              'halfspace_absolute_product_sums_A': absolute_sums.cpu().tolist(),
              'halfspace_roundoff_allowances_A': allowance.cpu().tolist(),
              'halfspace_conservative_dot_lower_bounds_A': conservative_dots.cpu().tolist(),
              'halfspace_bound_definition': 'rhs/conservative_dot; original rhs unchanged',
              'positive_halfspace_bounds_rounded_toward_zero': True,
              'positive_roundoff_allowances_rounded_up': True,
              'conservative_dots_rounded_down': True,
              'roundoff_model_scope': 'finite normal floating-point operations; subnormal and cast endpoints still require strict verification',
              'origin_is_feasible': True, 'halfspace_dot_displacements_A': dots.cpu().tolist(),
              'halfspace_alpha_upper_bounds_capped_at_one': half_bounds.cpu().tolist(),
              'atom_ball_alpha_upper_bounds_capped_at_one': ball_bounds.cpu().tolist(),
              'limiting_halfspace_indices': (half_bounds == alpha).nonzero(as_tuple=False).flatten().tolist() if alpha < 1. else [],
              'limiting_atom_ball_indices': (ball_bounds == alpha).nonzero(as_tuple=False).flatten().tolist() if alpha < 1. else [],
              'unit_interval_limit_active': alpha == 1., 'alpha_upper_bound': alpha,
              'output_dtype_epsilon': torch.finfo(output_dtype).eps, 'fixed_epsilon_multiplier': 64,
              'interior_shrink_factor': shrink, 'applied_global_scale': scale,
              'zero_direction': zero, 'stationary_reason': 'zero_contracted_direction' if zero else None,
              'geometry_thresholds_relaxed': False, 'extra_solver_sweeps': 0,
              'nearest_projection_optimality_claimed': False,
              'finite_roundoff_guarantee_claimed': False}
    return returned, record


@torch.no_grad()
def project_feasible_displacement(frac_coords, cell, displacement, contact_audit, *,
                                  contact_latched, minimum_latched, min_distance_A,
                                  max_step_A, max_sweeps):
    """Project a displacement onto current-image halfspaces and atom balls.

    Only independently latched guards become halfspaces. Missing contact
    groups never acquire reference bounds. A finite Dykstra iterate is not a
    proof of the exact nearest projection; every returned dtype is checked
    against every linear constraint and ball with strict comparisons. A fixed
    global contraction toward the feasible origin interiorizes the final iterate;
    this is not an exact nearest projection. Real periodic endpoints must still pass the original exact geometry guards.
    """
    if type(max_sweeps) is not int or not 1 <= max_sweeps <= 32:
        raise ValueError('max_sweeps must be an integer1..32')
    if type(contact_latched) is not bool or type(minimum_latched) is not bool:
        raise ValueError('Latches must be explicit booleans')
    _positive(min_distance_A, 'min_distance_A'); _positive(max_step_A, 'max_step_A')
    if (not torch.is_tensor(frac_coords) or frac_coords.dtype not in (torch.float32, torch.float64)
            or frac_coords.ndim != 2 or frac_coords.shape[1] != 3 or not len(frac_coords)
            or not bool(torch.isfinite(frac_coords).all())):
        raise ValueError('Coordinates must be finite [N,3] float32/64')
    if (not torch.is_tensor(displacement) or displacement.shape != frac_coords.shape
            or displacement.dtype != frac_coords.dtype or displacement.device != frac_coords.device
            or not bool(torch.isfinite(displacement).all())):
        raise ValueError('Displacement must match finite coordinates/device/dtype')
    if (not torch.is_tensor(cell) or cell.shape != (3, 3) or cell.device != frac_coords.device
            or cell.dtype != frac_coords.dtype or not bool(torch.isfinite(cell).all())):
        raise ValueError('Fixed cell must match coordinate dtype/device')
    audit = {'method': 'finite_dykstra_current_image_halfspaces_and_atom_balls',
        'max_sweeps': max_sweeps, 'sweeps': [], 'coordinate_dtype': str(frac_coords.dtype),
        'solver_dtype': 'float64', 'contact_latched': contact_latched, 'minimum_latched': minimum_latched,
        'max_step_A': float(max_step_A), 'min_distance_A': float(min_distance_A),
        'input_frac_coords': frac_coords.cpu().tolist(), 'cell': cell.cpu().tolist(),
        'raw_cartesian_displacement': displacement.cpu().tolist(),
        'coverage_complete': bool(isinstance(contact_audit, dict) and contact_audit.get('coverage_complete')),
        'full_physical_validity_claimed': False, 'nearest_projection_optimality_claimed': False,
        'minimum_image_policy': 'same exact bounded current MIC branch; exact all-image endpoint guard still required',
        'status': 'preparing', 'usable': False, 'numerical_fixed_point': False}

    def fail(status, message):
        audit.update(status=status, error=message, usable=False, actual_sweeps=len(audit['sweeps']))
        raise FeasibleDisplacementError(message, audit)

    try:
        determinant = torch.linalg.det(cell.double())
        if not bool(torch.isfinite(determinant)) or float(determinant) <= 0:
            fail('invalid_cell', 'Cell determinant must be finite and positive')
        try:
            inverse = torch.linalg.inv(cell)
        except RuntimeError as error:
            fail('invalid_cell_inverse', str(error))
        if not bool(torch.isfinite(inverse).all()):
            fail('invalid_cell_inverse', 'Nonfinite inverse cell')

        def candidate(value):
            result = torch.remainder(frac_coords + value.to(frac_coords) @ inverse, 1.)
            if not bool(torch.isfinite(result).all()):
                fail('nonfinite_candidate', 'Nonfinite candidate coordinates')
            return result.cpu().tolist()

        audit['raw_candidate_frac_coords'] = candidate(displacement)
        count = len(frac_coords)
        left, right = torch.triu_indices(count, count, 1, device=frac_coords.device)
        total_pairs = len(left)
        lower_by_pair = {}; scopes = {}
        if contact_latched:
            if (not isinstance(contact_audit, dict) or not contact_audit.get('has_constraints')
                    or contact_audit.get('measured_pair_count') != total_pairs
                    or contact_audit.get('planned_distinct_site_pair_count') != total_pairs
                    or len(contact_audit.get('pairs', [])) != total_pairs):
                fail('incomplete_contact_audit', 'Latched contact audit must cover the entire current distinct-site pair inventory')
            seen = set()
            for row in contact_audit['pairs']:
                if not isinstance(row, dict) or not isinstance(row.get('pair'), (list, tuple)) or type(row.get('reference_covered')) is not bool:
                    fail('invalid_contact_pair_inventory', 'Each contact row needs an explicit pair and boolean coverage')
                pair = tuple(row['pair'])
                if len(pair) != 2 or any(type(x) is not int for x in pair) or not 0 <= pair[0] < pair[1] < count or pair in seen:
                    fail('invalid_contact_pair_inventory', 'Invalid or duplicate contact pair identity')
                seen.add(pair)
                if row['reference_covered'] is True:
                    bound = row.get('lower_bound_A')
                    try:
                        _positive(bound, 'contact lower bound')
                    except ValueError as error:
                        fail('invalid_contact_lower_bound', str(error))
                    lower_by_pair[pair] = float(bound); scopes[pair] = ['covered_contact']
            if not lower_by_pair:
                fail('missing_contact_constraints', 'Latched covered contacts have no supplied lower bounds')
        audit['covered_contact_constraint_count'] = len(lower_by_pair)
        if minimum_latched:
            for i, j in zip(left.tolist(), right.tolist()):
                pair = (i, j)
                lower_by_pair[pair] = max(lower_by_pair.get(pair, 0.), float(min_distance_A))
                scopes.setdefault(pair, []).append('minimum_separation')
        pairs = sorted(lower_by_pair)
        audit['planned_distinct_site_pair_count'] = total_pairs
        audit['minimum_separation_constraint_count'] = total_pairs if minimum_latched else 0
        audit['constraint_count'] = len(pairs)
        audit['no_latched_halfspaces'] = not bool(pairs)
        edges = torch.tensor(pairs, dtype=torch.long, device=frac_coords.device).reshape(-1, 2)
        vectors = torch.empty((0, 3), dtype=torch.float64, device=frac_coords.device)
        if pairs:
            try:
                vectors = nearest_image_displacements(frac_coords.double()[edges[:, 1]]-frac_coords.double()[edges[:, 0]], cell.double())
            except (ValueError, RuntimeError) as error:
                fail('undefined_periodic_normal', str(error))
        distances = torch.linalg.vector_norm(vectors, dim=-1)
        undefined = (~torch.isfinite(distances)) | (distances <= 0)
        if bool(undefined.any()):
            audit['undefined_pair_indices'] = undefined.nonzero(as_tuple=False).flatten().tolist()
            fail('undefined_periodic_normal', 'Coincident or nonfinite pair direction; no arbitrary separating direction chosen')
        normals = vectors/distances[:, None]
        normal_square = normals.square().sum(-1)
        lower = torch.tensor([lower_by_pair[p] for p in pairs], dtype=torch.float64, device=frac_coords.device)
        rhs = lower-distances
        audit['constraints'] = [{'pair': list(pair), 'scopes': scopes[pair], 'lower_A': lower_by_pair[pair],
                                'current_distance_A': float(distances[k]), 'normal': normals[k].cpu().tolist(),
                                'rhs_A': float(rhs[k])} for k, pair in enumerate(pairs)]
        if bool((rhs > 0).any()):
            audit['initial_violated_constraint_indices'] = (rhs > 0).nonzero(as_tuple=False).flatten().tolist()
            fail('latched_current_infeasible', 'Current state violates a declared latched constraint')
        redundant = -rhs >= 2*max_step_A*normal_square.sqrt()
        active = (~redundant).nonzero(as_tuple=False).flatten().tolist()
        audit.update(redundancy_rule='slack_A >= 2*max_step_A*normal_norm; full constraints rechecked every sweep',
                     redundant_constraint_indices=redundant.nonzero(as_tuple=False).flatten().tolist(),
                     iterated_constraint_indices=active,
                     halfspace_correction_constraint_indices=active)
        value = displacement.double().clone()
        dual = torch.zeros(len(pairs), dtype=torch.float64, device=value.device)
        ball_dual = torch.zeros_like(value)
        change_tolerance = 64*torch.finfo(torch.float64).eps*max(1., max_step_A, float(value.abs().max()))
        audit['fixed_point_change_tolerance_A'] = change_tolerance
        audit['feasibility_tolerance_A'] = 0.

        def residual(current):
            half = rhs-(normals*(current[edges[:, 1]]-current[edges[:, 0]])).sum(-1)
            balls = torch.linalg.vector_norm(current, dim=-1)-max_step_A
            finite = bool(torch.isfinite(half).all() and torch.isfinite(balls).all())
            return {'finite': finite, 'feasible': finite and not bool((half > 0).any() or (balls > 0).any()),
                    'maximum_halfspace_violation_A': max(0., float(half.max())) if len(half) else 0.,
                    'maximum_ball_violation_A': max(0., float(balls.max())),
                    'violated_halfspace_indices': (half > 0).nonzero(as_tuple=False).flatten().tolist(),
                    'violated_atom_balls': (balls > 0).nonzero(as_tuple=False).flatten().tolist()}

        for sweep in range(max_sweeps):
            previous, old_dual, old_balls = value.clone(), dual.clone(), ball_dual.clone()
            for index in active:
                i, j = pairs[index]; unit = normals[index]
                yi, yj = value[i]-dual[index]*unit, value[j]+dual[index]*unit
                gap = rhs[index]-(unit*(yj-yi)).sum()
                correction = torch.clamp_min(gap, 0)/(2*normal_square[index])
                value[i], value[j] = yi-correction*unit, yj+correction*unit
                dual[index] = -correction
            shifted = value+ball_dual
            norms = torch.linalg.vector_norm(shifted, dim=-1, keepdim=True)
            if not bool(torch.isfinite(shifted).all() and torch.isfinite(norms).all()):
                fail('nonfinite_projection', 'Nonfinite Dykstra ball update')
            projected = shifted*torch.clamp(max_step_A/norms.clamp_min(torch.finfo(torch.float64).tiny), max=1.)
            ball_dual = shifted-projected; value = projected
            state = residual(value)
            change = max(float((value-previous).abs().max()), float((dual-old_dual).abs().max()) if len(dual) else 0.,
                         float((ball_dual-old_balls).abs().max()))
            audit['sweeps'].append({'sweep': sweep, **state, 'maximum_primal_dual_change_A': change,
                'projected_cartesian_displacement': value.cpu().tolist(),
                'projected_candidate_frac_coords': candidate(value),
                'halfspace_corrections': dual[active].cpu().tolist(), 'ball_corrections': ball_dual.cpu().tolist()})
            if not state['finite']:
                fail('nonfinite_projection', 'Projection residual is nonfinite')
            if state['feasible'] and change <= change_tolerance:
                audit['numerical_fixed_point'] = True
                break
        audit.update(dykstra_cartesian_displacement=value.cpu().tolist(),
                     dykstra_final_residual=residual(value),
                     dykstra_iterate_status='numerical_fixed_point' if audit['numerical_fixed_point'] else 'finite_sweep_budget_exhausted')
        returned, contraction = _contract_toward_origin(value, edges, normals, rhs, max_step_A, displacement.dtype)
        final = residual(returned.double())
        audit.update(origin_contraction=contraction, returned_dtype_residual=final,
                     actual_sweeps=len(audit['sweeps']),
                     projected_cartesian_displacement=returned.cpu().tolist(),
                     projected_candidate_frac_coords=candidate(returned))
        if not final['feasible']:
            fail('projection_residual_infeasible', 'Contracted direction does not satisfy every original strict halfspace and atom ball at the returned dtype')
        audit.update(status='contracted_stationary' if contraction['zero_direction'] else 'contracted_direction',
                     stationary=contraction['zero_direction'], descent_claimed=False, usable=True)
        return {'displacement': returned, 'audit': audit}
    except FeasibleDisplacementError:
        raise
    except (ValueError, RuntimeError, TypeError, KeyError, IndexError) as error:
        fail('projection_evaluation_failed', f'{type(error).__name__}: {error}')
