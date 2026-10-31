"""Fixed-cell PDM refinement with explicit reference bond and angle intervals."""
import math

import torch

from cdvae.common.differentiable_pdm import PairPDMDescriptor
from cdvae.common.feasible_displacement import FeasibleDisplacementError, project_feasible_displacement
from cdvae.common.local_geometry import LocalGeometryError, local_geometry_penalty
from cdvae.common.profile_pair_contacts import ProfileContactError, profile_pair_lower_bound_penalty
from cdvae.common.reference_geometry_profile import _validate_profile
from cdvae.common.pdm_guidance import (
    _config, _finite_scalar, _geometry_summary, _objective, _validate_target_bounds,
)


def _covered_contacts_feasible(audit):
    """Read strict float64 MIC feasibility; missing measurements stay unknown."""
    if (not audit or not audit.get('has_constraints')
            or type(audit.get('planned_distinct_site_pair_count')) is not int
            or type(audit.get('covered_pair_count')) is not int or audit['covered_pair_count'] <= 0
            or audit.get('measured_pair_count') != audit.get('planned_distinct_site_pair_count')
            or type(audit.get('declared_lower_bounds_satisfied')) is not bool):
        return None
    return audit['declared_lower_bounds_satisfied']


def _contact_guard_proposal_evidence(coords, audit):
    """Retain the tested coordinate dtype and every measured violated bound."""
    return {'proposal_frac_coords': coords.detach().cpu().tolist(),
            'coordinate_dtype': str(coords.dtype),
            'covered_pair_count': audit.get('covered_pair_count'),
            'coverage_complete': audit.get('coverage_complete', False),
            'violated_pairs': [row for row in audit.get('pairs', []) if row.get('violation') is True]}


def refine_pdm_with_local_geometry(frac_coords, atom_types, num_atoms, cell_matrices,
                                   target_raw, external_std, schema, config=None, *,
                                   local_constraints, bond_weight, angle_weight,
                                   contact_profile=None, contact_source=None, contact_weight=None,
                                   preserve_feasible_contacts=False, contact_projection_sweeps=None):
    """Refine each declared graph under one fixed combined objective.

    ``local_constraints`` contains global-batch bond_pairs, bond_bounds_A,
    angle_triples and angle_bounds_deg from an external reference profile.
    Optional contact_profile/source/weight jointly add source-specific lower
    bounds across all distinct-site pairs, including newly approaching sites.
    This term applies no upper bound to nonbonded pairs. The optional
    preserve_feasible_contacts guard rejects any proposal that leaves an
    already feasible set of covered lower bounds; absent groups stay unknown.
    It compares strict distances at the proposed coordinate dtype, not loss.
    Each graph retains its cell, species, topology and intervals. The objective
    is train-std soft-PDM MSE plus the configured short-distance repulsion and
    explicitly weighted local bond/angle penalties. Every tested backtracking
    proposal and each objective component is recorded. Graphs without declared
    local constraints retain their original coordinates and coverage status
    unless the optional all-pair lower-bound term supplies constraints.
    """
    for name, weight in [('bond_weight', bond_weight), ('angle_weight', angle_weight)]:
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(name + ' must be explicitly positive and finite')
    if type(preserve_feasible_contacts) is not bool:
        raise ValueError('preserve_feasible_contacts must be boolean')
    if contact_projection_sweeps is not None and (type(contact_projection_sweeps) is not int
            or not 1 <= contact_projection_sweeps <= 32 or not preserve_feasible_contacts):
        raise ValueError('contact_projection_sweeps must be an integer1..32 with preserve_feasible_contacts=True')
    contacts_enabled = any(value is not None for value in (contact_profile, contact_source, contact_weight))
    if contacts_enabled:
        if (contact_profile is None or contact_source is None or isinstance(contact_weight, bool)
                or not isinstance(contact_weight, (int, float)) or not math.isfinite(contact_weight) or contact_weight <= 0):
            raise ValueError('Contact profile, source and a positive finite contact_weight must be supplied together')
        _validate_profile(contact_profile, contact_source)
    if preserve_feasible_contacts and not contacts_enabled:
        raise ValueError('preserve_feasible_contacts requires the explicit contact profile/source/weight')
    keys = {'bond_pairs', 'bond_bounds_A', 'angle_triples', 'angle_bounds_deg'}
    if not isinstance(local_constraints, dict) or set(local_constraints) != keys:
        raise ValueError('Explicit complete local_constraints required')
    cfg = _config(config)
    descriptor = PairPDMDescriptor(schema).to(frac_coords.device)
    with torch.no_grad():
        initial_counts = descriptor(frac_coords, atom_types, num_atoms, cell_matrices, hard=True)
    std = torch.as_tensor(external_std, dtype=frac_coords.dtype, device=frac_coords.device).detach()
    targets = torch.as_tensor(target_raw, dtype=frac_coords.dtype, device=frac_coords.device).detach()
    if targets.shape != initial_counts.shape or std.shape != (initial_counts.shape[1],):
        raise ValueError('Raw targets and training std must match the PDM schema')
    if not bool(torch.isfinite(targets).all() & torch.isfinite(std).all() & (std > 0).all()):
        raise ValueError('Targets must be finite and training std strictly positive')
    _validate_target_bounds(targets, atom_types, num_atoms, cell_matrices, descriptor)
    # Validate complete batch topology before splitting or starting any graph.
    try:
        global_audit = local_geometry_penalty(frac_coords.detach(), num_atoms, cell_matrices,
                                             **local_constraints)['audit']
    except LocalGeometryError as error:
        global_audit = error.audit
    bonds = torch.tensor(global_audit['bond_pairs'], dtype=torch.long, device=frac_coords.device).reshape(-1, 2)
    angles = torch.tensor(global_audit['angle_triples'], dtype=torch.long, device=frac_coords.device).reshape(-1, 3)
    bond_bounds = torch.tensor(global_audit['bond_bounds_A'], dtype=torch.float64, device=frac_coords.device).reshape(-1, 2)
    angle_bounds = torch.tensor(global_audit['angle_bounds_deg'], dtype=torch.float64, device=frac_coords.device).reshape(-1, 2)
    result, reports, audits, offset = [], [], [], 0
    for graph, count in enumerate(num_atoms.tolist()):
        current = frac_coords[offset:offset+count].detach().clone()
        cell = cell_matrices[graph].detach()
        types = atom_types[offset:offset+count]
        counts = num_atoms.new_tensor([count])
        bond_mask = (bonds[:, 0] >= offset) & (bonds[:, 0] < offset+count)
        angle_mask = (angles[:, 1] >= offset) & (angles[:, 1] < offset+count)
        constraints = {'bond_pairs': bonds[bond_mask]-offset, 'bond_bounds_A': bond_bounds[bond_mask],
                       'angle_triples': angles[angle_mask]-offset, 'angle_bounds_deg': angle_bounds[angle_mask]}
        declared = bool(int(bond_mask.sum()) + int(angle_mask.sum()))
        history, first, final, initial_audit, final_audit = [], None, None, None, None
        initial_contact_audit, final_contact_audit = None, None
        contact_guard_rejections, contact_guard_latched, contact_guard_first_feasible = 0, False, None
        guard_last_proposal = None
        status, failure = 'steps_completed', None
        inverse = torch.linalg.inv(cell)
        geometry = _geometry_summary(current, cell, cfg['min_distance_A'])

        def objective(coords):
            base, pdm, repulsion = _objective(coords, types, cell, targets[graph], std, descriptor, cfg)
            local = local_geometry_penalty(coords, counts, cell[None], **constraints)
            total = base + bond_weight*local['bond_loss'] + angle_weight*local['angle_loss']
            contact = (profile_pair_lower_bound_penalty(coords, types, counts, cell[None],
                contact_profile, source=contact_source) if contacts_enabled else None)
            if contact is not None:
                total = total + contact_weight * contact['loss']
            terms = {'loss': total, 'soft_pdm_loss': pdm, 'repulsion_loss': repulsion,
                    'bond_loss': local['bond_loss'], 'angle_loss': local['angle_loss'],
                    'weighted_repulsion_loss': cfg['geometry_weight']*repulsion,
                    'weighted_bond_loss': bond_weight*local['bond_loss'],
                    'weighted_angle_loss': angle_weight*local['angle_loss']}
            if contact is not None:
                terms.update(contact_lower_bound_loss=contact['loss'], weighted_contact_lower_bound_loss=contact_weight*contact['loss'])
            return terms, local['audit'], contact['audit'] if contact is not None else None

        def scalars(terms):
            return {key: _finite_scalar(value.detach()) for key, value in terms.items()}

        try:
            with torch.no_grad():
                first, initial_audit, initial_contact_audit = objective(current)
                final_audit, final_contact_audit = initial_audit, initial_contact_audit
                if preserve_feasible_contacts and _covered_contacts_feasible(initial_contact_audit) is True:
                    contact_guard_latched, contact_guard_first_feasible = True, -1
            if not declared and not (initial_contact_audit and initial_contact_audit['has_constraints']):
                status = 'missing_local_constraints'
            else:
                for iteration in range(cfg['steps']):
                    with torch.enable_grad():
                        variable = current.detach().requires_grad_(True)
                        terms, before_audit, before_contact_audit = objective(variable)
                        row = {'iteration': iteration, **scalars(terms), 'accepted': False, 'proposals': [],
                               'interval_constraints_satisfied': before_audit['interval_constraints_satisfied']}
                        if before_contact_audit is not None:
                            row.update(contact_violation_pair_count=before_contact_audit['violation_pair_count'],
                                contact_covered_pair_count=before_contact_audit['covered_pair_count'])
                        if preserve_feasible_contacts:
                            row['contact_guard_active'] = contact_guard_latched
                            row['covered_contacts_feasible'] = _covered_contacts_feasible(before_contact_audit)
                            if contact_guard_latched and row['covered_contacts_feasible'] is not True:
                                status, failure = 'contact_guard_evaluation_failed', 'Previously feasible covered contacts could not be verified'
                                row['contact_guard_failure_evidence'] = _contact_guard_proposal_evidence(current, before_contact_audit or {})
                                history.append(row); break
                        if not bool(torch.isfinite(terms['loss'])):
                            status = 'nonfinite_objective'; history.append(row); break
                        gradient, = torch.autograd.grad(terms['loss'], variable)
                    cartesian_gradient = gradient.detach() @ inverse.T
                    if not bool(torch.isfinite(cartesian_gradient).all()):
                        row['gradient_rms'] = None
                        status = 'nonfinite_gradient'; history.append(row); break
                    # Scale before squaring in double precision, then normalize
                    # there before returning the step to the coordinate dtype.
                    gradient64 = cartesian_gradient.double()
                    scale = gradient64.abs().max()
                    rms = scale * (gradient64/scale).square().sum(-1).mean().sqrt() if float(scale) else scale
                    row['gradient_rms'] = _finite_scalar(rms)
                    if not bool(torch.isfinite(rms)):
                        status = 'nonfinite_gradient_norm'; history.append(row); break
                    if float(rms) <= 1e-12:
                        status = 'stationary'; history.append(row); break
                    displacement = (-cfg['max_step_A'] * gradient64/rms).to(current)
                    if not bool(torch.isfinite(displacement).all()):
                        status = 'nonfinite_displacement'; history.append(row); break
                    norms = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
                    displacement *= torch.clamp(cfg['max_step_A']/norms.clamp_min(1e-15), max=1.)
                    if contact_projection_sweeps is not None:
                        try:
                            projection = project_feasible_displacement(current, cell, displacement, before_contact_audit,
                                contact_latched=contact_guard_latched,
                                minimum_latched=bool(cfg.get('preserve_feasible_geometry', False) and geometry['geometry_feasible']),
                                min_distance_A=cfg['min_distance_A'], max_step_A=cfg['max_step_A'],
                                max_sweeps=contact_projection_sweeps)
                            row['contact_projection'] = projection['audit']
                            displacement = projection['displacement']
                            descent = (gradient64*displacement.double()).sum()
                            row['contact_projection']['gradient_dot_displacement'] = _finite_scalar(descent)
                            row['contact_projection']['descent_verified'] = bool(torch.isfinite(descent)) and float(descent) < 0
                            if not bool(torch.isfinite(descent)) or float(descent) >= 0:
                                status, failure = 'projected_direction_not_descent', 'Projected displacement is zero, nonfinite or not a descent direction'
                                history.append(row); break
                            norms = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
                        except FeasibleDisplacementError as error:
                            row['contact_projection'] = error.audit
                            status, failure = 'contact_projection_failed', str(error)
                            history.append(row); break
                    with torch.no_grad():
                        for backtrack in range(cfg['line_search_steps']):
                            proposal = torch.remainder(current + (displacement * (0.5**backtrack)) @ inverse, 1.)
                            attempt = {'backtrack': backtrack, 'accepted': False,
                                       'max_cartesian_step_A': float(norms.clamp_max(cfg['max_step_A']).max())*(0.5**backtrack)}
                            if preserve_feasible_contacts:
                                guard_last_proposal = proposal.detach().clone()
                                attempt['contact_guard_evidence'] = _contact_guard_proposal_evidence(proposal, {})
                            try:
                                proposed, proposed_audit, proposed_contact_audit = objective(proposal)
                                attempt.update(scalars(proposed), interval_constraints_satisfied=proposed_audit['interval_constraints_satisfied'])
                                if proposed_contact_audit is not None:
                                    attempt.update(contact_violation_pair_count=proposed_contact_audit['violation_pair_count'],
                                        contact_covered_pair_count=proposed_contact_audit['covered_pair_count'])
                                if preserve_feasible_contacts:
                                    attempt['covered_contacts_feasible'] = _covered_contacts_feasible(proposed_contact_audit)
                                    attempt['contact_guard_active'] = contact_guard_latched
                                    attempt['contact_guard_evidence'] = _contact_guard_proposal_evidence(proposal, proposed_contact_audit or {})
                                if not bool(torch.isfinite(proposed['loss'])):
                                    attempt['reason'] = 'nonfinite_objective'
                                elif (preserve_feasible_contacts and contact_guard_latched
                                      and attempt['covered_contacts_feasible'] is not True):
                                    attempt['reason'] = 'covered_contact_feasibility_regression'
                                    contact_guard_rejections += 1
                                elif float(proposed['loss']) > row['loss']:
                                    attempt['reason'] = 'combined_objective_increased'
                                else:
                                    proposed_geometry = _geometry_summary(proposal, cell, cfg['min_distance_A'])
                                    if cfg.get('preserve_feasible_geometry', False) and geometry['geometry_feasible'] and not proposed_geometry['geometry_feasible']:
                                        attempt['reason'] = 'minimum_separation_regression'
                                    else:
                                        current, geometry = proposal, proposed_geometry
                                        if preserve_feasible_contacts and _covered_contacts_feasible(proposed_contact_audit) is True:
                                            if not contact_guard_latched:
                                                contact_guard_first_feasible = iteration
                                            contact_guard_latched = True
                                        final_audit, final_contact_audit = proposed_audit, proposed_contact_audit
                                        attempt.update(accepted=True, reason='combined_objective_nonincreasing')
                                        row.update(accepted=True, backtracks=backtrack, accepted_loss=float(proposed['loss']))
                            except ProfileContactError as error:
                                attempt.update(reason='undefined_pair_contacts', error=str(error), pair_contact_audit=error.audit)
                                if preserve_feasible_contacts:
                                    attempt['contact_guard_evidence'] = _contact_guard_proposal_evidence(proposal, error.audit)
                            except LocalGeometryError as error:
                                attempt.update(reason='undefined_local_geometry', error=str(error), local_geometry_audit=error.audit)
                            row['proposals'].append(attempt)
                            if attempt['accepted']:
                                break
                    history.append(row)
                    if not row['accepted']:
                        status = 'line_search_stalled'; break
            with torch.no_grad():
                final, final_audit, final_contact_audit = objective(current)
                if (preserve_feasible_contacts and contact_guard_latched
                        and _covered_contacts_feasible(final_contact_audit) is not True):
                    status, failure = 'contact_guard_evaluation_failed', 'Returned coordinates violate or cannot verify latched covered-contact feasibility'
        except ProfileContactError as error:
            status, failure, final_contact_audit = 'undefined_pair_contacts', str(error), error.audit
            if initial_contact_audit is None:
                initial_contact_audit = error.audit
        except LocalGeometryError as error:
            status, failure, final_audit = 'undefined_local_geometry', str(error), error.audit
            if initial_audit is None:
                initial_audit = error.audit
        except (ValueError, RuntimeError) as error:
            status, failure = 'evaluation_failed', str(error)
        report = {'graph': graph, 'global_atom_offset': offset, 'num_atoms': count,
                  'status': status, 'error': failure, 'iterations': len(history),
                  'has_local_constraints': declared, 'optimizer_stationary': status == 'stationary',
                  'interval_constraints_satisfied': final_audit.get('interval_constraints_satisfied') if final_audit else None,
                  'profile_completeness_checked': False,
                  'initial_components': scalars(first) if first is not None else None,
                  'final_components': scalars(final) if final is not None else None,
                  'initial_local_geometry_audit': initial_audit,
                  'local_constraints': {key: value.tolist() for key, value in constraints.items()},
                  'global_bond_constraint_indices': bond_mask.nonzero(as_tuple=False).reshape(-1).tolist(),
                  'global_angle_constraint_indices': angle_mask.nonzero(as_tuple=False).reshape(-1).tolist(),
                  'minimum_separation': _geometry_summary(current, cell, cfg['min_distance_A']),
                  'history': history}
        if contacts_enabled:
            report.update(initial_contact_audit=initial_contact_audit, final_contact_audit=final_contact_audit,
                contact_weight=float(contact_weight), contact_source=contact_source)
        if preserve_feasible_contacts:
            report['contact_feasibility_guard'] = {
                'enabled': True, 'scope': 'covered distinct-site minimum-image lower bounds only',
                'distance_dtype': 'float64', 'coordinate_dtype': str(current.dtype),
                'comparison': 'strict distance_A >= supplied lower_bound_A; no tolerance',
                'initial_feasible': _covered_contacts_feasible(initial_contact_audit),
                'ever_feasible': contact_guard_latched,
                'first_feasible_iteration': contact_guard_first_feasible,
                'rejected_proposals': contact_guard_rejections,
                'final_feasible': _covered_contacts_feasible(final_contact_audit),
                'coverage_complete': bool(final_contact_audit and final_contact_audit.get('coverage_complete')),
                'full_physical_validity_claimed': False}
            if failure is not None:
                report['contact_guard_failure_evidence'] = {
                    'input_frac_coords': frac_coords[offset:offset+count].detach().cpu().tolist(),
                    'last_frac_coords': current.detach().cpu().tolist(),
                    'last_proposal_frac_coords': None if guard_last_proposal is None else guard_last_proposal.cpu().tolist(),
                    'cell': cell.cpu().tolist(), 'atom_types': types.cpu().tolist(),
                    'coordinate_dtype': str(current.dtype), 'error': failure}
        if contact_projection_sweeps is not None:
            measured = [row['contact_projection'] for row in history if 'contact_projection' in row]
            report['contact_projection'] = {'max_sweeps_per_direction': contact_projection_sweeps,
                'directions_attempted': len(measured), 'actual_sweeps': sum(len(item['sweeps']) for item in measured),
                'direction_statuses': [item['status'] for item in measured],
                'nearest_projection_optimality_claimed': False}
        result.append(current.detach())
        reports.append(report); audits.append(final_audit)
        offset += count
    coordinates = torch.cat(result)
    with torch.no_grad():
        hard_counts = descriptor(coordinates, atom_types, num_atoms, cell_matrices, hard=True)
    returned = {'frac_coords': coordinates, 'input_frac_coords': frac_coords.detach().clone(),
            'atom_types': atom_types.detach().clone(), 'cell_matrices': cell_matrices.detach().clone(),
            'final_hard_raw': hard_counts, 'reports': reports, 'local_geometry_audits': audits,
            'config': cfg, 'bond_weight': float(bond_weight), 'angle_weight': float(angle_weight),
            'method': 'fixed_cell_pdm_local_geometry_refinement_v1'}
    if contacts_enabled:
        returned.update(contact_weight=float(contact_weight), contact_source=contact_source,
            method='fixed_cell_pdm_local_geometry_contacts_refinement_v1')
    if preserve_feasible_contacts:
        returned['preserve_feasible_contacts'] = True
    if contact_projection_sweeps is not None:
        returned['contact_projection_sweeps'] = contact_projection_sweeps
    return returned
