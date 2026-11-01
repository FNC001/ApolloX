"""Fixed-budget staged PDM refinement with source-profile topology refresh."""
import copy
import hashlib
import json
import math

from ase import Atoms
import torch

from cdvae.common.differentiable_pdm import PairPDMDescriptor, _validate_inputs
from cdvae.common.local_geometry import _contains_boolean
from cdvae.common.pdm_geometry_guidance import refine_pdm_with_local_geometry, _covered_contacts_feasible
from cdvae.common.pdm_guidance import _config, _validate_target_bounds
from cdvae.common.profile_pair_contacts import ProfileContactError, profile_pair_lower_bound_penalty
from cdvae.common.reference_geometry_profile import (
    _validate_profile, audit_frozen_topology, build_local_constraints,
)


CONTINUING_STATUSES = frozenset(('steps_completed', 'stationary', 'line_search_stalled'))


def _atoms(coords, types, cell):
    return Atoms(numbers=types.detach().cpu().tolist(),
                 scaled_positions=coords.detach().double().cpu().numpy(),
                 cell=cell.detach().double().cpu().numpy(), pbc=True)


def _independent_intervals(binding):
    """Audit refreshed covered intervals using ASE topology distances and angles."""
    topology = binding['frozen_topology']
    if topology is None:
        return {'status': 'unsupported_periodic_topology', 'coverage_complete': False,
                'interval_constraints_satisfied': None, 'reason': binding.get('reason')}
    constraints = binding['constraints']
    bond_map = {tuple(row['pair']): row['length_A'] for row in topology['bonds']}
    angle_map = {tuple(row['triple']): row['angle_deg'] for row in topology['angles']}
    bond_values = [bond_map[tuple(pair)] for pair in constraints['bond_pairs']]
    angle_values = [angle_map[tuple(triple)] for triple in constraints['angle_triples']]
    bad_bonds = [i for i, (value, bounds) in enumerate(zip(bond_values, constraints['bond_bounds_A']))
                 if value < bounds[0] or value > bounds[1]]
    bad_angles = [i for i, (value, bounds) in enumerate(zip(angle_values, constraints['angle_bounds_deg']))
                  if value < bounds[0] or value > bounds[1]]
    has = bool(bond_values or angle_values)
    any_bad = bool(bad_bonds or bad_angles)
    return {'status': 'computed' if has else 'no_declared_constraints',
            'method': 'ASE periodic MIC vectors and acos angle values from independently rebuilt topology',
            'comparison': 'strict supplied interval endpoints',
            'has_constraints': has, 'coverage_complete': binding['coverage_complete'],
            'bond_pairs': copy.deepcopy(constraints['bond_pairs']),
            'bond_bounds_A': copy.deepcopy(constraints['bond_bounds_A']),
            'angle_triples': copy.deepcopy(constraints['angle_triples']),
            'angle_bounds_deg': copy.deepcopy(constraints['angle_bounds_deg']),
            'bond_distances_A': bond_values, 'angle_degrees': angle_values,
            'bond_violation_indices': bad_bonds, 'angle_violation_indices': bad_angles,
            'declared_intervals_satisfied': not any_bad if has else None,
            'interval_constraints_satisfied': False if any_bad else True if binding['coverage_complete'] else None}


def refine_pdm_with_topology_refresh(frac_coords, atom_types, num_atoms, cell_matrices,
                                     target_raw, external_std, schema, config=None, *,
                                     profile, source, structure_id, stage_steps,
                                     bond_weight, angle_weight, contact_weight=None,
                                     preserve_feasible_contacts=False, contact_projection_sweeps=None):
    """Refine one structure using a declared partition of the total step budget.

    Every stage binds the current ASE covalent-radius neighbors and center CN
    to the same supplied training profile. The stage's topology is then fixed
    during refinement. Positive integer stage budgets must sum to config.steps;
    unused iterations from stationary or stalled stages are never reassigned.
    Stage losses refer to their own fixed objectives and are recorded separately.

    A supplied contact_weight uses the same profile/source for all-site lower
    bounds. Cell, species, target, scaler and profile snapshots stay fixed.
    Every stage retains starting/ending coordinates, binding and complete core
    result. Final coverage and intervals are independently recomputed on the
    delivered topology; no further refinement follows that audit.
    """
    if type(preserve_feasible_contacts) is not bool:
        raise ValueError('preserve_feasible_contacts must be boolean')
    if contact_projection_sweeps is not None and (type(contact_projection_sweeps) is not int
            or not 1 <= contact_projection_sweeps <= 32 or not preserve_feasible_contacts):
        raise ValueError('contact_projection_sweeps must be an integer1..32 with preserve_feasible_contacts=True')
    if preserve_feasible_contacts and contact_weight is None:
        raise ValueError('preserve_feasible_contacts requires contact_weight')
    cfg = _config(config)
    if (not isinstance(stage_steps, (list, tuple)) or not stage_steps
            or any(type(steps) is not int or steps <= 0 for steps in stage_steps)
            or sum(stage_steps) != cfg['steps']):
        raise ValueError('stage_steps must be positive integers summing exactly to config.steps')
    budgets = list(stage_steps)
    for name, value in [('bond_weight', bond_weight), ('angle_weight', angle_weight)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(name+' must be positive and finite')
    if contact_weight is not None and (isinstance(contact_weight, bool)
            or not isinstance(contact_weight, (int, float)) or not math.isfinite(contact_weight) or contact_weight <= 0):
        raise ValueError('contact_weight must be None or positive and finite')
    if not isinstance(structure_id, str) or not structure_id.strip():
        raise ValueError('An explicit structure_id is required')
    if not all(torch.is_tensor(x) for x in (frac_coords, atom_types, num_atoms, cell_matrices)):
        raise ValueError('Coordinates, species, counts and cell must be explicit tensors')
    if frac_coords.dtype not in (torch.float32, torch.float64):
        raise ValueError('Coordinates must be float32 or float64')
    _validate_inputs(frac_coords, atom_types, num_atoms, cell_matrices)
    if len(num_atoms) != 1:
        raise ValueError('Topology refresh accepts exactly one structure')
    if not bool(torch.isfinite(torch.linalg.det(cell_matrices.detach().double())).all()):
        raise ValueError('Cell determinant must be finite')
    _validate_profile(profile, source)
    fixed_profile = copy.deepcopy(profile)
    profile_digest = hashlib.sha256(json.dumps(fixed_profile, sort_keys=True, separators=(',', ':'),
                                               allow_nan=False).encode()).hexdigest()
    fixed_schema = copy.deepcopy(schema)
    types, counts, cells = (x.detach().clone() for x in (atom_types, num_atoms, cell_matrices))
    current = frac_coords.detach().clone()
    descriptor = PairPDMDescriptor(fixed_schema).to(current.device)
    with torch.no_grad():
        initial_hard = descriptor(current, types, counts, cells, hard=True)
    if _contains_boolean(target_raw) or _contains_boolean(external_std):
        raise ValueError('Targets and training standard deviations cannot be boolean')
    if torch.as_tensor(target_raw).is_complex() or torch.as_tensor(external_std).is_complex():
        raise ValueError('Targets and training standard deviations must be real')
    targets = torch.as_tensor(target_raw, dtype=current.dtype, device=current.device).detach().clone()
    std = torch.as_tensor(external_std, dtype=current.dtype, device=current.device).detach().clone()
    if targets.shape != initial_hard.shape or std.shape != (initial_hard.shape[1],):
        raise ValueError('Raw targets and training standard deviations must match the PDM schema')
    if not bool(torch.isfinite(targets).all() & torch.isfinite(std).all() & (std > 0).all()):
        raise ValueError('Targets must be finite and training standard deviations positive')
    _validate_target_bounds(targets, types, counts, cells, descriptor)

    stages, total_iterations = [], 0
    terminal, failure, first_topology = 'stages_completed', None, None
    for index, allocated in enumerate(budgets):
        start = current.detach().clone()
        entry = {'stage': index, 'allocated_steps': allocated, 'iterations': 0,
                 'unused_allocated_steps': allocated, 'start_frac_coords': start,
                 'end_frac_coords': start.detach().clone(), 'binding': None,
                 'status': 'binding_pending', 'error': None, 'result': None,
                 'iteration_accounting_complete': True,
                 'objective_scope': 'this stage fixed topology only'}
        stages.append(entry)
        try:
            binding = build_local_constraints(_atoms(start, types, cells[0]), fixed_profile,
                source=source, structure_id=f'{structure_id}-stage{index}')
            entry.update(binding=binding, status=binding['status'])
            if first_topology is None and binding['frozen_topology'] is not None:
                first_topology = copy.deepcopy(binding['frozen_topology'])
            stage_cfg = dict(cfg, steps=allocated)
            kwargs = {'local_constraints': copy.deepcopy(binding['constraints']),
                      'bond_weight': bond_weight, 'angle_weight': angle_weight}
            if contact_weight is not None:
                kwargs.update(contact_profile=copy.deepcopy(fixed_profile), contact_source=source,
                              contact_weight=contact_weight)
            if preserve_feasible_contacts:
                kwargs['preserve_feasible_contacts'] = True
            if contact_projection_sweeps is not None:
                kwargs['contact_projection_sweeps'] = contact_projection_sweeps
        except Exception as error:
            entry.update(status='stage_binding_failed', error=f'{type(error).__name__}: {error}')
            if hasattr(error, 'audit'):
                entry['failure_audit'] = error.audit
            terminal, failure = 'stopped', entry['error']
            break
        if binding['status'] == 'unsupported_periodic_topology':
            terminal = 'stopped'
            failure = binding.get('reason', binding['status'])
            entry['error'] = failure
            break
        try:
            entry['iteration_accounting_complete'] = False
            result = refine_pdm_with_local_geometry(current.detach().clone(), types.clone(), counts.clone(),
                cells.clone(), targets.clone(), std.clone(), copy.deepcopy(fixed_schema), stage_cfg, **kwargs)
            entry['result'] = result
            if not isinstance(result, dict) or not isinstance(result.get('reports'), list) or len(result['reports']) != 1:
                raise ValueError('A stage must return exactly one structure report')
            report = result['reports'][0]
            if preserve_feasible_contacts:
                entry['contact_feasibility_guard'] = copy.deepcopy(report.get('contact_feasibility_guard'))
            if contact_projection_sweeps is not None:
                entry['contact_projection'] = copy.deepcopy(report.get('contact_projection'))
            iterations = report.get('iterations')
            entry['reported_iterations'] = iterations
            if type(iterations) is not int or not 0 <= iterations <= allocated:
                raise ValueError('Stage reported an invalid iteration count for its allocated budget')
            if contact_projection_sweeps is not None:
                projection = entry['contact_projection']
                if (not isinstance(projection, dict) or projection.get('max_sweeps_per_direction') != contact_projection_sweeps
                        or type(projection.get('directions_attempted')) is not int
                        or not 0 <= projection['directions_attempted'] <= iterations
                        or type(projection.get('actual_sweeps')) is not int
                        or not 0 <= projection['actual_sweeps'] <= projection['directions_attempted']*contact_projection_sweeps
                        or len(projection.get('direction_statuses', [])) != projection['directions_attempted']):
                    raise ValueError('Missing or invalid finite contact projection accounting')
            entry.update(iterations=iterations, unused_allocated_steps=allocated-iterations,
                         status=report['status'], error=report.get('error'), iteration_accounting_complete=True)
            total_iterations += iterations
            candidate = result.get('frac_coords')
            if (not torch.is_tensor(candidate) or candidate.shape != current.shape or candidate.dtype != current.dtype
                    or candidate.device != current.device or not bool(torch.isfinite(candidate).all())):
                raise ValueError('Stage returned invalid coordinates; last available coordinates retained')
            if not torch.equal(result['atom_types'], types) or not torch.equal(result['cell_matrices'], cells):
                raise ValueError('Stage changed fixed cell or atom identities')
            current = candidate.detach().clone()
            entry['end_frac_coords'] = current.detach().clone()
            if preserve_feasible_contacts:
                guard = entry.get('contact_feasibility_guard')
                if (not isinstance(guard, dict) or guard.get('enabled') is not True
                        or type(guard.get('rejected_proposals')) is not int or guard['rejected_proposals'] < 0
                        or type(guard.get('ever_feasible')) is not bool
                        or any(guard.get(key) is not None and type(guard.get(key)) is not bool
                               for key in ('initial_feasible', 'final_feasible'))):
                    raise ValueError('Missing or invalid covered-contact guard stage evidence')
                previously_feasible = any((stage.get('contact_feasibility_guard') or {}).get('ever_feasible') is True
                                          for stage in stages[:-1])
                if previously_feasible and guard.get('initial_feasible') is not True:
                    raise ValueError('Covered-contact feasibility was lost between stages')
                if guard.get('ever_feasible') is True and guard.get('final_feasible') is not True:
                    raise ValueError('Stage did not preserve latched covered-contact feasibility')
            if report['status'] not in CONTINUING_STATUSES:
                terminal, failure = 'stopped', report.get('error') or report['status']
                break
        except Exception as error:
            entry.update(status='stage_evaluation_failed', error=f'{type(error).__name__}: {error}')
            if hasattr(error, 'audit'):
                entry['failure_audit'] = error.audit
            terminal, failure = 'stopped', entry['error']
            break
    final_atoms = final_binding = topology_changes = contact_audit = final_hard = None
    final_audit = {'status': 'unavailable', 'coverage_complete': False,
                   'interval_constraints_satisfied': None}
    diagnostic_errors = []

    def diagnostic_failure(scope, error):
        row = {'scope': scope, 'error': f'{type(error).__name__}: {error}'}
        if hasattr(error, 'audit'):
            row['partial_audit'] = error.audit
        diagnostic_errors.append(row)

    try:
        final_atoms = _atoms(current, types, cells[0])
    except Exception as error:
        diagnostic_failure('final_atoms', error)
    if final_atoms is not None:
        try:
            final_binding = build_local_constraints(final_atoms, fixed_profile, source=source,
                                                   structure_id=f'{structure_id}-final')
        except Exception as error:
            diagnostic_failure('final_reference_binding', error)
        if final_binding is not None:
            try:
                final_audit = _independent_intervals(final_binding)
            except Exception as error:
                diagnostic_failure('final_local_geometry', error)
        if first_topology is not None:
            try:
                topology_changes = audit_frozen_topology(final_atoms, first_topology)
            except Exception as error:
                diagnostic_failure('final_topology_changes', error)
    if contact_weight is not None:
        try:
            contact_audit = profile_pair_lower_bound_penalty(current, types, counts, cells,
                                                             fixed_profile, source=source)['audit']
            if (preserve_feasible_contacts
                    and any((stage.get('contact_feasibility_guard') or {}).get('ever_feasible') is True for stage in stages)
                    and _covered_contacts_feasible(contact_audit) is not True):
                raise ProfileContactError('Final coordinates do not preserve latched covered-contact feasibility', contact_audit)
        except ProfileContactError as error:
            contact_audit = error.audit
            diagnostic_failure('final_pair_contacts', error)
        except Exception as error:
            diagnostic_failure('final_pair_contacts', error)
    try:
        with torch.no_grad():
            final_hard = descriptor(current, types, counts, cells, hard=True)
    except Exception as error:
        diagnostic_failure('final_hard_pdm', error)
    if diagnostic_errors:
        if terminal == 'stages_completed':
            terminal = 'final_diagnostics_failed'
        if failure is None:
            failure = diagnostic_errors[0]['error']
    returned = {'method': 'fixed_budget_profile_topology_refresh_v1', 'status': terminal, 'error': failure,
            'frac_coords': current, 'input_frac_coords': frac_coords.detach().clone(),
            'atom_types': types, 'num_atoms': counts, 'cell_matrices': cells,
            'target_raw': targets, 'external_std': std, 'final_hard_raw': final_hard,
            'config': copy.deepcopy(cfg), 'stage_steps': budgets, 'stages': stages,
            'planned_total_steps': cfg['steps'], 'total_iterations': total_iterations,
            'unused_planned_iterations': cfg['steps']-total_iterations,
            'iteration_accounting_complete': all(stage['iteration_accounting_complete'] for stage in stages),
            'budget_respected': total_iterations <= cfg['steps'] and all(stage['iteration_accounting_complete'] for stage in stages),
            'completed_stage_count': sum(stage['status'] in CONTINUING_STATUSES for stage in stages),
            'profile_content_sha256': profile_digest, 'source': source,
            'bond_weight': float(bond_weight), 'angle_weight': float(angle_weight),
            'contact_weight': float(contact_weight) if contact_weight is not None else None,
            'cross_stage_objective_comparison': False, 'profile_refit_performed': False,
            'final_reference': final_binding, 'final_local_geometry_audit': final_audit,
            'topology_changes_from_initial': topology_changes, 'final_contact_audit': contact_audit,
            'final_diagnostics': {'complete': not diagnostic_errors, 'errors': diagnostic_errors}}
    if preserve_feasible_contacts:
        guards = [stage.get('contact_feasibility_guard') for stage in stages]
        measured = [item for item in guards if isinstance(item, dict)]
        returned['preserve_feasible_contacts'] = True
        returned['contact_feasibility_guard'] = {
            'enabled': True, 'scope': 'covered distinct-site minimum-image lower bounds only',
            'initial_feasible': measured[0]['initial_feasible'] if measured else None,
            'ever_feasible': any(item.get('ever_feasible') is True for item in measured),
            'rejected_proposals': sum(item.get('rejected_proposals', 0) for item in measured),
            'reported_stage_count': len(measured), 'attempted_stage_count': len(stages),
            'history_complete': len(measured) == len(stages),
            'final_feasible': _covered_contacts_feasible(contact_audit),
            'coverage_complete': bool(contact_audit and contact_audit.get('coverage_complete')),
            'full_physical_validity_claimed': False}
    if contact_projection_sweeps is not None:
        returned['contact_projection_sweeps'] = contact_projection_sweeps
    return returned
