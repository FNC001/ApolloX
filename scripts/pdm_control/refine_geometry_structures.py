#!/usr/bin/env python3
"""Refine POSCARs with PDM targets and source-specific bond/angle intervals."""
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / 'cond-cdvae'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.pdm_control.refine_structures import (
    filename, hard_pdm, parser as pdm_parser, read_csv, require, sha,
    targets_for_files, write_json,
)


def parser():
    result = pdm_parser()
    result.description = __doc__
    result.add_argument('--profile', type=Path, required=True, help='Training-reference interval profile JSON')
    result.add_argument('--profile-sha256', required=True, help='Exact SHA256 of the reference profile file')
    result.add_argument('--source', required=True, help='Source matching the reference profile')
    result.add_argument('--bond-weight', type=float, required=True)
    result.add_argument('--angle-weight', type=float, required=True)
    result.add_argument('--contact-weight', type=float, default=None,
                        help='Positive weight for profile lower bounds on all distinct-site contacts')
    result.add_argument('--preserve-feasible-contacts', action='store_true',
                        help='Preserve already satisfied covered contact lower bounds; requires --contact-weight')
    result.add_argument('--contact-projection-sweeps', type=int, default=None,
                        help='Use1..32 finite feasible-displacement sweeps; requires --preserve-feasible-contacts')
    result.add_argument('--topology-stage-steps', type=int, nargs='+', default=None,
                        help='Positive stage budgets summing to config.steps; refresh neighbors between stages')
    result.add_argument('--audit-dir', type=Path, help='Optional separate directory for full iteration and input records')
    return result


def _json(value):
    return json.dumps(value, allow_nan=False, separators=(',', ':'))


def _serializable(value):
    import numpy as np
    import torch
    if torch.is_tensor(value):
        return _serializable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _serializable(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else {'nonfinite': str(value)}
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def _ase(structure):
    from ase import Atoms
    return Atoms(numbers=[site.specie.Z for site in structure],
                 scaled_positions=structure.frac_coords, cell=structure.lattice.matrix, pbc=True)


def _refresh_ledger(result):
    """Keep stage outcomes in the final status table without iteration traces."""
    stages = result.get('stages', [])
    ledger = {
        'stages': [{**{key: stage.get(key) for key in
            ('stage', 'allocated_steps', 'iterations', 'unused_allocated_steps', 'status',
             'error', 'iteration_accounting_complete')},
            'reference_status': (stage.get('binding') or {}).get('status'),
            'reference_coverage_complete': (stage.get('binding') or {}).get('coverage_complete'),
            'reference_coverage': (stage.get('binding') or {}).get('coverage')} for stage in stages],
        'final_diagnostics': {'complete': result.get('final_diagnostics', {}).get('complete'),
            'errors': [{key: row.get(key) for key in ('scope', 'error')}
                       for row in result.get('final_diagnostics', {}).get('errors', [])]},
    }
    if result.get('preserve_feasible_contacts') is True:
        for compact, stage in zip(ledger['stages'], stages):
            compact['contact_feasibility_guard'] = stage.get('contact_feasibility_guard')
    if result.get('contact_projection_sweeps') is not None:
        for compact, stage in zip(ledger['stages'], stages):
            compact['contact_projection'] = stage.get('contact_projection')
    return ledger


def _contact_geometry_fields(audit):
    """Summarize only measured final contacts; incomplete measurements stay unknown."""
    graphs = audit.get('graphs', [])
    total = audit.get('planned_distinct_site_pair_count')
    measured = audit.get('measured_pair_count', 0)
    all_measured = total is not None and measured == total
    covered = audit.get('covered_pair_count')
    if covered is None and len(graphs) == audit.get('planned_graph_count'):
        covered = sum(row['covered_pair_count'] for row in graphs)
    missing = {}
    for graph in graphs:
        for key, count in graph.get('missing_group_pair_counts', {}).items():
            missing[key] = missing.get(key, 0) + count
    return {
        'contact_status': audit.get('status', 'unavailable'),
        'contact_pair_scope': audit.get('pair_scope'),
        'contact_distinct_site_pair_count': total,
        'contact_measured_pair_count': measured,
        'contact_covered_pair_count': covered,
        'contact_uncovered_pair_count': total-covered if total is not None and covered is not None else None,
        'contact_coverage_complete': audit.get('coverage_complete', False),
        'contact_violation_pair_count': len(audit.get('violation_pair_indices', [])) if all_measured else None,
        'contact_undefined_pair_count': len(audit.get('undefined_pair_indices', [])) if all_measured else None,
        'contact_declared_lower_bounds_satisfied': audit.get('declared_lower_bounds_satisfied'),
        'contact_lower_bound_constraints_satisfied': audit.get('lower_bound_constraints_satisfied'),
        'contact_missing_group_pair_counts': _json(missing),
    }


def run(args):
    import numpy as np
    import torch
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar
    from cdvae.common.differentiable_pdm import validate_schema
    from cdvae.common.local_geometry import LocalGeometryError, local_geometry_penalty
    from cdvae.common.pdm_geometry_guidance import refine_pdm_with_local_geometry
    from cdvae.common.pdm_topology_refresh import refine_pdm_with_topology_refresh
    from cdvae.common.profile_pair_contacts import ProfileContactError, profile_pair_lower_bound_penalty
    from cdvae.common.pdm_guidance import _config
    from cdvae.common.pdm_schema import read_external_stats, read_pdm_schema
    from cdvae.common.reference_geometry_profile import (
        audit_frozen_topology, build_local_constraints, load_reference_profile,
    )

    source, model = args.input_dir.expanduser().resolve(), args.model_path.expanduser().resolve()
    output = args.output_dir.expanduser().absolute()
    audit_dir = args.audit_dir.expanduser().absolute() if args.audit_dir else None
    require(source.is_dir() and model.is_dir(), 'Input and model paths must be directories')
    for destination in (output, audit_dir):
        if destination is None:
            continue
        require(not os.path.lexists(destination), f'Output path already exists: {destination}')
        resolved = destination.resolve()
        require(not resolved.is_relative_to(source) and not source.is_relative_to(resolved),
                'Input and output directories must not contain one another')
    if audit_dir:
        require(not audit_dir.resolve().is_relative_to(output.resolve())
                and not output.resolve().is_relative_to(audit_dir.resolve()),
                'Audit and final-output directories must be separate')
    for name in ('bond_weight', 'angle_weight'):
        value = getattr(args, name)
        require(not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) and value > 0, name + ' must be positive and finite')
    contact_weight = getattr(args, 'contact_weight', None)
    stage_steps = getattr(args, 'topology_stage_steps', None)
    contact_guard = getattr(args, 'preserve_feasible_contacts', False)
    contact_projection = getattr(args, 'contact_projection_sweeps', None)
    require(contact_projection is None or (type(contact_projection) is int and 1 <= contact_projection <= 32 and contact_guard),
            'contact_projection_sweeps requires an integer1..32 and preserve_feasible_contacts')
    require(type(contact_guard) is bool, 'preserve_feasible_contacts must be boolean')
    require(not contact_guard or contact_weight is not None,
            'preserve_feasible_contacts requires a positive contact_weight')
    if contact_weight is not None:
        require(not isinstance(contact_weight, bool) and isinstance(contact_weight, (int, float))
                and math.isfinite(contact_weight) and contact_weight > 0,
                'contact_weight must be positive and finite')
    profile_path = args.profile.expanduser().resolve()
    profile = load_reference_profile(profile_path, source=args.source, expected_sha256=args.profile_sha256)
    paths = sorted(path for path in source.iterdir() if path.suffix.lower() == '.vasp')
    require(paths, 'No .vasp structures found')
    require(all(path.is_file() and not path.is_symlink() for path in paths), 'POSCAR inputs must be regular files')
    names = [filename(path.name) for path in paths]
    schema_path, stats_path = model / 'pdm_schema.json', model / 'scaler_stats.txt'
    schema_json = json.loads(schema_path.read_text())
    schema = read_pdm_schema(schema_path, len(schema_json.get('features', [])))
    validate_schema(schema)
    mean, std = read_external_stats(stats_path, len(schema['features']))
    config_input = json.loads(args.config.expanduser().read_text())
    config_input.setdefault('preserve_feasible_geometry', True)
    config = _config(config_input)
    if stage_steps is not None:
        require(isinstance(stage_steps, (list, tuple)) and bool(stage_steps)
                and all(type(steps) is int and steps > 0 for steps in stage_steps)
                and sum(stage_steps) == config['steps'],
                'topology_stage_steps must be positive integers summing exactly to config.steps')
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda'), 'Device must be cpu or cuda[:index]')
    torch.empty(0, device=device)
    target_file = (args.target or args.targets).expanduser().resolve()
    targets = targets_for_files(names, target_file if args.target else None,
                               target_file if args.targets else None, len(schema['features']), mean, std, args.target_space)
    identities, manifest_fields = {}, []
    manifest = source / 'structure_manifest.csv'
    if manifest.exists():
        manifest_fields, records = read_csv(manifest)
        require('filename' in manifest_fields, 'Identity manifest needs a filename column')
        for row in records:
            name = filename(row['filename'])
            require(name not in identities, 'Duplicate identity filename: ' + name)
            require('source' not in row or row['source'] == args.source, 'Identity manifest source mismatch')
            identities[name] = row
        require(set(identities) == set(names), 'Identity manifest must match every input exactly')
    identity_columns = [key for key in ('material_id', 'eval_index', 'sample_index', 'job_id') if key in manifest_fields]
    output.mkdir(parents=True, exist_ok=False)
    generated = output / 'gen'
    generated.mkdir()
    if audit_dir:
        audit_dir.mkdir(parents=True, exist_ok=False)
    if manifest_fields:
        shutil.copyfile(manifest, generated / 'structure_manifest.csv')
    metadata = {'status': 'RUNNING', 'planned_files': names, 'planned': len(paths),
                'method': 'fixed_cell_pdm_local_geometry_refinement_v1', 'source': args.source,
                'minimum_distance_scope': 'distinct_site_minimum_image_pairs',
                'profile_sha256': args.profile_sha256, 'target_space': args.target_space,
                'target_file_sha256': sha(target_file), 'schema': schema, 'config': config,
                'bond_weight': args.bond_weight, 'angle_weight': args.angle_weight,
                'model_contract_sha256': {'pdm_schema.json': sha(schema_path), 'scaler_stats.txt': sha(stats_path)}}
    if contact_weight is not None:
        metadata.update(contact_weight=contact_weight, method='fixed_cell_pdm_local_geometry_contacts_refinement_v1')
    if stage_steps is not None:
        metadata.update(topology_stage_steps=list(stage_steps), method='fixed_budget_profile_topology_refresh_v1')
    if contact_guard:
        metadata['preserve_feasible_contacts'] = True
    if contact_projection is not None:
        metadata['contact_projection_sweeps'] = contact_projection
    write_json(output / 'run.json', metadata)
    identity_fields = ['filename', *identity_columns]
    geometry_fields = [*identity_fields, 'source', 'reference_status', 'reference_coverage_complete',
        'total_bonds', 'covered_bonds', 'total_angles', 'covered_angles', 'zero_coordination_count',
        'geometry_status', 'declared_intervals_satisfied', 'bond_violation_count', 'angle_violation_count',
        'bond_distances_A', 'angle_degrees', 'minimum_distance_A', 'minimum_separation_satisfied',
        'topology_status', 'missing_bonds', 'omitted_angles', 'zero_coordination_atoms',
        'final_topology_reference_status', 'final_topology_coverage_complete',
        'final_topology_total_bonds', 'final_topology_covered_bonds', 'final_topology_total_angles',
        'final_topology_covered_angles', 'final_topology_zero_coordination_count',
        'final_topology_geometry_status', 'final_topology_intervals_satisfied',
        'final_topology_bond_violation_count', 'final_topology_angle_violation_count',
        'final_topology_bond_distances_A', 'final_topology_angle_degrees',
        'final_topology_missing_bonds', 'final_topology_omitted_angles', 'final_topology_zero_coordination_atoms', 'error']
    status_fields = [*identity_fields, 'input_sha256', 'output_sha256', 'status', 'optimizer_status',
                     'coordinate_state', 'audit_status', 'audit_error', 'error']
    if contact_weight is not None:
        geometry_fields.extend(['contact_status', 'contact_pair_scope', 'contact_distinct_site_pair_count',
            'contact_measured_pair_count', 'contact_covered_pair_count', 'contact_uncovered_pair_count',
            'contact_coverage_complete', 'contact_violation_pair_count', 'contact_undefined_pair_count',
            'contact_declared_lower_bounds_satisfied', 'contact_lower_bound_constraints_satisfied',
            'contact_missing_group_pair_counts', 'contact_error'])
    if stage_steps is not None:
        status_fields.extend(['topology_stage_ledger', 'topology_planned_steps', 'topology_iterations',
            'topology_budget_respected', 'topology_final_diagnostics_complete'])
    if contact_guard:
        status_fields.extend(['contact_guard_enabled', 'contact_guard_rejected_proposals',
            'contact_guard_ever_feasible', 'contact_guard_initial_feasible',
            'contact_guard_final_feasible', 'contact_guard_evidence_complete'])
    completed, errors, limitations, retained = 0, 0, 0, 0
    with (output/'pdm.csv').open('x', newline='') as ph, (output/'geometry.csv').open('x', newline='') as gh, (output/'status.csv').open('x', newline='') as sh:
        pw = csv.DictWriter(ph, fieldnames=['label', 'formula', *identity_columns, 'pdm_status', *schema['features'], 'error'])
        gw, sw = csv.DictWriter(gh, fieldnames=geometry_fields), csv.DictWriter(sh, fieldnames=status_fields)
        pw.writeheader(); gw.writeheader(); sw.writeheader()
        for path in paths:
            identity = {key: identities[path.name][key] for key in identity_columns}
            state = {'filename': path.name, **identity, 'input_sha256': '', 'status': 'error',
                     'optimizer_status': '', 'coordinate_state': 'unavailable',
                     'audit_status': 'pending' if audit_dir else 'not_requested', 'audit_error': '', 'error': ''}
            geometry = {'filename': path.name, **identity, 'source': args.source, 'geometry_status': 'unavailable'}
            if contact_weight is not None:
                geometry['contact_status'] = 'unavailable'
            if stage_steps is not None:
                state.update(topology_stage_ledger=_json({'stages': [], 'final_diagnostics': {'complete': False, 'errors': []}}),
                             topology_planned_steps=config['steps'])
            if contact_guard:
                state.update(contact_guard_enabled=True, contact_guard_evidence_complete=False)
            pdm = {'label': path.name, **identity, 'pdm_status': 'unavailable'}
            evidence, built, result = {'filename': path.name}, None, None
            final_path, temp = generated/path.name, generated/('.'+path.name+'.tmp')
            try:
                state['input_sha256'] = sha(path)
                evidence['input_sha256'] = state['input_sha256']
                shutil.copyfile(path, final_path)
                state['coordinate_state'] = 'input_retained'
                structure = Poscar.from_file(path, check_for_potcar=False).structure
                hard_pdm(structure, schema)
                evidence['input_frac_coords'] = structure.frac_coords.tolist()
                built = build_local_constraints(_ase(structure), profile, source=args.source, structure_id=path.name)
                evidence['reference'] = built
                geometry.update(reference_status=built['status'], reference_coverage_complete=built['coverage_complete'],
                    **{key: built['coverage'][key] for key in ('total_bonds','covered_bonds','total_angles','covered_angles','zero_coordination_count')},
                    missing_bonds=_json(built['missing_bonds']), omitted_angles=_json(built['omitted_angles']),
                    zero_coordination_atoms=_json(built['zero_coordination_atoms']))
                state['status'] = built['status']
                if (built['has_constraints'] or stage_steps is not None
                        or (contact_weight is not None and built['status'] != 'unsupported_periodic_topology')):
                    cell = torch.tensor(np.asarray(structure.lattice.matrix), dtype=torch.float64, device=device)
                    atom_types = torch.tensor([site.specie.Z for site in structure], dtype=torch.long, device=device)
                    inputs = (
                        torch.tensor(np.asarray(structure.frac_coords), dtype=torch.float64, device=device), atom_types,
                        torch.tensor([len(structure)], dtype=torch.long, device=device), cell[None],
                        torch.tensor([targets[path.name]], dtype=torch.float64, device=device),
                        torch.tensor(std, dtype=torch.float64, device=device), schema, config)
                    if stage_steps is None:
                        kwargs = {'local_constraints': built['constraints'], 'bond_weight': args.bond_weight,
                                  'angle_weight': args.angle_weight}
                        if contact_weight is not None:
                            kwargs.update(contact_profile=profile, contact_source=args.source, contact_weight=contact_weight)
                        if contact_guard:
                            kwargs['preserve_feasible_contacts'] = True
                        if contact_projection is not None:
                            kwargs['contact_projection_sweeps'] = contact_projection
                        result = refine_pdm_with_local_geometry(*inputs, **kwargs)
                        evidence['refinement'] = _serializable(result)
                        report = result['reports'][0]
                        optimizer_ok = report['status'] in ('steps_completed', 'stationary', 'line_search_stalled')
                        reference_complete = built['coverage_complete']
                    else:
                        guard_kwargs = {'preserve_feasible_contacts': True} if contact_guard else {}
                        if contact_projection is not None:
                            guard_kwargs['contact_projection_sweeps'] = contact_projection
                        result = refine_pdm_with_topology_refresh(*inputs, profile=profile, source=args.source,
                            structure_id=path.name, stage_steps=list(stage_steps), bond_weight=args.bond_weight,
                            angle_weight=args.angle_weight, contact_weight=contact_weight, **guard_kwargs)
                        evidence['refinement'] = _serializable(result)
                        report = result
                        final_diagnostics = result.get('final_diagnostics', {})
                        optimizer_ok = (result.get('status') == 'stages_completed'
                            and result.get('budget_respected') is True
                            and result.get('iteration_accounting_complete') is True
                            and final_diagnostics.get('complete') is True
                            and not final_diagnostics.get('errors')
                            and result.get('planned_total_steps') == config['steps']
                            and result.get('stage_steps') == list(stage_steps)
                            and type(result.get('total_iterations')) is int
                            and 0 <= result['total_iterations'] <= config['steps']
                            and len(result.get('stages', [])) == len(stage_steps)
                            and result.get('completed_stage_count') == len(stage_steps))
                        reference_complete = (built['coverage_complete']
                            and all(stage.get('binding', {}).get('coverage_complete') is True
                                    for stage in result.get('stages', []) if stage.get('binding') is not None)
                            and all(stage.get('binding') is not None for stage in result.get('stages', [])))
                        state.update(topology_stage_ledger=_json(_serializable(_refresh_ledger(result))),
                            topology_planned_steps=result.get('planned_total_steps'),
                            topology_iterations=result.get('total_iterations'),
                            topology_budget_respected=result.get('budget_respected'),
                            topology_final_diagnostics_complete=final_diagnostics.get('complete'))
                    state['optimizer_status'] = report['status']
                    require(torch.equal(result['cell_matrices'], cell[None]) and torch.equal(result['atom_types'], atom_types),
                            'Refinement changed cell or atom identities')
                    coords = result['frac_coords'].detach().cpu().numpy()
                    require(coords.shape == (len(structure), 3) and np.isfinite(coords).all(), 'Invalid refined coordinates')
                    refined = Structure(structure.lattice, structure.species, coords, site_properties=structure.site_properties)
                    Poscar(refined, comment='PDM and local-geometry refined '+path.name).write_file(temp, significant_figures=16)
                    delivered = Poscar.from_file(temp, check_for_potcar=False).structure
                    require(delivered.species == structure.species and np.allclose(delivered.lattice.matrix, structure.lattice.matrix, rtol=0, atol=1e-10),
                            'Export changed cell or atom identities')
                    os.replace(temp, final_path)
                    if contact_guard:
                        guard = report.get('contact_feasibility_guard') or {}
                        guard_complete = (guard.get('enabled') is True
                            and type(guard.get('rejected_proposals')) is int and guard['rejected_proposals'] >= 0
                            and type(guard.get('ever_feasible')) is bool
                            and all(guard.get(key) is None or type(guard.get(key)) is bool
                                    for key in ('initial_feasible', 'final_feasible'))
                            and (guard.get('ever_feasible') is not True or guard.get('final_feasible') is True)
                            and (stage_steps is None or guard.get('history_complete') is True))
                        state.update(contact_guard_rejected_proposals=guard.get('rejected_proposals'),
                            contact_guard_ever_feasible=guard.get('ever_feasible'),
                            contact_guard_initial_feasible=guard.get('initial_feasible'),
                            contact_guard_final_feasible=guard.get('final_feasible'),
                            contact_guard_evidence_complete=guard_complete)
                        if not guard_complete:
                            optimizer_ok = False
                            state['error'] = 'Incomplete covered-contact feasibility guard evidence'
                    if optimizer_ok:
                        state['coordinate_state'] = 'optimized'
                        state['status'] = 'refined' if reference_complete else 'partial_reference_coverage'
                    else:
                        failure = state['error'] or report.get('error') or report['status']
                        if stage_steps is not None and report.get('status') == 'stages_completed':
                            failure = 'Incomplete topology budget or final diagnostics'
                        state.update(status='error', coordinate_state='last_iterate', error=failure)
            except Exception as error:
                if (contact_weight is not None or stage_steps is not None) and hasattr(error, 'audit'):
                    evidence['refinement_failure_audit'] = _serializable(error.audit)
                state.update(status='error', error=f'{type(error).__name__}: {error}')
            finally:
                try:
                    if temp.exists():
                        temp.unlink()
                except OSError as error:
                    state.update(status='error', error=state['error']+'; temporary_file_retained: '+str(error))
            try:
                delivered = Poscar.from_file(final_path, check_for_potcar=False).structure
                values = hard_pdm(delivered, schema)
                symbols = [site.specie.symbol for site in delivered]
                pdm.update(formula=''.join(f'{s}{symbols.count(s)}' for s in sorted(set(symbols))), pdm_status='computed', **values)
                atoms = _ase(delivered)
                distances = atoms.get_all_distances(mic=True)
                minimum = float(distances[np.triu_indices(len(atoms), 1)].min()) if len(atoms)>1 else None
                geometry.update(minimum_distance_A=minimum, minimum_separation_satisfied=
                    None if minimum is None else minimum>=config['min_distance_A'])
                evidence['delivered_frac_coords'] = delivered.frac_coords.tolist()
                if built is not None and built['frozen_topology'] is not None:
                    topology = audit_frozen_topology(atoms, built['frozen_topology'])
                    evidence['final_topology'] = topology
                    geometry['topology_status'] = topology['status']
                    try:
                        audit = local_geometry_penalty(torch.tensor(delivered.frac_coords, dtype=torch.float64),
                            torch.tensor([len(delivered)]), torch.tensor(delivered.lattice.matrix, dtype=torch.float64)[None],
                            **built['constraints'])['audit']
                        geometry.update(geometry_status='computed' if built['has_constraints'] else 'no_declared_constraints',
                            declared_intervals_satisfied=audit['interval_constraints_satisfied'],
                            bond_violation_count=len(audit['bond_violation_indices']), angle_violation_count=len(audit['angle_violation_indices']),
                            bond_distances_A=_json(audit['bond_distances_A']), angle_degrees=_json(audit['angle_degrees']))
                        evidence['delivered_local_geometry'] = audit
                    except LocalGeometryError as error:
                        geometry.update(geometry_status='undefined_local_geometry', declared_intervals_satisfied=False, error=str(error))
                        evidence['delivered_local_geometry'] = error.audit
                        state.update(status='error', error=state['error'] or str(error))
                # Diagnose current neighbors with the same fixed profile;
                # these constraints are never sent back into optimization.
                current_reference = build_local_constraints(atoms, profile, source=args.source, structure_id=path.name)
                evidence['delivered_reference'] = current_reference
                if current_reference['status']=='unsupported_periodic_topology':
                    geometry['minimum_separation_satisfied'] = None
                geometry.update(final_topology_reference_status=current_reference['status'],
                    final_topology_coverage_complete=current_reference['coverage_complete'],
                    **{'final_topology_'+key: current_reference['coverage'][key] for key in
                       ('total_bonds','covered_bonds','total_angles','covered_angles','zero_coordination_count')},
                    final_topology_missing_bonds=_json(current_reference['missing_bonds']),
                    final_topology_omitted_angles=_json(current_reference['omitted_angles']),
                    final_topology_zero_coordination_atoms=_json(current_reference['zero_coordination_atoms']),
                    final_topology_geometry_status='unavailable')
                if state['status']=='refined' and not current_reference['coverage_complete']:
                    state['status']='partial_final_reference_coverage'
                if current_reference['frozen_topology'] is not None:
                    try:
                        current_audit = local_geometry_penalty(torch.tensor(delivered.frac_coords,dtype=torch.float64),
                            torch.tensor([len(delivered)]),torch.tensor(delivered.lattice.matrix,dtype=torch.float64)[None],
                            **current_reference['constraints'])['audit']
                        geometry.update(final_topology_geometry_status='computed' if current_reference['has_constraints'] else 'no_declared_constraints',
                            final_topology_intervals_satisfied=current_audit['interval_constraints_satisfied'],
                            final_topology_bond_violation_count=len(current_audit['bond_violation_indices']),
                            final_topology_angle_violation_count=len(current_audit['angle_violation_indices']),
                            final_topology_bond_distances_A=_json(current_audit['bond_distances_A']),
                            final_topology_angle_degrees=_json(current_audit['angle_degrees']))
                        evidence['delivered_current_local_geometry'] = current_audit
                    except LocalGeometryError as error:
                        geometry.update(final_topology_geometry_status='undefined_local_geometry',
                            final_topology_intervals_satisfied=False,error=str(error))
                        evidence['delivered_current_local_geometry'] = error.audit
                        state.update(status='error',error=state['error'] or str(error))
            except Exception as error:
                message = f'{type(error).__name__}: {error}'
                if pdm['pdm_status'] == 'unavailable':
                    pdm['error'] = message
                geometry['error'] = message
                state.update(status='error', error=state['error'] or message)
            if contact_weight is not None:
                try:
                    contact_structure = Poscar.from_file(final_path, check_for_potcar=False).structure
                    with torch.no_grad():
                        contact_result = profile_pair_lower_bound_penalty(
                            torch.tensor(contact_structure.frac_coords, dtype=torch.float64),
                            torch.tensor([site.specie.Z for site in contact_structure], dtype=torch.long),
                            torch.tensor([len(contact_structure)], dtype=torch.long),
                            torch.tensor(contact_structure.lattice.matrix, dtype=torch.float64)[None],
                            profile, source=args.source)
                    contact_audit = contact_result['audit']
                    geometry.update(_contact_geometry_fields(contact_audit))
                    evidence['delivered_pair_contacts'] = _serializable(contact_audit)
                    if contact_guard:
                        state['contact_guard_final_feasible'] = contact_audit.get('declared_lower_bounds_satisfied')
                        if (state.get('contact_guard_ever_feasible') is True
                                and contact_audit.get('declared_lower_bounds_satisfied') is not True):
                            state.update(status='error', coordinate_state='last_iterate',
                                error=state['error'] or 'Exported coordinates do not preserve latched covered-contact feasibility')
                    if state['status'] == 'refined' and not contact_audit['coverage_complete']:
                        state['status'] = 'partial_contact_reference_coverage'
                except ProfileContactError as error:
                    geometry.update(_contact_geometry_fields(error.audit), contact_error=str(error))
                    evidence['delivered_pair_contacts'] = _serializable(error.audit)
                    state.update(status='error', error=state['error'] or str(error))
                except Exception as error:
                    message = f'{type(error).__name__}: {error}'
                    geometry.update(contact_status='evaluation_failed', contact_error=message)
                    state.update(status='error', error=state['error'] or message)
            state['output_sha256'] = ''
            output_present = False
            try:
                output_present = final_path.is_file()
                if output_present:
                    state['output_sha256'] = sha(final_path)
            except OSError as error:
                state.update(status='error',error=state['error']+'; output_sha256_error: '+str(error))
            retained += int(output_present)
            if audit_dir:
                evidence.update(status=state, final_pdm=pdm, final_geometry=geometry)
                state['audit_status'] = 'written'
                try:
                    write_json(audit_dir/(path.name+'.json'), evidence)
                except (OSError, TypeError, ValueError) as error:
                    state.update(status='error',audit_status='error',audit_error=f'{type(error).__name__}: {error}',
                                 error=state['error']+'; audit_write_error: '+str(error))
            if state['status'] == 'refined':
                completed += 1
            elif state['status'] == 'error':
                errors += 1
            else:
                limitations += 1
            pw.writerow(pdm); gw.writerow(geometry); sw.writerow(state)
            ph.flush(); gh.flush(); sh.flush()
    metadata.update(status='COMPLETE' if not errors and not limitations else 'COMPLETE_WITH_ERRORS' if errors else 'COMPLETE_WITH_LIMITATIONS',
                    refined=completed, errors=errors, limitations=limitations, retained_files=retained)
    write_json(output/'run.json', metadata)
    print(json.dumps({'output_dir': str(output), 'planned': len(paths), 'refined': completed,
                      'errors': errors, 'limitations': limitations, 'retained_files': retained}))
    return 1 if errors or limitations else 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        print(f'PDM geometry refinement error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
