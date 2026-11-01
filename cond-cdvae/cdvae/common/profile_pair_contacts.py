"""All-site periodic lower-distance penalties from training-reference profiles."""
from collections import Counter

from ase.data import chemical_symbols
import torch

from cdvae.common.differentiable_pdm import nearest_image_displacements
from cdvae.common.reference_geometry_profile import _validate_profile


class ProfileContactError(ValueError):
    """Retain pair coverage and measured distances for an undefined geometry."""

    def __init__(self, message, audit):
        self.audit = audit
        super().__init__(message)


def profile_pair_lower_bound_penalty(frac_coords, atom_types, num_atoms,
                                     cell_matrices, profile, *, source):
    """Penalize covered element-pair lower bounds across every within-graph i<j.

    Each covered pair contributes ``(relu(lower_A-distance_A)/lower_A)**2``.
    Terms are summed within each graph and divided by that graph's atom count;
    all graph losses are then averaged equally. Far pairs contribute zero
    without changing the atom-count denominator. No upper-distance term is
    applied. Missing profile groups remain explicit uncovered pairs.

    Geometry uses float64 and exact bounded periodic minimum images. Cell,
    species and profile values are fixed detached inputs; coordinate gradients
    are retained. The pair scope excludes self periodic images. Empty coverage
    contributes a differentiable zero and an unavailable coverage status.
    """
    if (not torch.is_tensor(frac_coords) or frac_coords.ndim != 2 or frac_coords.shape[1] != 3
            or frac_coords.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(frac_coords).all())):
        raise ValueError('frac_coords must be finite [N,3] float32/float64')
    device = frac_coords.device
    if (not torch.is_tensor(num_atoms) or num_atoms.ndim != 1 or not len(num_atoms)
            or num_atoms.dtype not in (torch.int32, torch.int64) or num_atoms.device != device
            or not bool((num_atoms > 0).all()) or int(num_atoms.sum()) != len(frac_coords)):
        raise ValueError('num_atoms must be positive integer [B], sum to N and share the coordinate device')
    if (not torch.is_tensor(atom_types) or atom_types.shape != (len(frac_coords),)
            or atom_types.dtype not in (torch.int32, torch.int64) or atom_types.device != device
            or bool(((atom_types < 1) | (atom_types > 118)).any())):
        raise ValueError('atom_types must be integer atomic numbers1..118 [N] on the coordinate device')
    if (not torch.is_tensor(cell_matrices) or cell_matrices.shape != (len(num_atoms), 3, 3)
            or cell_matrices.dtype not in (torch.float32, torch.float64)
            or cell_matrices.device != device or not bool(torch.isfinite(cell_matrices).all())):
        raise ValueError('cell_matrices must be finite [B,3,3] float32/float64 on the coordinate device')
    cells = cell_matrices.detach().double()
    determinants = torch.linalg.det(cells)
    if not bool(torch.isfinite(determinants).all() & (determinants > 0).all()):
        raise ValueError('Cells must have finite positive determinants')
    _validate_profile(profile, source)

    coords = frac_coords.double()
    numbers = atom_types.detach().tolist()
    counts = num_atoms.detach().tolist()
    audit = {'status': 'evaluating', 'source': source, 'bounds_source': 'source_specific_training_profile',
        'profile_version': profile['version'], 'fixed_cell': True, 'fixed_species': True,
        'geometry_dtype': 'float64', 'pair_scope': 'all within-graph distinct-site i<j minimum-image pairs',
        'self_periodic_images_included': False, 'upper_distance_penalty': False,
        'minimum_image_search': {'max_image_range': 8, 'pair_chunk_size': 512},
        'planned_graph_count': len(counts),
        'planned_distinct_site_pair_count': sum(count*(count-1)//2 for count in counts),
        'measured_pair_count': 0,
        'normalization': 'sum((relu(lower_A-distance_A)/lower_A)^2)/atom_count per graph, then equal mean over all graphs',
        'pairs': [], 'graphs': [], 'undefined_pair_indices': [], 'violation_pair_indices': [],
        'has_constraints': False, 'coverage_complete': False,
        'declared_lower_bounds_satisfied': None, 'lower_bound_constraints_satisfied': None}
    losses = []
    offset = 0
    zero = coords.reshape(-1)[0] * 0
    for graph, count in enumerate(counts):
        left, right = torch.triu_indices(count, count, offset=1, device=device)
        start = len(audit['pairs'])
        supported_indices, lower_values, missing_keys = [], [], Counter()
        for local_index, (i, j) in enumerate(zip(left.tolist(), right.tolist())):
            pair = [offset + i, offset + j]
            key = '|'.join(sorted((chemical_symbols[numbers[pair[0]]], chemical_symbols[numbers[pair[1]]])))
            group = profile['bond_groups'].get(key)
            covered = group is not None and group['covered']
            lower = float(group['bounds'][0]) if covered else None
            reasons = [] if covered else (list(group['reasons']) if group else ['missing_reference_group'])
            audit['pairs'].append({'graph': graph, 'pair': pair, 'group_key': key,
                'reference_covered': covered, 'lower_bound_A': lower, 'missing_reasons': reasons,
                'distance_A': None, 'deficit_A': None, 'violation': None})
            if covered:
                supported_indices.append(local_index)
                lower_values.append(lower)
            else:
                missing_keys[key] += 1
        covered_count, total_count = len(supported_indices), len(left)
        graph_row = {'graph': graph, 'atom_count': count, 'distinct_site_pair_count': total_count,
            'covered_pair_count': covered_count, 'uncovered_pair_count': total_count-covered_count,
            'missing_group_pair_counts': dict(missing_keys), 'violation_pair_count': 0,
            'has_constraints': bool(covered_count), 'coverage_complete': bool(total_count and covered_count == total_count),
            'declared_lower_bounds_satisfied': None, 'lower_bound_constraints_satisfied': None,
            'status': 'no_distinct_site_pairs' if not total_count else 'missing_reference_lower_bounds' if not covered_count
                      else 'partial_reference_coverage' if covered_count < total_count else 'complete_reference_coverage'}
        audit['graphs'].append(graph_row)
        audit['has_constraints'] = audit['has_constraints'] or bool(covered_count)
        if not total_count:
            losses.append(zero)
            offset += count
            continue
        try:
            deltas = coords[offset+right] - coords[offset+left]
            if not bool(torch.isfinite(deltas).all()):
                raise ValueError('Nonfinite fractional pair displacement')
            vectors = nearest_image_displacements(deltas, cells[graph])
            distances = torch.linalg.vector_norm(vectors, dim=-1)
        except (ValueError, RuntimeError) as error:
            audit.update(status='undefined_periodic_distance', failure=str(error), failed_graph=graph)
            raise ProfileContactError('Unable to evaluate exact periodic pair distances', audit) from error
        if not bool(torch.isfinite(distances).all()):
            audit.update(status='undefined_periodic_distance', failure='nonfinite_distance', failed_graph=graph)
            raise ProfileContactError('Nonfinite periodic pair distance', audit)
        for local_index, distance in enumerate(distances.detach().tolist()):
            row = audit['pairs'][start+local_index]
            row['distance_A'] = distance
            if row['reference_covered']:
                row['deficit_A'] = max(row['lower_bound_A']-distance, 0.)
                row['violation'] = distance < row['lower_bound_A']
                if row['violation']:
                    audit['violation_pair_indices'].append(start+local_index)
                    graph_row['violation_pair_count'] += 1
            if distance == 0.:
                audit['undefined_pair_indices'].append(start+local_index)
        audit['measured_pair_count'] += total_count
        graph_row['declared_lower_bounds_satisfied'] = not graph_row['violation_pair_count'] if covered_count else None
        graph_row['lower_bound_constraints_satisfied'] = (False if graph_row['violation_pair_count']
            else True if graph_row['coverage_complete'] else None)
        if audit['undefined_pair_indices']:
            audit.update(status='undefined_distance_direction', failure='coincident_distinct_sites',
                         failed_graph=graph, lower_bound_constraints_satisfied=False)
            raise ProfileContactError('Coincident distinct sites have an undefined separating direction', audit)
        if covered_count:
            selected = torch.tensor(supported_indices, dtype=torch.long, device=device)
            lower = torch.tensor(lower_values, dtype=torch.float64, device=device)
            terms = (torch.relu(lower-distances[selected])/lower).square()
            loss = terms.sum()/count
        else:
            loss = zero
        if not bool(torch.isfinite(loss)):
            audit.update(status='nonfinite_lower_bound_loss', failed_graph=graph)
            raise ProfileContactError('Nonfinite lower-bound contact loss', audit)
        losses.append(loss)
        offset += count
    per_graph = torch.stack(losses)
    covered_total = sum(row['covered_pair_count'] for row in audit['graphs'])
    all_covered = all(row['coverage_complete'] for row in audit['graphs'])
    audit.update(status='complete_reference_coverage' if all_covered else 'partial_reference_coverage' if covered_total else 'unavailable_reference_coverage',
        has_constraints=bool(covered_total), coverage_complete=all_covered,
        graph_count=len(counts), distinct_site_pair_count=len(audit['pairs']),
        covered_pair_count=covered_total, uncovered_pair_count=len(audit['pairs'])-covered_total,
        violation_pair_count=len(audit['violation_pair_indices']),
        declared_lower_bounds_satisfied=not audit['violation_pair_indices'] if covered_total else None,
        lower_bound_constraints_satisfied=False if audit['violation_pair_indices'] else True if all_covered else None)
    return {'loss': per_graph.mean(), 'per_graph_loss': per_graph, 'audit': audit}
