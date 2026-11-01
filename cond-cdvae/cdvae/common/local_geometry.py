"""Differentiable interval penalties for explicitly declared local geometry.

Bounds and topology are supplied by an external reference profile. Computes
periodic interval penalties and coverage audits.
"""
import math
from itertools import product

import numpy as np
import torch

from cdvae.common.differentiable_pdm import nearest_image_displacements


class LocalGeometryError(ValueError):
    """Retain a partial audit when declared local geometry is undefined."""
    def __init__(self, message, audit):
        self.audit = audit
        super().__init__(message)


def _contains_boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if torch.is_tensor(value):
        return value.dtype == torch.bool
    if isinstance(value, np.ndarray):
        return value.dtype == np.bool_ or (value.dtype == object and any(_contains_boolean(x) for x in value.flat))
    return isinstance(value, (list, tuple)) and any(_contains_boolean(x) for x in value)


def _indices(value, width, device, name):
    if _contains_boolean(value):
        raise ValueError(name + ' cannot contain boolean indices')
    value = torch.as_tensor(value, device=device)
    if value.ndim == 1 and value.numel() == 0:
        value = torch.empty((0, width), dtype=torch.long, device=device)
    if value.ndim != 2 or value.shape[1] != width or value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f'{name} must be integer [N,{width}]')
    return value.detach().long().clone()


def _bounds(value, count, device, name, angles=False):
    supplied_tensor = torch.is_tensor(value)
    def has_boolean(item):
        return isinstance(item, bool) or (isinstance(item, (list, tuple)) and any(has_boolean(x) for x in item))
    if not supplied_tensor and has_boolean(value):
        raise ValueError(name + ' cannot contain boolean bounds')
    inferred = torch.as_tensor(value, device=device)
    if inferred.is_complex():
        raise ValueError(name + ' cannot contain complex bounds')
    value = inferred if supplied_tensor or inferred.dtype == torch.bool else torch.as_tensor(value, dtype=torch.float64, device=device)
    if value.ndim == 1 and value.numel() == 0:
        value = torch.empty((0, 2), dtype=torch.float64, device=device)
    if value.shape != (count, 2) or value.dtype == torch.bool or not bool(torch.isfinite(value).all()):
        raise ValueError(name + ' must be finite [N,2] intervals')
    value = value.detach().double().clone()
    if bool((value[:, 0] > value[:, 1]).any()):
        raise ValueError(name + ' lower bound exceeds upper bound')
    if angles:
        if bool(((value < 0) | (value > 180)).any()):
            raise ValueError('Angle bounds must be within [0,180] degrees')
    elif bool((value <= 0).any()):
        raise ValueError('Bond-length bounds must be strictly positive')
    return value



def _ambiguous_angle_images(deltas, cell):
    """Find equal-nearest images with distinct directions, using float64 decisions.

    Squared-distance tie tolerance is 64 * eps64 * max(1 Å², cell row norm²,
    wrapped-vector norm²). Direction tolerance is 128 * eps64 times the
    corresponding length scale divided by nearest length. These tolerances
    cover arithmetic roundoff, not a bond-length or angle acceptance margin.
    The exact bounded image search includes the entire tolerance shell.
    """
    if not len(deltas):
        return []
    with torch.no_grad():
        q = deltas.detach().double()
        q = q - torch.floor(q + .5)
        c = cell.detach().double()
        eps = torch.finfo(torch.float64).eps
        r0_squared = (q @ c).square().sum(-1)
        cell_scale_squared = c.square().sum(-1).max().clamp_min(1.)
        scale_squared = torch.maximum(r0_squared, cell_scale_squared)
        distance_tolerance = 64 * eps * scale_squared
        radius = (r0_squared + distance_tolerance).sqrt()
        reciprocal = torch.linalg.vector_norm(torch.linalg.inv(c), dim=0)
        bounds = torch.floor((radius[:, None] * reciprocal + q.abs()).amax(dim=0) + 1e-10).long()
        if int(bounds.max()) > 8:
            raise ValueError('Exact angle-image search exceeds configured limit 8; cell is too skew/ill-conditioned')
        shifts = torch.tensor(list(product(*(range(-int(b), int(b) + 1) for b in bounds))),
                              device=cell.device, dtype=torch.float64)
        ambiguous = []
        for start in range(0, len(q), 512):
            vectors = (q[start:start+512, None, :] + shifts[None, :, :]) @ c
            squared = vectors.square().sum(-1)
            minima, selected = squared.min(-1)
            tolerances = distance_tolerance[start:start+512]
            tied = squared <= minima[:, None] + tolerances[:, None]
            for local in ((tied.sum(-1) > 1) & (minima > 0)).nonzero(as_tuple=False).flatten().tolist():
                candidates = vectors[local, tied[local]]
                directions = candidates / torch.linalg.vector_norm(candidates, dim=-1, keepdim=True)
                chosen = vectors[local, selected[local]] / minima[local].sqrt()
                direction_tolerance = 128 * eps * scale_squared[start+local].sqrt() / minima[local].sqrt()
                if bool((torch.linalg.vector_norm(directions-chosen, dim=-1) > direction_tolerance).any()):
                    ambiguous.append({'index': start+local,
                        'candidate_vectors_A': candidates.tolist(),
                        'minimum_squared_distance_A2': float(minima[local]),
                        'distance_tie_tolerance_A2': float(tolerances[local]),
                        'direction_tolerance': float(direction_tolerance)})
        return ambiguous

def local_geometry_penalty(frac_coords, num_atoms, cell_matrices,
                           bond_pairs, bond_bounds_A,
                           angle_triples, angle_bounds_deg):
    """Evaluate declared bond/angle intervals in each graph's fixed cell.

    Indices are global batch indices. Angles are [neighbor, center, neighbor]
    and must reference two declared bonds. Edges are unordered; reversing an
    edge or the two angle neighbors cannot create a distinct constraint.

    Geometry and losses use float64, preserving gradients to input coordinates.
    Cells, bounds and topology are detached fixed inputs. Bond loss per edge is
    squared interval-distance divided by the squared interval midpoint (Å).
    Angle loss is squared hinge distance to the allowed cosine interval; there
    is no acos in its gradient path or division by a possibly zero interval
    width. Each loss first averages constraints within a graph, then averages
    all graphs equally. A graph with no constraints of that type contributes
    zero; the audit reports per-graph constraint counts and uncovered graphs.
    """
    if (not torch.is_tensor(frac_coords) or frac_coords.ndim != 2 or frac_coords.shape[1] != 3
            or frac_coords.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(frac_coords).all())):
        raise ValueError('frac_coords must be finite [N,3] float32/float64')
    device = frac_coords.device
    if (not torch.is_tensor(num_atoms) or num_atoms.ndim != 1 or not len(num_atoms)
            or num_atoms.dtype not in (torch.int32, torch.int64) or num_atoms.device != device
            or not bool((num_atoms > 0).all()) or int(num_atoms.sum()) != len(frac_coords)):
        raise ValueError('num_atoms must be positive integer [B] on the coordinate device and sum to N')
    if (not torch.is_tensor(cell_matrices) or cell_matrices.shape != (len(num_atoms), 3, 3)
            or cell_matrices.device != device or cell_matrices.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(cell_matrices).all())):
        raise ValueError('cell_matrices must be finite [B,3,3] float32/float64 on the coordinate device')
    cells = cell_matrices.detach().double()
    if not bool((torch.linalg.det(cells) > 0).all()):
        raise ValueError('Cells must have positive determinants')
    bonds = _indices(bond_pairs, 2, device, 'bond_pairs')
    angles = _indices(angle_triples, 3, device, 'angle_triples')
    bond_bounds = _bounds(bond_bounds_A, len(bonds), device, 'bond_bounds_A')
    angle_bounds = _bounds(angle_bounds_deg, len(angles), device, 'angle_bounds_deg', angles=True)
    graph = torch.arange(len(num_atoms), device=device).repeat_interleave(num_atoms.long())
    for name, indices in [('bond_pairs', bonds), ('angle_triples', angles)]:
        if bool(((indices < 0) | (indices >= len(frac_coords))).any()):
            raise ValueError(name + ' contains out-of-range indices')
        if len(indices) and not bool((graph[indices] == graph[indices[:, :1]]).all()):
            raise ValueError(name + ' contains a cross-graph constraint')
        if any(len(set(row)) != len(row) for row in indices.tolist()):
            raise ValueError(name + ' contains repeated/self indices')
    canonical_bonds = [tuple(sorted(pair)) for pair in bonds.tolist()]
    if len(canonical_bonds) != len(set(canonical_bonds)):
        raise ValueError('Duplicate unordered bond edge')
    canonical_angles = [(min(a, c), b, max(a, c)) for a, b, c in angles.tolist()]
    if len(canonical_angles) != len(set(canonical_angles)):
        raise ValueError('Duplicate angle with reversed or repeated neighbors')
    edge_set = set(canonical_bonds)
    if any(tuple(sorted((a,b))) not in edge_set or tuple(sorted((c,b))) not in edge_set
           for a,b,c in angles.tolist()):
        raise ValueError('Every angle arm must correspond to a declared bond edge')

    coords = frac_coords.double()
    bond_graph = graph[bonds[:, 0]]
    angle_graph = graph[angles[:, 1]]

    def vectors(deltas, owners):
        result = deltas.new_zeros(deltas.shape)
        for index in range(len(num_atoms)):
            ids = (owners == index).nonzero(as_tuple=False).reshape(-1)
            if len(ids):
                result = result.index_copy(0, ids, nearest_image_displacements(deltas[ids], cells[index]))
        return result

    bond_vectors = vectors(coords[bonds[:, 1]] - coords[bonds[:, 0]], bond_graph)
    distances = torch.linalg.vector_norm(bond_vectors, dim=-1)
    first = vectors(coords[angles[:, 0]] - coords[angles[:, 1]], angle_graph)
    second = vectors(coords[angles[:, 2]] - coords[angles[:, 1]], angle_graph)
    first_length, second_length = torch.linalg.vector_norm(first, dim=-1), torch.linalg.vector_norm(second, dim=-1)
    bond_counts = [int((bond_graph == i).sum()) for i in range(len(num_atoms))]
    angle_counts = [int((angle_graph == i).sum()) for i in range(len(num_atoms))]
    audit = {'fixed_cell': True, 'geometry_dtype': 'float64', 'bounds_source': 'caller_supplied_reference_intervals',
        'has_constraints': bool(len(bonds) or len(angles)),
        'coverage': {'graph_count':len(num_atoms), 'bond_constraint_counts':bond_counts,
                    'angle_constraint_counts':angle_counts,
                    'unconstrained_graph_indices':[i for i,(b,a) in enumerate(zip(bond_counts,angle_counts)) if b+a == 0],
                    'profile_completeness_checked':False},
        'normalization': {'bond':'squared hinge divided by squared positive interval midpoint in angstrom',
                          'angle':'squared hinge in dimensionless cosine interval',
                          'graphs':'mean within graph, then mean over all B graphs; absent type contributes zero'},
        'bond_pairs':bonds.tolist(), 'angle_triples':angles.tolist(),
        'bond_bounds_A':bond_bounds.tolist(), 'angle_bounds_deg':angle_bounds.tolist(),
        'bond_distances_A':distances.detach().tolist()}
    zero_bonds = (distances == 0).nonzero(as_tuple=False).reshape(-1).tolist()
    undefined = ((first_length == 0) | (second_length == 0)).nonzero(as_tuple=False).reshape(-1).tolist()
    if zero_bonds or undefined:
        audit.update(interval_constraints_satisfied=False, failure='coincident_bond_or_undefined_angle',
                     coincident_bond_indices=zero_bonds, undefined_angle_indices=undefined)
        raise LocalGeometryError('Coincident bonded sites or zero-length angle arm: direction/angle is undefined', audit)
    if not bool(torch.isfinite(distances).all() & torch.isfinite(first_length).all() & torch.isfinite(second_length).all()):
        audit.update(interval_constraints_satisfied=False, failure='nonfinite_local_geometry')
        raise LocalGeometryError('Nonfinite local geometry', audit)

    ambiguous_arms = []
    for index in range(len(num_atoms)):
        ids = (angle_graph == index).nonzero(as_tuple=False).flatten()
        for arm, neighbor_column in [('first', 0), ('second', 2)]:
            deltas = coords[angles[ids, neighbor_column]] - coords[angles[ids, 1]]
            for item in _ambiguous_angle_images(deltas, cells[index]):
                item['angle_index'] = int(ids[item.pop('index')])
                item['arm'] = arm
                item['graph'] = index
                ambiguous_arms.append(item)
    audit['angle_image_uniqueness'] = {'checked': True, 'decision_dtype': 'float64',
        'distance_tolerance': '64*eps64*max(1 A^2, max cell row norm^2, wrapped vector norm^2)',
        'direction_tolerance': '128*eps64*sqrt(distance scale)/nearest length',
        'ambiguous_arms': ambiguous_arms}
    if ambiguous_arms:
        audit.update(interval_constraints_satisfied=False, failure='ambiguous_periodic_angle_image',
                     undefined_angle_indices=sorted({item['angle_index'] for item in ambiguous_arms}))
        raise LocalGeometryError('Equal-nearest periodic images have distinct angle-arm directions', audit)

    cosine = ((first * second).sum(-1) / (first_length * second_length)).clamp(-1., 1.)
    if not bool(torch.isfinite(cosine).all()):
        audit.update(interval_constraints_satisfied=False, failure='nonfinite_angle')
        raise LocalGeometryError('Nonfinite angle cosine', audit)
    bond_hinge = torch.relu(bond_bounds[:, 0]-distances) + torch.relu(distances-bond_bounds[:, 1])
    midpoint = bond_bounds[:, 0]/2 + bond_bounds[:, 1]/2
    bond_terms = (bond_hinge/midpoint).square()
    cosine_low = torch.cos(angle_bounds[:, 1] * (math.pi/180))
    cosine_high = torch.cos(angle_bounds[:, 0] * (math.pi/180))
    angle_terms = (torch.relu(cosine_low-cosine) + torch.relu(cosine-cosine_high)).square()
    zero = coords.sum()*0
    per_graph_bond = torch.stack([bond_terms[bond_graph == i].mean() if bond_counts[i] else zero for i in range(len(num_atoms))])
    per_graph_angle = torch.stack([angle_terms[angle_graph == i].mean() if angle_counts[i] else zero for i in range(len(num_atoms))])
    bond_loss, angle_loss = per_graph_bond.mean(), per_graph_angle.mean()
    if not bool(torch.isfinite(bond_loss) & torch.isfinite(angle_loss)):
        audit.update(interval_constraints_satisfied=False, failure='nonfinite_interval_loss')
        raise LocalGeometryError('Nonfinite local interval loss', audit)
    with torch.no_grad():
        degrees = torch.rad2deg(torch.acos(cosine.detach()))
        bad_bonds = (distances.detach() < bond_bounds[:, 0]) | (distances.detach() > bond_bounds[:, 1])
        # Match the stable cosine inequalities used for optimization exactly.
        bad_angles = (cosine.detach() < cosine_low) | (cosine.detach() > cosine_high)
        graph_rows = []
        for i in range(len(num_atoms)):
            b_bad = int(bad_bonds[bond_graph == i].sum()); a_bad = int(bad_angles[angle_graph == i].sum())
            graph_rows.append({'graph':i,'bond_constraint_count':bond_counts[i],'angle_constraint_count':angle_counts[i],
                'bond_violation_count':b_bad,'angle_violation_count':a_bad,
                'has_constraints':bool(bond_counts[i]+angle_counts[i]),
                'interval_constraints_satisfied': not (b_bad or a_bad) if bond_counts[i]+angle_counts[i] else None})
        any_bad = bool(bad_bonds.any() | bad_angles.any())
        audit.update(angle_degrees=degrees.tolist(), angle_cosines=cosine.detach().tolist(),
            bond_violation_indices=bad_bonds.nonzero(as_tuple=False).reshape(-1).tolist(),
            angle_violation_indices=bad_angles.nonzero(as_tuple=False).reshape(-1).tolist(),graphs=graph_rows,
            all_declared_constraints_satisfied=(not any_bad) if audit['has_constraints'] else None,
            interval_constraints_satisfied=False if any_bad else (None if audit['coverage']['unconstrained_graph_indices'] else True))
    return {'bond_loss':bond_loss,'angle_loss':angle_loss,'per_graph_bond_loss':per_graph_bond,
            'per_graph_angle_loss':per_graph_angle,'audit':audit}
