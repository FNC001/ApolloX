"""Optional target-PDM refinement, reported separately from model-only sampling.

The fixed deterministic optimizer receives a composition, a PDM and a generated
structure. It never receives a reference structure, changes the cell/species,
or restarts sampling. Hard-count evaluation remains an independent operation.
"""
import math
from itertools import product
import torch

from cdvae.common.differentiable_pdm import PairPDMDescriptor, nearest_image_displacements


DEFAULT_CONFIG = {
    'steps': 200, 'temperature': 0.1, 'max_step_A': 0.04,
    'min_distance_A': 1.0, 'geometry_weight': 10.0,
    'line_search_steps': 6,
}


def _config(values):
    config = dict(DEFAULT_CONFIG)
    if values:
        if set(values) - set(config):
            raise ValueError('Unknown PDM refinement configuration')
        config.update(values)
    for name in ('steps', 'line_search_steps'):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(name + ' must be a positive integer')
    for name in ('temperature', 'max_step_A', 'min_distance_A', 'geometry_weight'):
        if isinstance(config[name], bool) or not math.isfinite(float(config[name])) or config[name] <= 0:
            raise ValueError(name + ' must be positive and finite')
    return config



def _finite_scalar(value):
    """Keep failure traces valid JSON without disguising nonfinite values."""
    number = float(value)
    return number if math.isfinite(number) else None


def _validate_target_bounds(targets, types, num_atoms, cells, descriptor):
    """Reject known impossible counts; this is not a full realizability solver.

    Fractional targets (e.g. a training-set mean) are allowed inside the bounds.
    They need not correspond to an exactly attainable integer hard descriptor.
    """
    if bool((targets < 0).any()):
        raise ValueError('Raw PDM targets cannot be negative')
    offset = 0
    lookup = descriptor.feature_lookup.to(types.device)
    corners = torch.tensor(list(product((-.5, .5), repeat=3)), device=cells.device, dtype=cells.dtype)
    for graph, count in enumerate(num_atoms.tolist()):
        graph_types = types[offset:offset + count]
        left, right = torch.triu_indices(count, count, 1, device=types.device)
        feature = lookup[graph_types[left], graph_types[right]]
        upper = torch.bincount(feature, minlength=len(descriptor.features)).to(targets)
        if bool((targets[graph] > upper).any()):
            raise ValueError(f'Graph {graph}: PDM target exceeds composition pair-count bounds')
        # Any fractional displacement can be wrapped into this central box.
        # If all its corners are strictly within cutoff, every distinct pair
        # must be counted regardless of coordinates. Do not assume this bound
        # is the exact covering radius for a skew cell.
        radius_bound = torch.linalg.vector_norm(corners @ cells[graph], dim=-1).max()
        if float(radius_bound) < descriptor.cutoff and bool((targets[graph] != upper).any()):
            raise ValueError(f'Graph {graph}: fixed cell forces all site pairs inside the PDM cutoff')
        offset += count


def _objective(frac, types, cell, target, std, descriptor, config):
    counts = torch.tensor([len(frac)], dtype=torch.long, device=frac.device)
    prediction = descriptor(frac, types, counts, cell[None], temperature=config['temperature'])[0]
    pdm = ((prediction - target) / std).square().mean()
    left, right = torch.triu_indices(len(frac), len(frac), 1, device=frac.device)
    vectors = nearest_image_displacements(frac[right] - frac[left], cell)
    distances = torch.linalg.vector_norm(vectors, dim=-1)
    repulsion = torch.relu(config['min_distance_A'] - distances).square().sum() / len(frac)
    return pdm + config['geometry_weight'] * repulsion, pdm, repulsion


def refine_pdm(frac_coords, atom_types, num_atoms, cell_matrices,
               target_raw, external_std, schema, config=None):
    """Return every input's refined coordinates plus a complete optimizer trace.

    Loss is train-std weighted soft-PDM MSE plus a short-distance penalty. Each
    iteration proposes a bounded Cartesian step, with a fixed backtracking
    schedule accepting only non-increasing total soft loss. No hard evaluation
    score is used for selection. Cells and atom identities stay fixed.

    A stalled/failed optimizer retains its last coordinates and a status; it is
    not replaced by a new sample. This operation cannot establish that the
    neural network learned control; evaluate model-only outputs separately.
    Necessary composition/cell count bounds reject known impossible targets,
    but passing these checks does not prove geometric realizability. At exact
    atomic coincidence the norm has zero gradient; such stationary collisions
    remain unchanged and are explicitly reported, never randomly perturbed.
    """
    cfg = _config(config)
    descriptor = PairPDMDescriptor(schema).to(frac_coords.device)
    with torch.no_grad():
        initial = descriptor(frac_coords, atom_types, num_atoms, cell_matrices, hard=True)
    std = torch.as_tensor(external_std, dtype=frac_coords.dtype, device=frac_coords.device)
    targets = torch.as_tensor(target_raw, dtype=frac_coords.dtype, device=frac_coords.device).detach()
    if targets.shape != initial.shape or std.shape != (initial.shape[1],):
        raise ValueError('Raw target and training std must match the PDM schema')
    if not bool(torch.isfinite(targets).all() & torch.isfinite(std).all() & (std > 0).all()):
        raise ValueError('Targets must be finite and training std strictly positive')
    _validate_target_bounds(targets, atom_types, num_atoms, cell_matrices, descriptor)
    result, reports, offset = [], [], 0
    for graph, count in enumerate(num_atoms.tolist()):
        current = frac_coords[offset:offset + count].detach().clone()
        types = atom_types[offset:offset + count]
        cell = cell_matrices[graph].detach()
        inverse = torch.linalg.inv(cell)
        history = []
        status = 'steps_completed'
        with torch.no_grad():
            first, first_pdm, first_repulsion = _objective(current, types, cell, targets[graph], std, descriptor, cfg)
        for iteration in range(cfg['steps']):
            with torch.enable_grad():
                variable = current.detach().requires_grad_(True)
                objective, pdm, repulsion = _objective(variable, types, cell, targets[graph], std, descriptor, cfg)
                gradient_frac, = torch.autograd.grad(objective, variable)
            gradient_cart = gradient_frac.detach() @ inverse.T
            rms = gradient_cart.square().sum(-1).mean().sqrt()
            row = {'iteration': iteration, 'loss': _finite_scalar(objective.detach()),
                   'soft_pdm_loss': _finite_scalar(pdm.detach()), 'repulsion_loss': _finite_scalar(repulsion.detach()),
                   'gradient_rms': _finite_scalar(rms), 'accepted': False}
            if not bool(torch.isfinite(objective) & torch.isfinite(gradient_cart).all()):
                status = 'nonfinite_objective' if not bool(torch.isfinite(objective)) else 'nonfinite_gradient'
                history.append(row); break
            if float(rms) <= 1e-12:
                status = 'stationary_with_collisions' if float(repulsion.detach()) > 0 else 'stationary'
                history.append(row); break
            displacement = -cfg['max_step_A'] * gradient_cart / rms
            norms = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
            displacement = displacement * torch.clamp(cfg['max_step_A'] / norms.clamp_min(1e-15), max=1.)
            with torch.no_grad():
                for backtrack in range(cfg['line_search_steps']):
                    proposal = torch.remainder(current + (displacement * (0.5 ** backtrack)) @ inverse, 1.)
                    proposed, _, _ = _objective(proposal, types, cell, targets[graph], std, descriptor, cfg)
                    if bool(torch.isfinite(proposed)) and float(proposed) <= row['loss']:
                        current = proposal
                        row.update(accepted=True, backtracks=backtrack, accepted_loss=float(proposed))
                        break
            history.append(row)
            if not row['accepted']:
                status = 'line_search_stalled'; break
        with torch.no_grad():
            final, final_pdm, final_repulsion = _objective(current, types, cell, targets[graph], std, descriptor, cfg)
        result.append(current)
        reports.append({'graph': graph, 'status': status, 'iterations': len(history),
                        'initial_loss': _finite_scalar(first), 'final_loss': _finite_scalar(final),
                        'initial_soft_pdm_loss': _finite_scalar(first_pdm), 'final_soft_pdm_loss': _finite_scalar(final_pdm),
                        'initial_repulsion_loss': _finite_scalar(first_repulsion), 'final_repulsion_loss': _finite_scalar(final_repulsion),
                        'history': history})
        offset += count
    coordinates = torch.cat(result)
    with torch.no_grad():
        final_counts = descriptor(coordinates, atom_types, num_atoms, cell_matrices, hard=True)
    return {'frac_coords': coordinates, 'cell_matrices': cell_matrices.detach().clone(),
            'initial_hard_raw': initial, 'final_hard_raw': final_counts,
            'reports': reports, 'config': cfg,
            'method': 'fixed_cell_target_pdm_refinement_v1'}
