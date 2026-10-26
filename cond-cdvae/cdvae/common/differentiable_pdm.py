"""Differentiable pair-PDM geometry using the versioned V2 schema.

Counts unordered site pairs once using their nearest periodic distance. The soft
variant replaces 1[d < cutoff] with a sigmoid, so finite-temperature results are
an approximation, not the exact descriptor used to evaluate generated structures.
"""
from itertools import combinations_with_replacement, product
import math
import torch

_SYMBOLS = ('X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe '
            'Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn '
            'Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W '
            'Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf '
            'Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og').split()
_ATOMIC_NUMBER = {s: z for z, s in enumerate(_SYMBOLS)}


def validate_schema(schema):
    if schema.get('schema_version') != 2 or schema.get('geometry') != 'periodic-minimum-image-clique-v2':
        raise ValueError('Expected the V2 periodic minimum-image schema')
    if schema.get('modes') != ['pair']:
        raise ValueError('This differentiable descriptor supports pair mode only')
    elements = schema.get('elements')
    if not elements or elements != sorted(set(elements)) or any(s not in _ATOMIC_NUMBER or s == 'X' for s in elements):
        raise ValueError('Schema elements must be valid, sorted and unique')
    features = sorted(''.join(p) for p in combinations_with_replacement(elements, 2))
    if schema.get('features') != features or len(features) != len(set(features)):
        raise ValueError('Expected canonical complete pair feature order')
    cutoff = float(schema['cutoff'])
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('Cutoff must be finite and positive')
    return features


def _validate_inputs(frac, atom_types, num_atoms, cells):
    if frac.ndim != 2 or frac.shape[1] != 3 or not frac.is_floating_point():
        raise ValueError('frac_coords must have shape [N,3] and floating dtype')
    if atom_types.ndim != 1 or len(atom_types) != len(frac) or atom_types.dtype not in (torch.int32, torch.int64):
        raise ValueError('atom_types must be integer atomic numbers with shape [N]')
    if num_atoms.ndim != 1 or num_atoms.dtype not in (torch.int32, torch.int64):
        raise ValueError('num_atoms must be an integer [B] tensor')
    if not bool((num_atoms > 0).all()) or int(num_atoms.sum()) != len(frac):
        raise ValueError('num_atoms must be positive and sum to N')
    if cells.shape != (len(num_atoms), 3, 3) or cells.dtype != frac.dtype:
        raise ValueError('cell_matrices must be [B,3,3], row vectors, same dtype as coordinates')
    if any(x.device != frac.device for x in (atom_types, num_atoms, cells)):
        raise ValueError('All input tensors must share a device')
    if not bool(torch.isfinite(frac).all() & torch.isfinite(cells).all()):
        raise ValueError('Coordinates and cells must be finite')
    if not bool((torch.linalg.det(cells) > 0).all()):
        raise ValueError('Cells must have positive determinant')


def nearest_image_displacements(frac_deltas, cell, max_image_range=8, pair_chunk_size=512):
    """Exact nearest-image vectors, piecewise differentiable in deltas and cell.

    For wrapped fractional delta q, use r0=||q C|| as an upper bound on the
    shortest distance. Every better image has fractional coordinate component
    |(q+n)_k| <= r0 ||C^{-1}[:,k]||. Thus integer |n_k| <= floor(r0*reciprocal_norm
    + |q_k|), yielding a finite exact search box. Selection is discrete and
    detached; reconstruction retains gradients in the chosen image. Nearest-
    image ties are nondifferentiable, as for the physical minimum distance.
    No silent 27-image assumption is used for highly skew cells.
    """
    if max_image_range < 1 or pair_chunk_size < 1:
        raise ValueError('Image limit and chunk size must be positive')
    if frac_deltas.ndim != 2 or frac_deltas.shape[1] != 3 or cell.shape != (3, 3):
        raise ValueError('Expected deltas [P,3], cell [3,3]')
    if len(frac_deltas) == 0:
        return frac_deltas @ cell
    with torch.no_grad():
        # Numerical decisions use double precision even for float32 training.
        detached = frac_deltas.detach().double()
        wrap = torch.floor(detached + .5)
        q = detached - wrap
        c = cell.detach().double()
        r0 = torch.linalg.vector_norm(q @ c, dim=-1)
        reciprocal = torch.linalg.vector_norm(torch.linalg.inv(c), dim=0)
        bounds = torch.floor((r0[:, None] * reciprocal + q.abs()).amax(dim=0) + 1e-10).long()
        if int(bounds.max()) > max_image_range:
            raise ValueError(f'Exact image search exceeds configured limit {max_image_range}; cell is too skew/ill-conditioned')
        shifts = torch.tensor(list(product(*(range(-int(b), int(b) + 1) for b in bounds))), device=cell.device, dtype=torch.float64)
        chosen = []
        for chunk in q.split(pair_chunk_size):
            vectors = (chunk[:, None, :] + shifts[None, :, :]) @ c
            indices = vectors.square().sum(-1).argmin(-1)
            chosen.append(shifts[indices])
        image_shift = torch.cat(chosen) - wrap
    return (frac_deltas + image_shift.to(frac_deltas)) @ cell


class PairPDMDescriptor(torch.nn.Module):
    def __init__(self, schema, max_image_range=8, pair_chunk_size=512):
        super().__init__()
        self.features = validate_schema(schema)
        self.schema = dict(schema)
        self.cutoff = float(schema['cutoff'])
        self.max_image_range = max_image_range
        self.pair_chunk_size = pair_chunk_size
        lookup = torch.full((len(_SYMBOLS), len(_SYMBOLS)), -1, dtype=torch.long)
        feature_index = {name: i for i, name in enumerate(self.features)}
        for left, right in combinations_with_replacement(schema['elements'], 2):
            a, b = _ATOMIC_NUMBER[left], _ATOMIC_NUMBER[right]
            lookup[a, b] = lookup[b, a] = feature_index[left + right]
        self.register_buffer('feature_lookup', lookup, persistent=False)

    def forward(self, frac_coords, atom_types, num_atoms, cell_matrices, temperature=0.1, hard=False):
        _validate_inputs(frac_coords, atom_types, num_atoms, cell_matrices)
        if not hard and (not math.isfinite(float(temperature)) or temperature <= 0):
            raise ValueError('Soft cutoff temperature in angstrom must be finite and positive')
        lookup = self.feature_lookup.to(atom_types.device)
        if bool(((atom_types < 1) | (atom_types >= len(_SYMBOLS))).any()):
            raise ValueError('Invalid atomic number')
        if bool((lookup[atom_types, atom_types] < 0).any()):
            raise ValueError('Atomic element absent from PDM schema')
        outputs, start = [], 0
        for graph, n in enumerate(num_atoms.tolist()):
            coords = frac_coords[start:start + n]
            types = atom_types[start:start + n]
            ii, jj = torch.triu_indices(n, n, offset=1, device=frac_coords.device)
            vectors = nearest_image_displacements(coords[jj] - coords[ii], cell_matrices[graph], self.max_image_range, self.pair_chunk_size)
            distance = torch.linalg.vector_norm(vectors, dim=-1)
            weights = (distance < self.cutoff).to(frac_coords.dtype) if hard else torch.sigmoid((self.cutoff - distance) / temperature)
            feature = lookup[types[ii], types[jj]]
            counts = torch.zeros(len(self.features), device=frac_coords.device, dtype=frac_coords.dtype)
            outputs.append(counts.scatter_add(0, feature, weights) + coords.sum() * 0.)
            start += n
        return torch.stack(outputs)


def soft_pair_pdm(frac_coords, atom_types, num_atoms, cell_matrices, schema, temperature=0.1, **kwargs):
    return PairPDMDescriptor(schema, **kwargs)(frac_coords, atom_types, num_atoms, cell_matrices, temperature=temperature)


def standardized_pdm(raw, external_mean, external_std):
    """External train-fitted normalization, before the separate model scaler."""
    mean = torch.as_tensor(external_mean, device=raw.device, dtype=raw.dtype)
    std = torch.as_tensor(external_std, device=raw.device, dtype=raw.dtype)
    if raw.ndim != 2 or mean.shape != (raw.shape[1],) or std.shape != mean.shape:
        raise ValueError('Expected raw [B,D] and mean/std [D]')
    if not bool(torch.isfinite(mean).all() & torch.isfinite(std).all() & (std > 0).all()):
        raise ValueError('Training mean/std must be finite and std positive')
    return (raw - mean) / std


def raw_pdm_from_model_condition(model_values, model_mean, model_std, external_mean, external_std):
    """Invert the model property scaler AND external train-fitted scaler."""
    model_mean = torch.as_tensor(model_mean, device=model_values.device, dtype=model_values.dtype)
    model_std = torch.as_tensor(model_std, device=model_values.device, dtype=model_values.dtype)
    if model_mean.shape != (model_values.shape[1],) or model_std.shape != model_mean.shape:
        raise ValueError('Model property scaler must match PDM feature dimension')
    if not bool(torch.isfinite(model_mean).all() & torch.isfinite(model_std).all() & (model_std > 0).all()):
        raise ValueError('Model property scaler must be finite with positive std')
    # Validate both external vectors through the same canonical transform.
    standardized_pdm(model_values, external_mean, external_std)
    return (model_values * model_std + model_mean) * torch.as_tensor(external_std, device=model_values.device, dtype=model_values.dtype) + torch.as_tensor(external_mean, device=model_values.device, dtype=model_values.dtype)


def denoised_fractional_coordinates(noisy_frac, pred_cart_coord_diff, sigmas_per_atom, num_atoms, cells):
    """x0 = x_sigma + sigma * decoder_output under the native coord_loss scaling.

    Native loss compares (target-x_sigma)/sigma**2 with decoder_output/sigma.
    Consequently a perfect decoder outputs (target-x_sigma)/sigma, not a score
    divided by sigma**2. Do not multiply this output by sigma**2 a second time.
    Coordinates remain unwrapped; the PDM implementation handles periodicity.
    """
    if pred_cart_coord_diff.shape != noisy_frac.shape or sigmas_per_atom.shape != (len(noisy_frac),):
        raise ValueError('Expected decoder [N,3], sigma [N]')
    if not bool(torch.isfinite(sigmas_per_atom).all() & (sigmas_per_atom > 0).all()):
        raise ValueError('Noise scales must be finite and positive')
    inverse_per_atom = torch.linalg.inv(cells).repeat_interleave(num_atoms, dim=0)
    cart_shift = sigmas_per_atom[:, None] * pred_cart_coord_diff
    frac_shift = torch.bmm(cart_shift[:, None, :], inverse_per_atom).squeeze(1)
    return noisy_frac + frac_shift


def denoised_pdm_auxiliary_loss(descriptor, noisy_frac, pred_cart_coord_diff, sigmas_per_atom, atom_types, num_atoms, cells, target_raw_pdm, external_mean, external_std, temperature=0.1, max_sigma=None):
    """Per-feature train-std weighted MSE on the actual denoised structure PDM.

    Optional max_sigma is a predeclared training mask, not selected using test
    precision. Loss averages only eligible structures and returns the count.
    A zero eligible batch returns a differentiable zero, never NaN.
    No additional trainable head or latent property regression is involved.
    """
    x0 = denoised_fractional_coordinates(noisy_frac, pred_cart_coord_diff, sigmas_per_atom, num_atoms, cells)
    if target_raw_pdm.shape != (len(num_atoms), len(descriptor.features)) or not bool(torch.isfinite(target_raw_pdm).all()):
        raise ValueError('Target raw PDM must be finite with shape [B,D]')
    if max_sigma is None:
        mask = torch.ones(len(num_atoms), dtype=torch.bool, device=noisy_frac.device)
    else:
        if not math.isfinite(float(max_sigma)) or max_sigma <= 0:
            raise ValueError('max_sigma must be positive and finite')
        chunks = sigmas_per_atom.split(num_atoms.tolist())
        if any(not bool(torch.allclose(s, s[:1].expand_as(s))) for s in chunks):
            raise ValueError('Expected one common sigma per structure')
        mask = torch.stack([s[0] <= max_sigma for s in chunks])
    eligible = int(mask.sum())
    raw_prediction = x0.new_zeros((len(num_atoms), len(descriptor.features)))
    per_graph_loss = x0.new_zeros(len(num_atoms))
    if eligible:
        atom_mask = mask.repeat_interleave(num_atoms)
        prediction = descriptor(x0[atom_mask], atom_types[atom_mask], num_atoms[mask], cells[mask], temperature=temperature)
        residual = standardized_pdm(prediction, external_mean, external_std) - standardized_pdm(target_raw_pdm[mask].detach(), external_mean, external_std)
        eligible_losses = residual.square().mean(dim=-1)
        raw_prediction = raw_prediction.index_copy(0, mask.nonzero().flatten(), prediction)
        per_graph_loss = per_graph_loss.index_copy(0, mask.nonzero().flatten(), eligible_losses)
        loss = eligible_losses.mean()
    else:
        loss = x0.sum() * 0.0
    # Excluded rows in raw_prediction/per_graph_loss are placeholders; use mask.
    return {'loss': loss, 'per_graph_loss': per_graph_loss, 'eligible_graphs': eligible, 'mask': mask, 'raw_prediction': raw_prediction, 'x0_frac_coords': x0}
