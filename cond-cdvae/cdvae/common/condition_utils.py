"""Validation and scaling shared by command-line and CSV PDM generation.

Inputs use the externally standardized PDM values stored in the training data.
The saved model property scaler is applied afterwards, just as in CrystDataset.
This module deliberately depends only on PyTorch and the standard library.
"""

import json

import torch


def validate_vector_tensor(values, n_in, name="element_values"):
    """Require one finite, fixed-width vector per sample without reshaping it."""
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a tensor with shape [B, {n_in}]")
    if values.ndim != 2 or values.shape[1] != n_in:
        raise ValueError(
            f"{name} must have shape [B, {n_in}], got {tuple(values.shape)}"
        )
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite numbers")
    return values


def pdm_input_dim(model):
    """Read the active embedding, rather than the repository's default config."""
    keys = list(model.multiemb.cond_keys)
    if "element_values" not in keys:
        return None
    embedding = model.multiemb.sub_emb_list[keys.index("element_values")]
    n_in = getattr(embedding, "n_in", None)
    if not isinstance(n_in, int) or isinstance(n_in, bool) or n_in < 1:
        raise ValueError("The active element_values embedding needs a positive n_in")
    return n_in


def _parse_pdm_row(value, n_in):
    try:
        if isinstance(value, str):
            value = value.strip()
            value = json.loads(value) if value.startswith("[") else value.split(",")
            value = [float(item) for item in value]
        tensor = torch.as_tensor(value, dtype=torch.get_default_dtype())
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("element_values must be a numeric vector") from exc
    if tensor.ndim != 1:
        raise ValueError("Each element_values input must be a one-dimensional vector")
    return validate_vector_tensor(tensor.unsqueeze(0), n_in)


def pdm_tensor_for_model(model, values, *, batched=False):
    """Validate supplied PDM against the loaded model, without normalizing it."""
    n_in = pdm_input_dim(model)
    if n_in is None:
        if values is not None:
            raise ValueError(
                "element_values was supplied, but the loaded model does not enable "
                "PDM conditioning. Use a model trained with element_values, or omit "
                "the PDM input for composition-only generation."
            )
        return None
    if values is None:
        raise ValueError(
            f"The loaded model requires element_values with {n_in} values; supply "
            "--element_values for gen or an element_values column for target."
        )
    if not batched:
        return _parse_pdm_row(values, n_in)
    rows = []
    for index, value in enumerate(values):
        try:
            rows.append(_parse_pdm_row(value, n_in))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid element_values at batch row {index}: {exc}") from exc
    if not rows:
        raise ValueError("element_values batch must contain at least one row")
    return torch.cat(rows, dim=0)


def prepare_pdm_values(model, values, prop_names, prop_scalers, *, batched=False):
    """Validate and apply the same saved PDM scaler used by the training dataset."""
    tensor = pdm_tensor_for_model(model, values, batched=batched)
    if tensor is None:
        return None
    names = list(prop_names)
    if len(names) != len(prop_scalers):
        raise ValueError("Model property names and saved property scalers do not match")
    if names.count("element_values") != 1:
        raise ValueError("The PDM model must have exactly one element_values property scaler")
    scaler = prop_scalers[names.index("element_values")]
    n_in = tensor.shape[1]
    for label in ("means", "stds"):
        stats = torch.as_tensor(getattr(scaler, label))
        if stats.shape != (n_in,) or not torch.isfinite(stats).all():
            raise ValueError(
                f"Saved element_values scaler {label} must be a finite vector of length {n_in}"
            )
        if label == "stds" and (stats <= 0).any():
            raise ValueError("Saved element_values scaler stds must be positive")
    scaler.match_device(tensor)
    transformed = scaler.transform(tensor)
    if transformed.shape != tensor.shape:
        raise ValueError("Saved element_values scaler must preserve the input batch shape")
    return validate_vector_tensor(transformed, n_in)


def prepare_scalar_values(name, values, prop_names, prop_scalers):
    """Scale a CSV scalar column exactly as the training dataset does."""
    names = list(prop_names)
    if len(names) != len(prop_scalers) or names.count(name) != 1:
        raise ValueError(f"Condition {name!r} requires exactly one saved property scaler")
    try:
        tensor = torch.as_tensor(values, dtype=torch.get_default_dtype())
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Condition {name!r} must contain numeric scalar values") from exc
    if tensor.ndim != 1 or not torch.isfinite(tensor).all():
        raise ValueError(f"Condition {name!r} must contain one finite scalar per sample")
    tensor = tensor.unsqueeze(1)
    scaler = prop_scalers[names.index(name)]
    for label in ("means", "stds"):
        stats = torch.as_tensor(getattr(scaler, label))
        if stats.numel() != 1 or not torch.isfinite(stats).all():
            raise ValueError(f"Saved {name} scaler {label} must be a finite scalar")
        if label == "stds" and (stats <= 0).any():
            raise ValueError(f"Saved {name} scaler stds must be positive")
    scaler.match_device(tensor)
    result = scaler.transform(tensor)
    if result.shape != tensor.shape or not torch.isfinite(result).all():
        raise ValueError(f"Saved {name} scaler must preserve finite scalar batch values")
    return result
