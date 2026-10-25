"""Validated, named PDM tables shared by the evaluation entrypoints."""
import ast
import csv
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apollox_common.pdm import METADATA_COLUMNS


def read_table(path):
    with open(path, newline='') as stream:
        columns = next(csv.reader(stream), [])
    if len(columns) != len(set(columns)):
        raise ValueError(f'Duplicate column names in {path}')
    table = pd.read_csv(path, dtype={'label': str, 'material_id': str}, keep_default_na=False)
    if 'label' not in table and 'material_id' in table:
        table = table.rename(columns={'material_id': 'label'})
    if 'label' not in table:
        raise ValueError(f'{path} requires a label or material_id column')
    if table.empty:
        raise ValueError(f'{path} contains no structures')
    if table.label.isna().any() or table.label.str.strip().eq('').any():
        raise ValueError(f'{path} contains missing structure IDs')
    return table


def feature_columns(table):
    columns = [name for name in table if name not in METADATA_COLUMNS and name != 'job_id']
    if not columns:
        raise ValueError('No PDM features found')
    return columns


def feature_matrix(table, columns=None):
    actual = feature_columns(table)
    columns = actual if columns is None else list(columns)
    if set(actual) != set(columns):
        raise ValueError(f'PDM feature names differ: expected {columns}, received {actual}')
    matrix = table[columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError('PDM features must be finite numeric values')
    return matrix


def parse_vector(value, expected_size=None):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError('element_values must be a numeric vector') from exc
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError('element_values must be a numeric vector')
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
        raise ValueError('element_values must be a nonempty, finite 1D vector')
    if expected_size is not None and len(vector) != expected_size:
        raise ValueError(f'PDM vector has {len(vector)} features; expected {expected_size}')
    return vector


def relative_distances(vector, matrix):
    difference = np.abs(matrix - vector)
    return np.minimum(np.mean(difference / (np.abs(vector) + 1e-8), axis=1),
                      np.mean(difference / (np.abs(matrix) + 1e-8), axis=1))
