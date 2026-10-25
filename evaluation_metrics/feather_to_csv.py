"""Restore named PDM counts from vectors using the training scaler and schema."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apollox_common.pdm import load_schema
from evaluation_metrics.common import parse_vector, read_table, feature_columns
from prepare_dataset.standardize import load_scaler_stats


def parse_element_values(value):
    return parse_vector(value)


def main(feather_path, reference_csv, scaler_stats_txt, output_csv, schema_path=None, allow_legacy_order=False):
    frame = pd.read_feather(feather_path)
    schema_path = Path(schema_path) if schema_path else Path(scaler_stats_txt).parent / 'pdm_schema.json'
    source_schema_path = Path(feather_path).parent / 'pdm_schema.json'
    source_scaler_path = Path(feather_path).parent / 'scaler_stats.txt'
    if schema_path.is_file():
        schema = load_schema(schema_path)
        if not source_schema_path.is_file() or not source_scaler_path.is_file():
            raise FileNotFoundError('V2 Feather input must carry its source pdm_schema.json and scaler_stats.txt')
        if load_schema(source_schema_path) != schema:
            raise ValueError('Feather source schema differs from the requested scaler schema')
        source_stats = load_scaler_stats(source_scaler_path)
        requested_stats = load_scaler_stats(scaler_stats_txt)
        if any(not np.array_equal(source, requested) for source, requested in zip(source_stats, requested_stats)):
            raise ValueError('Feather source scaler statistics differ from the requested scaler statistics')
        columns = schema['features']
        if reference_csv is not None:
            reference_columns = feature_columns(read_table(reference_csv))
            if set(reference_columns) != set(columns):
                raise ValueError('Reference feature names differ from the training schema')
    elif allow_legacy_order and reference_csv is not None and not source_schema_path.exists():
        columns = feature_columns(read_table(reference_csv))
    else:
        raise FileNotFoundError('Training pdm_schema.json is required; legacy use requires --allow-legacy-order and --ref_csv with the verified training column order')
    means, stds = load_scaler_stats(scaler_stats_txt)
    if len(means) != len(columns) or len(stds) != len(columns):
        raise ValueError('Scaler statistics length does not match the training PDM schema')
    if 'material_id' not in frame or 'element_values' not in frame:
        raise ValueError('Feather requires material_id and element_values columns')
    if frame.material_id.isna().any() or frame.material_id.astype(str).str.strip().eq('').any() or frame.material_id.duplicated().any():
        raise ValueError('Feather structure IDs must be nonempty and unique')
    vectors = [parse_vector(value, len(columns)) for value in frame.element_values]
    matrix = np.stack(vectors) if vectors else np.empty((0, len(columns)))
    restored = matrix * np.asarray(stds) + np.asarray(means)
    if not np.isfinite(restored).all():
        raise ValueError('Inverse scaling produced nonfinite PDM values')
    result = pd.DataFrame(restored, columns=columns)
    result.insert(0, 'label', frame.material_id.to_numpy())
    if 'formula' in frame:
        formula = frame.formula.to_numpy()
    else:
        formula = frame.material_id.astype(str).str.split('-').str[1].fillna('').str.replace('_', '', regex=False).to_numpy()
    result.insert(1, 'formula', formula)
    result.to_csv(output_csv, index=False)
    print(f'Successfully saved to {output_csv}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feather', required=True)
    parser.add_argument('--ref_csv', help='Optional reference table; required only for explicitly permitted legacy scalers')
    parser.add_argument('--scaler_stats', required=True)
    parser.add_argument('--schema', help='Defaults to pdm_schema.json next to the scaler')
    parser.add_argument('--allow-legacy-order', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.feather, args.ref_csv, args.scaler_stats, args.output, args.schema, args.allow_legacy_order)
