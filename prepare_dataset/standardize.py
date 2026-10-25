import pandas as pd
import numpy as np
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apollox_common.pdm import METADATA_COLUMNS, align_features, load_schema, save_schema

def load_scaler_stats(filepath):
    with Path(filepath).expanduser().open() as f:
        text = f.read()

    # Extract mean and standard deviation values
    mean_match = re.search(r'^Train Mean:[ \t]*([^\r\n]+)', text, re.MULTILINE)
    std_match = re.search(r'^Train Std:[ \t]*([^\r\n]+)', text, re.MULTILINE)

    if not mean_match or not std_match:
        raise ValueError("Invalid format in scaler_stats.txt: missing 'Train Mean' or 'Train Std'")

    means = np.array([float(value) for value in mean_match.group(1).split(',')])
    stds = np.array([float(value) for value in std_match.group(1).split(',')])
    if len(means) != len(stds):
        raise ValueError("Scaler means and standard deviations must have the same length")
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError("Scaler statistics must be finite with strictly positive standard deviations")

    return means, stds

def main(input_csv, scaler_file, output_csv, schema_path=None, allow_legacy_order=False):
    """Align raw PDM by the training schema, then apply its saved scaler."""
    data = pd.read_csv(input_csv, dtype={key: str for key in METADATA_COLUMNS
                                       if key not in {'eval_index', 'sample_index', 'Energy'}},
                       keep_default_na=False)
    schema_path = Path(schema_path or Path(scaler_file).expanduser().parent / 'pdm_schema.json').expanduser()
    schema = load_schema(schema_path) if schema_path.exists() else None
    if schema is not None:
        for companion in (
            Path(scaler_file).expanduser().parent / 'pdm_schema.json',
            Path(input_csv).expanduser().parent / 'pdm_schema.json',
        ):
            if companion.exists() and load_schema(companion) != schema:
                raise ValueError(f"Input/scaler PDM schema differs from the training schema: {companion}")
        features = align_features(data, schema)
        metadata = data[[column for column in data if column in METADATA_COLUMNS and column != 'Energy']]
    elif allow_legacy_order:
        metadata = data.iloc[:, :2]
        features = data.iloc[:, 2:].drop(columns=['Energy'], errors='ignore')
    else:
        raise ValueError(
            f"Training PDM schema not found: {schema_path}. Regenerate V2 data, or explicitly "
            "use --allow-legacy-order only for an old model with verified legacy column order."
        )

    # Drop the 'energy' column if it exists
    features = features.drop(columns=['Energy'], errors='ignore')
    if features.shape[1] == 0:
        raise ValueError("No PDM feature columns found after the two metadata columns")
    features = features.apply(pd.to_numeric, errors='raise')
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("PDM features must contain only finite numeric values")

    # Load scaler statistics
    means, stds = load_scaler_stats(scaler_file)

    if features.shape[1] != len(means):
        raise ValueError(f"Mismatch: input data has {features.shape[1]} PDM features, but {len(means)} scaler values were provided")

    # Standardize the data
    standardized_features = (features - means) / stds
    result=pd.concat([metadata,standardized_features],axis=1)
    if not np.isfinite(standardized_features.to_numpy()).all():
        raise ValueError("PDM scaling produced non-finite values")
    if schema is not None:
        save_schema(Path(output_csv).expanduser().parent / 'pdm_schema.json', schema)
    # Save to CSV
    result.to_csv(output_csv, index=False)
    print(f"Standardized data saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standardize data using mean and std from scaler_stats.txt")
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--scaler', type=str, default='scaler_stats.txt', help='Path to scaler statistics file')
    parser.add_argument('--output', type=str, default='standardized_optimized_particles_distribution.csv', help='Path to output CSV file')
    parser.add_argument('--schema', help='Training pdm_schema.json (default: beside scaler file)')
    parser.add_argument('--allow-legacy-order', action='store_true', help='Explicitly accept an old scaler without a feature schema')
    args = parser.parse_args()

    main(args.input, args.scaler, args.output, args.schema, args.allow_legacy_order)
