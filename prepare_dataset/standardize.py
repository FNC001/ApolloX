import pandas as pd
import numpy as np
import argparse
import re

def load_scaler_stats(filepath):
    with open(filepath, 'r') as f:
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

def main(input_csv, scaler_file, output_csv):
    """Scale PDM columns in their training order, preserving the first two metadata columns.

    The legacy scaler file contains no feature names, so callers must supply the
    same PDM columns in the same order used to fit the training scaler.
    """
    data = pd.read_csv(input_csv)
    metadata=data.iloc[:,:2]
    features=data.iloc[:,2:]

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
    # Save to CSV
    result.to_csv(output_csv, index=False)
    print(f"Standardized data saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standardize data using mean and std from scaler_stats.txt")
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--scaler', type=str, default='scaler_stats.txt', help='Path to scaler statistics file')
    parser.add_argument('--output', type=str, default='standardized_optimized_particles_distribution.csv', help='Path to output CSV file')
    args = parser.parse_args()

    main(args.input, args.scaler, args.output)
