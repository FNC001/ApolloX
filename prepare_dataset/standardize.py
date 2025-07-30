import pandas as pd
import numpy as np
import argparse
import re

def load_scaler_stats(filepath):
    with open(filepath, 'r') as f:
        text = f.read()

    # Extract mean and standard deviation values
    mean_match = re.search(r'Train Mean:\s*([0-9.,\s]+)', text)
    std_match = re.search(r'Train Std:\s*([0-9.,\s]+)', text)

    if not mean_match or not std_match:
        raise ValueError("Invalid format in scaler_stats.txt: missing 'Train Mean' or 'Train Std'")

    means = np.fromstring(mean_match.group(1), sep=',')
    stds = np.fromstring(std_match.group(1), sep=',')

    return means, stds

def main(input_csv, scaler_file, output_csv):
    data = pd.read_csv(input_csv)
    metadata=data.iloc[:,:2]
    features=data.iloc[:,2:]

    # Drop the 'energy' column if it exists
    features = features.drop(columns=['Energy'], errors='ignore')

    # Load scaler statistics
    means, stds = load_scaler_stats(scaler_file)

    if features.shape[1] != len(means):
        raise ValueError(f"Mismatch: input data has {data.shape[1]} columns, but {len(means)} mean values were provided")

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
