import pandas as pd
import argparse
import ast

def parse_element_values(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    return []

def load_scaler_stats(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    mean_line = next((line for line in lines if line.startswith('Train Mean:')), None)
    std_line = next((line for line in lines if line.startswith('Train Std:')), None)

    if not mean_line or not std_line:
        raise ValueError("Scaler stats file must contain 'Train Mean:' and 'Train Std:' lines.")

    means = list(map(float, mean_line.replace('Train Mean:', '').strip().split(',')))
    stds = list(map(float, std_line.replace('Train Std:', '').strip().split(',')))

    return means, stds

def main(feather_path, reference_csv, scaler_stats_txt, output_csv):
    # Load feather data
    df = pd.read_feather(feather_path)

    # Load reference CSV and get column names from the 3rd column onward
    ref_df = pd.read_csv(reference_csv)
    pair_columns = ref_df.columns[2:].tolist()

    # Load scaler stats
    means, stds = load_scaler_stats(scaler_stats_txt)
    if len(means) != len(pair_columns) or len(stds) != len(pair_columns):
        raise ValueError("Mismatch between scaler stats length and number of feature columns.")

    # Prepare output DataFrame
    output_df = pd.DataFrame()
    output_df['label'] = df['material_id']
    output_df['formula'] = df['material_id'].str.split('-').str[1].str.replace('_', '', regex=False)

    # Parse and inverse transform element_values
    element_values = df['element_values'].apply(parse_element_values).tolist()

    for i, col in enumerate(pair_columns):
        restored_values = [
            row[i] * stds[i] + means[i] if i < len(row) else None
            for row in element_values
        ]
        output_df[col] = restored_values

    # Save to CSV
    output_df.to_csv(output_csv, index=False)
    print(f"Successfully saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert feather file to CSV and inverse-transform features.")
    parser.add_argument('--feather', type=str, required=True, help='Path to the .feather input file')
    parser.add_argument('--ref_csv', type=str, required=True, help='Path to the reference .csv file')
    parser.add_argument('--scaler_stats', type=str, required=True, help='Path to the scaler_stats.txt file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output .csv file')

    args = parser.parse_args()
    main(args.feather, args.ref_csv, args.scaler_stats, args.output)


