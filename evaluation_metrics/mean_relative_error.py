import pandas as pd
import numpy as np
import argparse

def compute_average_relative_error(true_csv, regenerate_csv, output_csv, eps=1e-8):
    # Load both CSVs
    df_true = pd.read_csv(true_csv)
    df_pred = pd.read_csv(regenerate_csv)

    if df_true.shape != df_pred.shape:
        raise ValueError("The CSV files must have the same shape.")

    # Extract labels and values
    labels = df_true.iloc[:, 0]
    true_values = df_true.iloc[:, 2:].values
    pred_values = df_pred.iloc[:, 2:].values

    # Compute relative errors
    abs_diff = np.abs(pred_values - true_values)
    rel_error = abs_diff / (np.abs(true_values) + eps)

    # Mean relative error per row
    mean_rel_error = np.mean(rel_error, axis=1)

    # Output DataFrame
    output_df = pd.DataFrame({
        'label': labels,
        'mean_relative_error': mean_rel_error
    })

    # Save
    output_df.to_csv(output_csv, index=False)
    print(f"Average relative errors saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute row-wise mean relative error between two CSVs.")
    parser.add_argument("--true_csv", type=str, required=True, help="Path to the ground truth CSV file.")
    parser.add_argument("--regenerate_csv", type=str, required=True, help="Path to the predicted CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save output CSV.")

    args = parser.parse_args()
    compute_average_relative_error(args.true_csv, args.regenerate_csv, args.output)
