import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime


def analyze_and_save(file_path, threshold, output_csv, output_json):
    """
    Analyzes item uniqueness, saves detailed results to CSV, and summary to JSON.

    Args:
        file_path (str): The path to the input CSV file.
        threshold (float): The Euclidean distance threshold for clustering.
        output_csv (str): The path for the output CSV file.
        output_json (str): The path for the output JSON summary file.
    """
    try:
        # Read data from the input CSV file
        df = pd.read_csv(file_path)

        # Assume numerical data for comparison starts from the 3rd column (index 2)
        X = df.iloc[:, 2:].values

        n_structures = X.shape[0]
        if n_structures == 0:
            print("Input file is empty. Nothing to analyze.")
            return

        # --- Clustering Algorithm ---
        eps = 1e-8  # 防止除以 0
        # --- Clustering Algorithm ---
        cluster_ids = [-1] * n_structures
        current_cluster_id = 0
        for i in range(n_structures):
            if cluster_ids[i] == -1:
                current_cluster_id += 1
                cluster_ids[i] = current_cluster_id
                for j in range(i + 1, n_structures):
                    if cluster_ids[j] == -1:
                        # 平均相对误差
                        diff = np.abs(X[i] - X[j])
                        mre_ab = np.mean(diff / (np.abs(X[j]) + eps))  # A相对于B
                        mre_ba = np.mean(diff / (np.abs(X[i]) + eps))  # B相对于A
                        distance = min(mre_ab, mre_ba)  # 取更小的

                        if distance < threshold:
                            cluster_ids[j] = current_cluster_id

        # --- Prepare CSV Output ---
        result_df = df.copy()
        result_df['cluster_id'] = cluster_ids
        result_df['status'] = 'duplicate'
        first_indices = result_df.drop_duplicates(subset='cluster_id', keep='first').index
        result_df.loc[first_indices, 'status'] = 'unique_representative'

        # Save the detailed results to the CSV file
        result_df.to_csv(output_csv, index=False)

        # --- Prepare JSON Summary Output ---
        unique_count = len(result_df['cluster_id'].unique())
        uniqueness_percentage = (unique_count / n_structures) * 100 if n_structures > 0 else 0

        summary_data = {
            "analysis_summary": {
                "input_file": file_path,
                "analysis_timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "total_items_analyzed": n_structures,
                "uniqueness_threshold": threshold,
                "unique_clusters_found": unique_count,
                "uniqueness_percentage": round(uniqueness_percentage, 2)
            },
            "output_files": {
                "csv_results": output_csv
            }
        }

        # Save the summary dictionary to a JSON file
        with open(output_json, 'w') as f:
            json.dump(summary_data, f, indent=4)

        # --- Print Summary to Console ---
        print("--- Analysis Complete ---")
        print(f"Input File: {file_path}")
        print(f"Total items analyzed: {n_structures}")
        print(f"Uniqueness Threshold: {threshold}")
        print(f"Number of unique clusters found: {unique_count}")
        print(f"Uniqueness Percentage: {uniqueness_percentage:.2f}%")
        print("\n✅ Detailed results saved to:", output_csv)
        print("✅ Analysis summary saved to:", output_json)

    except FileNotFoundError:
        print(f"Error: Input file not found at '{file_path}'. Please check the path.")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster items in a CSV, save detailed results to CSV, and summary to JSON.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("file_path", type=str, help="Path to the input CSV file.")

    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.2,
        help="Euclidean distance threshold for clustering.\n(default: 0.2)"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="uniqueness_results.csv",
        help="Path for the output CSV file with detailed results.\n(default: results.csv)"
    )

    parser.add_argument(
        "-j", "--json",
        type=str,
        default="uniqueness_report.json",
        help="Path for the output JSON file with the analysis summary.\n(default: summary.json)"
    )

    args = parser.parse_args()

    analyze_and_save(args.file_path, args.threshold, args.output, args.json)