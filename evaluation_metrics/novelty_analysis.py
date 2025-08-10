# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import argparse
import json
import os
from joblib import Parallel, delayed

def calculate_min_distance(new_vec, known_vectors):
    """计算单个向量到所有已知向量的最小相对误差距离（双向取小）"""
    eps = 1e-8  # 防止除 0
    diff = np.abs(known_vectors - new_vec)
    mre_known_to_new = np.mean(diff / (np.abs(new_vec) + eps), axis=1)        # B 相对 A
    mre_new_to_known = np.mean(diff / (np.abs(known_vectors) + eps), axis=1)  # A 相对 B
    distances = np.minimum(mre_known_to_new, mre_new_to_known)
    return float(np.min(distances))

def calculate_novelty(known_csv_path, new_csv_path, threshold, n_jobs):
    """
    Calculates the novelty fraction of a set of new items against a known database.
    Uses parallel processing.
    """
    try:
        # --- 1. Load Data Files ---
        print("Loading data files...")
        df_known = pd.read_csv(known_csv_path)
        df_new = pd.read_csv(new_csv_path)

        # --- 2. Extract Feature Vectors ---
        known_vectors = df_known.iloc[:, 2:].values
        new_vectors = df_new.iloc[:, 2:].values

        if len(new_vectors) == 0:
            print("Error: The file with new candidates is empty.")
            return

        if known_vectors.shape[1] != new_vectors.shape[1]:
            print(f"Error: Feature dimensions do not match between files!")
            print(f"  - Known reference file has {known_vectors.shape[1]} features.")
            print(f"  - New candidates file has {new_vectors.shape[1]} features.")
            return

        print(f"Loaded reference DB: {len(known_vectors)} items")
        print(f"Loaded candidate set: {len(new_vectors)} items")
        print("-" * 40)

        # --- 3. Parallel Calculation ---
        print(f"Calculating novelty in parallel (n_jobs={n_jobs})...")
        min_distances = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
            delayed(calculate_min_distance)(new_vec, known_vectors) for new_vec in new_vectors
        )

        # --- 4. Evaluate Novelty ---
        results = []
        novel_count = 0
        for idx, min_distance in enumerate(min_distances):
            is_novel = min_distance > threshold
            if is_novel:
                novel_count += 1
            results.append({
                "structure_id": df_new.iloc[idx, 0],
                "min_distance": min_distance,
                "is_novel": int(is_novel)
            })

        # --- 5. Summary ---
        total_new = len(new_vectors)
        novelty_fraction = (novel_count / total_new) * 100 if total_new > 0 else 0

        report_data = {
            "known_csv": os.path.abspath(known_csv_path),
            "new_csv": os.path.abspath(new_csv_path),
            "threshold": threshold,
            "total_known": len(known_vectors),
            "total_new": total_new,
            "novel_count": novel_count,
            "novelty_fraction_percent": novelty_fraction
        }

        with open("novelty_report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
        print("Saved novelty_report.json")

        df_results = pd.DataFrame(results)
        df_results.to_csv("novelty_results.csv", index=False)
        print("Saved novelty_results.csv")

        print("\n--- Novelty Analysis Report ---")
        print(json.dumps(report_data, indent=4))

    except FileNotFoundError as e:
        print(f"Error: File not found: {e.filename}")
    except Exception as e:
        print(f"An unknown error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate novelty fraction in parallel and save results to JSON and CSV.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--known", type=str, required=True,
                        help="Path to the CSV file of the known reference database.")
    parser.add_argument("--new", type=str, required=True,
                        help="Path to the CSV file of new candidates.")
    parser.add_argument("-t", "--threshold", type=float, default=0.2,
                        help="Novelty threshold (default: 0.2)")
    parser.add_argument("-j", "--n_jobs", type=int, default=-1,
                        help="Number of parallel jobs (-1 means all cores)")

    args = parser.parse_args()
    calculate_novelty(args.known, args.new, args.threshold, args.n_jobs)
