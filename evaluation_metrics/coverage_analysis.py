import pandas as pd
import numpy as np
import argparse

def calculate_coverage(known_csv_path, new_csv_path, threshold, k_list):
    """
    Calculates coverage recall and precision between known and new structures.

    Args:
        known_csv_path (str): Path to known structures CSV.
        new_csv_path (str): Path to new structures CSV.
        threshold (float): Distance threshold for considering "covered".
        k_list (list of int): Number of top-k known structures to consider for coverage.
    """
    try:
        print("Loading data files...")
        df_known = pd.read_csv(known_csv_path)
        df_new = pd.read_csv(new_csv_path)

        known_vectors = df_known.iloc[:, 2:].values
        new_vectors = df_new.iloc[:, 2:].values

        if known_vectors.shape[1] != new_vectors.shape[1]:
            print("Feature dimension mismatch.")
            return

        num_known = len(known_vectors)
        num_new = len(new_vectors)

        print(f"Known structures: {num_known}")
        print(f"New structures:   {num_new}")
        print(f"Threshold for coverage: {threshold}")
        print("-" * 40)

        # 计算两两之间的 mean absolute difference (等效于1范数的平均)
        eps = 1e-8  # 防止除以 0
        print("Calculating pairwise distances (mean relative error)...")
        distance_matrix = np.zeros((num_new, num_known))
        for i_new, new_vec in enumerate(new_vectors):
            diff = np.abs(known_vectors - new_vec)
            mre_known_to_new = np.mean(diff / (np.abs(new_vec) + eps), axis=1)  # B 相对 A
            mre_new_to_known = np.mean(diff / (np.abs(known_vectors) + eps), axis=1)  # A 相对 B
            distance_matrix[i_new] = np.minimum(mre_known_to_new, mre_new_to_known)

        results = []  # 用于存储最终结果

        # 针对每个k值，计算覆盖率
        for k in k_list:
            print(f"\n--- Evaluating Top-{k} Coverage ---")

            known_covered = np.zeros(num_known, dtype=bool)
            new_covered = np.zeros(num_new, dtype=bool)

            for i_new, distances in enumerate(distance_matrix):
                top_k_indices = np.argsort(distances)[:k]
                top_k_distances = distances[top_k_indices]

                if np.all(top_k_distances < threshold):
                    known_covered[top_k_indices] = True
                    new_covered[i_new] = True

            recall = 100 * np.sum(known_covered) / num_known
            precision = 100 * np.sum(new_covered) / num_new

            print(f"Coverage Recall:    {recall:.2f}%")
            print(f"Coverage Precision: {precision:.2f}%")
            print("-" * 40)

            # 存入结果列表
            results.append({
                "k": k,
                "recall": recall,
                "precision": precision
            })

        # 保存为 CSV
        df_results = pd.DataFrame(results)
        df_results.to_csv("coverage_results.csv", index=False)
        print("Saved results to coverage_results.csv")

    except Exception as e:
        print(f"Error during coverage calculation: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate coverage recall and precision of generated structures against known ones."
    )
    parser.add_argument("--known", type=str, required=True, help="Path to known CSV.")
    parser.add_argument("--new", type=str, required=True, help="Path to new generated CSV.")
    parser.add_argument("-t", "--threshold", type=float, default=0.2, help="Threshold for considering a structure as covered.")
    parser.add_argument("-k", "--top_k", type=int, nargs='+', default=[1],, help="List of k values to evaluate (e.g., -k 1 5 10).")

    args = parser.parse_args()
    calculate_coverage(args.known, args.new, args.threshold, args.top_k)
