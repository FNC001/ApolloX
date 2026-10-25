"""Evaluate top-k threshold coverage using named, finite PDM features.

A generated structure is covered when all of its k nearest reference structures
are within the threshold. Recall counts the union of these covered neighbors.
For k=1 this preserves the existing nearest-neighbor coverage definition.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_metrics.common import read_table, feature_columns, feature_matrix, relative_distances


def calculate_coverage(known_csv_path, new_csv_path, threshold, k_list):
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Coverage threshold must be finite and positive')
    known, generated = read_table(known_csv_path), read_table(new_csv_path)
    columns = feature_columns(known)
    reference, candidates = feature_matrix(known, columns), feature_matrix(generated, columns)
    if any(not isinstance(k, int) or k < 1 or k > len(reference) for k in k_list):
        raise ValueError('Each top-k value must be between 1 and the reference population size')
    distances = np.array([relative_distances(vector, reference) for vector in candidates])
    results = []
    for k in k_list:
        known_covered = np.zeros(len(reference), dtype=bool)
        new_covered = np.zeros(len(candidates), dtype=bool)
        for i, row in enumerate(distances):
            nearest = np.argsort(row, kind='stable')[:k]
            if np.all(row[nearest] < threshold):
                known_covered[nearest] = True
                new_covered[i] = True
        results.append({'k': k, 'recall': 100 * known_covered.mean(), 'precision': 100 * new_covered.mean()})
    frame = pd.DataFrame(results)
    frame.to_csv('coverage_results.csv', index=False)
    print(frame.to_string(index=False))
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--known', required=True)
    parser.add_argument('--new', required=True)
    parser.add_argument('-t', '--threshold', type=float, default=0.2)
    parser.add_argument('-k', '--top_k', type=int, nargs='+', default=[1])
    args = parser.parse_args()
    calculate_coverage(args.known, args.new, args.threshold, args.top_k)
