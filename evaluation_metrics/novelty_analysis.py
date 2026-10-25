"""Evaluate novelty with feature-name alignment and finite PDM validation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_metrics.common import read_table, feature_columns, feature_matrix, relative_distances


def calculate_min_distance(new_vec, known_vectors):
    return float(relative_distances(new_vec, known_vectors).min())


def calculate_novelty(known_csv_path, new_csv_path, threshold, n_jobs):
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('Novelty threshold must be finite and nonnegative')
    known, generated = read_table(known_csv_path), read_table(new_csv_path)
    columns = feature_columns(known)
    reference, candidates = feature_matrix(known, columns), feature_matrix(generated, columns)
    distances = Parallel(n_jobs=n_jobs)(delayed(calculate_min_distance)(vector, reference) for vector in candidates)
    novel = np.asarray(distances) > threshold
    result = pd.DataFrame({'structure_id': generated.label, 'min_distance': distances, 'is_novel': novel.astype(int)})
    report = {'known_csv': str(Path(known_csv_path).resolve()), 'new_csv': str(Path(new_csv_path).resolve()),
              'threshold': threshold, 'total_known': len(known), 'total_new': len(generated),
              'novel_count': int(novel.sum()), 'novelty_fraction_percent': float(100 * novel.mean())}
    Path('novelty_report.json').write_text(json.dumps(report, indent=2))
    result.to_csv('novelty_results.csv', index=False)
    print(json.dumps(report, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--known', required=True)
    parser.add_argument('--new', required=True)
    parser.add_argument('-t', '--threshold', type=float, default=0.2)
    parser.add_argument('-j', '--n_jobs', type=int, default=-1)
    args = parser.parse_args()
    calculate_novelty(args.known, args.new, args.threshold, args.n_jobs)
