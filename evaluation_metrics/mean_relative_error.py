"""Reconstruction errors aligned by structure ID and PDM feature name."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_metrics.common import read_table, feature_columns, feature_matrix


def compute_average_relative_error(true_csv, regenerate_csv, output_csv, eps=1e-8, eval_index=None):
    truth, predicted = read_table(true_csv), read_table(regenerate_csv)
    if 'eval_index' in predicted:
        if predicted.eval_index.isna().any():
            raise ValueError('Missing eval_index values in reconstruction data')
        if eval_index is not None:
            predicted = predicted.loc[predicted.eval_index == eval_index].copy()
            if predicted.empty:
                raise ValueError(f'No reconstruction rows for eval_index={eval_index}')
        elif predicted.eval_index.nunique() > 1:
            raise ValueError('Multiple reconstruction evaluations found; select --eval-index')
    elif eval_index is not None:
        raise ValueError('--eval-index requires an eval_index column')
    for name, table in [('ground truth', truth), ('reconstruction', predicted)]:
        if table.label.duplicated().any():
            raise ValueError(f'Duplicate structure IDs in {name}')
    if set(truth.label) != set(predicted.label):
        raise ValueError('Ground truth and reconstruction must contain exactly the same structure IDs')
    columns = feature_columns(truth)
    predicted = predicted.set_index('label').loc[truth.label].reset_index()
    true_values = feature_matrix(truth, columns)
    predicted_values = feature_matrix(predicted, columns)
    errors = np.mean(np.abs(predicted_values - true_values) / (np.abs(true_values) + eps), axis=1)
    result = pd.DataFrame({'label': truth.label.to_numpy(), 'mean_relative_error': errors})
    result.to_csv(output_csv, index=False)
    print(f'Average relative errors saved to: {output_csv}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--true_csv', required=True)
    parser.add_argument('--regenerate_csv', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--eval-index', type=int, help='Select one evaluation from a multi-evaluation reconstruction')
    args = parser.parse_args()
    compute_average_relative_error(args.true_csv, args.regenerate_csv, args.output, eval_index=args.eval_index)
