import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apollox_common.pdm import align_features, load_schema, save_schema


def split_dataset(input_csv, output_dir, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1, schema_path=None):
    ratios = np.asarray([train_ratio, test_ratio, val_ratio], dtype=float)
    if not np.isfinite(ratios).all() or (ratios <= 0).any() or not np.isclose(ratios.sum(), 1):
        raise ValueError("Train/test/validation ratios must be positive and sum to 1")
    schema = load_schema(schema_path or Path(input_csv).parent / 'pdm_schema.json')
    data = pd.read_csv(input_csv, dtype={'material_id': str, 'cif_file': str}, keep_default_na=False)
    if not {'material_id', 'cif_file'} <= set(data):
        raise ValueError("Dataset must have material_id and cif_file columns")
    if data.material_id.str.strip().eq('').any() or data.material_id.duplicated().any():
        raise ValueError("Dataset material_id must be present and unique before splitting")
    features = align_features(data, schema, allow_missing=False)
    data = pd.concat([data[['material_id', 'cif_file']], features], axis=1)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size, test_size = int(len(data) * train_ratio), int(len(data) * test_ratio)
    splits = [data.iloc[:train_size], data.iloc[train_size:train_size + test_size], data.iloc[train_size + test_size:]]
    if any(frame.empty for frame in splits):
        raise ValueError("Dataset/ratios must produce non-empty train, test, and validation splits")
    scaler = StandardScaler().fit(splits[0][schema['features']])
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_schema(output_dir / 'pdm_schema.json', schema)
    for name, frame in zip(['train', 'test', 'val'], splits):
        result = frame[['material_id', 'cif_file']].reset_index(drop=True)
        result[schema['features']] = scaler.transform(frame[schema['features']])
        result.to_csv(output_dir / f'{name}_set_scaled.csv', index=False)
    (output_dir / 'scaler_stats.txt').write_text(
        "Train Mean: " + ','.join(format(value, '.17g') for value in scaler.mean_) + "\n" +
        "Train Std: " + ','.join(format(value, '.17g') for value in scaler.scale_) + "\n"
    )
    return scaler


def main():
    parser = argparse.ArgumentParser(description="Split data and fit a schema-bound scaler on training rows only")
    parser.add_argument('--input', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--schema')
    args = parser.parse_args()
    split_dataset(args.input, args.output_dir, args.train_ratio, args.test_ratio, args.val_ratio, args.schema)


if __name__ == '__main__':
    main()
