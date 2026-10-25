"""Merge completed split workers with a single shared feature schema."""
import argparse
from pathlib import Path
import sys

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apollox_common.pdm import align_features, load_schema, save_schema


def collect_and_merge_csvs(current_directory='.', output_path=None):
    directory = Path(current_directory).resolve()
    folders = sorted(path for path in directory.iterdir() if path.is_dir() and path.name.startswith('split'))
    if not folders:
        raise ValueError(f'No split folders in {directory}')
    frames, schema = [], None
    for folder in folders:
        path = folder / 'all_structures_summary.csv'
        candidate = load_schema(folder / 'pdm_schema.json')
        if schema is None:
            schema = candidate
        elif candidate != schema:
            raise ValueError('Split PDM schemas differ; recompute workers with one shared --schema')
        frame = pd.read_csv(path, dtype={'material_id': str, 'cif_file': str}, keep_default_na=False)
        align_features(frame, schema, allow_missing=False)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    if merged['material_id'].duplicated().any():
        raise ValueError('Duplicate material_id across split folders')
    output = Path(output_path).resolve() if output_path else directory.parent / 'all_structures_summary.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    save_schema(output.parent / 'pdm_schema.json', schema)
    merged.to_csv(output, index=False)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', default='.')
    parser.add_argument('--output')
    args = parser.parse_args()
    print(collect_and_merge_csvs(args.input_dir, args.output))
