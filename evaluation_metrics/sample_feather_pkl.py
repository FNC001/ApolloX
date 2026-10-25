"""Sample aligned V2 Feather/cache records and preserve their cache provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apollox_common.pdm import load_schema, save_schema


def contract(feather, schema):
    canonical = json.dumps(schema, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
    return {'schema_sha256': hashlib.sha256(canonical).hexdigest(),
            'source_sha256': hashlib.sha256(Path(feather).read_bytes()).hexdigest()}


def sample(feather, pkl, n, feather_out, pkl_out):
    feather, pkl, feather_out, pkl_out = map(Path, [feather, pkl, feather_out, pkl_out])
    schema = load_schema(feather.parent / 'pdm_schema.json')
    source_contract = json.loads(Path(str(pkl) + '.meta.json').read_text())
    if any(source_contract.get(key) != value for key, value in contract(feather, schema).items()):
        raise ValueError('Source cache does not match the Feather data and V2 schema; rebuild the cache first')
    frame = pd.read_feather(feather)
    with pkl.open('rb') as stream:
        records = pickle.load(stream)
    if not isinstance(records, list) or len(frame) != len(records):
        raise ValueError('Feather rows and cache records must have equal length')
    if not 0 < n <= len(frame):
        raise ValueError('n must be between 1 and the source population size')
    for output in [feather_out, pkl_out, Path(str(pkl_out) + '.meta.json')]:
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite {output}')
    if feather_out.resolve() == pkl_out.resolve():
        raise ValueError('Feather and cache output paths must differ')
    feather_out.parent.mkdir(parents=True, exist_ok=True)
    pkl_out.parent.mkdir(parents=True, exist_ok=True)
    save_schema(feather_out.parent / 'pdm_schema.json', schema)
    source_scaler = feather.parent / 'scaler_stats.txt'
    target_scaler = feather_out.parent / 'scaler_stats.txt'
    if source_scaler.is_file() and source_scaler.resolve() != target_scaler.resolve():
        if target_scaler.exists() and target_scaler.read_bytes() != source_scaler.read_bytes():
            raise ValueError('Output directory has a different training scaler')
        if not target_scaler.exists():
            shutil.copy2(source_scaler, target_scaler)
    indices = np.random.choice(len(frame), size=n, replace=False)
    frame.iloc[indices].reset_index(drop=True).to_feather(feather_out)
    with pkl_out.open('wb') as stream:
        pickle.dump([records[index] for index in indices], stream)
    metadata = dict(cache_version=2, **contract(feather_out, schema))
    Path(str(pkl_out) + '.meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved {n} aligned samples to {feather_out} and {pkl_out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feather', required=True)
    parser.add_argument('--pkl', required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--feather_out', required=True)
    parser.add_argument('--pkl_out', required=True)
    args = parser.parse_args()
    sample(args.feather, args.pkl, args.n, args.feather_out, args.pkl_out)


if __name__ == '__main__':
    main()
