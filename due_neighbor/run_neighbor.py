"""Compute neighbor descriptors using the shared V2 geometry and schema."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apollox_common.pdm import calculate_distances, calculate_sro, compute_folder, read_poscar


def process_all_poscar_files(input_dir='.', schema_path=None, cutoff=None):
    return compute_folder(
        input_dir, Path(input_dir) / 'all_structures_summary.csv',
        cutoff=cutoff, modes=None if schema_path else ['pair', 'triple', 'quadruple'],
        n_jobs=1, starts_with='POSCAR', schema_path=schema_path,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', default='.')
    parser.add_argument('--schema')
    parser.add_argument('--cutoff', type=float)
    args = parser.parse_args()
    process_all_poscar_files(args.input_dir, args.schema, args.cutoff)
