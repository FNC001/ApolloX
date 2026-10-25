import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apollox_common.pdm import calculate_distances, calculate_sro, compute_folder, read_poscar


def main():
    parser = argparse.ArgumentParser(description="Compute PDM with a complete, versioned feature schema")
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_csv', '--output', default='all_structures_summary.csv')
    parser.add_argument('--cutoff', type=float, default=None)
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument('--mode', nargs='+', choices=['pair', 'triple', 'quadruple'], default=None)
    parser.add_argument('--starts_with', default='')
    parser.add_argument('--ends_with', default='')
    parser.add_argument('--schema', help='Existing training pdm_schema.json; controls feature order and geometry')
    args = parser.parse_args()
    frame, schema = compute_folder(
        args.input_dir, args.output_csv, cutoff=args.cutoff, modes=args.mode,
        n_jobs=args.n_jobs, starts_with=args.starts_with, ends_with=args.ends_with,
        schema_path=args.schema, formula_metadata=True,
    )
    print(f"Saved {len(frame)} structures and {len(schema['features'])} PDM features to {args.output_csv}")


if __name__ == '__main__':
    main()
