import pandas as pd
import os
import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apollox_common.pdm import align_features, load_schema, save_schema

def read_cif_content(file_name, cif_directory):
    file_path = os.path.join(cif_directory, file_name + '.cif')
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CIF missing for material {file_name}: {file_path}") from exc

def process_file(input_csv, output_csv, cif_directory):
    data = pd.read_csv(input_csv, dtype={'material_id': str, 'cif_file': str}, keep_default_na=False)
    schema = load_schema(Path(input_csv).parent / 'pdm_schema.json')
    features = align_features(data, schema, allow_missing=False)

    # Add .cif file content
    data['cif_file'] = data.apply(lambda row: read_cif_content(row['material_id'], cif_directory), axis=1)

    # Convert interaction columns (skip first two columns)
    data['element_values'] = [json.dumps(row.tolist()) for row in features.to_numpy()]

    save_schema(Path(output_csv).parent / 'pdm_schema.json', schema)
    data.to_csv(output_csv, index=False)
    print(f"Saved with CIF and element vectors: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Append .cif content and element values to dataset.")
    parser.add_argument('--input_csvs', nargs='+', required=True, help="List of input CSV files")
    parser.add_argument('--output_names', nargs='+', required=True, help="List of output CSV names (without extension)")
    parser.add_argument('--cif_dir', required=True, help="Directory containing .cif files")
    parser.add_argument('--output_dir', default='.', help="Directory to save output files")

    args = parser.parse_args()

    if len(args.input_csvs) != len(args.output_names):
        raise ValueError("The number of input CSVs and output names must match.")

    for input_file, out_name in zip(args.input_csvs, args.output_names):
        output_path = os.path.join(args.output_dir, f"with_cif_{out_name}.csv")
        process_file(input_file, output_path, args.cif_dir)

if __name__ == "__main__":
    main()
