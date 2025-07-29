import pandas as pd
import os
import argparse
from tqdm import tqdm

def read_cif_content(file_name, cif_directory):
    file_path = os.path.join(cif_directory, file_name + '.cif')
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return 'File not found'

def process_file(input_csv, output_csv, cif_directory):
    data = pd.read_csv(input_csv)

    # Add .cif file content
    data['cif_file'] = data.apply(lambda row: read_cif_content(row['material_id'], cif_directory), axis=1)

    # Convert interaction columns (skip first two columns)
    interaction_columns = data.columns[2:]
    data['element_values'] = [list(row[interaction_columns].values) for _, row in tqdm(data.iterrows(), total=data.shape[0])]

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
