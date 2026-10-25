import pandas as pd
import os
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apollox_common.pdm import load_schema

def process_csv_files(input_dir):
    schema = load_schema(Path(input_dir) / 'pdm_schema.json')
    files = ['with_cif_train.csv', 'with_cif_test.csv', 'with_cif_val.csv']
    names = ['train', 'test', 'val']

    for i in range(3):
        csv_path = os.path.join(input_dir, files[i])
        feather_path = os.path.join(input_dir, f"{names[i]}.feather")

        # Read CSV
        data = pd.read_csv(csv_path, dtype={'material_id': str, 'cif_file': str}, keep_default_na=False)

        # Select specific columns
        newdata = data[["material_id", "cif_file", "element_values"]].copy()
        vectors = []
        for index, value in enumerate(newdata['element_values']):
            vector = np.asarray(json.loads(value), dtype=float)
            if vector.ndim != 1 or vector.size != len(schema['features']) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid PDM vector at {csv_path}, row {index}")
            vectors.append(vector.tolist())
        newdata['element_values'] = vectors

        # Rename column
        newdata.rename(columns={"cif_file": "cif"}, inplace=True)

        # Add pressure column
        newdata["pressure"] = 0

        # Save as Feather format
        newdata.to_feather(feather_path)
        print(f"Saved {feather_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert CSV files with CIF to Feather format.")
    parser.add_argument('--input_dir', required=True, help="Directory containing the with_cif_*.csv files")

    args = parser.parse_args()
    process_csv_files(args.input_dir)

if __name__ == "__main__":
    main()
