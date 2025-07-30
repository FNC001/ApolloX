import pandas as pd
import os
import argparse

def process_csv_files(input_dir):
    files = ['with_cif_train.csv', 'with_cif_test.csv', 'with_cif_val.csv']
    names = ['train', 'test', 'val']

    for i in range(3):
        csv_path = os.path.join(input_dir, files[i])
        feather_path = os.path.join(input_dir, f"{names[i]}.feather")

        # Read CSV
        data = pd.read_csv(csv_path)

        # Select specific columns
        newdata = data[["material_id", "cif_file", "element_values"]].copy()

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
