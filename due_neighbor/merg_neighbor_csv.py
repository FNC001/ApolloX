import os
import pandas as pd
from tqdm import tqdm

def collect_and_merge_csvs():
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    merged_df = pd.DataFrame()

    for folder in os.listdir(current_directory):
        if folder.startswith("split") and os.path.isdir(os.path.join(current_directory, folder)):
            csv_path = os.path.join(current_directory, folder, 'all_calculations_summary.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                merged_df = pd.concat([merged_df, df], ignore_index=True)

    tqdm.pandas(desc="Merging CSVs")
    merged_df.progress_apply(lambda x: x)
    merged_output_path = os.path.join(parent_directory, 'all_calculations_summary.csv')
    merged_df.to_csv(merged_output_path, index=False)
    print(f"All calculations merged successfully into {merged_output_path}")

collect_and_merge_csvs()