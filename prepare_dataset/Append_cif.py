import pandas as pd
import json
import os
from tqdm import tqdm

file_path = ['./train_set_scaled.csv','./test_set_scaled.csv','./val_set_scaled.csv']
n=['train','test','val']

def read_cif_content(file_name):
    cif_directory = "Apollox/cif"  # make sure the folder path is right
    file_name_with_suffix = os.path.join(cif_directory, file_name + '.cif')
    try:
        with open(file_name_with_suffix, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return 'File not found'
for i in file_path:
    data = pd.read_csv(i)
    data['cif_file'] = data.apply(lambda row: read_cif_content(row['material_id']), axis=1)
    interaction_columns = data.columns[2:]
    data['element_values'] = [list(row[interaction_columns].values) for index, row in tqdm(data.iterrows(), total=data.shape[0])]
    data.to_csv(f'./with_cif{n[file_path.index(i)]}.csv', index=False)
