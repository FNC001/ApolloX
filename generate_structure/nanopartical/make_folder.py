import os
import shutil
from math import ceil
from tqdm import tqdm  # 引入tqdm库

# Define the directory containing the files
source_dir = "./"  # 修改为您的文件夹路径

# Get a list of all files
files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Number of splits
num_splits = 5

# Calculate the number of files per split (rounding up to ensure all files are included)
files_per_split = ceil(len(files) / num_splits)

# Create target directories and distribute files
for I in range(num_splits):
    # Create a subdirectory for each split
    split_dir = os.path.join(source_dir, f'split_{I+1}')
    os.makedirs(split_dir, exist_ok=True)

    # Calculate the slice of files to move to this directory
    start_index = I * files_per_split
    end_index = min((I + 1) * files_per_split, len(files))

    # Move the files with progress bar
    for f in tqdm(files[start_index:end_index], desc=f'Moving files to split_{I+1}'):
        shutil.move(os.path.join(source_dir, f), os.path.join(split_dir, f))

print("All files have been distributed into the specified splits.")
