import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import os

# === Command-line argument parsing ===
parser = argparse.ArgumentParser(description="Standardize dataset and split into train/test/val")
parser.add_argument('--input', type=str, required=True, help="Input CSV file path")
parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output files")
parser.add_argument('--train_ratio', type=float, default=0.8, help="Proportion of training set")
parser.add_argument('--test_ratio', type=float, default=0.1, help="Proportion of test set")
parser.add_argument('--val_ratio', type=float, default=0.1, help="Proportion of validation set")
args = parser.parse_args()

# === Check if ratios are valid ===
total_ratio = args.train_ratio + args.test_ratio + args.val_ratio
if not np.isclose(total_ratio, 1.0):
    raise ValueError(f"The sum of train, test, and val ratios must be 1.0, but got {total_ratio}")

# === Create output directory ===
os.makedirs(args.output_dir, exist_ok=True)

# === Read data and split columns ===
df = pd.read_csv(args.input)
non_data_cols = df.iloc[:, :2]
data_cols = df.iloc[:, 2:]

# === Shuffle data ===
# Concatenate non-data and data columns to keep them aligned during shuffling
data_with_index = pd.concat([non_data_cols, data_cols], axis=1)
data_with_index = data_with_index.sample(frac=1, random_state=42).reset_index(drop=True)

non_data_cols = data_with_index.iloc[:, :2]
data_cols = data_with_index.iloc[:, 2:]

# === Split dataset ===
total_rows = len(data_cols)
train_size = int(args.train_ratio * total_rows)
test_size = int(args.test_ratio * total_rows)
val_size = total_rows - train_size - test_size  # 剩下的归为 val

train_data = data_cols[:train_size]
test_data = data_cols[train_size:train_size + test_size]
val_data = data_cols[train_size + test_size:]

# === Standardization process ===
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
train_mean = scaler.mean_
train_std = scaler.scale_
test_scaled = scaler.transform(test_data)
val_scaled = scaler.transform(val_data)

# === Convert to DataFrame and merge with original index columns ===
train_scaled_df = pd.DataFrame(train_scaled, columns=train_data.columns)
test_scaled_df = pd.DataFrame(test_scaled, columns=test_data.columns)
val_scaled_df = pd.DataFrame(val_scaled, columns=val_data.columns)

train_final_df = pd.concat([non_data_cols.iloc[:train_size].reset_index(drop=True), train_scaled_df], axis=1)
test_final_df = pd.concat([non_data_cols.iloc[train_size:train_size + test_size].reset_index(drop=True), test_scaled_df], axis=1)
val_final_df = pd.concat([non_data_cols.iloc[train_size + test_size:].reset_index(drop=True), val_scaled_df], axis=1)

# === Save results ===
train_final_df.to_csv(os.path.join(args.output_dir, 'train_set_scaled.csv'), index=False)
test_final_df.to_csv(os.path.join(args.output_dir, 'test_set_scaled.csv'), index=False)
val_final_df.to_csv(os.path.join(args.output_dir, 'val_set_scaled.csv'), index=False)


with open(os.path.join(args.output_dir, "scaler_stats.txt"), "w") as f:
    f.write("Train Mean: " + ','.join(f"{x:.6f}" for x in train_mean) + "\n")
    f.write("Train Std: " + ','.join(f"{x:.6f}" for x in train_std) + "\n")
