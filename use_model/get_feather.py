import pandas as pd
import pickle
import argparse
import numpy as np

# -------------------------
# Command-line arguments
# -------------------------
parser = argparse.ArgumentParser(description="Randomly select N rows from a Feather and PKL file, keeping them aligned.")
parser.add_argument("--feather", required=True, help="Input Feather file path")
parser.add_argument("--pkl", required=True, help="Input PKL file path")
parser.add_argument("--n", type=int, required=True, help="Number of rows to select")
parser.add_argument("--feather_out", required=True, help="Output Feather file path")
parser.add_argument("--pkl_out", required=True, help="Output PKL file path")
args = parser.parse_args()

# -------------------------
# Load Feather file
# -------------------------
df = pd.read_feather(args.feather)

# Load PKL file
with open(args.pkl, "rb") as f:
    data_list = pickle.load(f)

if not isinstance(data_list, list):
    raise TypeError("Error: The PKL file content is not a list.")

if len(df) != len(data_list):
    raise ValueError("Error: Feather and PKL files have different lengths.")

# -------------------------
# Random sampling
# -------------------------
indices = np.random.choice(len(df), size=args.n, replace=False)

df_sub = df.iloc[indices].reset_index(drop=True)
data_sub = [data_list[i] for i in indices]

# -------------------------
# Save results
# -------------------------
df_sub.to_feather(args.feather_out)
with open(args.pkl_out, "wb") as f:
    pickle.dump(data_sub, f)

print(f"Randomly selected {args.n} rows have been saved to {args.feather_out} and {args.pkl_out}")
