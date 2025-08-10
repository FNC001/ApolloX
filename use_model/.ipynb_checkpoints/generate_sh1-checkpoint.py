import os
import pandas as pd
import yaml
from pathlib import Path
import subprocess
import sys
import time

def find_apollox_path(start_path):
    """
    Traverse upward from start_path until a directory containing cond-cdvae/ is found.
    Returns the parent directory as apollox_path.
    """
    current = Path(start_path).resolve()
    while current != current.parent:
        if (current / "cond-cdvae").is_dir():
            return str(current)
        current = current.parent
    raise FileNotFoundError("Could not find a parent directory containing 'cond-cdvae'. Please check your project structure.")

def run_cmd(cmd):
    """
    Run a shell command for pre-processing, print its output, and return the result.
    This is used for sequential pre-processing steps.
    """
    print(f"\n[COMMAND] Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("ERROR:", result.stderr)
    return result

# --- 1. Setup and Configuration ---
print("[INFO] Starting script...")
try:
    apollox_path = find_apollox_path(Path(__file__).parent)
    print(f"[INFO] Automatically detected apollox_path: {apollox_path}")
except FileNotFoundError as e:
    print(f"[ERROR] {e}", file=sys.stderr)
    sys.exit(1)

config_path = Path(apollox_path) / "use_model" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

mode_choice = config.get("mode_choice")
batch_size = config["batch_size"]
num_batches = config["num_batches_to_samples"]
# Get max parallel submissions, default to 1 (sequential) for safety.
max_parallel = config.get("max_parallel_submissions", 1)
if not isinstance(max_parallel, int) or max_parallel <= 0:
    max_parallel = 1

df_processed = None
print(f"[INFO] Mode selected: '{mode_choice}'")
print(f"[INFO] Maximum parallel submissions set to: {max_parallel}")


# --- 2. Data Preparation Stage ---
# This stage populates the `df_processed` DataFrame based on the mode.
try:
    if mode_choice == 'structure':
        params = config["structure_mode_params"]
        pdm_output_csv = "all_structures_summary.csv"
        cmd_pdm = f"python {apollox_path}/prepare_dataset/compute_pdm_formula.py --input_dir {params['input_dir']} --output {pdm_output_csv} --cutoff {params['cutoff']} --n_jobs {params['n_jobs']} --mode {params['compute_mode']} --starts_with \"{params['starts_with']}\" --ends_with \"{params['ends_with']}\""
        if run_cmd(cmd_pdm).returncode != 0: sys.exit(1)

        standardized_output_csv = "standardized.csv"
        cmd_std = f"python {apollox_path}/prepare_dataset/standardize.py --input {pdm_output_csv} --scaler {params['scaler_path']} --output {standardized_output_csv}"
        if run_cmd(cmd_std).returncode != 0: sys.exit(1)
        
        df = pd.read_csv(standardized_output_csv)
        df.rename(columns={'material_id': 'label_for_script'}, inplace=True)
        df_processed = df

    elif mode_choice == 'unscaled_pdm':
        params = config["unscaled_pdm_mode_params"]
        standardized_output_csv = "standardized.csv"
        cmd_std = f"python {apollox_path}/prepare_dataset/standardize.py --input {params['input_csv']} --scaler {params['scaler_path']} --output {standardized_output_csv}"
        if run_cmd(cmd_std).returncode != 0: sys.exit(1)
        
        df = pd.read_csv(standardized_output_csv)
        df.rename(columns={'label': 'label_for_script'}, inplace=True)
        df_processed = df

    elif mode_choice == 'feather':
        params = config["feather_mode_params"]
        df = pd.read_feather(params['feather_path'])
        df['formula'] = df['material_id'].str.split('-').str[1]
        df['label_for_script'] = df['material_id']
        last_col_name = df.columns[-1]
        df['element_values_str'] = df[last_col_name].apply(lambda x: ",".join(map(str, x)))
        df_processed = df
        
    else:
        print(f"[ERROR] Invalid 'mode_choice': {mode_choice}. Please choose from 'structure', 'feather', 'unscaled_pdm'.", file=sys.stderr)
        sys.exit(1)

except KeyError as e:
    print(f"[ERROR] Missing parameter in config.yaml for mode '{mode_choice}': {e}", file=sys.stderr)
    sys.exit(1)
except FileNotFoundError as e:
    print(f"[ERROR] File not found: {e}", file=sys.stderr)
    sys.exit(1)

# --- 3. Script Generation and PARALLEL Execution Stage ---
if df_processed is None:
    print("[ERROR] Data processing failed, DataFrame is empty.", file=sys.stderr)
    sys.exit(1)

sh_dir = Path("sh_files_multi_pdm")
sh_dir.mkdir(exist_ok=True)
log_dir = Path("parallel_logs")
log_dir.mkdir(exist_ok=True)

print(f"\n[INFO] Data loaded. Found {len(df_processed)} materials to process.")
print(f"[INFO] Starting parallel job submission (max {max_parallel} jobs at a time)...")

active_processes = []
for idx, row in df_processed.iterrows():
    # --- Check and manage the process pool ---
    while len(active_processes) >= max_parallel:
        # Check for finished processes and remove them from the list
        finished_procs = [p for p in active_processes if p.poll() is not None]
        for p in finished_procs:
            active_processes.remove(p)
            print(f"[INFO] Process PID {p.pid} finished. Pool size: {len(active_processes)}")
        if len(active_processes) < max_parallel:
            break # A slot has opened up
        time.sleep(2) # Wait a moment before checking again to avoid busy-waiting

    # --- Prepare and launch a new process ---
    label_for_script = row["label_for_script"]
    formula = row["formula"]

    if 'element_values_str' in df_processed.columns:
        element_values_str = row['element_values_str']
    else:
        try:
            formula_col_index = df_processed.columns.get_loc('formula')
            element_cols = df_processed.columns[formula_col_index + 1:]
            element_cols = [c for c in element_cols if c not in ['label_for_script', 'element_values_str']]
            element_values = [f"{col}:{row[col]}" for col in element_cols if pd.notna(row[col])]
            element_values_str = ",".join(element_values)
        except KeyError:
            print(f"[WARNING] 'formula' column not found for row {idx}. Skipping.", file=sys.stderr)
            continue
    
    sh_content = (
        "#!/bin/bash\n"
        "# This script is auto-generated.\n"
        f"CUDA_VISIBLE_DEVICES=$(({idx} % {max(1, torch.cuda.device_count())}))\n" # Basic GPU balancing
        f"python {apollox_path}/cond-cdvae/scripts/evaluate.py --model_path `pwd` --tasks gen \\\n"
        f"    --formula=\"{formula}\" --pressure=0 --label=\"{label_for_script}\" \\\n"
        f"    --element_values=\"{element_values_str}\" --batch_size={batch_size} \\\n"
        f"    --num_batches_to_samples={num_batches}\n"
    )

    sh_file_name = f"{Path(label_for_script).stem}.sh"
    sh_file = sh_dir / sh_file_name
    with open(sh_file, "w") as f:
        f.write(sh_content)
    os.chmod(sh_file, 0o755)

    # Redirect output to log files
    log_file = log_dir / f"{Path(label_for_script).stem}.log"
    err_file = log_dir / f"{Path(label_for_script).stem}.err"
    
    with open(log_file, "w") as log_f, open(err_file, "w") as err_f:
        # Launch the script as a new process
        p = subprocess.Popen(str(sh_file), shell=True, stdout=log_f, stderr=err_f)
        active_processes.append(p)
        print(f"--- Launched job for: {label_for_script} (PID: {p.pid}). Running jobs: {len(active_processes)} ---")


# --- Final Cleanup: Wait for any remaining processes ---
print("\n[INFO] All jobs have been launched. Waiting for the last batch to finish...")
for p in active_processes:
    p.wait() # Wait for each of the last running processes to complete

print(f"\n[SUCCESS] All parallel jobs have completed. Check the '{log_dir}' directory for logs.")
