import os
import pandas as pd
import yaml
from pathlib import Path
import subprocess
import sys
import time
import torch
import shutil
import ast


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
    raise FileNotFoundError(
        "Could not find a parent directory containing 'cond-cdvae'. Please check your project structure.")


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
        cmd_pdm = f"python {apollox_path}/use_model/compute_pdm_formula.py --input_dir {params['input_dir']} --output {pdm_output_csv} --cutoff {params['cutoff']} --n_jobs {params['n_jobs']} --mode {params['compute_mode']} --starts_with \"{params['starts_with']}\" --ends_with \"{params['ends_with']}\""
        if run_cmd(cmd_pdm).returncode != 0: sys.exit(1)

        standardized_output_csv = "standardized.csv"
        cmd_std = f"python {apollox_path}/prepare_dataset/standardize.py --input {pdm_output_csv} --scaler {params['scaler_path']} --output {standardized_output_csv}"
        if run_cmd(cmd_std).returncode != 0: sys.exit(1)

        df = pd.read_csv(standardized_output_csv)
        df.rename(columns={'label': 'label_for_script'}, inplace=True)
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
        df['formula'] = df['material_id'].str.split('-').str[1].str.replace('_', '', regex=False)
        df['label_for_script'] = df['material_id']
        df['element_values_str'] = df['element_values'].apply(
            lambda x: ",".join(map(str, ast.literal_eval(x))) if isinstance(x, str) else ",".join(map(str, x)))
        df_processed = df

    else:
        print(
            f"[ERROR] Invalid 'mode_choice': {mode_choice}. Please choose from 'structure', 'feather', 'unscaled_pdm'.",
            file=sys.stderr)
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


active_processes = []  # [(Popen对象, label_for_script)]
processed_count = 0
final_structures_dir = Path("final_generated_structures")
final_structures_dir.mkdir(exist_ok=True)

for idx, row in df_processed.iterrows():
    # --- 控制并行数 ---
    while len(active_processes) >= max_parallel:
        finished = [proc for proc, label in active_processes if proc.poll() is not None]
        for proc, label_prefix in list(active_processes):
            if proc in finished:
                active_processes.remove((proc, label_prefix))
                # 任务完成 → 立刻处理输出
                pt_file = Path(f"eval_gen_{label_prefix}.pt")
                if pt_file.is_file():
                    print(f"[INFO] Processing output for {label_prefix}")
                    cmd_extract = f"python {apollox_path}/cond-cdvae/scripts/extract_gen.py {pt_file}"
                    run_cmd(cmd_extract)

                    extracted_dir = Path(f"eval_gen_{label_prefix}")
                    source_gen_dir = extracted_dir / 'gen'
                    if source_gen_dir.is_dir():
                        for structure_file in source_gen_dir.iterdir():
                            if structure_file.is_file():
                                new_name = f"{label_prefix}_{structure_file.name}"
                                shutil.move(str(structure_file), str(final_structures_dir / new_name))
                                processed_count += 1
                    if extracted_dir.is_dir():
                        shutil.rmtree(extracted_dir)
                    os.remove(pt_file)
                else:
                    print(f"[WARNING] .pt file not found for {label_prefix}")

        time.sleep(1)  # 稍等，避免占满 CPU

    # --- 生成 sh 脚本并启动任务 ---
    label_for_script = row["label_for_script"]
    formula = row["formula"]

    if 'element_values_str' in df_processed.columns:
        element_values_str = row['element_values_str']
    else:
        formula_col_index = df_processed.columns.get_loc('formula')
        element_cols = df_processed.columns[formula_col_index + 1:]
        element_cols = [c for c in element_cols if c not in ['label_for_script', 'element_values_str']]
        element_values = [str(row[col]) for col in element_cols if pd.notna(row[col])]
        element_values_str = ",".join(element_values)

    sh_content = (
        "#!/bin/bash\n"
        f"CUDA_VISIBLE_DEVICES=$(({idx} % {max(1, torch.cuda.device_count())}))\n"
        f"python {apollox_path}/cond-cdvae/scripts/evaluate.py --model_path `pwd` --tasks gen \\\n"
        f"    --formula=\"{formula}\" --pressure=0 --label=\"{label_for_script}\" \\\n"
        f"    --element_values=\"{element_values_str}\" --batch_size={batch_size} \\\n"
        f"    --num_batches_to_samples={num_batches}\n"
    )

    sh_file = sh_dir / f"{Path(label_for_script).stem}.sh"
    with open(sh_file, "w") as f:
        f.write(sh_content)
    os.chmod(sh_file, 0o755)

    log_file = log_dir / f"{Path(label_for_script).stem}.log"
    err_file = log_dir / f"{Path(label_for_script).stem}.err"

    with open(log_file, "w") as log_f, open(err_file, "w") as err_f:
        proc = subprocess.Popen(str(sh_file), shell=True, stdout=log_f, stderr=err_f)
        active_processes.append((proc, Path(label_for_script).stem))
        print(f"--- Launched job for: {label_for_script} (PID: {proc.pid}). Running jobs: {len(active_processes)} ---")

# --- 等待剩余任务并处理 ---
while active_processes:
    for proc, label_prefix in list(active_processes):
        if proc.poll() is not None:
            active_processes.remove((proc, label_prefix))
            pt_file = Path(f"eval_gen_{label_prefix}.pt")
            if pt_file.is_file():
                print(f"[INFO] Processing output for {label_prefix}")
                cmd_extract = f"python {apollox_path}/cond-cdvae/scripts/extract_gen.py {pt_file}"
                run_cmd(cmd_extract)
                extracted_dir = Path(f"eval_gen_{label_prefix}")
                source_gen_dir = extracted_dir / 'gen'
                if source_gen_dir.is_dir():
                    for structure_file in source_gen_dir.iterdir():
                        if structure_file.is_file():
                            new_name = f"{label_prefix}_{structure_file.name}"
                            shutil.move(str(structure_file), str(final_structures_dir / new_name))
                            processed_count += 1
                if extracted_dir.is_dir():
                    shutil.rmtree(extracted_dir)
                os.remove(pt_file)
            else:
                print(f"[WARNING] .pt file not found for {label_prefix}")
    time.sleep(1)

print(f"\n[SUCCESS] All jobs done. {processed_count} structures consolidated into '{final_structures_dir}'.")

try:
    shutil.rmtree(sh_dir)
    print(f"[INFO] Removed directory '{sh_dir}'")
except Exception as e:
    print(f"[WARNING] Failed to remove '{sh_dir}': {e}")

