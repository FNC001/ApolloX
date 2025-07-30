import os
import shutil
import subprocess
import yaml
import concurrent.futures
import time
import psutil
import pandas as pd
from pathlib import Path

# === Locate ApolloX root directory ===
def find_apollox_root(current_path):
    while current_path != current_path.parent:
        if (current_path / "ApolloX").is_dir():
            return current_path / "ApolloX"
        current_path = current_path.parent
    raise RuntimeError("ApolloX directory not found in any parent directory.")
def has_enough_memory(threshold_gb):
    mem = psutil.virtual_memory()
    return mem.available / 1e9 >= threshold_gb

def run_chgnet_for_structure(structure_file: Path, chgnet_script: Path, chgnet_cfg: dict, work_root: Path):
    """
    给定单个结构文件，创建子文件夹拷贝该结构，运行 chgnet_gpu.py。
    输出 sorted_energies.csv 到该子文件夹。
    """
    structure_name = structure_file.name
    # 子文件夹命名用结构名去除后缀
    subfolder_name = structure_file.stem+"_dir"
    subfolder = work_root / subfolder_name

    subfolder.mkdir(exist_ok=True)

    # 拷贝结构文件到子文件夹
    target_structure_file = subfolder / structure_name
    if not target_structure_file.exists():
        shutil.copy(structure_file, target_structure_file)

    cmd_chgnet = [
        "python", str(chgnet_script),
        "--input_pattern", structure_name,
        "--mlp_optstep", str(chgnet_cfg.get("mlp_optstep", 1)),
        "--fmax", str(chgnet_cfg.get("fmax", 0.01)),
        "--output_csv", "sorted_energies.csv"
    ]
    print(f"[CHGNet] Running optimization for {structure_name} in {subfolder} ...")
    subprocess.run(cmd_chgnet, cwd=subfolder, check=True)
    return subfolder / "sorted_energies.csv"

def run_parallel_chgnet(output_dir: Path, chgnet_script: Path, chgnet_cfg: dict, max_workers=2, min_free_mem_gb=4.0):
    # 找到所有结构文件，默认用 input_pattern 匹配
    structure_files = [
        f for f in output_dir.glob(chgnet_cfg["input_pattern"])
        if f.is_file()
    ]
    print(f"[CHGNet] Found {len(structure_files)} structure files to optimize.")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for sf in structure_files:
            while not has_enough_memory(min_free_mem_gb):
                print("[MEMORY] Not enough memory, waiting 10 seconds...")
                time.sleep(10)
            future = executor.submit(run_chgnet_for_structure, sf, chgnet_script, chgnet_cfg, output_dir)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                result_path = future.result()
                results.append(result_path)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] ChgNet failed: {e}")

    return results

def merge_all_sorted_energies(sorted_files, merged_output_path):
    dfs = []
    for f in sorted_files:
        if f.exists():
            df = pd.read_csv(f)
            dfs.append(df)
        else:
            print(f"[WARN] Sorted energy file not found: {f}")
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(merged_output_path, index=False)
        print(f"[MERGE] Combined sorted energies saved to {merged_output_path}")
    else:
        print("[MERGE] No sorted energies files to combine.")
# === Setup paths ===
script_path = Path(__file__).resolve()
apollox_root = find_apollox_root(script_path.parent)
pso_dir = apollox_root / "PSO"

# 当前运行目录
for folder_name in ["temp", "pdm_and_energy", "poscars"]:
    folder_path = Path.cwd() / folder_name
    if folder_path.exists() and folder_path.is_dir():
        print(f"[INIT] Removing existing directory: {folder_path}")
        shutil.rmtree(folder_path)
run_dir = Path.cwd()
pdm_energy_dir = run_dir / "pdm_and_energy"
generation_dir = run_dir / "poscars" / "generation1"
pdm_energy_dir.mkdir(parents=True, exist_ok=False)
generation_dir.mkdir(parents=True, exist_ok=False)

# === Load config ===
with open(pso_dir / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

poscar_name = config["poscar_name"]
gen_num = config["gen_num"]
structure_num_per_gen = config["structure_num_per_gen"]
num_files = gen_num * structure_num_per_gen

original_structure_dir = apollox_root / "original_structures"
opt_script = config["opt_script"]
pdm_script = apollox_root / "prepare_dataset" / "compute_pdm.py"
merge_script = pso_dir / "merge1.py"
output_dir = pso_dir / "poscar"
if output_dir.exists():
    print(f"[INIT] Removing existing directory: {output_dir}")
    shutil.rmtree(output_dir)

output_dir.mkdir(parents=True, exist_ok=False)

chgnet_cfg = config["opt"]
pdm_cfg = config["pdm"]

# === Step 1: Generate structures ===
subprocess.run([
    "python", str(apollox_root / "generate_structure/bulk/generate_single_component.py"),
    "--input", str(original_structure_dir / poscar_name),
    "--num", str(num_files),
    "--outdir", str(pso_dir)
], check=True)

# === Step 2: Run ChgNet optimization ===
shutil.copy(merge_script, output_dir)
sorted_energy_files = run_parallel_chgnet(
    output_dir,
    apollox_root / "opt" / opt_script,
    chgnet_cfg,
    max_workers=chgnet_cfg.get("max_workers", 2),
    min_free_mem_gb=chgnet_cfg.get("min_free_mem_gb", 4.0)
)

merged_sorted_energies = output_dir / "sorted_energies.csv"
merge_all_sorted_energies(sorted_energy_files, merged_sorted_energies)
for subfolder in output_dir.glob("*_dir"):
    if subfolder.is_dir():
        for file in subfolder.glob("*.optdone"):
            target = output_dir / file.name
            print(f"[CLEANUP] Moving {file.name} to {output_dir}")
            shutil.move(str(file), str(target))
        # 删除子目录
        try:
            shutil.rmtree(subfolder)
            print(f"[CLEANUP] Removed directory {subfolder}")
        except Exception as e:
            print(f"[ERROR] Failed to remove directory {subfolder}: {e}")
# === Step 3: Compute PDM ===
subprocess.run([
    "python", str(pdm_script),
    "--input_dir", str(output_dir),
    "--cutoff", str(pdm_cfg["cutoff"]),
    "--n_jobs", str(pdm_cfg["n_jobs"]),
    "--mode", pdm_cfg["mode"],
    "--starts_with", pdm_cfg["starts_with"],
    "--ends_with", pdm_cfg["ends_with"],
    "--output_csv", str(output_dir / pdm_cfg["output_csv"])
], cwd=pso_dir, check=True)

# === Step 4: Merge energy and structure info ===
subprocess.run(["python", str(merge_script)], cwd=output_dir, check=True)

# === Step 5: Extract first batch ===
merged_file = output_dir / "Merged_all_structures_with_energy.csv"
batch_file = pdm_energy_dir / "updated_all_structures_summary_batch_1.csv"

merged_df = pd.read_csv(merged_file)
batch_df = merged_df.iloc[:structure_num_per_gen]
remaining_df = merged_df.iloc[structure_num_per_gen:]

# Save
batch_df.to_csv(batch_file, index=False)
remaining_df.to_csv(merged_file, index=False)

shutil.move(str(merged_file), str(pso_dir / "Merged_all_structures_with_energy.csv"))

# === Step 6: Copy selected structures ===
material_ids = batch_df["material_id"].astype(str).tolist()
for mat_id in material_ids:
    source_file = output_dir / f"{mat_id}"
    target_file=generation_dir/mat_id
    if source_file.exists():
        if target_file.exists():
            print(f"[Info] Target file exists. Removing:{target_file}")
            target_file.unlink()
        shutil.move(source_file, generation_dir)
    else:
        print(f"[Warning] File not found: {source_file}")
