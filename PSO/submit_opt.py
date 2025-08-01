import argparse
import subprocess
from pathlib import Path
import sys
import yaml
import pandas as pd
import os
import shutil
import concurrent.futures
import time
import psutil 
def find_apollox_root():
    current = Path(__file__).resolve()
    for parent in [current]+list(current.parents):
        if parent.name=="ApolloX":
            return parent
    raise FileNotFoundError("ApolloX directory not found in parent paths.")

def load_config(apollox_root):
    config_path = apollox_root/"PSO"/ "config.yaml"
    if not config_path.exists():
        print(f"Error: config.yaml not found at {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    try:
        chgnet_cfg = config["opt"]
        pdm_cfg=config["pdm"]
        structure_num_per_gen = config["structure_num_per_gen"]
        return chgnet_cfg,pdm_cfg,structure_num_per_gen
    except KeyError:
        print("Error: 'chgnet' section not found in config.yaml")
        sys.exit(1)
def copy_selected_poscars(apollox_root, base_dir, g, combined_csv_path):
    dest_dir = Path(os.getcwd()) / "poscars" / f"generation{g+1}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(combined_csv_path)
    material_ids = df["material_id"].astype(str).unique()

    poscar_dir = apollox_root / "PSO" / "poscar"
    for mid in material_ids:
        candidates = list(poscar_dir.glob(f"{mid}*"))
        if not candidates:
            print(f"[COPY] Warning: No file found for material_id={mid} in {poscar_dir}")
            continue
        for c in candidates:
            shutil.copy(c, dest_dir / c.name)
            print(f"[COPY] Copied {c} to {dest_dir}")

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue
        gen_folder = folder / "gen"
        merged_csv = gen_folder / "Merged_all_structures_with_energy.csv"
        if not merged_csv.exists():
            continue

        df_merged = pd.read_csv(merged_csv)
        merged_material_ids = df_merged["material_id"].astype(str).unique()
        intersect_ids = set(material_ids) & set(merged_material_ids)
        if not intersect_ids:
            continue
    
        for file_path in gen_folder.glob("*.optdone"):
            if file_path.name in intersect_ids:
                shutil.move(file_path, dest_dir / file_path.name)
                print(f"[MOVE] Copied {file_path} to {dest_dir}")

def run_task(gen_dir: Path, structure_name: str, chgnet_script: Path, chgnet_cfg: dict,
             pdm_script: Path, pdm_cfg: dict, apollox_root: Path):
    # 1. chgnet_gpu.py
    cmd_chgnet = [
        "python", str(chgnet_script),
        "--input_pattern", structure_name,
        "--mlp_optstep", str(chgnet_cfg.get("mlp_optstep", 1)),
        "--fmax", str(chgnet_cfg.get("fmax", 0.01)),
        "--output_csv", "sorted_energies.csv"
    ]
    print(f"[CHGNet] Running in {gen_dir} with input {structure_name}...")
    subprocess.run(cmd_chgnet, cwd=gen_dir, check=True)

    # 2. compute_pdm.py
    cmd_pdm = [
        "python", str(pdm_script),
        "--input_dir", ".",
        "--cutoff", str(pdm_cfg["cutoff"]),
        "--n_jobs", str(pdm_cfg["n_jobs"]),
        "--mode", pdm_cfg["mode"],
        "--starts_with",str("POSCAR"),
        "--ends_with",str("optdone"),
        "--output_csv", str(pdm_cfg["output_csv"])
    ]
    print(f"[PDM] Running compute_pdm.py for {structure_name}...")
    subprocess.run(cmd_pdm, cwd=gen_dir, check=True)
    sorted_csv_path = gen_dir / "sorted_energies.csv"
    # 3. merge1.py 
    if sorted_csv_path.exists():
        merge_script = apollox_root / "PSO" / "merge1.py"
        cmd_merge = ["python", str(merge_script)]
        print(f"[MERGE] Running merge1.py in {gen_dir}...")
        subprocess.run(cmd_merge, cwd=gen_dir, check=True)
    else:
        print(f"[MERGE] Skipping merge1.py since {sorted_csv_path} not found.")


def run_chgnet_parallel(base_dir: Path, chgnet_script: Path, chgnet_cfg: dict,
                        pdm_script: Path, pdm_cfg: dict, apollox_root: Path,
                        max_workers: int = 2, min_free_mem_gb: float = 4.0):
    tasks = []
    for folder in sorted(base_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("eval_gen_"):
            structure_name = folder.name[len("eval_gen_"):] + ".vasp"
            gen_dir = folder / "gen"
            if not gen_dir.exists():
                print(f"[CHGNet] Warning: {gen_dir} does not exist, skipping.")
                continue

            files = list(gen_dir.glob("*.vasp"))
            if not files:
                print(f"[CHGNet] Warning: No .vasp file in {gen_dir}, skipping.")
                continue

            old_path = files[0]
            new_path = gen_dir / structure_name
            old_path.rename(new_path)

            tasks.append((gen_dir, structure_name))

    def has_enough_memory(threshold_gb):
        mem = psutil.virtual_memory()
        return mem.available / 1e9 >= threshold_gb

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for gen_dir, structure_name in tasks:
            while not has_enough_memory(min_free_mem_gb):
                print("[MEMORY] Not enough free memory, waiting 10 seconds...")
                time.sleep(10)

            future = executor.submit(run_task, gen_dir, structure_name,
                                     chgnet_script, chgnet_cfg,
                                     pdm_script, pdm_cfg, apollox_root)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except subprocess.CalledProcessError as e:
                print(f"[CHGNet-ERROR] Failed in one task: {e}")

                
def run_chgnet_and_pdm(g: int):
    base_dir = Path("temp") / f"pt_files_{g}"
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        sys.exit(1)

    apollox_root = find_apollox_root()
    chgnet_script = apollox_root / "opt" / "chgnet_gpu.py"
    pdm_script = apollox_root / "prepare_dataset" / "compute_pdm.py"
    #pso_dir = apollox_root / "PSO"

    chgnet_cfg, pdm_cfg,n = load_config(apollox_root)

    run_chgnet_parallel(base_dir, chgnet_script, chgnet_cfg,
                        pdm_script, pdm_cfg, apollox_root,
                        max_workers=chgnet_cfg.get("max_workers", 2),
                        min_free_mem_gb=chgnet_cfg.get("min_free_mem_gb", 4.0))

    all_data = []
    for folder in sorted(base_dir.iterdir()):
        if folder.is_dir():
            merged_path = folder / "gen" / "Merged_all_structures_with_energy.csv"
            if merged_path.exists():
                df = pd.read_csv(merged_path)
                all_data.append(df)
            else:
                print(f"[MERGE-COMBINE] Warning: {merged_path} not found.")

    backup_csv = apollox_root / "PSO" / "Merged_all_structures_with_energy.csv"
    if not backup_csv.exists():
        print(f"Error: {backup_csv} not found.")
        sys.exit(1)

    g_next = g + 1
    output_dir = Path(os.getcwd()) / "pdm_and_energy"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / f"updated_all_structures_summary_batch_{g_next}.csv"

    if not all_data:
        print("[MERGE-COMBINE] No valid data found. Using backup directly.")
        df_backup = pd.read_csv(backup_csv)
        final_df = df_backup.head(n)
        final_df.to_csv(output_csv_path, index=False)
        print(f"[FINAL] Used {n} rows from backup only. Saved to {output_csv_path}")
    else:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.drop_duplicates(inplace=True)

        l = len(combined_df)
        if l >= n:
            print(f"[FINAL] Enough structures ({l}) already collected, no need to add more.")
            final_df = combined_df
        else:
            remain = n - l
            df_backup = pd.read_csv(backup_csv)
            df_extra = df_backup.head(remain)
            final_df = pd.concat([combined_df, df_extra], ignore_index=True)
            print(f"[FINAL] Added {remain} rows from backup.")

        final_df.to_csv(output_csv_path, index=False)
        print(f"[MERGE-COMBINE] Saved merged file to {output_csv_path}")

    copy_selected_poscars(apollox_root, base_dir, g, output_csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--g", type=int, required=True, help="g index (used to find pt_files_g)")
    args = parser.parse_args()
    run_chgnet_and_pdm(args.g)




