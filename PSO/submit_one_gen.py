import os
import subprocess
import argparse
import yaml
from pathlib import Path
import shutil
def run_cmd(command, cwd=None):
    print(f"\nRunning: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed: {command}")
        exit(1)

def find_apollox_root(start_path="."):
    current = Path(start_path).resolve()
    for parent in [current] + list(current.parents):
        if parent.name == "ApolloX":
            return parent
    raise FileNotFoundError("ApolloX directory not found in parent paths.")

def get_composition_tag(g):
    gen_dir = Path(f"./poscars/generation{g}")
    for file in gen_dir.iterdir():
        if file.name.startswith("POSCAR-"):
            return file.name.replace("POSCAR-", "")
    raise FileNotFoundError(f"No POSCAR-XXXX file found in {gen_dir}")


def find_latest_pt():
    pt_files = sorted(Path(".").glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pt_files[0] if pt_files else None

def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--g', type=int, required=True, help="Generation number")
    args = parser.parse_args()
    g = args.g
    config = load_config()
    apollox_root = find_apollox_root()
    scaler_path = config["scaler_path"]
    pso_params = config["PSO"]

    composition_tag = get_composition_tag(g)
    extract_script = apollox_root / "cond-cdvae/scripts/extract_gen.py"

    # === 替换 run_cmd 部分 ===
    run_cmd(
        f"python {apollox_root}/PSO/pso_optimizer.py "
        f"--g {g} "
        f"--target_ratio {pso_params['target_ratio']} "
        f"--min_bound_scale {pso_params['min_bound_scale']} "
        f"--max_bound_scale {pso_params['max_bound_scale']} "
        f"--max_iter {pso_params['max_iter']}"
    )

    run_cmd(
        f"python {apollox_root}/prepare_dataset/standardize.py "
        f"--input temp/optimized_particles_distribution_with_ids_{g}.csv "
        f"--scaler {scaler_path} "
        f"--output temp/standardized_optimized_particles_distribution{g}.csv"
    )

    run_cmd(f"python {apollox_root}/PSO/make_sh.py "
           f"--g {g}")

    # === 执行 .sh 脚本 ===
    sh_dir = Path("temp") / f"sh_files_{g}"
    sh_dir.mkdir(parents=True, exist_ok=True)
    
    pt_output_dir = Path("temp") / f"pt_files_{g}"
    pt_output_dir.mkdir(parents=True, exist_ok=True)
    
    for sh_file in sorted(sh_dir.glob("run_evaluation_*.sh")):
        print(f"Executing {sh_file}...")
        os.chmod(sh_file, 0o755)
        run_cmd(str(sh_file))
    
        pt_file = find_latest_pt()
        if pt_file:
            print(f"Processing .pt file: {pt_file}")
            run_cmd(f"python {extract_script} {pt_file}")
    
            result_folder = pt_file.stem
            gen_path = Path(result_folder) / "gen"
            if gen_path.exists():
                print(f"Entering gen folder in {result_folder}...")
                # run_cmd("python ../../bulk.py", cwd=gen_path)
            else:
                print(f"Error: gen folder not found in {result_folder}")
    
            # === 移动 .pt 文件和结果文件夹到 temp 目录 ===
            dest_pt = pt_output_dir / pt_file.name
            dest_result = pt_output_dir / result_folder
    
            shutil.move(str(pt_file), dest_pt)

            dest_result_path = Path(dest_result)
            if dest_result_path.exists():
                print(f"[Info] Target folder {dest_result_path} exists, removing before move.")
                if dest_result_path.is_dir():
                    shutil.rmtree(dest_result_path)
                else:
                    dest_result_path.unlink()
            
            if Path(result_folder).exists():
                shutil.move(result_folder, dest_result)
                print(f"Moved {pt_file} and folder {result_folder} to {pt_output_dir}")
            else:
                print(f"[Warning] Result folder {result_folder} does not exist, skipped moving.")
        else:
            print("Error: no .pt file found.")
    
        print("\nAll tasks completed.")

if __name__ == "__main__":
    main()


