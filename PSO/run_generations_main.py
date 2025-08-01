import subprocess
import yaml
from pathlib import Path

def find_apollox_root(start_path="."):
    current = Path(start_path).resolve()
    for parent in [current] + list(current.parents):
        if parent.name == "ApolloX":
            return parent
    raise FileNotFoundError("ApolloX directory not found in parent paths.")

def read_gen_num(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["gen_num"]

def main():
    apollox_root = find_apollox_root()
    config_path = apollox_root /"PSO"/"config.yaml"
    gen_num = read_gen_num(config_path)

    # Step 1: Run initial_structures.py once
    print(f"[INFO] Running initial_structures.py")
    subprocess.run(["python", str(apollox_root /"PSO"/"initial_structures.py")], check=True)

    # Step 2: Loop over generations and run submit_one_gen.py and submit_opt.py
    for g in range(1, gen_num + 1):
        print(f"[INFO] Running generation {g}...")

        # Run submit_one_gen.py --g g
        subprocess.run(["python", str(apollox_root /"PSO"/"submit_one_gen.py"), "--g", str(g)], check=True)

        # Run submit_opt.py --g g
        subprocess.run(["python", str(apollox_root /"PSO"/"submit_opt.py"), "--g", str(g)], check=True)
    print(f"\n {gen_num} generation structures have been generated.")
    print("PDMs and energies are saved in `pdm_and_energy`.")
    print("Structures are saved in `poscars`.")

if __name__ == "__main__":
    main()

