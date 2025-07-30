import pandas as pd
import os
import argparse
from pathlib import Path

def find_apollox_path():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.name == "ApolloX":
            return str(parent)
    raise FileNotFoundError("ApolloX directory not found in parent paths.")

def get_composition_from_poscar(g):
    poscar_dir = f'./poscars/generation{g}/'
    for file in os.listdir(poscar_dir):
        if file.startswith('POSCAR'):
            with open(os.path.join(poscar_dir, file), 'r') as f:
                lines = f.readlines()
                if len(lines) > 6:
                    elements = lines[5].strip().split()
                    counts = lines[6].strip().split()
                    if len(elements) == len(counts):
                        return ''.join(f'{el}{ct}' for el, ct in zip(elements, counts))
    raise FileNotFoundError(f"No valid POSCAR file found in {poscar_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a .sh file for each line of data.")
    parser.add_argument('--g', type=int, required=True, help="The input parameter g, used to read the data file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    g = args.g  
    apollox_path = find_apollox_path()
    formula = get_composition_from_poscar(g)

    data = pd.read_csv(f'./temp/standardized_optimized_particles_distribution{g}.csv')

    output_dir = os.path.join("temp", f"sh_files_{g}")
    os.makedirs(output_dir, exist_ok=True)

    for idx, row in data.iterrows():
        material_id = row['material_id']
        element_values = row.iloc[2:-1]
        element_values_str = ",".join(map(str, element_values))


        sh_content = "#!/bin/bash\nCUDA_VISIBLE_DEVICES=0\n"
        sh_content += f"python {apollox_path}/cond-cdvae/scripts/evaluate.py --model_path `pwd` --tasks gen \\\n"
        sh_content += f"    --formula={formula} \\\n"
        sh_content += f"    --pressure=0 \\\n"
        sh_content += f"    --label=\"{material_id}\" \\\n"
        sh_content += f"    --element_values=\"{element_values_str}\" \\\n"
        sh_content += "    --batch_size=1 \\\n"
        sh_content += "    --num_batches_to_samples=1"

        sh_path = os.path.join(output_dir, f'run_evaluation_{idx}.sh')
        with open(sh_path, 'w') as f:
            f.write(sh_content)

        print(f"Shell script saved to {sh_path}")

    print(f"All shell scripts saved to {output_dir}")

