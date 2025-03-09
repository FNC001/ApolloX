# Make the input files of the generative model for structures in the file “updated_all_structures_summary_batch_*.csv”.
import pandas as pd
import os
import argparse

# Set up command-line argument parsing.
def parse_args():
    parser = argparse.ArgumentParser(description="Generate a .sh file for each line of data.")
    parser.add_argument('--g', type=int, required=True, help="The input parameter g, which is used to read the data file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    g = args.g  
    #Read data
    data = pd.read_csv(f'standardized_optimized_particles_distribution{g}.csv')

    output_dir = f'sh_files_{g}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate a .sh file for each line of data
    for idx, row in data.iterrows():
        material_id = row['material_id']
        element_values = row.iloc[1:-1] 
        element_values_str = ",".join(map(str, element_values))

        # The content of the .sh file
        sh_content = "#!/bin/bash\nCUDA_VISIBLE_DEVICES=0\n"
        sh_content += "python ~/cond-cdvae-main/scripts/evaluate.py --model_path `pwd` --tasks gen \\\n"
        sh_content += "    --formula=B12Mo12Co12Fe12Ni12O60 \\\n"
        sh_content += f"    --pressure=0 \\\n"
        sh_content += f"    --label=\"{material_id}\" \\\n"
        sh_content += f"    --element_values=\"{element_values_str}\" \\\n"
        sh_content += "    --batch_size=1 \\\n"
        sh_content += "    --num_batches_to_samples=1"

        # Save ".sh" files
        sh_file_path = os.path.join(output_dir, f'run_evaluation_{idx}.sh')
        with open(sh_file_path, 'w') as f:
            f.write(sh_content)

        print(f"Shell script saved to {sh_file_path}")

    print(f"All shell scripts saved to {output_dir}")
