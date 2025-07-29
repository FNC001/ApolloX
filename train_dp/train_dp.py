import dpdata
import os

# Define folder paths (can be modified to the actual folder list).
folders = ['POSCAR-shuffled-10-ori_dir', 'POSCAR-shuffled-12-ori_dir', 'POSCAR-shuffled-14-ori_dir', 'POSCAR-shuffled-16-ori_dir', 'POSCAR-shuffled-11-ori_dir', 'POSCAR-shuffled-13-ori_dir', 'POSCAR-shuffled-15-ori_dir']

# Define the output folder.
output_folder = "processed_data"
os.makedirs(output_folder, exist_ok=True)

step_interval = 25

for folder in folders:
    outcar_path = os.path.join(folder, 'OUTCAR')
    if os.path.exists(outcar_path):
        print(f"Processing {outcar_path}...")
        try:
            # Load OUTCAR data
            data = dpdata.LabeledSystem(outcar_path, fmt='vasp/outcar')

            # Extract subset.
            subset_data = data.sub_system(range(0, len(data), step_interval))

            # Save in DeepMD-kit format.
            output_path = os.path.join(output_folder, f"{os.path.basename(folder)}_subset")
            subset_data.to('deepmd/npy', output_path)

            print(f"Processed {outcar_path} successfully! Data saved in {output_path}")
        except Exception as e:
            print(f"Error processing {outcar_path}: {e}")
    else:
        print(f"OUTCAR not found in {folder}")

print("All processing done!")

