import os
import shutil
from pymatgen.io.vasp import Poscar
from tqdm import tqdm

def convert_vasp_to_cif_and_copy():
    current_directory = os.getcwd()

    cif_directory = os.path.join(current_directory, 'cif')
    if not os.path.exists(cif_directory):
        os.makedirs(cif_directory)

    for folder in os.listdir(current_directory):
        if folder.startswith("split") and os.path.isdir(os.path.join(current_directory, folder)):
            folder_path = os.path.join(current_directory, folder)
            for file in tqdm([f for f in os.listdir(folder_path) if f.endswith('.vasp') and not f.startswith('._')], desc=f"Converting and copying files in {folder}"):
                vasp_path = os.path.join(folder_path, file)
                try:
                    poscar = Poscar.from_file(vasp_path)
                    structure = poscar.structure
                    cif_filename = file.replace('.vasp', '.cif')
                    cif_path = os.path.join(folder_path, cif_filename)
                    structure.to(fmt="cif", filename=cif_path)

                    target_cif_path = os.path.join(cif_directory, cif_filename)
                    shutil.copy(cif_path, target_cif_path)

                    print(f"Converted and copied {vasp_path} to {target_cif_path}")
                except Exception as e:
                    print(f"Failed to process {vasp_path}: {e}")

convert_vasp_to_cif_and_copy()