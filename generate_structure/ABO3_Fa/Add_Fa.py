import os
from pymatgen.io.vasp import Poscar
from pymatgen.io.xyz import XYZ
from tqdm import tqdm

input_directory = './str/no_fa' 
output_directory = './str/with_fa'
xyz_filename = './str/Fa.xyz'  # Put fa.cif here

os.makedirs(output_directory, exist_ok=True)
molecule_structure = XYZ.from_file(xyz_filename).molecule
molecule_centroid = molecule_structure.center_of_mass
poscar_files = [f for f in os.listdir(input_directory) if f.startswith('Changed_POSCAR_')]

# Check POSCAR
if not poscar_files:
    print("No POSCAR files found in the specified directory.")
    exit()

# Due POSCAR
for poscar_file in tqdm(poscar_files, desc="Processing POSCAR files"):
    poscar_path = os.path.join(input_directory, poscar_file)
    original_structure = Poscar.from_file(poscar_path).structure

    # Find Ir position repalce Ir atom as what you want
    ir_indices = [i for i, site in enumerate(original_structure) if site.species_string == "Ir"]

    # Replace Ir with cif
    for index in sorted(ir_indices, reverse=True):
        ir_site = original_structure.pop(index)
        translation_vector = ir_site.coords - molecule_centroid
        molecule_to_add = molecule_structure.copy()
        molecule_to_add.translate_sites(list(range(len(molecule_to_add))), translation_vector)
        for site in molecule_to_add:
            original_structure.append(site.species_string, site.coords, coords_are_cartesian=True)

    output_path = os.path.join(output_directory, f'POSCAR-{poscar_file.split("_")[-1]}.vasp')
    Poscar(original_structure).write_file(output_path)

    print(f"Processed {poscar_path} and saved to {output_path}")