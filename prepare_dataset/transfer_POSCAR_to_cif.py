import os
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter
from multiprocessing import Pool, cpu_count


def convert_poscar_to_cif(poscar_file, output_dir='.'):
    try:
        # Read POSCAR files
        poscar = Poscar.from_file(poscar_file)

        # Acquire information of structures
        structure = poscar.structure

        # Generate a CifWriter object.
        cif_writer = CifWriter(structure)

        # Get the output file name.
        cif_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(poscar_file))[0] + '.cif')

        # Write the CIF to the file.
        cif_writer.write_file(cif_filename)

        print(f"Converted {poscar_file} to {cif_filename}")
    except Exception as e:
        print(f"Error converting {poscar_file}: {e}")


def convert_all_poscar_files(poscar_dir, output_dir='.'):
    # Get the paths of all POSCAR files.
    poscar_files = [os.path.join(poscar_dir, file_name) for file_name in os.listdir(poscar_dir) if file_name.startswith('POSCAR')]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use a multiprocessing pool for parallel processing.
    with Pool(processes=8) as pool:
        pool.starmap(convert_poscar_to_cif, [(poscar_file, output_dir) for poscar_file in poscar_files])


if __name__ == "__main__":
    poscar_directory = './poscar'  # The directory where the POSCAR files are located.
    output_directory = './cif'  # The directory where the CIF files are located.

    # Batch parallel conversion.
    convert_all_poscar_files(poscar_directory, output_directory)
