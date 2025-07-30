import os
import argparse
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter
from multiprocessing import Pool


def convert_poscar_to_cif(poscar_file, output_dir):
    try:
        poscar = Poscar.from_file(poscar_file)
        structure = poscar.structure
        cif_writer = CifWriter(structure)

        cif_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(poscar_file))[0] + '.cif')
        cif_writer.write_file(cif_filename)

        print(f"[✔] {poscar_file} → {cif_filename}")
    except Exception as e:
        print(f"[✘] Error converting {poscar_file}: {e}")


def convert_all_poscars(base_dir, nproc=4):
    poscar_dir = os.path.join(base_dir, 'poscar')
    output_dir = os.path.join(base_dir, 'cif')

    if not os.path.exists(poscar_dir):
        print(f"[Error] Directory not found: {poscar_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    poscar_files = [
        os.path.join(poscar_dir, f)
        for f in os.listdir(poscar_dir)
        if f.startswith("POSCAR") and os.path.isfile(os.path.join(poscar_dir, f))
    ]

    args = [(f, output_dir) for f in poscar_files]

    with Pool(processes=nproc) as pool:
        pool.starmap(convert_poscar_to_cif, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert all POSCAR files under poscar/ to CIF format under cif/")
    parser.add_argument('base_path', type=str, help="Path: The directory containing the 'poscar' folder, e.g., ./data")
    parser.add_argument('--n', type=int, default=4, help="Number of parallel processes, default is 4")

    args = parser.parse_args()

    convert_all_poscars(args.base_path, args.n)
