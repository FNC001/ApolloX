import argparse
from pathlib import Path
import sys

from pymatgen.io.vasp import Poscar
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from apollox_common.pdm import shuffled_poscar


def extract_composition_string(file_path):
    poscar = Poscar.from_file(file_path, check_for_potcar=False)
    return '_'.join(f'{symbol}{count}' for symbol, count in zip(poscar.site_symbols, poscar.natoms))


def shuffle_poscar_lines(file_path, num_files, out_dir, output_name='poscar'):
    if num_files < 1:
        raise ValueError('num_files must be positive')
    source = Poscar.from_file(file_path, check_for_potcar=False)
    poscar_dir = Path(out_dir).expanduser() / output_name
    if poscar_dir.exists() and any(poscar_dir.iterdir()):
        raise FileExistsError(f'Use an empty output folder; refusing to delete existing files: {poscar_dir}')
    poscar_dir.mkdir(parents=True, exist_ok=True)
    composition = extract_composition_string(file_path)
    for index in range(num_files):
        output = poscar_dir / f'POSCAR-{composition}-{index + 1}'
        shuffled_poscar(source).write_file(output)
        print(f'Generated: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shuffle complete POSCAR sites without mixing coordinate and velocity blocks')
    parser.add_argument('--input', required=True)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--outdir', default='shuffled_poscars')
    parser.add_argument('--output_name', default='poscar')
    args = parser.parse_args()
    shuffle_poscar_lines(args.input, args.num, args.outdir, args.output_name)
