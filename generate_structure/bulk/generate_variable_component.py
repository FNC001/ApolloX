import argparse
from pathlib import Path
import random
import sys

from pymatgen.io.vasp import Poscar
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from apollox_common.pdm import shuffled_poscar


def generate_random_numbers(total, count):
    if count < 1 or total < count:
        raise ValueError('Atom total must allow at least one atom of every element')
    cuts = [0, *sorted(random.sample(range(1, total), count - 1)), total]
    return [right - left for left, right in zip(cuts[:-1], cuts[1:])]


def shuffle_and_save_poscar(original_path, output_dir, num_files, output_name='poscar'):
    if num_files < 1:
        raise ValueError('num_files must be positive')
    source = Poscar.from_file(original_path, check_for_potcar=False)
    symbols = list(dict.fromkeys(site.specie.symbol for site in source.structure))
    poscar_dir = Path(output_dir).expanduser() / output_name
    if poscar_dir.exists() and any(poscar_dir.iterdir()):
        raise FileExistsError(f'Use an empty output folder; refusing to delete existing files: {poscar_dir}')
    poscar_dir.mkdir(parents=True, exist_ok=True)
    for index in range(num_files):
        counts = generate_random_numbers(len(source.structure), len(symbols))
        composition = '_'.join(f'{symbol}{count}' for symbol, count in zip(symbols, counts))
        output = poscar_dir / f'POSCAR-{composition}-{index + 1}'
        shuffled_poscar(source, counts).write_file(output)
        print(f'Generated: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shuffle POSCAR sites and assign positive element counts')
    parser.add_argument('--input', required=True)
    parser.add_argument('--outdir', default='./')
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--output_name', default='poscar')
    args = parser.parse_args()
    shuffle_and_save_poscar(args.input, args.outdir, args.num, args.output_name)
