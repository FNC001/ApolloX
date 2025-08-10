import os
import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
def extract_formula(element_types, element_counts):
    return ''.join(f"{el}{count}" for el, count in zip(element_types, element_counts))

def read_poscar(poscar_path):
    with open(poscar_path, 'r') as file:
        lines = file.readlines()

    scale = float(lines[1].strip())
    lattice = np.array([list(map(float, lines[i].split())) for i in range(2, 5)]) * scale
    element_types = lines[5].split()
    element_counts = list(map(int, lines[6].split()))
    coord_type_line = lines[7].strip().lower()
    coord_type = 'direct' if 'direct' in coord_type_line else 'cartesian'
    total_atoms = sum(element_counts)
    coords = np.array([list(map(float, line.split()[:3])) for line in lines[8:8 + total_atoms]])

    if coord_type == 'direct':
        coords = coords @ lattice

    atom_types = []
    for el, count in zip(element_types, element_counts):
        atom_types.extend([el] * count)

    return element_types, element_counts, coords, atom_types, lattice


def calculate_distances(positions, lattice_vectors):
    distances = squareform(pdist(positions))
    num_atoms = len(positions)
    inv_lattice = np.linalg.inv(lattice_vectors)

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            vec = positions[j] - positions[i]
            vec -= np.round(vec @ inv_lattice) @ lattice_vectors
            d = np.linalg.norm(vec)
            distances[i, j] = distances[j, i] = d
    return distances


def get_neighbors(distances, cutoff):
    return {
        i: [j for j in range(len(distances)) if i != j and distances[i, j] < cutoff]
        for i in range(len(distances))
    }


def update_descriptor(descriptor_dict, atoms, label):
    key = ''.join(sorted(atoms))
    descriptor_dict[key] = descriptor_dict.get(key, 0) + 1


def calculate_sro(distances, atom_types, cutoff, modes):
    descriptors = {}
    neighbors = get_neighbors(distances, cutoff)
    n = len(atom_types)

    for i in range(n):
        ni = neighbors[i]

        if 'pair' in modes:
            for j in ni:
                if j > i:
                    update_descriptor(descriptors, [atom_types[i], atom_types[j]], 'pair')

        if 'triple' in modes:
            for j, k in combinations(ni, 2):
                if j > i and k > i and distances[j, k] < cutoff:
                    update_descriptor(descriptors, [atom_types[i], atom_types[j], atom_types[k]], 'triple')

        if 'quadruple' in modes:
            for j, k, l in combinations(ni, 3):
                if j > i and k > i and l > i:
                    if all(distances[x, y] < cutoff for x, y in combinations([j, k, l], 2)):
                        update_descriptor(descriptors, [atom_types[i], atom_types[j], atom_types[k], atom_types[l]], 'quadruple')

    return descriptors


def process_file(filepath, cutoff, modes):
    try:
        element_types, element_counts, coords, atom_types, lattice = read_poscar(filepath)
        formula = extract_formula(element_types, element_counts)
        distances = calculate_distances(coords, lattice)
        descriptors = calculate_sro(distances, atom_types, cutoff, modes)
        filename = os.path.basename(filepath)
        return filename, formula, descriptors
    except Exception as e:
        print(f"[ERROR] {filepath}: {e}")
        return os.path.basename(filepath), "error", {}


def main():
    parser = argparse.ArgumentParser(description="SRO Descriptor Extractor (multi-mode)")
    parser.add_argument('--input_dir', required=True, help='Directory containing POSCAR files')
    parser.add_argument('--output_csv', default='all_structures_summary.csv', help='Output CSV filename')
    parser.add_argument('--cutoff', type=float, default=5.0, help='Distance cutoff in angstrom')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--mode', nargs='+', choices=['pair', 'triple', 'quadruple'],
                        default=['pair'], help='One or more SRO modes')
    parser.add_argument('--starts_with', default='', help='Only include files starting with this prefix')
    parser.add_argument('--ends_with', default='', help='Only include files ending with this suffix')

    args = parser.parse_args()

    poscar_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.startswith(args.starts_with) and f.endswith(args.ends_with)
    ]

    if not poscar_files:
        print("No POSCAR files matched the criteria.")
        return

    print(f"Found {len(poscar_files)} files. Running with {args.n_jobs} threads and modes {args.mode}...")

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_file)(f, args.cutoff, args.mode) for f in poscar_files
    )

    all_keys = set()
    for _,_, desc in results:
        all_keys.update(desc.keys())
    all_keys = sorted(all_keys)

    rows = []
    for fname, formula, desc in results:
        row = [fname, formula] + [desc.get(k, 0) for k in all_keys]
        rows.append(row)
    
    df = pd.DataFrame(rows, columns=['label', 'formula'] + all_keys)

    df.to_csv(args.output_csv, index=False)
    print(f"✅ Done. Output saved to {args.output_csv}")


if __name__ == '__main__':
    main()
