import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from concurrent.futures import ProcessPoolExecutor, as_completed


def read_poscar(poscar_path):
    with open(poscar_path, 'r') as file:
        lines = file.readlines()

    scale = float(lines[1].strip())
    lattice_vectors = np.array([list(map(float, line.split())) for line in lines[2:5]]) * scale
    element_types = lines[5].split()
    element_counts = list(map(int, lines[6].split()))
    positions = [list(map(float, line.split()[:3])) for line in lines[8:8 + sum(element_counts)]]

    positions = np.dot(np.array(positions), lattice_vectors)  # Convert to Cartesian coordinates
    atom_types = []
    for element_type, count in zip(element_types, element_counts):
        atom_types.extend([element_type] * count)

    return element_types, element_counts, positions, atom_types, lattice_vectors


def calculate_distances(positions, lattice_vectors):
    # Use minimum image convention to consider periodic boundary conditions
    distances = squareform(pdist(positions, metric='euclidean'))
    for i in range(len(positions)):
        for j in range(len(positions)):
            if i != j:
                vector = positions[j] - positions[i]
                vector -= np.round(vector @ np.linalg.inv(lattice_vectors)) @ lattice_vectors
                distances[i, j] = np.linalg.norm(vector)
    return distances


def calculate_sro(distances, cutoff, atom_types):
    num_atoms = len(distances)
    descriptors = {}

    # Compute descriptors for pairs
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if distances[i, j] < cutoff:
                update_descriptor(descriptors, [atom_types[i], atom_types[j]], "pair")
                # Uncomment for higher-order descriptors
                # for k in range(j + 1, num_atoms):
                #     if distances[i, k] < cutoff and distances[j, k] < cutoff:
                #         update_descriptor(descriptors, [atom_types[i], atom_types[j], atom_types[k]], "triple")
                #         for l in range(k + 1, num_atoms):
                #             if distances[i, l] < cutoff and distances[j, l] < cutoff and distances[k, l] < cutoff:
                #                 update_descriptor(descriptors, [atom_types[i], atom_types[j], atom_types[k], atom_types[l]], "quadruple")

    return descriptors


def update_descriptor(desc, elements, type):
    key = ''.join(sorted(elements))
    desc[key] = desc.get(key, 0) + 1


def process_poscar_file(filename):
    print(f"Processing {filename}")
    element_types, element_counts, positions, atom_types, lattice_vectors = read_poscar(filename)
    distances = calculate_distances(positions, lattice_vectors)
    cutoff = 5
    descriptors = calculate_sro(distances, cutoff, atom_types)
    return filename, descriptors


def process_all_poscar_files():
    all_data = []
    all_keys = set()
    poscar_files = [f for f in os.listdir('.') if f.endswith(".optdone.vasp")]

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_poscar_file, filename): filename for filename in poscar_files}

        for future in as_completed(futures):
            filename, descriptors = future.result()
            all_keys.update(descriptors.keys())
            row = [filename] +[filename + '.cif']+ [descriptors.get(key, 0) for key in sorted(all_keys)]
            all_data.append(row)

    headers = ['material_id'] + ['cif_file'] +sorted(all_keys)
    df = pd.DataFrame(all_data, columns=headers)
    df.to_csv('all_structures_summary.csv', index=False)
    print("All structures summary has been saved to all_structures_summary.csv.")


if __name__ == '__main__':
    process_all_poscar_files()

