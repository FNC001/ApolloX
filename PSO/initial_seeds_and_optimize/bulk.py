import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def read_poscar(poscar_path):
    """
    Read a POSCAR file and return:
    - element_types: a list of element symbols
    - element_counts: a list of the number of atoms for each element
    - positions: the Cartesian coordinates (numpy array)
    - atom_types: a list mapping each atom index to its element type
    - lattice_vectors: the 3x3 lattice vectors (in Cartesian)
    """
    with open(poscar_path, 'r') as file:
        lines = file.readlines()

    # The second line contains the scaling factor
    scale = float(lines[1].strip())
    # Lines 3-5 (index 2:5) describe the lattice vectors
    lattice_vectors = np.array([list(map(float, line.split())) for line in lines[2:5]]) * scale
    # Line 6 (index 5) lists the element types
    element_types = lines[5].split()
    # Line 7 (index 6) lists the number of atoms for each element
    element_counts = list(map(int, lines[6].split()))
    # Lines 9 onwards (index 8:8+sum(element_counts)) are the fractional coordinates
    positions = [list(map(float, line.split()[:3])) for line in lines[8:8 + sum(element_counts)]]

    # Convert fractional coordinates to Cartesian coordinates
    positions = np.dot(np.array(positions), lattice_vectors)

    # Build a list that maps each atom to its element type
    atom_types = []
    for element_type, count in zip(element_types, element_counts):
        atom_types.extend([element_type] * count)

    return element_types, element_counts, positions, atom_types, lattice_vectors


def calculate_distances(positions, lattice_vectors):
    """
    Calculate pairwise distances between atoms,
    taking into account periodic boundary conditions
    using the minimum image convention.
    """
    # First, compute the raw Euclidean distances
    distances = squareform(pdist(positions, metric='euclidean'))
    num_atoms = len(positions)

    # Then apply the minimum image convention
    inv_lattice = np.linalg.inv(lattice_vectors)
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                vector = positions[j] - positions[i]
                # Shift the vector by a nearest periodic image
                vector -= np.round(vector @ inv_lattice) @ lattice_vectors
                distances[i, j] = np.linalg.norm(vector)
    return distances


def calculate_sro(distances, cutoff, atom_types):
    """
    Calculate the short-range ordering descriptors based on the given cutoff distance.
    This includes counting pairs, triples, and quadruples of atoms that are all within
    the specified cutoff distance of each other.
    """
    num_atoms = len(distances)
    descriptors = {}

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if distances[i, j] < cutoff:
                update_descriptor(descriptors, [atom_types[i], atom_types[j]], "pair")
                # for k in range(j + 1, num_atoms):
                #     if distances[i, k] < cutoff and distances[j, k] < cutoff:
                #         update_descriptor(descriptors, [atom_types[i], atom_types[j], atom_types[k]], "triple")
                #         for l in range(k + 1, num_atoms):
                #             if (distances[i, l] < cutoff and
                #                     distances[j, l] < cutoff and
                #                     distances[k, l] < cutoff):
                #                 update_descriptor(
                #                     descriptors,
                #                     [atom_types[i], atom_types[j], atom_types[k], atom_types[l]],
                #                     "quadruple"
                #                 )
    return descriptors


def update_descriptor(desc, elements, label):
    """
    Update the descriptor dictionary by sorting the element symbols
    into a consistent key and incrementing the count.
    The 'label' parameter is not used directly in the dictionary key,
    but left here for clarity or possible future extensions.
    """
    key = ''.join(sorted(elements))
    desc[key] = desc.get(key, 0) + 1


def process_all_poscar_files():
    """
    1. Iterate over all files in the current directory whose names start with 'POSCAR'
    2. Read and parse each file to calculate SRO descriptors
    3. Collect all keys (i.e., the sorted element combinations) from all files
    4. Build a DataFrame with one row per file and columns representing each key
    5. Save the DataFrame to 'all_structures_summary.csv'
    """
    all_entries = []  # A list of (filename, descriptors)
    all_keys = set()

    for filename in os.listdir('.'):
        if filename.endswith(".optdone.vasp"):
            print(f"Processing {filename}...")
            element_types, element_counts, positions, atom_types, lattice_vectors = read_poscar(filename)
            distances = calculate_distances(positions, lattice_vectors)

            # Set the cutoff distance
            cutoff = 5.0

            descriptors = calculate_sro(distances, cutoff, atom_types)
            all_entries.append((filename, descriptors))
            all_keys.update(descriptors.keys())

    all_keys = sorted(all_keys)
    data_rows = []
    for filename, descriptors in all_entries:
        row = [filename] +[filename+'.cif']+ [descriptors.get(k, 0) for k in all_keys]
        data_rows.append(row)

    # Create the DataFrame and save to CSV
    headers = ['material_id'] +['cif_file']+sorted(all_keys)
    df = pd.DataFrame(data_rows, columns=headers)
    df.to_csv('all_structures_summary.csv', index=False)
    print("All structures summary has been saved to all_structures_summary.csv.")


if __name__ == '__main__':
    process_all_poscar_files()

