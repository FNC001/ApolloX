import subprocess
import os
import csv
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from pymatgen.io.vasp import Poscar
from pymatgen.core.structure import Structure

def calculate_average_neighbors_for_element(poscar_file, target_element):
    structure = Poscar.from_file(poscar_file).structure
    search_radius = 3.0

    total_neighbor_counts = {}
    total_atoms = 0

    for site in structure:
        if site.species_string == target_element:
            total_atoms += 1
            neighbors = structure.get_neighbors(site, search_radius)
            for neighbor in neighbors:
                element = neighbor[0].species_string
                total_neighbor_counts[element] = total_neighbor_counts.get(element, 0) + 1

    if total_atoms > 0:
        average_neighbor_counts = {element: count / total_atoms for element, count in total_neighbor_counts.items()}
    else:
        average_neighbor_counts = {}

    return average_neighbor_counts

def process_element(poscar_file, element):
    results = calculate_average_neighbors_for_element(poscar_file, element)
    output_rows = []
    for neighbor_element, avg_neighbors in results.items():
        output_rows.append([poscar_file.name, element, neighbor_element, avg_neighbors])
    return output_rows

def main(njobs):
    poscar_files = list(Path(".").glob("POSCAR-*"))
    elements = ["Mo", "B", "Co", "Fe", "Ni", "O"]
    tasks = [(file, element) for file in poscar_files for element in elements]

    results = Parallel(njobs, backend="multiprocessing")(
        delayed(process_element)(file, element) for file, element in tqdm(tasks, ncols=79)
    )

    # Writing results to CSV
    csv_file_path = "all_calculations_summary.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["POSCAR File", "Element", "Neighbor Element", "Average Neighbors"])
        for result_rows in results:
            for row in result_rows:
                writer.writerow(row)

    print(f"All calculations have been summarized and saved to '{csv_file_path}'.")

if __name__ == "__main__":
    main(njobs=32)

