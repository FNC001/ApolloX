import numpy as np
from collections import Counter

A_replacements = {'Cs': 0.67, 'Rb': 0.22, 'Ir': 0.11}
B_replacements = {'Pb': 0.55, 'Ag': 0.17, 'Zn': 0.28}

# define atom number in POSCAR
num_A = 108  # A place
num_B = 108  # B place
num_O3 = 324  # O3 

# How many New POSCAR you want
num_files = 10000

# load origin POSCAR 
file_path = './SPOSCAR'
with open(file_path, 'r') as file:
    sposcar_content = file.readlines()

for i in range(num_files):
    # Gen random
    A_elements = np.random.choice(list(A_replacements.keys()), size=num_A, p=list(A_replacements.values()))
    B_elements = np.random.choice(list(B_replacements.keys()), size=num_B, p=list(B_replacements.values()))
    O3_elements = np.array(['Br'] * num_O3)  # Br not change

    np.random.shuffle(A_elements)
    np.random.shuffle(B_elements)

    A_elements_counts = Counter(A_elements)
    B_elements_counts = Counter(B_elements)
    O3_elements_counts = {'Br': num_O3}

    all_elements_counts = {**A_elements_counts, **B_elements_counts, **O3_elements_counts}
    new_elements_definition = ' '.join(all_elements_counts.keys())
    new_element_counts = ' '.join(str(count) for count in all_elements_counts.values())

    new_lines_corrected = [
        sposcar_content[0],
        sposcar_content[1],
        sposcar_content[2],
        sposcar_content[3],
        sposcar_content[4],
        new_elements_definition + '\n',
        new_element_counts + '\n',
        sposcar_content[7],
    ]

    A_coords = sposcar_content[8:8+num_A]
    B_coords = sposcar_content[8+num_A:8+num_A+num_B]
    O3_coords = sposcar_content[8+num_A+num_B:8+num_A+num_B+num_O3]
    np.random.shuffle(A_coords)
    np.random.shuffle(B_coords)
    new_lines_corrected += [f'  {coord.strip()}\n' for coord in A_coords + B_coords + O3_coords]

    final_file_path = f'./str/no_fa/Changed_POSCAR_{i+1}'
    with open(final_file_path, 'w') as new_file:
        new_file.writelines(new_lines_corrected)

    print("File saved to:", final_file_path)