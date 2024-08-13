import os
import random

original_poscar_path = './POSCAR-ori'
output_directory = './'
num_files = 250000  

def generate_random_numbers(total, count):
    while True:
        numbers = [random.randint(1, total - (count - 1)) for _ in range(count - 1)]
        numbers.append(total - sum(numbers))
        if all(n > 0 for n in numbers):
            random.shuffle(numbers)
            return numbers

def shuffle_and_save_poscar(original_path, output_dir, num_files):
    with open(original_path, 'r') as file:
        lines = file.readlines()

    direct_index = next(i for i, line in enumerate(lines) if "Cartesian" in line)
    initial_lines = lines[:direct_index + 1]

    for i in range(num_files):
        atom_positions = lines[direct_index + 1:]
        random.shuffle(atom_positions)
        shuffled_content = initial_lines + atom_positions
        shuffled_file_path = os.path.join(output_dir, f'POSCAR-shuffled-{i + 1}.vasp')
        with open(shuffled_file_path, 'w') as new_file:
            new_file.writelines(shuffled_content)
        modify_element_counts(shuffled_file_path, i + 1)  

def modify_element_counts(file_path, file_id):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    elements_line = lines[5].strip()
    element_counts_line = lines[6].strip()
    elements = elements_line.split()
    element_counts = list(map(int, element_counts_line.split()))
    new_counts = generate_random_numbers(328, len(element_counts)) # noticed 328 number change to your POSCAR true number of atom
    lines[6] = '    ' + '    '.join(map(str, new_counts)) + '\n'
    new_file_name = f'POSCAR-{"_".join(f"{elem}{count}" for elem, count in zip(elements, new_counts))}-{file_id}.vasp'
    new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)
    with open(new_file_path, 'w') as file:
        file.writelines(lines)
    os.remove(file_path)
    print(f'Processed and saved: {new_file_path}')


shuffle_and_save_poscar(original_poscar_path, output_directory, num_files)
