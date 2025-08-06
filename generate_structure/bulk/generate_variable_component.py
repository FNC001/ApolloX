import os
import random
import argparse

def generate_random_numbers(total, count):
    while True:
        numbers = [random.randint(1, total - (count - 1)) for _ in range(count - 1)]
        numbers.append(total - sum(numbers))
        if all(n > 0 for n in numbers):
            random.shuffle(numbers)
            return numbers

def find_coordinate_line_index(lines):
    for i, line in enumerate(lines):
        if "Direct" in line or "direct" in line:
            return i
    for i, line in enumerate(lines):
        if "Cartesian" in line or "cartesian" in line:
            return i
    raise ValueError("Not found 'Direct' or 'Cartesian' in POSCAR")

def shuffle_and_save_poscar(original_path, output_dir, num_files,output_name):
    poscar_dir = os.path.join(output_dir, output_name)
    if os.path.exists(poscar_dir):
        shutil.rmtree(poscar_dir)
    os.makedirs(poscar_dir)

    with open(original_path, 'r') as file:
        lines = file.readlines()

    direct_index = find_coordinate_line_index(lines)
    initial_lines = lines[:direct_index + 1]

    for i in range(num_files):
        atom_positions = lines[direct_index + 1:]
        random.shuffle(atom_positions)
        shuffled_content = initial_lines + atom_positions

        temp_file_path = os.path.join(poscar_dir, f'POSCAR-shuffled-{i + 1}')
        with open(temp_file_path, 'w') as new_file:
            new_file.writelines(shuffled_content)

        modify_element_counts(temp_file_path, i + 1, poscar_dir)

def modify_element_counts(file_path, file_id, poscar_dir):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    elements_line = lines[5].strip()
    element_counts_line = lines[6].strip()

    elements = elements_line.split()
    element_counts = list(map(int, element_counts_line.split()))
    total_atoms = sum(element_counts)

    new_counts = generate_random_numbers(total_atoms, len(element_counts))

    lines[6] = '    ' + '    '.join(map(str, new_counts)) + '\n'

    new_file_name = f'POSCAR-{"_".join(f"{elem}{count}" for elem, count in zip(elements, new_counts))}-{file_id}'
    new_file_path = os.path.join(poscar_dir, new_file_name)

    with open(new_file_path, 'w') as f:
        f.writelines(lines)

    os.remove(file_path)

    print(f"Generated: {new_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle POSCAR atom positions and modify element counts.")
    parser.add_argument("--input", type=str, required=True, help="Original POSCAR file path")
    parser.add_argument("--outdir", type=str, default="./", help="Output directory")
    parser.add_argument("--num", type=int, default=10, help="Number of shuffled files to generate")
    parser.add_argument("--output_name", type=int, default="poscar", help="name of the folder of random structures")
    args = parser.parse_args()

    shuffle_and_save_poscar(args.input, args.outdir, args.num,args.output_name)
