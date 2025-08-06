import random
import argparse
import os
import shutil
def extract_composition_string(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    element_line = lines[5].strip().split()
    count_line = lines[6].strip().split()

    if len(element_line) != len(count_line):
        raise ValueError("Mismatch between number of elements and atom counts in POSCAR.")

    composition_parts = [
        f"{el}{count}" for el, count in zip(element_line, count_line)
    ]
    return "_".join(composition_parts)

def shuffle_poscar_lines(file_path, num_files, out_dir):
    poscar_dir = os.path.join(out_dir, "poscar")
    if os.path.exists(poscar_dir):
        shutil.rmtree(poscar_dir)
    os.makedirs(poscar_dir)

    composition_str = extract_composition_string(file_path)

    for n in range(num_files):
        with open(file_path, 'r') as file:
            lines = file.readlines()


        for i, line in enumerate(lines):
            if "Direct" in line or "Cartesian" in line:
                direct_index = i
                break
        else:
            raise ValueError("Error: 'Direct' or 'Cartesian' not found in the file.")

        content_to_shuffle = lines[direct_index + 1:]
        random.shuffle(content_to_shuffle)
        new_content = lines[:direct_index + 1] + content_to_shuffle


        new_file_name = f"POSCAR-{composition_str}-{n+1}"
        new_file_path = os.path.join(poscar_dir, new_file_name)

        with open(new_file_path, 'w') as new_file:
            new_file.writelines(new_content)

        print(f"Generated: {new_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle atomic positions in POSCAR format.")
    parser.add_argument("--input", type=str, required=True, help="Path to the original POSCAR file.")
    parser.add_argument("--num", type=int, default=10, help="Number of shuffled files to generate.")
    parser.add_argument("--outdir", type=str, default="shuffled_poscars", help="Directory to save shuffled POSCAR files.")
    
    args = parser.parse_args()
    shuffle_poscar_lines(args.input, args.num, args.outdir)
