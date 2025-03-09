#Shuffling the atoms and getting structures.
import random
import argparse


def shuffle_poscar_lines(file_path, num_files):
    for n in range(num_files):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if "Cartesian" in line or "Direct" in line:
                direct_index = i
                break
        else:
            return "Error: 'Direct' not found in the file."

        content_to_shuffle = lines[direct_index + 1:]
        random.shuffle(content_to_shuffle)
        new_content = lines[:direct_index + 1] + content_to_shuffle
        new_file_path = file_path.replace('POSCAR', f'POSCAR-B12Mo12Co12Fe12Ni12O60-{n}')
        with open(new_file_path, 'w') as new_file:
            new_file.writelines(new_content)

        print(new_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle atomic positions in POSCAR file and generate multiple files.")
    parser.add_argument("file_path", type=str, help="Path to the POSCAR file")
    parser.add_argument("num_files", type=int, help="Number of shuffled files to generate")

    args = parser.parse_args()
    shuffle_poscar_lines(args.file_path, args.num_files)
