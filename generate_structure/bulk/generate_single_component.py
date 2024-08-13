import random

def shuffle_poscar_lines(file_path, num_files):
    
    for n in range(num_files):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if "Direct" in line:
                direct_index = i
                break
        else:
            
            return "Error: 'Direct' not found in the file."
        
        content_to_shuffle = lines[direct_index + 1:]
        random.shuffle(content_to_shuffle)
        new_content = lines[:direct_index + 1] + content_to_shuffle
        new_file_path = file_path.replace('POSCAR', f'POSCAR-shuffled-{n+1}')
        with open(new_file_path, 'w') as new_file:
            new_file.writelines(new_content)

        print(new_file_path)

file_path = './POSCAR-ori'
num_files = 100
shuffle_poscar_lines(file_path, num_files)
