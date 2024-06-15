import os
import random

# 定义原始POSCAR文件路径和输出目录
original_poscar_path = './POSCAR-ori'
output_directory = './'
num_files = 1000  # 生成文件的数量

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

    # 找到“Direct”一行
    #direct_index = next(i for i, line in enumerate(lines) if "Direct" in line)
    #initial_lines = lines[:direct_index + 1]
    
    direct_index = next(i for i, line in enumerate(lines) if "Cartesian" in line)
    initial_lines = lines[:direct_index + 1]

    for i in range(num_files):
        # 打乱“Direct”后的行
        atom_positions = lines[direct_index + 1:]
        random.shuffle(atom_positions)
        shuffled_content = initial_lines + atom_positions

        # 保存打乱后的文件
        shuffled_file_path = os.path.join(output_dir, f'POSCAR-shuffled-{i + 1}.vasp')
        with open(shuffled_file_path, 'w') as new_file:
            new_file.writelines(shuffled_content)

        # 修改组分并重新命名文件
        modify_element_counts(shuffled_file_path, i + 1)  # 添加一个参数传递当前文件的序号

def modify_element_counts(file_path, file_id):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    elements_line = lines[5].strip()
    element_counts_line = lines[6].strip()
    elements = elements_line.split()
    element_counts = list(map(int, element_counts_line.split()))

    # 生成新的元素数量总和为128
    new_counts = generate_random_numbers(328, len(element_counts))
    lines[6] = '    ' + '    '.join(map(str, new_counts)) + '\n'

    # 生成新文件名，包含文件ID
    new_file_name = f'POSCAR-{"_".join(f"{elem}{count}" for elem, count in zip(elements, new_counts))}-{file_id}.vasp'
    new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

    # 保存修改后的文件
    with open(new_file_path, 'w') as file:
        file.writelines(lines)

    # 删除旧文件，仅保留重命名后的文件
    os.remove(file_path)

    print(f'Processed and saved: {new_file_path}')

# 执行函数
shuffle_and_save_poscar(original_poscar_path, output_directory, num_files)
