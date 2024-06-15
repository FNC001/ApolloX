import os
import shutil
from glob import glob
from tqdm import tqdm  # 引入tqdm库

# 定义原始POSCAR文件所在的目录
source_directory = './'
# 定义目标目录的根路径
target_root_directory = './'

def distribute_poscar_files(source_dir, target_root, num_batches):
    # 获取所有POSCAR文件
    poscar_files = glob(os.path.join(source_dir, 'POSCAR-*.vasp'))
    num_files = len(poscar_files)

    # 确保目标目录存在
    if not os.path.exists(target_root):
        os.makedirs(target_root)

    # 创建指定数量的子目录
    for i in range(num_batches):
        os.makedirs(os.path.join(target_root, f'batch_{i+1}'), exist_ok=True)

    # 分配文件到指定数量的子目录，并显示进度条
    for file in tqdm(poscar_files, desc="Distributing files"):
        index = poscar_files.index(file)
        batch_index = index % num_batches  # 计算当前文件应该分配到的批次
        target_dir = os.path.join(target_root, f'batch_{batch_index + 1}')
        shutil.move(file, target_dir)

    print(f'All POSCAR files have been distributed into {num_batches} batches under {target_root}')

# 调用函数进行文件分配，指定子目录的数量
distribute_poscar_files(source_directory, target_root_directory, 10)  
