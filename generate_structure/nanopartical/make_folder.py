import os
import shutil
from glob import glob

# 定义原始POSCAR文件所在的目录
source_directory = './'
# 定义目标目录的根路径
target_root_directory = './'

def distribute_poscar_files(source_dir, target_root):
    # 获取所有POSCAR文件
    poscar_files = glob(os.path.join(source_dir, 'POSCAR-*.vasp'))
    num_files = len(poscar_files)
    
    # 确保目标目录存在
    if not os.path.exists(target_root):
        os.makedirs(target_root)

    # 创建五个子目录
    for i in range(5):
        os.makedirs(os.path.join(target_root, f'batch_{i+1}'), exist_ok=True)

    # 分配文件到五个子目录
    for index, file in enumerate(poscar_files):
        batch_index = index % 10  # 计算当前文件应该分配到的批次
        target_dir = os.path.join(target_root, f'batch_{batch_index + 1}')
        shutil.move(file, target_dir)

    print(f'All POSCAR files have been distributed into five batches under {target_root}')

# 调用函数进行文件分配
distribute_poscar_files(source_directory, target_root_directory)
