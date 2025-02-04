import os
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifWriter
from multiprocessing import Pool, cpu_count


def convert_poscar_to_cif(poscar_file, output_dir='.'):
    try:
        # 使用 Poscar 读取 POSCAR 文件
        poscar = Poscar.from_file(poscar_file)

        # 获取结构信息
        structure = poscar.structure

        # 生成 CifWriter 对象
        cif_writer = CifWriter(structure)

        # 获取输出文件名
        cif_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(poscar_file))[0] + '.cif')

        # 将 CIF 写入文件
        cif_writer.write_file(cif_filename)

        print(f"Converted {poscar_file} to {cif_filename}")
    except Exception as e:
        print(f"Error converting {poscar_file}: {e}")


def convert_all_poscar_files(poscar_dir, output_dir='.'):
    # 获取所有 POSCAR 文件的路径
    poscar_files = [os.path.join(poscar_dir, file_name) for file_name in os.listdir(poscar_dir) if file_name.startswith('POSCAR')]

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用多进程池并行处理
    with Pool(processes=8) as pool:
        pool.starmap(convert_poscar_to_cif, [(poscar_file, output_dir) for poscar_file in poscar_files])


if __name__ == "__main__":
    poscar_directory = './poscar'  # POSCAR 文件所在的目录
    output_directory = './cif'  # CIF 文件的保存目录

    # 批量并行转换
    convert_all_poscar_files(poscar_directory, output_directory)