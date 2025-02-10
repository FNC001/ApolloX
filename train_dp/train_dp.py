import dpdata
import os

# 定义文件夹路径（可以修改为实际的文件夹列表）
folders = ['POSCAR-shuffled-10-ori_dir', 'POSCAR-shuffled-12-ori_dir', 'POSCAR-shuffled-14-ori_dir', 'POSCAR-shuffled-16-ori_dir', 'POSCAR-shuffled-11-ori_dir', 'POSCAR-shuffled-13-ori_dir', 'POSCAR-shuffled-15-ori_dir']

# 定义输出文件夹
output_folder = "processed_data"
os.makedirs(output_folder, exist_ok=True)

# 每十步提取一个结构
step_interval = 25

for folder in folders:
    outcar_path = os.path.join(folder, 'OUTCAR')
    if os.path.exists(outcar_path):
        print(f"Processing {outcar_path}...")
        try:
            # 加载 OUTCAR 数据
            data = dpdata.LabeledSystem(outcar_path, fmt='vasp/outcar')

            # 提取子集（每十步取一个结构）
            subset_data = data.sub_system(range(0, len(data), step_interval))

            # 保存为 DeepMD-kit 格式（或其他格式）
            output_path = os.path.join(output_folder, f"{os.path.basename(folder)}_subset")
            subset_data.to('deepmd/npy', output_path)

            print(f"Processed {outcar_path} successfully! Data saved in {output_path}")
        except Exception as e:
            print(f"Error processing {outcar_path}: {e}")
    else:
        print(f"OUTCAR not found in {folder}")

print("All processing done!")

