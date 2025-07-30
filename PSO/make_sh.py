import pandas as pd
import os
import argparse
from pathlib import Path

def find_apollox_path():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.name == "ApolloX":
            return str(parent)
    raise FileNotFoundError("ApolloX directory not found in parent paths.")

# 提取组分字符串（如 B12Mo12Co12Fe12Ni12O60）
def get_composition_from_poscar(g):
    poscar_dir = f'./poscars/generation{g}/'
    for file in os.listdir(poscar_dir):
        if file.startswith('POSCAR'):
            with open(os.path.join(poscar_dir, file), 'r') as f:
                lines = f.readlines()
                if len(lines) > 6:
                    elements = lines[5].strip().split()
                    counts = lines[6].strip().split()
                    if len(elements) == len(counts):
                        return ''.join(f'{el}{ct}' for el, ct in zip(elements, counts))
    raise FileNotFoundError(f"No valid POSCAR file found in {poscar_dir}")

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Generate a .sh file for each line of data.")
    parser.add_argument('--g', type=int, required=True, help="The input parameter g, used to read the data file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    g = args.g  
    apollox_path = find_apollox_path()
    # 获取组分信息
    formula = get_composition_from_poscar(g)

    # 读取结构数据
    data = pd.read_csv(f'./temp/standardized_optimized_particles_distribution{g}.csv')

    # 创建输出目录
    output_dir = os.path.join("temp", f"sh_files_{g}")
    os.makedirs(output_dir, exist_ok=True)

    # 为每一行生成 .sh 脚本
    for idx, row in data.iterrows():
        material_id = row['material_id']
        element_values = row.iloc[2:-1]  # 忽略 material_id 和最后一列（可能是 energy）
        element_values_str = ",".join(map(str, element_values))

        # 构建脚本内容
        sh_content = "#!/bin/bash\nCUDA_VISIBLE_DEVICES=0\n"
        sh_content += f"python {apollox_path}/cond-cdvae/scripts/evaluate.py --model_path `pwd` --tasks gen \\\n"
        sh_content += f"    --formula={formula} \\\n"
        sh_content += f"    --pressure=0 \\\n"
        sh_content += f"    --label=\"{material_id}\" \\\n"
        sh_content += f"    --element_values=\"{element_values_str}\" \\\n"
        sh_content += "    --batch_size=1 \\\n"
        sh_content += "    --num_batches_to_samples=1"

        # 写入脚本文件
        sh_path = os.path.join(output_dir, f'run_evaluation_{idx}.sh')
        with open(sh_path, 'w') as f:
            f.write(sh_content)

        print(f"Shell script saved to {sh_path}")

    print(f"All shell scripts saved to {output_dir}")

