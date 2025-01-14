import pandas as pd
import os
import argparse

# 设置命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="为每一行数据生成 .sh 文件")
    parser.add_argument('--g', type=int, required=True, help="传入的参数 g，用于读取数据文件")
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    g = args.g  # 获取传入的 g 参数

    # 读取数据
    data = pd.read_csv(f'standardized_optimized_particles_distribution{g}.csv')

    # 创建文件夹（如果不存在的话）
    output_dir = f'sh_files_{g}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 为每一行数据生成一个 .sh 文件
    for idx, row in data.iterrows():
        # 提取 material_id 和 element_values
        material_id = row['material_id']
        element_values = row.iloc[1:-1]  # 第二列到倒数第二列

        # 将 element_values 转换为字符串，使用逗号分割
        element_values_str = ",".join(map(str, element_values))

        # 生成 .sh 文件的内容
        sh_content = "#!/bin/bash\nCUDA_VISIBLE_DEVICES=0\n"
        sh_content += "python ~/cond-cdvae-main/scripts/evaluate.py --model_path `pwd` --tasks gen \\\n"
        sh_content += "    --formula=B12Mo12Co12Fe12Ni12O60 \\\n"
        sh_content += f"    --pressure=0 \\\n"
        sh_content += f"    --label=\"{material_id}\" \\\n"
        sh_content += f"    --element_values=\"{element_values_str}\" \\\n"
        sh_content += "    --batch_size=1 \\\n"
        sh_content += "    --num_batches_to_samples=1"

        # 保存 .sh 文件
        sh_file_path = os.path.join(output_dir, f'run_evaluation_{idx}.sh')
        with open(sh_file_path, 'w') as f:
            f.write(sh_content)

        print(f"Shell script saved to {sh_file_path}")

    print(f"All shell scripts saved to {output_dir}")
