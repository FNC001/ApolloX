import os
import pandas as pd
import yaml
from pathlib import Path
import subprocess

def run_cmd(cmd):
    """执行 shell 命令并打印输出"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("ERROR:", result.stderr)

# 读取 config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

apollox_path = config["apollox_path"]
batch_size = config["batch_size"]
num_batches = config["num_batches_to_samples"]
csv_path = config["csv_path"]

# 读取 CSV
df = pd.read_csv(csv_path, sep="\t")

# 创建输出文件夹
sh_dir = Path("sh_files_multi_pdm")
sh_dir.mkdir(exist_ok=True)

# 遍历每一行
for idx, row in df.iterrows():
    material_id = row["material_id"]
    formula = material_id  # 如有公式可替换

    # 拼接 element_values
    element_cols = df.columns[2:]  # 去掉 material_id, cif_file
    element_values = [f"{col}:{row[col]}" for col in element_cols]
    element_values_str = ",".join(element_values)

    # 生成 sh 文件内容
    sh_content = "#!/bin/bash\n"
    sh_content += "CUDA_VISIBLE_DEVICES=0\n"
    sh_content += (
        f"python {apollox_path}/cond-cdvae/scripts/evaluate.py --model_path `pwd` --tasks gen \\\n"
    )
    sh_content += f"    --formula={formula} \\\n"
    sh_content += "    --pressure=0 \\\n"
    sh_content += f"    --label=\"{material_id}\" \\\n"
    sh_content += f"    --element_values=\"{element_values_str}\" \\\n"
    sh_content += f"    --batch_size={batch_size} \\\n"
    sh_content += f"    --num_batches_to_samples={num_batches}\n"

    # 写入文件
    sh_file = sh_dir / f"{material_id}.sh"
    with open(sh_file, "w") as f:
        f.write(sh_content)

    # 改执行权限
    os.chmod(sh_file, 0o755)

    # 执行脚本
    run_cmd(str(sh_file))

print(f"所有脚本已生成在 {sh_dir} 并已执行。")

