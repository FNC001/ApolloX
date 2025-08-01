import yaml
import os
import sys
from pathlib import Path

def expand_path(path):
    return os.path.expanduser(path)

# === 自动获取 ApolloX 路径 ===
def find_apollox_path(current_path):
    """从当前路径向上搜索 ApolloX 根目录"""
    current = current_path.resolve()
    while current != current.parent:
        if (current / "generate_structure").exists() and (current / "prepare_dataset").exists():
            return current
        current = current.parent
    raise RuntimeError("ApolloX root directory not found")

# === 读取配置 ===
config_path = Path(__file__).resolve().parent / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# === 查找 ApolloX 根目录 ===
apollox_path = find_apollox_path(config_path.parent)

# === 校验 generation_type ===
valid_types = {"single", "variable"}
generation_type = config.get("generation_type", "").strip().lower()

if generation_type not in valid_types:
    print(f"[Error] generation_type must be one of: {valid_types}, but got '{generation_type}'")
    sys.exit(1)

# === 提取参数 ===
initial = config["initial_structure"]
num = config["num"]
dataset_path = expand_path(config["dataset_path"])
cutoff = config["cutoff"]
n_jobs = config["n_jobs"]
mode = config["mode"]
train_ratio = config["train_ratio"]
test_ratio = config["test_ratio"]
val_ratio = config["val_ratio"]

# === 组装命令 ===
commands = []

# 结构生成阶段
if generation_type == "single":
    commands.append(
        f"python {apollox_path}/generate_structure/bulk/generate_single_component.py "
        f"--input {apollox_path}/original_structures/{initial} --num {num} --outdir {dataset_path}"
    )
elif generation_type == "variable":
    commands.append(
        f"python {apollox_path}/generate_structure/bulk/generate_variable_component.py "
        f"--input {apollox_path}/original_structures/{initial} --num {num} --outdir {dataset_path}"
    )

# 后续处理流程
commands += [
    f"python {apollox_path}/prepare_dataset/compute_pdm.py "
    f"--input_dir {dataset_path}/poscar/ --cutoff {cutoff} --n_jobs {n_jobs} --mode {mode} "
    f"--starts_with POSCAR --output_csv {dataset_path}/all_structures_summary.csv",

    f"python {apollox_path}/prepare_dataset/transfer_POSCAR_to_cif.py {dataset_path} --n {n_jobs}",

    f"python {apollox_path}/prepare_dataset/split_dataset.py --input {dataset_path}/all_structures_summary.csv "
    f"--output_dir {dataset_path} --train_ratio {train_ratio} "
    f"--test_ratio {test_ratio} --val_ratio {val_ratio}",

    f"python {apollox_path}/prepare_dataset/append_cif.py --input_csvs {dataset_path}/train_set_scaled.csv "
    f"{dataset_path}/test_set_scaled.csv {dataset_path}/val_set_scaled.csv "
    f"--output_names train test val --cif_dir {dataset_path}/cif --output_dir {dataset_path}",

    f"python {apollox_path}/prepare_dataset/convert_CVS.py --input_dir {dataset_path}"
]

# === 执行命令 ===
for cmd in commands:
    print(f"\n[Running] {cmd}\n")
    os.system(cmd)

