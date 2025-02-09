#!/bin/bash
mkdir -p poscar
cp POSCAR poscar
cp bulk.py poscar
cp graph.pb poscar
cp run1.py poscar
cp merge1.py poscar

# 默认值
gen_num=15
structure_num_per_gen=100

# 解析参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gen_num) gen_num="$2"; shift ;;  
        --structure_num_per_gen) structure_num_per_gen="$2"; shift ;;  
        *) echo "Unknown parameter: $1" ; exit 1 ;;
    esac
    shift
done

# 计算 num_files
num_files=$((gen_num * structure_num_per_gen))

# 运行 Python 脚本
python rand.py ./poscar/POSCAR "$num_files"
cd poscar
rm POSCAR
python run1.py
python bulk.py
python merge1.py

# **确保 updated_all_structures_summary_batch_1.csv 为空**
> updated_all_structures_summary_batch_1.csv  # 清空文件，防止列名重复

# **复制列名**
head -n 1 Merged_all_structures_with_energy.csv > updated_all_structures_summary_batch_1.csv

# **从 merged_all_structures_with_energy.csv 获取前 structure_num_per_gen 行数据（不包括列名）**
tail -n +2 Merged_all_structures_with_energy.csv | head -n "$structure_num_per_gen" >> updated_all_structures_summary_batch_1.csv

# **确保 merged_all_structures_with_energy.csv 只删除数据行，不影响列名**
head -n 1 Merged_all_structures_with_energy.csv > temp_header.csv  # 备份列名
tail -n +2 Merged_all_structures_with_energy.csv | tail -n +"$((structure_num_per_gen + 1))" > temp_data.csv  # 删除前 structure_num_per_gen 行数据
cat temp_header.csv temp_data.csv > Merged_all_structures_with_energy.csv  # 重新组合文件

# **删除临时文件**
rm temp_header.csv temp_data.csv
mv updated_all_structures_summary_batch_1.csv ..
mv Merged_all_structures_with_energy.csv ..
echo "CSV 文件更新完成: 'updated_all_structures_summary_batch_1.csv' 已创建，'merged_all_structures_with_energy.csv' 已清理"

