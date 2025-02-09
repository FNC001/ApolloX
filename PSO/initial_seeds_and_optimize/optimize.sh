#!/bin/bash

# 默认值
structure_num_per_gen=100

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --structure_num_per_gen) structure_num_per_gen="$2"; shift ;;  
        *) echo "Unknown parameter: $1" ; exit 1 ;;
    esac
    shift
done

# 创建输出目录
g=1
output_dir="vasp_files_$g"
mkdir -p "$output_dir"

# 复制必要的文件到输出目录
cp graph.pb "$output_dir"
cd "$output_dir"

# 执行 Python 脚本
python ../run.py
python ../bulk.py
python ../merge.py
cd ..

# 计算已完成的文件数
optdone_count=$(ls vasp_files_$g/*.optdone.vasp 2>/dev/null | wc -l)

# 计算需要处理的行数（使用传入的结构数）
n=$((structure_num_per_gen - optdone_count))

# 如果有需要处理的行，则追加到目标文件
if [[ $n -gt 0 ]]; then
    # 检查目标文件是否已经有内容，如果没有则写入列名
    if [[ ! -f vasp_files_$g/merged_all_structures_with_energy.csv ]]; then
        head -n 1 Merged_all_structures_with_energy.csv > vasp_files_$g/merged_all_structures_with_energy.csv
    fi

    # 将 Merged_all_structures_with_energy.csv 中的前 $n 行（从第二行开始）追加到目标文件
    tail -n +2 Merged_all_structures_with_energy.csv | head -n "$n" >> vasp_files_$g/merged_all_structures_with_energy.csv
    echo "成功追加 $n 行到 merged_all_structures_with_energy.csv"
else
    echo "没有足够的行可以追加，n = $n"
fi

# 删除已追加到目标文件的行，保留列名
# 先备份列名
head -n 1 Merged_all_structures_with_energy.csv > temp_header.csv

# 删除前 $n 行数据
tail -n +2 Merged_all_structures_with_energy.csv | tail -n +"$((n + 1))" > temp_data.csv

# 合并列名和剩余数据
cat temp_header.csv temp_data.csv > Merged_all_structures_with_energy.csv

# 清理临时文件
rm temp_header.csv temp_data.csv

# 重命名文件
mv vasp_files_$g/merged_all_structures_with_energy.csv vasp_files_$g/updated_all_structures_summary_batch_$(($g+1)).csv
echo "文件已重命名为 updated_all_structures_summary_batch_$(($g+1)).csv"

# 移动文件到上一级目录
mv vasp_files_$g/updated_all_structures_summary_batch_$(($g+1)).csv . 
echo "文件已移动到上一级目录"

