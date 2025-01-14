#!/bin/bash
g=1
output_dir="vasp_files_$g"
cp graph.pb "$output_dir"
cd "$output_dir"
python ../run.py
cd "$output_dir"
python ../bulk.py
python ../merge.py
cd ..
# 计算已完成的文件数
optdone_count=$(ls vasp_files_$g/*.optdone.vasp 2>/dev/null | wc -l)

# 计算需要处理的行数
n=$((100 - optdone_count))

# 如果有需要处理的行，则追加到目标文件
if [[ $n -gt 0 ]]; then
    head -n $((n + 1)) split_csv_files/updated_all_structures_summary_batch_$(($g+1)).csv | tail -n $n >> vasp_files_$g/merged_all_structures_with_energy.csv
    echo "成功追加 $n 行到 merged_all_structures_with_energy.csv"
else
    echo "没有足够的行可以追加，n = $n"
fi

# 重命名文件
mv vasp_files_$g/merged_all_structures_with_energy.csv vasp_files_$g/updated_all_structures_summary_batch_$(($g+1)).csv
echo "文件已重命名为 updated_all_structures_summary_batch_$(($g+1)).csv"

# 移动文件到上一级目录
mv vasp_files_$g/updated_all_structures_summary_batch_$(($g+1)).csv .
echo "文件已移动到上一级目录"

