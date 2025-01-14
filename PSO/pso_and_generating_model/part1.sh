#!/bin/bash

# 设置 g 的值
g=1  # 你可以根据需要修改 g 的值

# 执行 pso_s.py，传递 g 参数
echo "Running pso_s.py with g=$g..."
python pso_s.py --g $g

# 执行 standadize_s.py，传递 g 参数
echo "Running standadize_s.py with g=$g..."
python standadize_s.py --g $g

# 执行 make_sh_s.py，传递 g 参数
echo "Running make_sh_s.py with g=$g..."
python make_sh_s.py --g $g

# 确保sh_files文件夹存在
sh_files_dir="./sh_files"
if [ ! -d "$sh_files_dir" ]; then
    mkdir -p "$sh_files_dir"
fi

# 遍历sh_files目录中的所有.sh文件
for sh_file in "$sh_files_dir"/run_evaluation_*.sh; do
    # 输出正在执行的.sh文件
    echo "Executing $sh_file..."
    
    # 赋予.sh文件执行权限
    chmod +x "$sh_file"
    
    # 执行.sh文件
    ./"$sh_file"
    
    # 获取返回的.pt文件（假设返回的文件有一定的规律，可以修改为实际规律）
    pt_file=$(ls -t *.pt | head -n 1)  # 获取最新生成的.pt文件，如果多个文件，取最新的

    # 检查.pt文件是否存在
    if [[ -f "$pt_file" ]]; then
        echo "Processing .pt file: $pt_file"

        # 执行 extract_gen.py
        python ~/cond-cdvae-main/scripts/extract_gen.py "$pt_file"
        
        # 获取 extract_gen.py 执行后的文件夹路径
        result_folder="${pt_file%.*}"  # 假设返回的文件夹与.pt文件名相同，去除扩展名
        
        # 检查返回的文件夹是否存在
        if [[ -d "$result_folder/gen" ]]; then
            echo "Entering gen folder in $result_folder..."
            
            # 进入返回的文件夹中的gen文件夹
            cd "$result_folder/gen"
            
            # 执行bulk.py（bulk.py在当前文件夹，即gen文件夹内）
            echo "Executing bulk.py in gen folder..."
            python ../../bulk.py  # 因为我们已经进入gen目录，所以使用相对路径执行
            
            # 返回到原来的目录
            cd - > /dev/null
        else
            echo "Error: gen folder does not exist in $result_folder."
        fi
    else
        echo "Error: $pt_file does not exist."
    fi
done
output_vasp_dir="./vasp_files_${g}"
if [ ! -d "$output_vasp_dir" ]; then
    mkdir -p "$output_vasp_dir"
fi


echo "Renaming and moving .vasp files..."
for dir in eval_gen_POSCAR-B12Mo12Co12Fe12Ni12O60-${g}_*; do
    
    if [[ "$dir" == *.pt ]]; then
        continue
    fi

    folder_g_value=$(echo $dir | grep -oP "${g}_\K[0-9]+")

   
    if [ -d "$dir/gen" ]; then
       
        if [ -f "$dir/gen/0.vasp" ]; then
            
            new_name="${g}_${folder_g_value}.vasp"
            mv "$dir/gen/0.vasp" "$output_vasp_dir/$new_name"
            echo "Moved and renamed $dir/gen/0.vasp to $output_vasp_dir/$new_name"
        else
            echo "Error: 0.vasp not found in $dir/gen"
        fi
    else
        echo "Error: $dir/gen folder does not exist"
    fi
done

echo "All tasks completed."
