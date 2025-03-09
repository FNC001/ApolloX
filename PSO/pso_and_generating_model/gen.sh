#!/bin/bash

# Set the number of generation "g"
g=1  

# Run pso_s.py, and pass the g parameter
echo "Running pso_s.py with g=$g..."
python pso_s.py --g $g

# Run standadize_s.py
echo "Running standadize_s.py with g=$g..."
python standadize_s.py --g $g

# Run make_sh_s.py
echo "Running make_sh_s.py with g=$g..."
python make_sh_s.py --g $g

sh_files_dir="./sh_files_$g"
if [ ! -d "$sh_files_dir" ]; then
    mkdir -p "$sh_files_dir"
fi

# Iterate through all `.sh` files in the `sh_files` directory.
for sh_file in "$sh_files_dir"/run_evaluation_*.sh; do
    #Output the currently executing .sh file.
    echo "Executing $sh_file..."
    
    #Grant execution permissions to the .sh file.
    chmod +x "$sh_file"
    
    #Run the "sh" file
    ./"$sh_file"
    
    # Obtain the returned .pt file.
    pt_file=$(ls -t *.pt | head -n 1) 

    # Check if the .pt file exists.
    if [[ -f "$pt_file" ]]; then
        echo "Processing .pt file: $pt_file"

        # Run extract_gen.py
        python ~/cond-cdvae-main/scripts/extract_gen.py "$pt_file"
        
        # Obtain the folder path after executing extract_gen.py.
        result_folder="${pt_file%.*}" 
        
        # Check if the returned folder exists.
        if [[ -d "$result_folder/gen" ]]; then
            echo "Entering gen folder in $result_folder..."
            
            # Enter the 'gen' folder inside the returned folder.
            cd "$result_folder/gen"
            
            # Run bulk.py（bulk.py is in the current folder, which is the 'gen' folder.）
            echo "Executing bulk.py in gen folder..."
            python ../../bulk.py  
            
            # Return to the original directory.
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
