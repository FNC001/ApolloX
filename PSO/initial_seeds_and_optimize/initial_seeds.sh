#!/bin/bash
mkdir -p poscar
cp POSCAR poscar
cp bulk.py poscar
cp graph.pb poscar
cp run1.py poscar
cp merge1.py poscar

# Default value.
gen_num=15
structure_num_per_gen=100

# Parse parameters
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gen_num) gen_num="$2"; shift ;;  
        --structure_num_per_gen) structure_num_per_gen="$2"; shift ;;  
        *) echo "Unknown parameter: $1" ; exit 1 ;;
    esac
    shift
done

# Calculate the number of required strucutres "num_files"
num_files=$((gen_num * structure_num_per_gen))

# Run Python scripts
python rand.py ./poscar/POSCAR "$num_files"
cd poscar
rm POSCAR
python run1.py
python bulk.py
python merge1.py

# **Ensure updated_all_structures_summary_batch_1.csv is empty**
> updated_all_structures_summary_batch_1.csv 

# **Copy the column names.**
head -n 1 Merged_all_structures_with_energy.csv > updated_all_structures_summary_batch_1.csv

# **Retrieve the first "structure_num_per_gen rows" of data (excluding the column names) from "merged_all_structures_with_energy.csv".**
tail -n +2 Merged_all_structures_with_energy.csv | head -n "$structure_num_per_gen" >> updated_all_structures_summary_batch_1.csv

# **Ensure that only the data rows are deleted from "merged_all_structures_with_energy.csv", without affecting the column names.**
head -n 1 Merged_all_structures_with_energy.csv > temp_header.csv  # Backup the column names.
tail -n +2 Merged_all_structures_with_energy.csv | tail -n +"$((structure_num_per_gen + 1))" > temp_data.csv  # Delete the first "structure_num_per_gen" rows of data.
cat temp_header.csv temp_data.csv > Merged_all_structures_with_energy.csv  # Recombine the file.

# **Delete the temporary files.**
rm temp_header.csv temp_data.csv
mv updated_all_structures_summary_batch_1.csv ..
mv Merged_all_structures_with_energy.csv ..
echo ""CSV file update complete: 'updated_all_structures_summary_batch_1.csv' has been created."

