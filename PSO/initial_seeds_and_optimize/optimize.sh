#!/bin/bash

# Default value
structure_num_per_gen=100

# Parse command-line arguments.
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --structure_num_per_gen) structure_num_per_gen="$2"; shift ;;  
        *) echo "Unknown parameter: $1" ; exit 1 ;;
    esac
    shift
done
# Change the value "g" to the contemporary number of generation.
g=1
output_dir="vasp_files_$g"
mkdir -p "$output_dir"

# Copy the necessary files to the output directory.
cp graph.pb "$output_dir"
cd "$output_dir"

# Run Python scripts
python ../run.py
python ../bulk.py
python ../merge.py
cd ..

# Calculate the number of completed files.
optdone_count=$(ls vasp_files_$g/*.optdone.vasp 2>/dev/null | wc -l)

# Calculate the number of rows to be processed (using the passed structure count).
n=$((structure_num_per_gen - optdone_count))

# If there are rows to be processed, append them to the target file.
if [[ $n -gt 0 ]]; then
    # Check if the target file already contains content; if not, write the column names.
    if [[ ! -f vasp_files_$g/merged_all_structures_with_energy.csv ]]; then
        head -n 1 Merged_all_structures_with_energy.csv > vasp_files_$g/merged_all_structures_with_energy.csv
    fi

    # Append the first $n rows (starting from the second row) of “Merged_all_structures_with_energy.csv" to the target file.
    tail -n +2 Merged_all_structures_with_energy.csv | head -n "$n" >> vasp_files_$g/merged_all_structures_with_energy.csv
    echo "Successfully appended $n rows to merged_all_structures_with_energy.csv"
else
    echo "There are not enough rows to append，n = $n"
fi

# Delete the rows that have been appended to the target file, keeping the column names.
# First, backup the column names.
head -n 1 Merged_all_structures_with_energy.csv > temp_header.csv

# Delete the first $n rows of data.
tail -n +2 Merged_all_structures_with_energy.csv | tail -n +"$((n + 1))" > temp_data.csv

# Merge the column names and the remaining data.
cat temp_header.csv temp_data.csv > Merged_all_structures_with_energy.csv

# Clean up temporary files.
rm temp_header.csv temp_data.csv

# Rename the file.
mv vasp_files_$g/merged_all_structures_with_energy.csv vasp_files_$g/updated_all_structures_summary_batch_$(($g+1)).csv
echo "The file has been renamed to updated_all_structures_summary_batch_$(($g+1)).csv"

# Move the file to the parent directory.
mv vasp_files_$g/updated_all_structures_summary_batch_$(($g+1)).csv . 
echo "The file has been moved to the parent directory."

