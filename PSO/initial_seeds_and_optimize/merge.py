# Merge the files “all_structures_summary.csv” and “sorted_energies.csv” of the structures after optimization of each generation.
import pandas as pd

# Read all_structures_summary.csv and sorted_energies.csv
all_structures_df = pd.read_csv('all_structures_summary.csv')
sorted_energies_df = pd.read_csv('sorted_energies.csv')

# Add the '.optdone.vasp' suffix to the 'name' column in sorted_energies.
sorted_energies_df['name_with_extension'] = sorted_energies_df['name'] + '.optdone.vasp'

# Match by 'material_id' and 'name_with_extension'.
merged_df = pd.merge(all_structures_df, sorted_energies_df[['name_with_extension', 'energy']], 
                     left_on='material_id', right_on='name_with_extension', how='left')

# Rename the 'energy' column to 'Energy'.
merged_df = merged_df.rename(columns={'energy': 'Energy'})

# Delete the temporary 'name_with_extension' column.
merged_df = merged_df.drop(columns=['name_with_extension'])

# Save the merged DataFrame to a new CSV file.
merged_df.to_csv('merged_all_structures_with_energy.csv', index=False)

print("Merge complete, the result has been saved to 'merged_all_structures_with_energy.csv'")

