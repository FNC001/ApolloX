import pandas as pd

def process_data(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Create the 'material_id' column by extracting the base file name without extension
    # Now considering the new file naming convention that includes an ID
    data['material_id'] = data['POSCAR File'].str.extract(r'(^POSCAR-[A-Za-z0-9_]+)-\d+')[0]

    # Create the 'cif_file' column
    data['cif_file'] = data['material_id'] + '.cif'

    # Create a pivot table to reshape the data
    pivot_data = data.pivot_table(index=['material_id', 'cif_file'],
                                  columns=['Element', 'Neighbor Element'],
                                  values='Average Neighbors',
                                  aggfunc='mean').reset_index()

    # Flatten the multi-level column names for element pairs
    pivot_data.columns = [''.join(col).strip() for col in pivot_data.columns.values]

    # Fill NaN values with 0 as there might be missing element pairs with no neighbors
    pivot_data.fillna(0, inplace=True)

    # Define the output path
    output_path = './data/processed_material_data.csv'
    pivot_data.to_csv(output_path, index=False)

    return output_path

# Replace 'your_file_path.csv' with the path to your CSV file
processed_file = process_data('./data/all_calculations_summary.csv')
print(f'Processed data saved to: {processed_file}')
