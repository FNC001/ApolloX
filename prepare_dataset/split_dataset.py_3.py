import pandas as pd
file=['with_ciftrain.csv','with_ciftest.csv','with_cifval.csv']
n=['train','test','val']
for i in range(3):
    # Read the csv file
    data = pd.read_csv(file[i])

    # Choose the specific column.
    newdata = data[["material_id", "cif_file", "element_values"]].copy()

    # Correct the possible mistake of the column name
    newdata.rename(columns={'cif_file': 'cif'}, inplace=True)

    # Add a new column
    newdata.loc[:, "pressure"] = 0

    # Save the data as a Feather format file, which is a binary file format.
    newdata.to_feather(f"./{n[i]}.feather")
