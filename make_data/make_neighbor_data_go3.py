import pandas as pd

data = pd.read_csv('./data/with_cif.csv')
newdata = data[["material_id", "cif_file", "element_values"]]
newdata.rename(columns={'cif_file': 'cif'}, inplace=True)
newdata["pressure"] = 0
newdata.to_feather("./data/with_cif.feather")