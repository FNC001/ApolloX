import pandas as pd
from sklearn.model_selection import train_test_split


file_path = './data/with_cif.feather'  
data = pd.read_feather(file_path)  

train, temp = train_test_split(data, test_size=0.2, random_state=42)

val, test = train_test_split(temp, test_size=0.5, random_state=42)

train.to_feather('./data/train.feather')
val.to_feather('./data/val.feather')
test.to_feather('./data/test.feather')