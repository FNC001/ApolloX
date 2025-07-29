import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./all_structures_summary.csv")
non_data_cols = df.iloc[:, :2] 
data_cols = df.iloc[:, 2:] 
# Shuffle the data (keeping the indexes of the first two columns).
data_with_index = pd.concat([non_data_cols, data_cols], axis=1)
data_with_index = data_with_index.sample(frac=1, random_state=42).reset_index(drop=True)

# Re-separate the first two columns and the data part.
non_data_cols = data_with_index.iloc[:, :2]  # the first two columns
data_cols = data_with_index.iloc[:, 2:]  # the data part

# Calculate the size of dataset
total_rows = len(data_cols)
train_size = int(0.8 * total_rows)
test_size = int(0.1 * total_rows)

# Split the dataset.
train_data = data_cols[:train_size]
test_data = data_cols[train_size:train_size + test_size]
val_data = data_cols[train_size + test_size:]

# Initialize StandardScaler
scaler = StandardScaler()
# Standardize the dataset
train_scaled = scaler.fit_transform(train_data)
# Acquire mean values and standard deviation
train_mean = scaler.mean_
train_std = scaler.scale_
# Normalize the validation and test sets using the mean and standard deviation of the training set.
test_scaled = scaler.transform(test_data)
val_scaled = scaler.transform(val_data)

# Convert the standardized data into a DataFrame.
train_scaled_df = pd.DataFrame(train_scaled, columns=train_data.columns)
test_scaled_df = pd.DataFrame(test_scaled, columns=test_data.columns)
val_scaled_df = pd.DataFrame(val_scaled, columns=val_data.columns)

# Merge the first two columns with the standardized data.
train_final_df = pd.concat([non_data_cols.iloc[:train_size].reset_index(drop=True), train_scaled_df], axis=1)
test_final_df = pd.concat([non_data_cols.iloc[train_size:train_size + test_size].reset_index(drop=True), test_scaled_df], axis=1)
val_final_df = pd.concat([non_data_cols.iloc[train_size + test_size:].reset_index(drop=True), val_scaled_df], axis=1)

train_final_df.to_csv('train_set_scaled.csv', index=False)
test_final_df.to_csv('test_set_scaled.csv', index=False)
val_final_df.to_csv('val_set_scaled.csv', index=False)

print("The dataset has been successfully standardized and saved as three CSV files！")
with open("scaler_stats.txt", "w") as file:
    file.write(f"Train Mean: {train_mean}\n")
    file.write(f"Train Std: {train_std}\n")
print(train_mean)
print(train_std)
