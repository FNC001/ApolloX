import pandas as pd
from sklearn.preprocessing import StandardScaler

# 读取 Excel 文件，指定引擎
df = pd.read_csv("D:\\pycharm\\atat\\poscar\\all_structures_summary.csv")  # 或使用 engine='xlrd'

# 分离出前两列
non_data_cols = df.iloc[:, :2]  # 前两列
data_cols = df.iloc[:, 2:]  # 数据部分
# 打乱数据（保持前两列的索引）
data_with_index = pd.concat([non_data_cols, data_cols], axis=1)
data_with_index = data_with_index.sample(frac=1, random_state=42).reset_index(drop=True)

# 重新分离前两列和数据部分
non_data_cols = data_with_index.iloc[:, :2]  # 前两列
data_cols = data_with_index.iloc[:, 2:]  # 数据部分

# 计算数据集的大小
total_rows = len(data_cols)
train_size = int(0.8 * total_rows)
test_size = int(0.1 * total_rows)

# 划分数据集
train_data = data_cols[:train_size]
test_data = data_cols[train_size:train_size + test_size]
val_data = data_cols[train_size + test_size:]

# 初始化 StandardScaler
scaler = StandardScaler()
# 对训练集进行标准化
train_scaled = scaler.fit_transform(train_data)
#获取平均值与标准差
train_mean = scaler.mean_
train_std = scaler.scale_
# 利用训练集的均值和标准差对验证集和测试集进行标准化
test_scaled = scaler.transform(test_data)
val_scaled = scaler.transform(val_data)

# 将标准化后的数据转换为 DataFrame
train_scaled_df = pd.DataFrame(train_scaled, columns=train_data.columns)
test_scaled_df = pd.DataFrame(test_scaled, columns=test_data.columns)
val_scaled_df = pd.DataFrame(val_scaled, columns=val_data.columns)

# 将前两列与标准化后的数据合并
train_final_df = pd.concat([non_data_cols.iloc[:train_size].reset_index(drop=True), train_scaled_df], axis=1)
test_final_df = pd.concat([non_data_cols.iloc[train_size:train_size + test_size].reset_index(drop=True), test_scaled_df], axis=1)
val_final_df = pd.concat([non_data_cols.iloc[train_size + test_size:].reset_index(drop=True), val_scaled_df], axis=1)

# 保存最终的数据集为不同的 Excel 文件
train_final_df.to_csv('train_set_scaled.csv', index=False)
test_final_df.to_csv('test_set_scaled.csv', index=False)
val_final_df.to_csv('val_set_scaled.csv', index=False)

print("数据集已成功标准化并保存为三个 csv 文件！")
with open("scaler_stats.txt", "w") as file:
    file.write(f"Train Mean: {train_mean}\n")
    file.write(f"Train Std: {train_std}\n")
print(train_mean)
print(train_std)