import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据
file_path = './data/with_cif.csv'  # 更新为完整的输入文件路径
data = pd.read_csv(file_path)

# 选择第一列、第二列和最后一列
data = data.iloc[:, [0, 1, -1]]

# 将数据分割为训练集（80%）和临时数据集（20%）
train, temp = train_test_split(data, test_size=0.2, random_state=42)

# 将临时数据集进一步分割为验证集和测试集（各占临时数据集的50%，因此各占总数据的10%）
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# 指定保存路径并保存数据集到CSV文件
train.to_csv('./data/train.csv', index=False)
val.to_csv('./data/val.csv', index=False)
test.to_csv('./data/test.csv', index=False)
