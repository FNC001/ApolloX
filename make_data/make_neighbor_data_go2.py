import pandas as pd
import json
import os
from tqdm import tqdm  # 引入进度条库

# 加载数据
file_path = './data/processed_material_data.csv'
data = pd.read_csv(file_path).head(10000)  # 限制为前10行进行测试

# 读取CIF文件内容的函数
def read_cif_content(file_name):
    # 指定CIF文件的存储目录
    cif_directory = './cif'  # 确保这个路径是正确的
    file_name_with_suffix = os.path.join(cif_directory, file_name + '.optdone.cif')
    try:
        with open(file_name_with_suffix, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return 'File not found'

# 更新 'cif_file' 列为CIF文件的内容
data['cif_file'] = data.apply(lambda row: read_cif_content(row['material_id']), axis=1)

# 自动检测元素互动数据的列（从第三列开始）
interaction_columns = data.columns[2:]

# 将元素互动数据转换成仅包含数值的列表
data['element_values'] = [list(row[interaction_columns].values) for index, row in tqdm(data.iterrows(), total=data.shape[0])]

# 保存修改后的DataFrame到一个新的CSV文件
data.to_csv('./data/with_cif.csv', index=False)