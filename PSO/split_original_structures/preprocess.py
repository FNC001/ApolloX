import pandas as pd
import numpy as np
import os

# 读取CSV文件
df = pd.read_csv('merged_all_structures_with_energy.csv')

# 每40行分割一次
batch_size = 50
num_batches = int(np.ceil(len(df) / batch_size))

# 创建保存CSV文件的文件夹
output_folder = 'split_csv_files'
os.makedirs(output_folder, exist_ok=True)

# 根据每40行分割成多个文件
for i in range(num_batches):
    # 计算当前分割的行范围
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(df))

    # 获取当前批次的数据
    batch_df = df.iloc[start_idx:end_idx]

    # 保存当前批次为CSV文件，文件名以批次编号命名
    batch_filename = os.path.join(output_folder, f'updated_all_structures_summary_batch_{i + 1}.csv')
    batch_df.to_csv(batch_filename, index=False)

print(f"分割完成，多个CSV文件已保存到文件夹：{output_folder}")
