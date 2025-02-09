import pandas as pd

# 读取 all_structures_summary.csv 和 sorted_energies.csv
all_structures_df = pd.read_csv('all_structures_summary.csv')
sorted_energies_df = pd.read_csv('sorted_energies.csv')

# 给 sorted_energies 中的 name 列加上 '.optdone.vasp' 后缀
sorted_energies_df['name_with_extension'] = sorted_energies_df['name'] + '.optdone.vasp'

# 通过 'material_id' 和 'name_with_extension' 进行匹配
merged_df = pd.merge(all_structures_df, sorted_energies_df[['name_with_extension', 'energy']], 
                     left_on='material_id', right_on='name_with_extension', how='left')

# 将 energy 列重命名为 Energy
merged_df = merged_df.rename(columns={'energy': 'Energy'})

# 删除临时的 'name_with_extension' 列
merged_df = merged_df.drop(columns=['name_with_extension'])

# 保存合并后的 DataFrame 到新的 CSV 文件
merged_df.to_csv('Merged_all_structures_with_energy.csv', index=False)

print("合并完成，结果已保存到 'merged_all_structures_with_energy.csv'")

