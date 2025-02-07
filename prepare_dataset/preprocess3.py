import pandas as pd
file=['with_ciftrain.csv','with_ciftest.csv','with_cifval.csv']
n=['train','test','val']
for i in range(3):
    # 读取CSV文件
    data = pd.read_csv(file[i])

    # 选择特定的列
    newdata = data[["material_id", "cif_file", "element_values"]].copy()

    # 修正列名错误
    newdata.rename(columns={'cif_file': 'cif'}, inplace=True)

    # 添加一个新的列
    newdata.loc[:, "pressure"] = 0

    # 将数据保存为Feather格式的文件，这是一种二进制文件格式
    newdata.to_feather(f"./{n[i]}.feather")