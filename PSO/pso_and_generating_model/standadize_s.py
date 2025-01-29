import pandas as pd
import numpy as np
import argparse

# 设置命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="标准化优化后的粒子分布数据")
    parser.add_argument('--g', type=int, required=True, help="传入的参数 g，用于读取数据文件")
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    g = args.g  # 获取传入的g参数

    # 定义给定的均值和标准差
    means = np.array([14.433375, 31.44525, 31.495625, 31.428375, 31.467125, 157.296875,
                      14.4625, 31.52275, 31.416, 31.4295, 157.2615, 14.389,
                      31.43425, 31.4445, 157.324875, 14.443875, 31.446875, 157.38675,
                      14.41325, 157.3855, 386.67225])

    std_devs = np.array([3.04619617, 4.50080575, 4.50263599, 4.51435154, 4.55583903, 7.87113335,
                         3.08825902, 4.4979976, 4.60447543, 4.53624622, 7.92454212, 3.04974245,
                         4.47240729, 4.51682076, 7.81332716, 3.13657456, 4.53061008, 7.88550407,
                         3.12333547, 7.9248432, 8.74691831])

    # 读取CSV文件
    df = pd.read_csv(f'./optimized_particles_distribution_with_ids_{g}.csv')

    # 提取需要标准化的数据（第二列到倒数第二列）
    data_to_standardize = df.iloc[:, 1:-1]

    # 标准化数据： (x - mean) / std
    standardized_data = (data_to_standardize - means) / std_devs

    # 用标准化后的数据替换原始数据
    df.iloc[:, 1:-1] = standardized_data

    # 保存为新的CSV文件
    output_file_path = f'./standardized_optimized_particles_distribution{g}.csv'
    df.to_csv(output_file_path, index=False)

    print(f"标准化后的数据已保存到 {output_file_path}")
