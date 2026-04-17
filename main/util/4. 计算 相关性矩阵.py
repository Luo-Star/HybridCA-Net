import os
import numpy as np
from tqdm import tqdm

def read_txt_file(file_path):
    """
    读取txt文件，并返回数据矩阵。

    参数:
    file_path: txt文件的路径。

    返回值:
    数据矩阵，形状为 (30, 400)。
    """
    return np.loadtxt(file_path)

def calculate_correlation_matrix_from_folders(folders):
    """
    计算多个文件夹中的所有txt文件之间的相关性矩阵。

    参数:
    folders: 包含txt文件的文件夹列表。

    返回值:
    相关性矩阵，形状为 (400, 400)。
    """
    data_all = []

    # 计算总的文件数量，用于进度条的总数
    total_files = sum([len([name for name in os.listdir(folder) if name.endswith('.txt')]) for folder in folders])

    with tqdm(total=total_files, desc="Processing files") as pbar:
        for folder in folders:
            for file_name in os.listdir(folder):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(folder, file_name)
                    data = read_txt_file(file_path)
                    data_all.append(data)
                    pbar.update(1)

    # 将所有数据矩阵进行垂直拼接，得到一个形状为 (len(data_all) * 30, 400) 的矩阵
    concatenated_data = np.vstack(data_all)

    # 计算相关系数矩阵
    correlation_matrix = np.corrcoef(concatenated_data.T)

    return correlation_matrix

# 定义起始和结束的 data_type 值
start = 80
end = 80

for i in range(start, end + 1, 10):
    data_type = f'WD{i}'

    # 使用变量替代原来的硬编码路径
    train_path = f'/data/home/202221000475/AD-fMRI-std-slices-data/{data_type}/train_{data_type}'
    val_path = f'/data/home/202221000475/AD-fMRI-std-slices-data/{data_type}/val_{data_type}'
    test_path = f'/data/home/202221000475/AD-fMRI-std-slices-data/{data_type}/test_{data_type}'
    adj_path = f'/data/home/202221000475/AD-fMRI-std-slices-data/{data_type}/adj_matrix_{data_type}.npy'

    # print(train_path, val_path, test_path, adj_path)
    # 配置文件夹路径
    data_folders = [train_path, val_path, test_path]

    # 计算相关性矩阵
    correlation_matrix = calculate_correlation_matrix_from_folders(data_folders)

    # 打印相关性矩阵
    print(f"{data_type} 相关性矩阵：")
    print(correlation_matrix)

    # 确保路径存在，不存在则创建
    os.makedirs(os.path.dirname(adj_path), exist_ok=True)
    # 将相关性矩阵保存到 .npy 文件
    np.save(adj_path, correlation_matrix)

    print(f"{data_type} 相关性矩阵已保存到 {adj_path} 文件中。")
