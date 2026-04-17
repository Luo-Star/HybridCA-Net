import os
import numpy as np
from tqdm import tqdm
import pandas as pd

def read_txt_file(file_path):
    """
    读取txt文件，并返回数据矩阵。

    参数:
    file_path: txt文件的路径。

    返回值:
    数据矩阵，形状为 (30, 400)。
    """
    return np.loadtxt(file_path)

def calc_zscore(matrix):
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    std[std == 0] = 1
    zscore = (matrix - mean) / std
    zscore[np.isnan(zscore)] = 0
    return zscore

def calculate_correlation_matrix_from_folders(folders):
    """
    计算多个文件夹中的所有txt文件之间的相关性矩阵。

    参数:
    folders: 包含txt文件的文件夹列表。

    返回值:
    相关性矩阵，形状为 (400, 400)。
    """
    data_all = None

    # 计算总的文件数量，用于进度条的总数
    total_files = sum([len([name for name in os.listdir(folder) if name.endswith('.txt')]) for folder in folders])

    with tqdm(total=total_files, desc="Processing files") as pbar:
        for folder in folders:
            for file_name in os.listdir(folder):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(folder, file_name)
                    data = read_txt_file(file_path)
                    data = calc_zscore(data)
                    if data_all is None:
                        data_all = np.transpose(data)
                    else:
                        data_all = np.concatenate((data_all, np.transpose(data)), axis=1)
                    pbar.update(1)

    n_regions = data_all.shape[0]
    A = np.zeros((n_regions, n_regions))

    for i in tqdm(range(n_regions), desc="Computing adjacency matrix"):
        for j in range(i, n_regions):
            if i == j:
                A[i][j] = 1
            else:
                A[i][j] = abs(np.corrcoef(data_all[i, :], data_all[j, :])[0][1])
                A[j][i] = A[i][j]

    return A

train_path = '/home/lwc/Alzheimer_Data/每个被试切分出来的数据划分到相同数据集/train'
val_path = '/home/lwc/Alzheimer_Data/每个被试切分出来的数据划分到相同数据集/val'
test_path = '/home/lwc/Alzheimer_Data/每个被试切分出来的数据划分到相同数据集/test'
adj_path = '/home/lwc/Alzheimer_Data/每个被试切分出来的数据划分到相同数据集/adj_matrix.npy'
# 配置文件夹路径
data_folders = [train_path, val_path, test_path]

# 计算相关性矩阵
correlation_matrix = calculate_correlation_matrix_from_folders(data_folders)

# 打印相关性矩阵
print("相关性矩阵：")
print(correlation_matrix)

# 将相关性矩阵保存到 .npy 文件
np.save(adj_path, correlation_matrix)

print("相关性矩阵已保存到 adj_matrix.npy 文件中。")
