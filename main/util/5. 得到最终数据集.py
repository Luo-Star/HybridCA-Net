import os
import numpy as np
from tqdm import tqdm

# 映射类型到标签
type_to_label = {'AD': 0, 'MCI': 1, 'NC': 2}
# type_to_label = {'AD': 0, 'NC': 1}
# type_to_label = {'CID': 0, 'HC': 1}

def read_txt_file(file_path):
    """
    读取txt文件，并返回数据矩阵。

    参数:
    file_path: txt文件的路径。

    返回值:
    数据矩阵，形状为 (30, 400)。
    """
    return np.loadtxt(file_path)

def process_folder(folder):
    """
    处理文件夹中的所有txt文件，返回数据和标签。

    参数:
    folder: 文件夹路径。

    返回值:
    data: 数据矩阵，形状为 (num, 1, 30, 400, 1)。
    labels: 标签数组，形状为 (num,)。
    """
    data = []
    labels = []

    files = [f for f in os.listdir(folder) if f.endswith('.txt')]
    for file_name in tqdm(files, desc=f"Processing {folder}"):
        file_path = os.path.join(folder, file_name)
        type_, id_, _ = file_name.split('_')
        label = type_to_label[type_]
        data_matrix = read_txt_file(file_path)
        data.append(data_matrix)
        labels.append(label)

    data = np.array(data)
    data = data[:, np.newaxis, :, :, np.newaxis]
    labels = np.array(labels)

    return data, labels

def shuffle_data(data, labels):
    """
    打乱数据和标签。

    参数:
    data: 数据矩阵。
    labels: 标签数组。

    返回值:
    打乱后的数据和标签。
    """
    permutation = np.random.permutation(len(labels))
    shuffled_data = data[permutation]
    shuffled_labels = labels[permutation]

    return shuffled_data, shuffled_labels

# 定义起始和结束的 data_type 值
start = 80
end = 80

for i in range(start, end + 1, 10):
    data_type = f'WD{i}'

    # 使用变量替代原来的硬编码路径
    train_path = f'/data/home/202221000475/AD-fMRI-std-slices-data/{data_type}/train_{data_type}'
    val_path = f'/data/home/202221000475/AD-fMRI-std-slices-data/{data_type}/val_{data_type}'
    test_path = f'/data/home/202221000475/AD-fMRI-std-slices-data/{data_type}/test_{data_type}'
    data_folders = [train_path, val_path, test_path]

    output_files = {
        train_path: (f'train_data_{data_type}.npy', f'train_label_{data_type}.npy'),
        val_path: (f'val_data_{data_type}.npy', f'val_label_{data_type}.npy'),
        test_path: (f'test_data_{data_type}.npy', f'test_label_{data_type}.npy')
    }

    # 获取当前工作目录
    current_directory = os.getcwd()

    # 处理每个文件夹并保存数据和标签
    for folder in data_folders:
        data, labels = process_folder(folder)
        data, labels = shuffle_data(data, labels)
        data_file, label_file = output_files[folder]
        data_file = os.path.join(current_directory, data_file)
        label_file = os.path.join(current_directory, label_file)

        # 确保路径存在，不存在则创建
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        os.makedirs(os.path.dirname(label_file), exist_ok=True)

        np.save(data_file, data)
        np.save(label_file, labels)
        print(f"{folder} 数据已保存到 {data_file} 和 {label_file} 文件中。")

print("所有数据处理完成。")
