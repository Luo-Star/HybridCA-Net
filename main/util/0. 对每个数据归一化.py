import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_filelist(dir_path):
    filelist = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".txt"):
                filelist.append(os.path.join(root, file))
    return filelist


def calc_zscore(matrix):
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    std[std == 0] = 1  # 防止除以零
    zscore = (matrix - mean) / std
    return zscore


def process_files(data_path, output_path):
    file_list = get_filelist(data_path)
    os.makedirs(output_path, exist_ok=True)

    nan_files = []

    for file_path in tqdm(file_list, desc="Processing files"):
        try:
            # 使用pandas读取txt文件中的数据
            data = pd.read_csv(file_path, delim_whitespace=True, header=None).values

            # 计算Z分数
            zscore_data = calc_zscore(data)

            # 检查是否存在NaN值
            has_nan = np.isnan(zscore_data).any()

            # 打印文件名和是否存在NaN值
            if has_nan:
                nan_files.append(os.path.basename(file_path))
                print(f"文件名: {os.path.basename(file_path)} 存在NaN值")
            else:
                print(f"文件名: {os.path.basename(file_path)} 不存在NaN值")
            print("-" * 50)

            # 保存标准化后的数据
            output_file_path = os.path.join(output_path, os.path.basename(file_path))
            pd.DataFrame(zscore_data).to_csv(output_file_path, sep='\t', header=False, index=False)

        except Exception as e:
            print(f"文件 {file_path} 处理失败: {e}")

    if nan_files:
        print("以下文件在标准化后存在NaN值：")
        for file in nan_files:
            print(file)


if __name__ == '__main__':
    # 数据路径
    data_path = "/data/home/202221000475/AD-fMRI-raw/NC_Signal_txt"
    output_path = "/data/home/202221000475/AD-fMRI-std/NC"

    # 调用处理函数
    process_files(data_path, output_path)
