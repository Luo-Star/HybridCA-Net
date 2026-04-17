import os
import shutil
import random
from collections import defaultdict


AD = "/data/home/202221000475/AD-fMRI-std-slices/WD80/AD"
MCI = "/data/home/202221000475/AD-fMRI-std-slices/WD80/MCI"
NC = "/data/home/202221000475/AD-fMRI-std-slices/WD80/NC"
# 配置文件夹路径和数据集划分比例
data_folders = [AD, MCI, NC]

train_path = '/data/home/202221000475/AD-fMRI-std-slices-data/train'
val_path = '/data/home/202221000475/AD-fMRI-std-slices-data/val'
test_path = '/data/home/202221000475/AD-fMRI-std-slices-data/test'
# data_folders = [AD, NC]
output_folders = [train_path, val_path, test_path]
split_ratio = [0.8, 0.1, 0.1]  # 训练集100%，验证集10%，测试集10%

# 创建输出文件夹
for folder in output_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 读取文件并按编号进行分类
files_by_id = defaultdict(list)

for folder in data_folders:
    for file_name in os.listdir(folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder, file_name)
            type_, id_, _ = file_name.split('_')
            files_by_id[id_].append(file_path)

# 将编号按比例划分到训练集、验证集和测试集中
ids = list(files_by_id.keys())
random.shuffle(ids)
train_split = int(len(ids) * split_ratio[0])
val_split = int(len(ids) * (split_ratio[0] + split_ratio[1]))

train_ids = ids[:train_split]
val_ids = ids[train_split:val_split]
test_ids = ids[val_split:]

# 将文件移动到对应的数据集中
for id_ in train_ids:
    for file_path in files_by_id[id_]:
        shutil.copy(file_path, train_path)

for id_ in val_ids:
    for file_path in files_by_id[id_]:
        shutil.move(file_path, val_path)

for id_ in test_ids:
    for file_path in files_by_id[id_]:
        shutil.move(file_path, test_path)

print("文件划分完成。")
