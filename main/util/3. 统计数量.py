import os
from collections import defaultdict


train_path = '/home/lwc/data/AD/train_WD100'
val_path = '/home/lwc/data/AD/val_WD100'
test_path = '/home/lwc/data/AD/test_WD100'
# 配置文件夹路径
data_folders = [train_path, val_path, test_path]
stats = {}

# 初始化统计结果的字典
for folder in data_folders:
    stats[folder] = {
        'type_count': defaultdict(int),
        'type_id_count': defaultdict(int)
    }

# 遍历文件夹进行统计
for folder in data_folders:
    for file_name in os.listdir(folder):
        if file_name.endswith('.txt'):
            type_, id_, _ = file_name.split('_')
            type_id = f"{type_}_{id_}"

            stats[folder]['type_count'][type_] += 1
            stats[folder]['type_id_count'][type_id] += 1

# 将统计结果写入文件
with open('dataset_stats.txt', 'w') as f:
    for folder in data_folders:
        f.write(f"Folder: {folder}\n")
        f.write("Type Count:\n")
        for type_, count in stats[folder]['type_count'].items():
            f.write(f"  {type_}: {count}\n")
        f.write("Type_ID Count:\n")
        for type_id, count in stats[folder]['type_id_count'].items():
            f.write(f"  {type_id}: {count}\n")
        f.write("\n")

print("统计结果已写入 dataset_stats.txt 文件中。")
