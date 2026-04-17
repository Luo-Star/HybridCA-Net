import numpy as np

def load_and_check_data(data_file, label_file):
    """
    加载数据和标签文件，并打印统计信息。

    参数:
    data_file: 数据文件路径。
    label_file: 标签文件路径。
    """
    # 加载数据和标签
    data = np.load(data_file)
    labels = np.load(label_file)

    # 打印数据和标签的形状
    print(f"数据文件: {data_file}")
    print(f"数据形状: {data.shape}")
    print(f"标签形状: {labels.shape}")

    # 打印数据和标签的一些统计信息
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("标签分布:")
    for label, count in zip(unique_labels, counts):
        print(f"  标签 {label}: {count} 个")
    print()

# 配置文件路径
data_files = {
    'train': ('train_data.npy', 'train_label.npy'),
    'val': ('val_data.npy', 'val_label.npy'),
    'test': ('test_data.npy', 'test_label.npy')
}

# 检查每个数据文件和标签文件
for name, (data_file, label_file) in data_files.items():
    load_and_check_data(data_file, label_file)

print("所有数据验证完成。")
