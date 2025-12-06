import os

import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

# 构建图数据集
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        filepath = os.path.join(self.root_dir, filename)

        # 从文件中读取数据并进行处理
        feature, adj = self.process_data(filepath)
        label = self.judge(filename)
        # 创建PyG的Data对象并返回
        # data = Data(x=feature, adj=adj)
        return feature, adj,label

    def process_data(self, filepath):
        # 从文件中读取数据并进行处理的逻辑
        # 假设数据存储为numpy数组
        feature = np.loadtxt(filepath)
        adj = self.cal_adj(feature)

        #转Tensor
        feature = torch.tensor(feature)
        adj = torch.tensor(adj)
        return feature, adj

    def cal_adj(self, feature):
        feature = np.transpose(feature)
        n_regions = 116
        A = np.zeros((n_regions, n_regions))
        for i in range(n_regions):
            for j in range(i, n_regions):
                if i == j:
                    A[i][j] = 1
                else:
                    A[i][j] = abs(np.corrcoef(feature[i, :], feature[j, :])[0][1])  # get value from corrcoef matrix
                    A[j][i] = A[i][j]
        return A

    def judge(self,filename):
        if "CID" in filename:
            return 1
        else:
            return 0


if __name__ == '__main__':
    # 指定数据存储的根目录
    root_dir = '/home/lwc/data/second/30TR5'

    # 创建自定义数据集对象
    dataset = GraphDataset(root_dir)

    # 创建数据加载器
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 遍历数据加载器
    for batch in dataloader:
        feature, adj, label = batch
        print(feature)
