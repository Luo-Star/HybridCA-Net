from random import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from net.st_gcn import Model


def a():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Model(1, 1, None, True)
    net.load_state_dict(torch.load('/home/lwc/6-7/NON-ECA/output/second/30TR20/non_eca_30TR20_best_checkpoint.pth'))
    net.to(device)
    TS = 64  # 每个测试样本的投票数
    batch_size = 32
    test_data = torch.from_numpy(np.load('/home/lwc/6-7/second_data/30TR20/test_data.npy')).float()
    test_label = torch.from_numpy(np.load('/home/lwc/6-7/second_data/30TR20/test_label.npy')).float()
    test_dataset = TensorDataset(test_data, test_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    acc = 0.0
    test_num = len(test_dataset)
    TP = 0  # 真阳性
    TN = 0  # 真阴性
    FP = 0  # 假阳性
    FN = 0  # 假阴性
    accuracies = []  # 用于保存每个样本的准确率
    specificities = []  # 用于保存每个样本的特异性
    sensitivities = []  # 用于保存每个样本的灵敏性
    with torch.no_grad():
        for val_data_batch, val_label_batch in test_loader:
            val_data_batch_dev = val_data_batch.to(device)
            val_label_batch_dev = val_label_batch.to(device)
            outputs = net(val_data_batch_dev)
            outputs = outputs.squeeze(-1)
            predict = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
            acc_batch = (predict == val_label_batch_dev).float().mean().item()
            accuracies.append(acc_batch)

            # 计算TP、TN、FP和FN的数量
            TP_tmp = torch.logical_and(predict == 1, val_label_batch_dev == 1).sum().item()
            TN_tmp = torch.logical_and(predict == 0, val_label_batch_dev == 0).sum().item()
            FP_tmp = torch.logical_and(predict == 1, val_label_batch_dev == 0).sum().item()
            FN_tmp = torch.logical_and(predict == 0, val_label_batch_dev == 1).sum().item()

            specificity = TN_tmp / (TN_tmp + FP_tmp)
            sensitivity = TP_tmp / (TP_tmp + FN_tmp)
            specificities.append(specificity)
            sensitivities.append(sensitivity)

            # 计算TP、TN、FP和FN的数量
            TP += TP_tmp
            TN += TN_tmp
            FP += FP_tmp
            FN += FN_tmp

    val_accuracy = np.mean(accuracies)
    val_accuracy_std = np.std(accuracies)
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    specificity_std = np.std(specificities)
    sensitivity_std = np.std(sensitivities)

    print("Validation Accuracy:", val_accuracy)
    print("Validation Accuracy Standard Deviation:", val_accuracy_std)
    print("Specificity:", specificity)
    print("Specificity Standard Deviation:", specificity_std)
    print("Sensitivity:", sensitivity)
    print("Sensitivity Standard Deviation:", sensitivity_std)

    # 保存 accuracies specificities sensitivities 到对应的txt文件
    np.savetxt("accuracies.txt", accuracies)
    np.savetxt("specificities.txt", specificities)
    np.savetxt("sensitivities.txt", sensitivities)
if __name__ == '__main__':
    a()