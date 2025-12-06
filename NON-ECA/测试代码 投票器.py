from random import random

import numpy as np
import torch
from net.st_gcn import Model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Model(1,1,None,True)
net.load_state_dict(torch.load('output/best_checkpoint.pth'))
net.to(device)
TS = 64 # number of voters per test subject
batch_size = 32
test_data = np.load('data/test_data.npy')
test_label = np.load('data/test_label.npy')


with torch.no_grad():
    idx_batch = np.random.permutation(int(test_data.shape[0]))
    idx_batch = idx_batch[:int(batch_size)]

    test_label_batch = test_label[idx_batch]
    prediction = np.zeros((test_data.shape[0],))
    voter = np.zeros((test_data.shape[0],))
    for v in range(TS):
        idx = np.random.permutation(int(test_data.shape[0]))

        # testing also performed batch by batch (otherwise it produces error)
        for k in range(int(test_data.shape[0] / batch_size)):
            idx_batch = idx[int(batch_size * k):int(batch_size * (k + 1))]

            # construct random sub-sequences from a batch of test subjects
            test_data_batch = np.zeros((batch_size, 1, 30, 116, 1))
            for i in range(batch_size):
                test_data_batch[i] = test_data[idx_batch[i], :, :, :, :]

            test_data_batch_dev = torch.from_numpy(test_data_batch).float().to(device)
            # test_data_batch_dev = torch.permute(test_data_batch_dev, [0, 2, 1, 3, 4])
            outputs = net(test_data_batch_dev)
            outputs = outputs.data.cpu().numpy()

            prediction[idx_batch] = prediction[idx_batch] + outputs[:, 0]
            voter[idx_batch] = voter[idx_batch] + 1

    # average voting
    prediction = prediction / voter
    print(sum((prediction>0.5)==test_label) / test_label.shape[0])

    # average voting
    prediction = prediction / voter

    # 计算真阳性（True Positive）
    tp = np.sum((prediction > 0.5) & (test_label == 1))

    # 计算真阴性（True Negative）
    tn = np.sum((prediction <= 0.5) & (test_label == 0))

    # 计算假阳性（False Positive）
    fp = np.sum((prediction > 0.5) & (test_label == 0))

    # 计算假阴性（False Negative）
    fn = np.sum((prediction <= 0.5) & (test_label == 1))

    # 计算特异性
    specificity = tn / (tn + fp)

    # 计算敏感性
    sensitivity = tp / (tp + fn)

    accuracy = np.sum((prediction > 0.5) == test_label) / test_label.shape[0]

    print("Accuracy:", accuracy)
    print("Specificity:", specificity)
    print("Sensitivity:", sensitivity)
