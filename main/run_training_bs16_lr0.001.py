from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from net.st_gcn import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

###### **model parameters**
W = 128  # window size
TS = 64  # number of voters per test subject

###### **training parameters**
LR = 1e-03  # learning rate
batch_size = 32

###### setup model & data
net = Model(1, 3, None, True)  # 修改输出为3类
net.to(device)

criterion = nn.CrossEntropyLoss()  # 修改损失函数为多分类
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.00005)

train_data = torch.from_numpy(np.load('/home/lwc/myown 代码优化/AD_data/WD80/train_data_WD80.npy')).float()
train_label = torch.from_numpy(np.load('/home/lwc/myown 代码优化/AD_data/WD80/train_label_WD80.npy')).long()  # 确保标签是长整型
val_data = torch.from_numpy(np.load('/home/lwc/myown 代码优化/AD_data/WD80/val_data_WD80.npy')).float()
val_label = torch.from_numpy(np.load('/home/lwc/myown 代码优化/AD_data/WD80/val_label_WD80.npy')).long()  # 确保标签是长整型

train_dataset = TensorDataset(train_data, train_label)
val_dataset = TensorDataset(val_data, val_label)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

print(train_data.shape)
print(val_data.shape)

###### start training model
training_loss = 0.0
val_loss = 0.0
best_acc = 0.0
train_acc = 0.0

val_num = len(val_dataset)
val_steps = len(val_loader)
train_steps = len(train_loader)
train_num = len(train_dataset)

for epoch in range(200):  # number of epochs
    net.train()
    for train_data_batch, train_label_batch in train_loader:
        train_data_batch_dev = train_data_batch.to(device)
        train_label_batch_dev = train_label_batch.to(device)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = net(train_data_batch_dev)
        loss = criterion(outputs, train_label_batch_dev)
        _, predicted = torch.max(outputs, 1)
        train_acc += (predicted == train_label_batch_dev).sum().item()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        # print training statistics
        training_loss += loss.item()

    train_acc_epoch = train_acc / train_num
    train_loss_epoch = training_loss / train_steps
    train_acc = 0.0
    training_loss = 0.0

    net.eval()
    acc = 0.0
    with torch.no_grad():
        for val_data_batch, val_label_batch in val_loader:
            val_data_batch_dev = val_data_batch.to(device)
            val_label_batch_dev = val_label_batch.to(device)
            outputs = net(val_data_batch_dev)
            loss = criterion(outputs, val_label_batch_dev)
            _, predicted = torch.max(outputs, 1)
            acc += (predicted == val_label_batch_dev).sum().item()
            val_loss += loss.item()

        val_accurate = acc / val_num
        val_loss_epoch = val_loss / val_steps
        if best_acc < val_accurate:
            best_acc = val_accurate
            torch.save(net.state_dict(), 'net/output/AD/WD80/AD_bs32_lr0.001.pth')

        results = f'[{epoch + 1}] train acc: {train_acc_epoch:.4f} training loss: {train_loss_epoch:.4f} val acc: {val_accurate:.4f} val loss: {val_loss_epoch:.4f}'
        print(results)
        with open('net/output/AD/WD80/AD_WD80_bs32_lr0.001.txt', 'a+') as f:
            time = datetime.now()
            f.write(f'{time} {results}\n')

        val_loss = 0.0
