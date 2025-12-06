from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from net.st_gcn import Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

###### **model parameters**
W = 128  # window size
TS = 64  # number of voters per test subject

###### **training parameters**
LR = 0.001  # learning rate
batch_size = 32

###### setup model & data
net = Model(1, 1, None, True)
net.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.0005)

train_data = torch.from_numpy(np.load('30TR5/train_data.npy')).float()
train_label = torch.from_numpy(np.load('30TR5/train_label.npy')).float()
val_data = torch.from_numpy(np.load('30TR5/val_data.npy')).float()
val_label = torch.from_numpy(np.load('30TR5/val_label.npy')).float()

train_dataset = TensorDataset(train_data, train_label)
val_dataset = TensorDataset(val_data, val_label)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=12)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=12)

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

for epoch in range(200):  # number of mini-batches
    # construct a mini-batch by sampling a window W for each subject
    for train_data_batch, train_label_batch in train_loader:
        train_data_batch_dev = train_data_batch.to(device)
        train_label_batch_dev = train_label_batch.to(device)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = net(train_data_batch_dev)
        loss = criterion(outputs.squeeze(-1), train_label_batch_dev)
        predict = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
        train_acc += (predict == train_label_batch_dev)[0].sum().item()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        # print training statistics
        training_loss += loss.item()

    # net.eval()
    acc = 0.0
    with torch.no_grad():
        for val_data_batch, val_label_batch in val_loader:
            val_data_batch_dev = val_data_batch.to(device)
            val_label_batch_dev = val_label_batch.to(device)
            outputs = net(val_data_batch_dev)
            loss = criterion(outputs.squeeze(-1), val_label_batch_dev)
            outputs = outputs.squeeze(-1)
            predict = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
            acc += (predict == val_label_batch_dev).sum().item()
            val_loss += loss.item()

        val_accurate = acc / val_num
        val_loss = val_loss / val_steps
        if best_acc < val_accurate:
            best_acc = val_accurate
            torch.save(net.state_dict(), 'output/30TR5/non_eca_30TR5_best_checkpoint.pth')
        results = '[' + str(epoch + 1) + ']' + "train acc " + str(train_acc / train_num) + " training loss:" + str(
            training_loss / train_steps) + " val acc: " + str(val_accurate) + " val_loss: " + str(val_loss)
        print(results)
        with open('output/30TR5/non_eca_30TR5_final.txt', 'a+') as f:
            time = datetime.now()
            f.write(str(time) + results)
            f.write("\n")
        train_acc = 0.0
        training_loss = 0.0
