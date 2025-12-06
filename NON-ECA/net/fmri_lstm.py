import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#import matplotlib.pyplot as plt
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class fMRI_LSTM(nn.Module):
    def __init__(self, hidden_dim, input_dim, target_size, batch_size):
        super(fMRI_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.hidden_init = self.init_hidden(batch_size=batch_size)
        self.dropout = nn.Dropout(p=0.5)
        self.target_size = target_size

    def init_hidden(self, batch_size):
        return (torch.randn(1, batch_size, self.hidden_dim).to(device), torch.randn(1, batch_size, self.hidden_dim).to(device))

    def forward(self, x):
        #x = x.T
        lstm_out, _ = self.lstm(x) # You don't really need to pass hidden layer state
        # lstm_out, _ = self.lstm(x,self.hidden_init) # or for init hidden state.
        lstm_out = lstm_out.squeeze()[:, -1, :]
        out = self.dropout(lstm_out)
        linear_output = self.linear(out)
        #linear_output = self.dropo(linear_output)
        #mean_pooling_output = F.avg_pool1d(linear_output, kernel_size=1)
        final_output = torch.sigmoid(linear_output)
        return final_output


