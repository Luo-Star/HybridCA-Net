import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.eca_module import eca_layer
from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph
import numpy as np

import pdb

class Model(nn.Module):
    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # **this is the adj matrix that computes correlation based on z-score of data for all 1200 timesteps**
        A = np.load('/home/lwc/第二个工作对比实验/adj_matrix_WD80.npy')
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)
        # ECA Attention
        self.eca = eca_layer(64)
        temp_matrix = np.zeros((1, A.shape[0], A.shape[0]))
        temp_matrix[0] = DAD
        A = torch.tensor(temp_matrix, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        # build networks (**number of layers, final output features, kernel size**)
        spatial_kernel_size = A.size(0)

        temporal_kernel_size = 11 # update temporal kernel size

        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 128, kernel_size, 1,  **kwargs),
            st_gcn(128, 256, kernel_size, 1,  **kwargs),
            st_gcn(256 , 512, kernel_size, 1,  **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            # self.edge_importance = nn.ParameterList([
            #     nn.Parameter(torch.ones(self.A.size()))
            #     for i in self.st_gcn_networks
            # ])
            self.edge_importance = nn.Parameter(torch.ones(self.A.size()))
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction (**number of fully connected layers**)
        self.fcn = nn.Conv2d(512, num_class, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        # for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
        #     x, _ = gcn(x, self.A * (importance + torch.transpose(importance,1,2)))
        #print(self.edge_importance.shape)
        for gcn in self.st_gcn_networks:
            x, _ = gcn(x, self.A * (self.edge_importance*self.edge_importance+torch.transpose(self.edge_importance*self.edge_importance,1,2)))
        x = self.eca(x)
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        # pdb.set_trace()
        # x = self.fcn(x)
        # x = self.sig(x)
        #
        x = x.view(x.size(0), -1)

        return x

class st_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0.5,
                 residual=True):
        super().__init__()
        print("Dropout={}".format(dropout))
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A
