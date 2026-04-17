# fusion.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.ResNet3D import resnet3d18  # Ensure the correct import path based on your project structure
from net.st_gcn import Model # Ensure the correct import path based on your project structure

class FusionModel(nn.Module):
    def __init__(self, resnet3d, stgcn, num_classes=10):
        super(FusionModel, self).__init__()
        self.resnet3d = resnet3d
        self.stgcn = stgcn
        self.fc = nn.Linear(512 + 512, num_classes)  # Adjust output features for the combined input
    def forward(self, x3d, xgcn):

        resnet_output = self.resnet3d(x3d)
        gcn_output = self.stgcn(xgcn)  # Ensure gcn_output shape matches expected

        # Adjust shapes if necessary
        resnet_output = torch.flatten(resnet_output, 1)
        gcn_output = torch.flatten(gcn_output, 1)


        # Concatenate outputs from both models
        combined_output = torch.cat((resnet_output, gcn_output), dim=1)
        # print(combined_output)
        output = self.fc(combined_output)
        return output

if __name__ == '__main__':
    # Instantiate models
    resnet3d_model = resnet3d18(num_classes=512, dropout_prob=0.5)
    stgcn_model = Model(1, 3, None, True)

    # Create fusion model
    fusion_model = FusionModel(resnet3d_model, stgcn_model, num_classes=3)

    # Sample inputs for both models
    input_tensor_3d = torch.randn(1, 1, 113, 137, 113)
    input_tensor_gcn = torch.randn(1, 1, 80, 400, 1)  # Adjust dimensions as necessary

    # Forward pass
    output = fusion_model(input_tensor_3d, input_tensor_gcn)
    print(output.shape)
