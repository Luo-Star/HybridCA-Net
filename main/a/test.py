import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义STGCN模型
class STGCN(nn.Module):
    def __init__(self):
        super(STGCN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.squeeze(-1)  # 移除单维度
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        return x

# 定义ResNet3D模型
class ResNet3D(nn.Module):
    def __init__(self):
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        return x

# 定义Transformer编码器层
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)

# 定义多模态融合模型
class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.stgcn = STGCN()
        self.resnet3d = ResNet3D()
        self.transformer = TransformerEncoder(input_dim=32, num_heads=4, ff_dim=128, num_layers=2)
        self.fc = nn.Linear(32, 10)

    def forward(self, x1, x2):
        out1 = self.stgcn(x1)  # 输入形状 (1, 1, 80, 400, 1)
        out2 = self.resnet3d(x2)  # 输入形状 (1, 1, 113, 137, 113)

        out1 = out1.flatten(2).permute(0, 2, 1)  # [batch_size, seq_len, input_dim]
        out2 = out2.flatten(2).permute(0, 2, 1)  # [batch_size, seq_len, input_dim]

        combined = torch.cat((out1, out2), dim=1)  # [batch_size, combined_seq_len, input_dim]

        fused = self.transformer(combined)
        fused = fused.mean(dim=1)

        output = self.fc(fused)
        return output

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化模型并移动到GPU
model = FusionModel().to(device)

# 示例输入数据并移动到GPU
input_data_2d = torch.randn(1, 1, 80, 400, 1).to(device)  # 2D输入 (batch_size, channels, height, width, depth)
input_data_3d = torch.randn(1, 1, 113, 137, 113).to(device)  # 3D输入 (batch_size, channels, depth, height, width)

# 执行前向传播
output = model(input_data_2d, input_data_3d)

print(output.shape)  # 输出的形状
