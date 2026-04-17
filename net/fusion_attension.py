import torch
import torch.nn as nn
import torch.nn.functional as F

from net.ResNet3D import resnet3d18  # 确保导入路径正确
from net.st_gcn import Model  # 确保导入路径正确

# 直接相加模块
class AddAttention(nn.Module):
    def __init__(self, d_model):
        super(AddAttention, self).__init__()

    def forward(self, x, y):
        # x: ResNet3D输出, y: ST-GCN输出
        out = x + y
        return out

# 直接相乘模块
class MultiplyAttention(nn.Module):
    def __init__(self, d_model):
        super(MultiplyAttention, self).__init__()

    def forward(self, x, y):
        # x: ResNet3D输出, y: ST-GCN输出
        out = x * y
        return out

# 交叉注意力模块
class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv1d(d_model, d_model // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(d_model, d_model // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        # x: ResNet3D输出, y: ST-GCN输出
        proj_query = self.query_conv(x).permute(0, 2, 1)
        proj_key = self.key_conv(y)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(y)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = self.gamma * out + x
        return out

# 一致性损失模块
class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()

    def forward(self, original, augmented):
        return F.mse_loss(original, augmented)

# 融合模型
class FusionModel(nn.Module):
    def __init__(self, resnet3d, stgcn, num_classes=10, attention_type="cross"):
        super(FusionModel, self).__init__()
        self.resnet3d = resnet3d
        self.stgcn = stgcn
        if attention_type == "cross":
            self.attention = CrossAttention(d_model=512)
        elif attention_type == "add":
            print(f"ttention_type  {attention_type}")
            self.attention = AddAttention(d_model=512)
        elif attention_type == "multiply":
            self.attention = MultiplyAttention(d_model=512)
        else:
            raise ValueError("Unsupported attention type. Choose from 'cross', 'add', or 'multiply'.")

        self.fc = nn.Linear(512 * 2, num_classes)  # 调整输出特征以适应组合输入

    def forward(self, x3d, xgcn):
        resnet_output = self.resnet3d(x3d)
        gcn_output = self.stgcn(xgcn)  # 确保 gcn_output 形状匹配预期

        # 调整形状（如有必要）
        resnet_output = torch.flatten(resnet_output, 1)
        gcn_output = torch.flatten(gcn_output, 1)

        # 应用选择的注意力机制
        attention_output = self.attention(resnet_output.unsqueeze(2), gcn_output.unsqueeze(2))
        attention_output = torch.flatten(attention_output, 1)

        # 拼接两个模型的输出
        combined_output = torch.cat((resnet_output, attention_output), dim=1)
        output = self.fc(combined_output)
        return output, resnet_output, gcn_output

if __name__ == '__main__':
    # 实例化模型
    resnet3d_model = resnet3d18(num_classes=512, dropout_prob=0.5)
    stgcn_model = Model(1, 2, None, True)

    # 创建融合模型
    fusion_model = FusionModel(resnet3d_model, stgcn_model, num_classes=2)

    # 样本输入
    input_tensor_3d = torch.randn(1, 1, 113, 137, 113)
    input_tensor_gcn = torch.randn(1, 1, 80, 400, 1)  # 根据需要调整维度

    # 前向传播
    output, resnet_output, gcn_output = fusion_model(input_tensor_3d, input_tensor_gcn)
    print(output.shape)

    # 自监督一致性损失计算
    consistency_loss_fn = ConsistencyLoss()
    consistency_loss = consistency_loss_fn(resnet_output, gcn_output)
    print(f"Consistency Loss: {consistency_loss.item()}")
