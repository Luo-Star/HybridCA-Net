import torch
import torch.nn as nn


class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.5):
        super(BasicBlock3D, self).__init__()
        # 定义第一个3D卷积层
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)  # 批归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        self.dropout = nn.Dropout3d(p=dropout_prob)  # Dropout层
        # 定义第二个3D卷积层
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)  # 批归一化层

        self.downsample = nn.Sequential()
        # 如果步长不为1或者输入通道数与输出通道数不一致，需要进行下采样
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x  # 保存输入值用于残差连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果存在下采样，则调整identity
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        # 激活函数
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=10, dropout_prob=0.5):
        super(ResNet3D, self).__init__()
        self.in_channels = 64  # 初始输入通道数

        # 初始卷积层，使用7x7x7卷积核，步长为2
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)  # 批归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)  # 最大池化层

        # 构建ResNet层
        self.layer1 = self._make_layer(block, 64, layers[0], dropout_prob)
        self.layer2 = self._make_layer(block, 128, layers[1], dropout_prob, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], dropout_prob, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], dropout_prob, stride=2)

        # 自适应平均池化层
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # 全连接层
        self.fc = nn.Linear(512, num_classes)

        # sofxmax
        self.soft = nn.Softmax()
    def _make_layer(self, block, out_channels, blocks, dropout_prob=0.5, stride=1):
        layers = []
        # 第一个块需要考虑步长和输入通道数变化
        layers.append(block(self.in_channels, out_channels, stride=stride, dropout_prob=dropout_prob))
        self.in_channels = out_channels  # 更新输入通道数
        # 剩下的块只需考虑通道数
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, dropout_prob=dropout_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播过程
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet3d18(num_classes=10, dropout_prob=0.5):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes, dropout_prob)


if __name__ == '__main__':

    # 测试模型，输入维度为 [1, 1, 113, 137, 113]
    input_tensor = torch.randn(1, 1, 113, 137, 113)
    model = resnet3d18(num_classes=10, dropout_prob=0.5)
    output = model(input_tensor)
    print(output.shape)
