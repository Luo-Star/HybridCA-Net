import re
import matplotlib.pyplot as plt

# 日志文件路径
log_file_path = 'training_20240713-195517.log'

# 初始化列表存储提取的数据
epochs = []
training_losses = []
validation_losses = []
validation_accuracies = []

# 定义正则表达式模式
pattern = re.compile(
    r"Epoch (\d+)/\d+ - Training Loss: ([\d\.]+) - Validation Loss: ([\d\.]+) - Validation Accuracy: ([\d\.]+)%"
)

# 读取日志文件并提取数据
with open(log_file_path, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            training_losses.append(float(match.group(2)))
            validation_losses.append(float(match.group(3)))
            validation_accuracies.append(float(match.group(4)))

# 绘制训练损失图表
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.plot(epochs, training_losses, label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# 绘制验证损失图表
plt.subplot(1, 3, 2)
plt.plot(epochs, validation_losses, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()

# 绘制验证准确率图表
plt.subplot(1, 3, 3)
plt.plot(epochs, validation_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy')
plt.legend()

# 调整图表布局并显示
plt.tight_layout()
plt.show()
