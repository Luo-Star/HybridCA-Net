import re
import matplotlib.pyplot as plt
import os

# 读取文件内容
file_path = '二分数据_同一被试在同一个集/二分数据_同一被试在同一个集.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 初始化数据列表
train_acc = []
train_loss = []
val_acc = []
val_loss = []

# 正则表达式匹配训练和验证数据
pattern = re.compile(r'train acc ([\d\.]+) training loss:([\d\.]+) val acc: ([\d\.]+) val_loss: ([\d\.]+)')

# 提取数据
for line in lines:
    match = pattern.search(line)
    if match:
        train_acc.append(float(match.group(1)))
        train_loss.append(float(match.group(2)))
        val_acc.append(float(match.group(3)))
        val_loss.append(float(match.group(4)))

# 绘制准确率曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()

# 绘制损失曲线
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()

# 调整布局
plt.tight_layout()

# 获取文件名前缀
file_prefix = os.path.splitext(file_path)[0]

# 保存图形
plt.savefig(f'{file_prefix}.png', transparent=True, dpi=600)

# 显示图形
plt.show()
