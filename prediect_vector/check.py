import torch

# 加载保存的文件
train_predictions_with_labels = torch.load('train_predictions_with_labels.pt')

# 查看数据

# 查看前10个数据样本的预测概率向量
print(train_predictions_with_labels[:10])
