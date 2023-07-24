import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# 加载数据集文件
train_predictions_with_labels = torch.load('train_predictions_with_labels.pt')
test_predictions_with_labels = torch.load('test_predictions_with_labels.pt')
train_predictions_with_labels_median_filter = torch.load('train_predictions_with_labels_median_filter.pt')
test_predictions_with_labels_median_filter = torch.load('test_predictions_with_labels_median_filter.pt')

# 创建标签
train_labels_0 = torch.zeros(train_predictions_with_labels.size(0), dtype=torch.long)
test_labels_0 = torch.zeros(test_predictions_with_labels.size(0), dtype=torch.long)
train_labels_1 = torch.ones(train_predictions_with_labels_median_filter.size(0), dtype=torch.long)
test_labels_1 = torch.ones(test_predictions_with_labels_median_filter.size(0), dtype=torch.long)

# 将数据和标签组合成新的数据集
train_dataset = torch.utils.data.TensorDataset(torch.cat([train_predictions_with_labels, train_predictions_with_labels_median_filter], dim=0),
                                               torch.cat([train_labels_0, train_labels_1], dim=0))

test_dataset = torch.utils.data.TensorDataset(torch.cat([test_predictions_with_labels, test_predictions_with_labels_median_filter], dim=0),
                                              torch.cat([test_labels_0, test_labels_1], dim=0))

# 创建数据加载器
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义分类器模型
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

# 创建分类器实例
input_dim = train_predictions_with_labels.size(1)
output_dim = 2
classifier = Classifier(input_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

# 训练分类器
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 测试分类器
classifier.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = classifier(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('测试集准确率: {:.2f}%'.format(100 * correct / total))
