import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

# 设置路径和文件名
data_dir = 'E:/ustc/CIFAR/dataset/cifar-10-batches-py/'
data_dir_medianfiltering = 'E:/ustc/CIFAR/dataset/cifar-10-batches-py-medianfiltering/'


# 超参数设置
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# 加载原始数据集
original_dataset = CIFAR10(root=data_dir, train=True, download=False, transform=transform)
original_data_file = os.path.join(data_dir, 'data_batch_1')
original_dataset.data = np.load(original_data_file, allow_pickle=True)
original_loader = data.DataLoader(original_dataset, batch_size=batch_size, shuffle=True)

# 加载中值滤波水印数据集
watermarked_dataset = CIFAR10(root=data_dir_medianfiltering, train=True, download=False, transform=transform)
watermarked_data_file = os.path.join(data_dir_medianfiltering, 'data_batch_1_medianfiltering')
watermarked_dataset.data = np.load(watermarked_data_file, allow_pickle=True)
watermarked_loader = data.DataLoader(watermarked_dataset, batch_size=batch_size, shuffle=True)

# 定义原始数据集模型
original_model = SimpleCNN()

# 定义中值滤波水印数据集模型
watermarked_model = SimpleCNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer_original = optim.Adam(original_model.parameters(), lr=learning_rate)
optimizer_watermarked = optim.Adam(watermarked_model.parameters(), lr=learning_rate)

# 训练原始数据集模型
total_step_original = len(original_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(original_loader):
        outputs = original_model(images)
        loss = criterion(outputs, labels)

        optimizer_original.zero_grad()
        loss.backward()
        optimizer_original.step()

        if (i+1) % 100 == 0:
            print('Original - Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step_original, loss.item()))

# 训练中值滤波水印数据集模型
total_step_watermarked = len(watermarked_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(watermarked_loader):
        outputs = watermarked_model(images)
        loss = criterion(outputs, labels)

        optimizer_watermarked.zero_grad()
        loss.backward()
        optimizer_watermarked.step()

        if (i+1) % 100 == 0:
            print('Watermarked - Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step_watermarked, loss.item()))

# 加载测试集进行预测
test_dataset = CIFAR10(root=data_dir, train=False, download=False, transform=transform)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_dataset_medianfiltering = CIFAR10(root=data_dir_medianfiltering, train=False, download=False, transform=transform)
test_loader_medianfiltering = data.DataLoader(test_dataset_medianfiltering, batch_size=batch_size, shuffle=False)

# 提取原始数据集的预测概率向量
original_prob_vectors = []
with torch.no_grad():
    for images, _ in test_loader:
        outputs = original_model(images)
        original_prob_vectors.append(outputs)

original_prob_vectors = torch.cat(original_prob_vectors, dim=0)

# 提取中值滤波水印数据集的预测概率向量
watermarked_prob_vectors = []
with torch.no_grad():
    for images, _ in test_loader_medianfiltering:
        outputs = watermarked_model(images)
        watermarked_prob_vectors.append(outputs)

watermarked_prob_vectors = torch.cat(watermarked_prob_vectors, dim=0)

# 创建标签
original_labels = torch.zeros(original_prob_vectors.size(0), dtype=torch.long)
watermarked_labels = torch.ones(watermarked_prob_vectors.size(0), dtype=torch.long)

# 合并数据和标签
features = torch.cat((original_prob_vectors, watermarked_prob_vectors), dim=0)
labels = torch.cat((original_labels, watermarked_labels), dim=0)

# 定义分类器模型
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc(x)

classifier = Classifier(input_dim=features.size(1))

# 定义损失函数和优化器
criterion_classifier = nn.CrossEntropyLoss()
optimizer_classifier = optim.Adam(classifier.parameters(), lr=learning_rate)

# 训练分类器
total_step_classifier = len(labels)
for epoch in range(num_epochs):
    optimizer_classifier.zero_grad()
    outputs = classifier(features)
    loss = criterion_classifier(outputs, labels)
    loss.backward()
    optimizer_classifier.step()

    print('Classifier - Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 测试分类器
classifier.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, _ in test_loader:
        outputs_original = original_model(images)
        outputs_watermarked = watermarked_model(images)
        inputs = torch.cat((outputs_original, outputs_watermarked), dim=1)
        outputs_classifier = classifier(inputs)
        _, predicted = torch.max(outputs_classifier.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('测试集准确率: {:.2f}%'.format(100 * correct / total))
