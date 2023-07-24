import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
# from sklearn.model_selection import train_test_split


# 设置路径和文件名
data_dir = 'E:/ustc/CIFAR/dataset/'
# original_data_dir = 'E:/ustc/CIFAR/dataset/cifar-10-batches-py/'
# watermarked_data_dir = 'E:/ustc/CIFAR/dataset/cifar-10-batches-py-medianfiltering/'

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


train_dataset = CIFAR10(root=data_dir, train=True, download=False, transform=transform)
# 加载测试集进行预测
test_dataset = CIFAR10(root=data_dir, train=False, download=False, transform=transform)

# 中值滤波处理函数
def median_filter(images):
    filtered_images = []
    for image in images:
        filtered_image = cv2.medianBlur(image, 3)  # 进行中值滤波
        filtered_images.append(filtered_image)
    return filtered_images

train_images, train_labels = train_dataset.data, train_dataset.targets
test_images, test_labels = test_dataset.data, test_dataset.targets

# 加载模型和数据集
original_model = SimpleCNN()
original_model.load_state_dict(torch.load('E:/ustc/CIFAR/model_save/model_user_ordinary.ckpt'))

median_filter_model = SimpleCNN()
median_filter_model.load_state_dict(torch.load('E:/ustc/CIFAR/model_save/model_user_medianfiltering.ckpt'))

# 中值滤波处理训练集和测试集的图像数据
train_filtered_images = median_filter(train_images)
test_filtered_images = median_filter(test_images)

train_filtered_images = np.array(train_filtered_images)
test_filtered_images = np.array(test_filtered_images)

# 进行必要的形状变换
train_filtered_images = train_filtered_images.reshape(-1, 3, 32, 32)
test_filtered_images = test_filtered_images.reshape(-1, 3, 32, 32)
train_filtered_images = np.array(train_filtered_images)
test_filtered_images = np.array(test_filtered_images)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# 将图像数据转换为浮点类型
train_filtered_images = train_filtered_images.astype(np.float32)
test_filtered_images = test_filtered_images.astype(np.float32)
train_images = train_images.astype(np.float32)
test_images = test_images.astype(np.float32)

# 归一化处理
train_filtered_images = (train_filtered_images / 255.0 - 0.5) / 0.5
test_filtered_images = (test_filtered_images / 255.0 - 0.5) / 0.5
train_images = (train_images / 255.0 - 0.5) / 0.5
test_images = (test_images / 255.0 - 0.5) / 0.5

# 构建新的数据集
train_filtered_dataset = data.TensorDataset(torch.from_numpy(train_filtered_images).float(), torch.from_numpy(train_labels).long())
test_filtered_dataset = data.TensorDataset(torch.from_numpy(test_filtered_images).float(), torch.from_numpy(test_labels).long())
train_dataset = data.TensorDataset(torch.from_numpy(train_images).float(), torch.from_numpy(train_labels).long())
test_dataset = data.TensorDataset(torch.from_numpy(test_images).float(), torch.from_numpy(test_labels).long())

# 创建数据加载器
train_filtered_loader = data.DataLoader(train_filtered_dataset, batch_size=batch_size, shuffle=True)
test_filtered_loader = data.DataLoader(test_filtered_dataset, batch_size=batch_size, shuffle=False)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 准备输入样本（例如，从测试集中获取）
original_data = []
watermarked_data = []
original_test_data = []
watermarked_test_data = []

# 获取原始数据集样本
for images, labels in train_loader:
    original_data.append(images)

original_data = torch.cat(original_data, dim=0)

for images, labels in test_loader:
    original_test_data.append(images)

original_test_data = torch.cat(original_test_data, dim=0)

# 获取中值滤波水印数据集样本
for images, labels in train_filtered_loader:
    watermarked_data.append(images)

watermarked_data = torch.cat(watermarked_data, dim=0)

for images, labels in test_filtered_loader:
    watermarked_test_data.append(images)

watermarked_test_data = torch.cat(watermarked_test_data, dim=0)

# 获取原始数据集样本的预测概率向量
original_probs = original_model(original_data)
original_probs_test = original_model(original_test_data)

# 获取中值滤波水印数据集样本的预测概率向量
watermarked_probs = median_filter_model(watermarked_data)
watermarked_probs_test = median_filter_model(watermarked_test_data)

# 准备标签
original_labels = torch.zeros(original_probs.size(0), dtype=torch.long)  # 原始数据集样本标签为0
watermarked_labels = torch.ones(watermarked_probs.size(0), dtype=torch.long)  # 中值滤波水印数据集样本标签为1
original_labels_test = torch.zeros(original_probs_test.size(0), dtype=torch.long)  # 原始数据集样本标签为0
watermarked_labels_test = torch.ones(watermarked_probs_test.size(0), dtype=torch.long)  # 中值滤波水印数据集样本标签为1

# 合并输入特征和标签
features = torch.cat((original_probs, watermarked_probs), dim=0)
labels = torch.cat((original_labels, watermarked_labels), dim=0)
features_test = torch.cat((original_probs_test, watermarked_probs_test), dim=0)
labels_test = torch.cat((original_labels_test, watermarked_labels_test), dim=0)

# 创建新的数据集
train_dataset = data.TensorDataset(features, labels)
test_dataset = data.TensorDataset(features_test, labels_test)

# 创建新的数据加载器
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 定义分类器模型
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(10, 2)  # 修改输入和输出的维度

    def forward(self, x):
        x = self.fc(x)
        return x

classifier = Classifier()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

# 训练分类器
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

# 加载测试集进行预测

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
