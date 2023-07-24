import os
import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

# 设置路径和文件名
data_dir = 'E:/ustc/CIFAR/dataset/'

# 超参数设置
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# 中值滤波处理函数
def median_filter(images):
    filtered_images = []
    for image in images:
        filtered_image = cv2.medianBlur(image, 3)  # 进行中值滤波
        filtered_images.append(filtered_image)
    return filtered_images


# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CIFAR10(root=data_dir, train=True, download=False, transform=transform)
# 加载测试集进行预测
test_dataset = CIFAR10(root=data_dir, train=False, download=False, transform=transform)

train_images, train_labels = train_dataset.data, train_dataset.targets
test_images, test_labels = test_dataset.data, test_dataset.targets

# # 选择要查看的图像索引
# image_index = 0

# # 显示中值滤波前的图像和标签
# original_image = train_images[image_index].reshape(32, 32, 3)
# original_label = train_labels[image_index]
# plt.subplot(1, 2, 1)
# plt.imshow(original_image)
# plt.title('Original Image')
# plt.xlabel(f'Label: {original_label}')

# plt.show()

# 中值滤波处理训练集和测试集的图像数据
train_filtered_images = median_filter(train_images)
test_filtered_images = median_filter(test_images)
# train_filtered_images = train_images
# test_filtered_images = test_images

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

# 归一化处理
train_filtered_images = (train_filtered_images / 255.0 - 0.5) / 0.5
test_filtered_images = (test_filtered_images / 255.0 - 0.5) / 0.5

# 构建新的数据集
train_dataset = data.TensorDataset(torch.from_numpy(train_filtered_images).float(), torch.from_numpy(train_labels).long())
test_dataset = data.TensorDataset(torch.from_numpy(test_filtered_images).float(), torch.from_numpy(test_labels).long())

# 创建数据加载器
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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

# 创建模型实例
model = SimpleCNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# 保存模型
save_path = 'E:/ustc/CIFAR/model_save/model_user_medianfiltering.ckpt'
torch.save(model.state_dict(), save_path)
print("模型已保存，保存路径为:", save_path)

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('测试集准确率: {:.2f}%'.format(100 * correct / total))
