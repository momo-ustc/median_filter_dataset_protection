import os
import cv2
import pickle

# 设置路径和文件名
data_dir = 'E:/ustc/CIFAR/dataset/cifar-10-batches-py/'
data_file = os.path.join(data_dir, 'test_batch')
new_data_file = os.path.join(data_dir, 'test_batch_medianfiltering')

# 加载原始数据集
with open(data_file, 'rb') as f:
    data = pickle.load(f, encoding='bytes')

# 获取图像数据和标签
images = data[b'data']
labels = data[b'labels']

# 进行中值滤波
filtered_images = []
for image in images:
    image_reshaped = image.reshape(3, 32, 32)  # 将图像形状调整为 3x32x32
    filtered_image = cv2.medianBlur(image_reshaped.transpose(1, 2, 0), 3)  # 进行中值滤波
    filtered_image_reshaped = filtered_image.transpose(2, 0, 1).reshape(-1)  # 将滤波后的图像形状调整回原来的形状
    filtered_images.append(filtered_image_reshaped)

# 创建新的数据集 data_batch_1_medianfiltering，并保存中值滤波后的图像数据和标签
new_data = {
    b'data': filtered_images,
    b'labels': labels
}

with open(new_data_file, 'wb') as f:
    pickle.dump(new_data, f)

print("新数据集已生成并保存在:", new_data_file)
