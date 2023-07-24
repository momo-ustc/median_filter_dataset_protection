import cv2
import matplotlib.pyplot as plt
import numpy as np

# 示例函数
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict

# 打开cifar-10数据集文件目录
def unpickle(file):
    import pickle
    with open("E:/ustc/CIFAR/dataset/cifar-10-batches-py/"+file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#打开cifar-10文件的data_batch_1
data_batch=unpickle("data_batch_1")

# data_batch为字典，包含四个字典键：
# b'batch_label' 
# b'labels' 标签
# b'data'  图片像素值
# b'filenames'
data_batch
cifar_label=data_batch[b'labels']
cifar_data=data_batch[b'data']

#把字典的值转成array格式，方便操作
cifar_label=np.array(cifar_label)
print(cifar_label.shape)
cifar_data=np.array(cifar_data)
print(cifar_data.shape)

label_name=['airplane','automobile','brid','cat','deer','dog','frog','horse','ship','truck']

# 拿第2个图片的np矩阵举例，将rgb矩阵转换为可显示图片
image = cifar_data[1]
# 分离出r,g,b：3*1024
image = image.reshape(-1,1024)
r = image[0,:].reshape(32,32) #红色分量
g = image[1,:].reshape(32,32) #绿色分量
b = image[2,:].reshape(32,32) #蓝色分量

# 特别注意点：cv2模块可以接受numpy数组,需要注意的是将 0-255 归一化到 0-1 ！！！
# 因此,您应该在代码中除以255

img = np.zeros((32,32,3))
img[:,:,0]=r/255
img[:,:,1]=g/255
img[:,:,2]=b/255

plt.imshow(img)
