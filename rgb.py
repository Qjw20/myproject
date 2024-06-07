import numpy as np
import os
import cv2

def normalize_rgb(image):
    epslon = 1e-6  # 防止RGB值全为0时出现除零错误
    normalize = np.zeros_like(image, dtype=np.float32)
    sum_rgb = np.sum(image, axis=2) + epslon  # 计算每个像素RGB通道值之和
    normalize[:, :, 0] = image[:, :, 0] / sum_rgb  # 归一化红色通道
    normalize[:, :, 1] = image[:, :, 1] / sum_rgb  # 归一化绿色通道
    normalize[:, :, 2] = image[:, :, 2] / sum_rgb  # 归一化蓝色通道
    return normalize

# 示例用法
# image是输入的RGB图像，数据类型为numpy数组
# 调用normalize_rgb函数进行归一化处理
# normalized_image是归一化后的图像，数据类型为numpy数组
images_dir = "./test/images"
save_dir = "./test/results"

# 获取文件夹下所有文件名
file_names = os.listdir(images_dir)

# 遍历文件名列表
for file_name in file_names:
    image_path = os.path.join(images_dir, file_name)
    # 载入图像
    image = cv2.imread(image_path)  # 替换为你的图像路径

    normalized_image = normalize_rgb(image)
    # 将浮点数类型转换为整数类型（0-255）
    normalized_image = (normalized_image * 255).astype(np.uint8)

    save_path = os.path.join(save_dir, file_name)

    cv2.imwrite(save_path, normalized_image)
