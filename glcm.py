import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops


def calculate_glcm(image_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    glcm = graycomatrix(image_gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    return glcm

images_dir = "./test/images"
save_dir = "./test/results"

# 获取文件夹下所有文件名
file_names = os.listdir(images_dir)

# 遍历文件名列表
for file_name in file_names:
    image_path = os.path.join(images_dir, file_name)

    # 读取图像
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算灰度共生矩阵
    glcm = calculate_glcm(image_gray)

    # 从灰度共生矩阵中提取纹理特征
    contrast = graycoprops(glcm, 'contrast')
    energy = graycoprops(glcm, 'energy')
    homogeneity = graycoprops(glcm, 'homogeneity')

    # save_path = os.path.join(save_dir, file_name[:-4] + "_contrast.jpg")
    # cv2.imwrite(save_path, contrast)

    # save_path = os.path.join(save_dir, file_name[:-4] + "_energy.jpg")
    # cv2.imwrite(save_path, energy)

    # save_path = os.path.join(save_dir, file_name[:-4] + "_homogeneity.jpg")
    # cv2.imwrite(save_path, homogeneity)

    print("Contrast:", contrast)
    print("Energy:", energy)
    print("Homogeneity:", homogeneity)

    print(file_name, "is done...")



