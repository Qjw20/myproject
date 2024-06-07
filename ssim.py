import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim


images_dir = "/opt/data/private/data/blueberry2_crop_padding_32/images"
save_dir = "/opt/data/private/data/blueberry2_crop_padding_32/images_ssim0.4"

# 获取文件夹下所有文件名
file_names = os.listdir(images_dir)

# 遍历文件名列表
for file_name in file_names:
    image_path = os.path.join(images_dir, file_name)
    # 载入图像
    original_image = cv2.imread(image_path)  # 替换为你的图像路径

    # 将图像转换为灰度图像进行SSIM计算
    gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 生成参考图像（例如通过模糊处理灰度图像）
    reference_image = cv2.GaussianBlur(gray_original, (15, 15), 0)

    # 计算SSIM地图
    score, ssim_map = ssim(gray_original, reference_image, full=True)

    # 设置阈值，标识低相似性区域
    threshold = 0.6
    low_similarity_mask = ssim_map < threshold

    # 对每个颜色通道分别进行处理
    result_image = original_image.copy()
    for i in range(3):  # 对B, G, R三个通道分别处理
        enhanced_channel = cv2.equalizeHist(original_image[:, :, i])
        result_image[:, :, i][low_similarity_mask] = enhanced_channel[low_similarity_mask]

    # 保存结果图像
    save_path = os.path.join(save_dir, file_name)
    cv2.imwrite(save_path, result_image)
    print("save ", file_name)