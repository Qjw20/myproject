import cv2
import numpy as np
import os

# images_dir = "/opt/data/private/data/blueberry2_crop_padding_32/images"
# save_dir = "/opt/data/private/data/blueberry2_crop_padding_32/images_ssim17"

images_dir = "./test/images"
save_dir = "./test/results"

# 获取文件夹下所有文件名
file_names = os.listdir(images_dir)

# 遍历文件名列表
for file_name in file_names:
    image_path = os.path.join(images_dir, file_name)
    # 载入图像
    image = cv2.imread(image_path)  # 替换为你的图像路径

    # 将RGB图像转换为灰度图像，因为拉普拉斯算子通常用于灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用拉普拉斯算子
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

    # 将拉普拉斯结果转换回uint8类型，以便显示
    abs_laplacian = np.abs(laplacian)
    abs_laplacian_8u = np.uint8(abs_laplacian)

    # 将灰度图转换回三个通道，模拟对RGB图像的每个通道进行处理
    laplacian_rgb = cv2.cvtColor(abs_laplacian_8u, cv2.COLOR_GRAY2BGR)

   # 保存结果图像
    save_path = os.path.join(save_dir, file_name)
    cv2.imwrite(save_path, laplacian_rgb)
    print("save ", file_name)
