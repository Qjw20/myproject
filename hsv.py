import cv2
import numpy as np
import os


image_dir = "/opt/data/private/data/blueberry2/images"
save_dir = "/opt/data/private/data/blueberry2/images_gamma2"

def adjust_gamma(image, gamma=1.0):
    # 构建查找表，用于将每个像素值进行伽马校正
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    
    # 应用查找表对图像进行伽马校正
    return cv2.LUT(image, table)

# 观察hsv空间上的结果
# 遍历文件夹中的所有文件
for imagename in os.listdir(image_dir):
    # 获取文件的完整路径
    image_path = os.path.join(image_dir, imagename)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)   
   
    # 应用不同的伽马值进行校正
    # gamma_corrected_05 = adjust_gamma(image, gamma=0.5)
    # gamma_corrected_10 = adjust_gamma(image, gamma=1.0)
    gamma_corrected_20 = adjust_gamma(image, gamma=2.0)

    # save_dir_h = os.path.join(save_dir, "h")
    # save_path = os.path.join(save_dir_h, imagename)
    # cv2.imwrite(save_path, gamma_corrected_05)

    # save_dir_s = os.path.join(save_dir, "s")
    # save_path = os.path.join(save_dir_s, imagename)
    # cv2.imwrite(save_path, gamma_corrected_10)

    save_path = os.path.join(save_dir, imagename)
    cv2.imwrite(save_path, gamma_corrected_20)
    print("save ", imagename)






