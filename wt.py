import cv2
import pywt
import os
 
def extract_texture_features(image):
    # 将图像转为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    # 进行小波变换
    coeffs = pywt.dwt2(gray_image, 'haar')
    cA, (cH, cV, cD) = coeffs
 
    # 提取纹理特征
    texture_features = {
        'approximation': cA,
        'horizontal_detail': cH,
        'vertical_detail': cV,
        'diagonal_detail': cD
    }
 
    return texture_features
 
images_dir = "./test/images"
save_dir = "./test/results"

# 获取文件夹下所有文件名
file_names = os.listdir(images_dir)

# 遍历文件名列表
for file_name in file_names:
    image_path = os.path.join(images_dir, file_name)
    # 载入图像
    image = cv2.imread(image_path)  # 替换为你的图像路径


    # 提取纹理特征
    texture_features = extract_texture_features(image)

    save_dir_a = os.path.join(save_dir, "a")
    save_path = os.path.join(save_dir_a, file_name)
    # Approximation
    cv2.imwrite(save_path, texture_features['approximation'])

    save_dir_h = os.path.join(save_dir, "h")
    save_path = os.path.join(save_dir_h, file_name) 
    # horizontal_detail
    cv2.imwrite(save_path, texture_features['horizontal_detail'])

    save_dir_v = os.path.join(save_dir, "v")
    save_path = os.path.join(save_dir_v, file_name)
    # vertical_detail
    cv2.imwrite(save_path, texture_features['vertical_detail'])

    save_dir_d = os.path.join(save_dir, "d")
    save_path = os.path.join(save_dir_d, file_name)
    # diagonal_detail
    cv2.imwrite(save_path, texture_features['diagonal_detail'])
