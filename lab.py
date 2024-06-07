import cv2
import os


images_dir = "/opt/data/private/data/blueberry2/images"
save_dir = "/opt/data/private/data/blueberry2/images_lab"

# images_dir = "./test/images"
# save_dir = "./test/results"

# 获取文件夹下所有文件名
file_names = os.listdir(images_dir)

# 遍历文件名列表
for file_name in file_names:
    image_path = os.path.join(images_dir, file_name)
    # 载入图像
    image = cv2.imread(image_path)  # 替换为你的图像路径

    # 将RGB图像转换为Lab颜色空间
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # 分离Lab通道
    L, a, b = cv2.split(lab_image)

    # 对a和b通道进行调整，增强绿色差异
    # 通过增加对比度或拉伸其直方图可以增强差异
    a_adjusted = cv2.equalizeHist(a)
    # b_adjusted = cv2.equalizeHist(b)
    b_adjusted = b

    # 合并调整后的Lab通道
    adjusted_lab_image = cv2.merge((L, a_adjusted, b_adjusted))

    # 将调整后的Lab图像转换回RGB颜色空间
    adjusted_rgb_image = cv2.cvtColor(adjusted_lab_image, cv2.COLOR_Lab2BGR)


    # 标准化L通道 (0-100) 到 (0-255)
    L_normalized = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX)

    # 标准化a通道 (-128-127) 到 (0-255)
    a_normalized = cv2.normalize(a_adjusted, None, 0, 255, cv2.NORM_MINMAX)

    # 标准化b通道 (-128-127) 到 (0-255)
    b_normalized = cv2.normalize(b_adjusted, None, 0, 255, cv2.NORM_MINMAX)

    save_path = os.path.join(save_dir, file_name)
    cv2.imwrite(save_path, adjusted_rgb_image)

    # save_dir_l = os.path.join(save_dir, "h")
    # save_path = os.path.join(save_dir_l, file_name)
    # cv2.imwrite(save_path, L_normalized)

    # save_dir_a = os.path.join(save_dir, "s")
    # save_path = os.path.join(save_dir_a, file_name)
    # cv2.imwrite(save_path, a_normalized)

    # save_dir_b = os.path.join(save_dir, "v")
    # save_path = os.path.join(save_dir_b, file_name)
    # cv2.imwrite(save_path, b_normalized)

    print(file_name, "is done...")


