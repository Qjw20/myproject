import cv2
import os

images_dir = "./test/images"
save_dir = "./test/results"

# 获取文件夹下所有文件名
file_names = os.listdir(images_dir)

# 遍历文件名列表
for file_name in file_names:
    image_path = os.path.join(images_dir, file_name)

    # 读取RGB图像
    rgb_image = cv2.imread(image_path)

    # 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)


    save_path = os.path.join(save_dir, file_name[:-4] + "_hsv_image.jpg")
    cv2.imwrite(save_path, hsv_image)


