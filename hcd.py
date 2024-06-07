import cv2
import numpy as np
import os

images_dir = "./test/images"
save_dir = "./test/results"

# 获取文件夹下所有文件名
file_names = os.listdir(images_dir)

# 遍历文件名列表
for file_name in file_names:
    image_path = os.path.join(images_dir, file_name)
    # 载入图像
    image = cv2.imread(image_path)  # 替换为你的图像路径

    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 预处理图像（可以根据需要进行调整）
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)

    # 使用霍夫圆检测算法检测圆形果实
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=200, param2=20, minRadius=20, maxRadius=80)

    # 绘制检测到的圆形果实区域
    if circles is not None:
        circles = np.uint16(np.around(circles))
        mask = np.zeros(gray_image.shape, dtype=np.uint8)
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(mask, center, radius, (255), -1)

    # 创建背景掩码
    _, background_mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY_INV)

    # 将背景掩码应用到原始图像上
    background = cv2.bitwise_and(image, image, mask=background_mask)

    save_path = os.path.join(save_dir, file_name)
    # 显示结果
    cv2.imwrite(save_path, background)
    print(file_name, "is done...")
