import cv2
import os

images_dir = "./test/images"
save_dir = "./test/results"

# 获取文件夹下所有文件名
file_names = os.listdir(images_dir)

# 遍历文件名列表
for file_name in file_names:
    image_path = os.path.join(images_dir, file_name)

    # 读取图像
    image = cv2.imread(image_path)

    # 预处理：边缘增强
    # 使用高斯滤波平滑图像
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # 使用拉普拉斯算子进行边缘增强
    edges = cv2.Laplacian(blurred, cv2.CV_8U)

    # 边缘检测
    # 使用Canny边缘检测算法
    canny = cv2.Canny(edges, 30, 150)

    # 轮廓查找
    contours, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    save_path = os.path.join(save_dir, file_name)
    # 显示结果
    cv2.imwrite(save_path, image)
    print(file_name, "is done...")
