import cv2
import numpy as np
import os

# images_dir = "/opt/data/private/data/blueberry2/images"
# save_dir = "/opt/data/private/data/blueberry2/images_mask"
images_dir = "./test/images"
save_dir = "./test/results"

# 获取文件夹下所有文件名
file_names = os.listdir(images_dir)

# 遍历文件名列表
for file_name in file_names:
    image_path = os.path.join(images_dir, file_name)
    # 载入图像
    image = cv2.imread(image_path)  # 替换为你的图像路径

    # 将彩色图像转换为YUV色彩空间
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # 应用直方图均衡化到Y通道
    image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
    # 将图像转换回BGR色彩空间
    equalized_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    
    # 1
    image_gray = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    # 预处理：阈值化
    _, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 形态学操作
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # 开运算，去除噪声
    sure_bg = cv2.dilate(opening, kernel, iterations=5)  # 膨胀，获取背景区域

    # 2
    # 将图像转换到HSV色彩空间
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # # 设定叶子的颜色范围（示例，根据实际情况调整）
    # lower_green = np.array([35, 50, 50])
    # upper_green = np.array([90, 255, 255])
    # # 根据颜色范围进行阈值化
    # leaf_mask = cv2.inRange(hsv, lower_green, upper_green)
    # # 进行形态学操作，去除噪声
    # kernel = np.ones((5, 5), np.uint8)
    # opening = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    # leaf_mask = cv2.dilate(opening, kernel, iterations=3)  # 膨胀，获取背景区域

    # 3
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    # # 预处理：阈值化
    # _, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # 预处理，强调叶子的形状特征（这里使用开运算）
    # kernel = np.ones((5, 5), np.uint8)
    # # 开运算
    # opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # sure_bg = cv2.dilate(opened, kernel, iterations=3)  # 膨胀，获取背景区域
    # # 闭运算
    # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # # 形态学梯度
    # gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    # # 顶帽运算
    # tophat = cv2.morphologyEx(image_gray, cv2.MORPH_TOPHAT, kernel)
    # # 黑帽运算
    # blackhat = cv2.morphologyEx(image_gray, cv2.MORPH_BLACKHAT, kernel)


    contours, _ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建空白掩码
    leaf_mask = np.zeros_like(image_gray, dtype=np.uint8)
    
    # 在掩码上绘制叶子轮廓
    cv2.drawContours(leaf_mask, contours, -1, (255), thickness=cv2.FILLED)

    # 在原始图像上mask掉叶子
    masked_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(leaf_mask))
         
    # 与背景区域的掩码进行按位与操作
    # masked_image = cv2.bitwise_and(image, image, mask=leaf_mask)

    save_path = os.path.join(save_dir, file_name)
    # 显示结果
    cv2.imwrite(save_path, equalized_image)
    print(file_name, "is done...")
