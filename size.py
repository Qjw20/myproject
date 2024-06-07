import cv2
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

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = cv2.Canny(gray_image, 100, 200)

    # 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 根据轮廓特征筛选出叶子和果实
    leaf_contours = []
    fruit_contours = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        area = cv2.contourArea(approx)
        if len(approx) > 9:  # 如果是叶子，则边缘点数大于3
            leaf_contours.append(contour)
        else:  # 如果是果实，则边缘点数小于等于3
            fruit_contours.append(contour)

    # 绘制叶子和果实的轮廓
    leaf_image = cv2.drawContours(image.copy(), leaf_contours, -1, (0, 255, 0), 2)
    fruit_image = cv2.drawContours(image.copy(), fruit_contours, -1, (0, 0, 255), 2)

    save_path = os.path.join(save_dir, file_name[:-4] + "_leaf.jpg")
    cv2.imwrite(save_path, leaf_image)

    save_path = os.path.join(save_dir, file_name[:-4] + "_fruit.jpg")
    cv2.imwrite(save_path, fruit_image)

    print(file_name, "is done...")

