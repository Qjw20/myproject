import cv2
import numpy as np
import os

images_dir = "/opt/data/private/data/blueberry2_crop_padding_32/images"
save_dir = "/opt/data/private/data/blueberry2_crop_padding_32/images_lut112"

# images_dir = "./test/images"
# save_dir = "./test/results"

# 获取文件夹下所有文件名
file_names = os.listdir(images_dir)

# 遍历文件名列表
for file_name in file_names:
    image_path = os.path.join(images_dir, file_name)
    # 载入图像
    image = cv2.imread(image_path)  # 替换为你的图像路径
    # 转换到Lab颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # 分离Lab通道
    L, a, b = cv2.split(lab)
    # print(a)
    # # 保存a通道值到文本文件
    # np.savetxt('./test/a_channel.txt', a, fmt='%d')

    # 创建查找表（LUT）进行颜色曲线调整
    def adjust_curve(value):
        # 这里定义了一个非线性变换，使绿色区域的对比度更明显
        if value < 120 and value > 112:
            return np.clip(value * 0.8, 0, 255).astype(np.uint8)
        else:
            return np.clip(value * 1.4, 0, 255).astype(np.uint8)

    # 应用查找表到绿色通道
    lut = np.array([adjust_curve(i) for i in range(256)], dtype=np.uint8)
    a_eq = cv2.LUT(a, lut)

    # # 创建CLAHE对象
    # clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(16, 16))
    # # 应用CLAHE到色调通道
    # a_eq = clahe.apply(a)


    # 合并调整后的Lab通道
    lab_eq = cv2.merge((L, a_eq, b))
    # 转换回BGR颜色空间
    enhanced_image = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)
   # 保存结果图像
    save_path = os.path.join(save_dir, file_name)
    cv2.imwrite(save_path, enhanced_image)
    print("save ", file_name)
