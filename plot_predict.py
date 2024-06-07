import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# 获取原图
# 文件夹路径和文件名
image_folder = '/opt/data/private/data/blueberry2/images'
density_folder = './results/predicts/predict_loca'
save_folder = "./results/plots/plot_prediction_loca"

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 不存在，已创建成功。")
    else:
        print(f"文件夹 '{folder_path}' 已存在。")

create_folder_if_not_exists(save_folder)

density_map_names = os.listdir(density_folder)

for density_map_name in density_map_names:
    # 分割出来文件名，pred_count，gt_count
    # 使用 '_' 分割文件名
    parts = density_map_name.split('_')
    # 提取 imagename、pred_count 和 gt_count
    imagename = '_'.join(parts[:-2])
    pred_count = float(parts[-2])
    gt_count = float(parts[-1][:-4])
    print("Image Name:", imagename)
    print("Pred Count:", pred_count)
    print("GT Count:", gt_count)
    image_name = imagename + ".jpg"  # 假设要读取的图像文件名
    # 读取图像
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    # print(image.shape)
    # 读取密度图
    density_map_path = os.path.join(density_folder, density_map_name)
    density_map = np.load(density_map_path)
    # print(density_map.shape)
    # 调整密度图大小与图像相同
    density_map_resized = cv2.resize(density_map, (image.shape[1], image.shape[0]))
    # print(density_map_resized.shape)
    # 归一化
    density_map_normalized = (density_map_resized - density_map_resized.min()) / (density_map_resized.max() - density_map_resized.min())
    # 创建形状相同的全零 BGR 图像
    density_map_bgr = np.zeros((density_map_resized.shape[0], density_map_resized.shape[1], 3), dtype=np.uint8)
    # 将密度图的值赋给 BGR 图像的红色通道
    density_map_bgr[:, :, 2] = density_map_normalized * 255  # 红色通道索引为 2，绿色通道索引为 1，蓝色通道索引为 0
    # 设置字体和文本大小
    h = image.shape[0]
    w = image.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 14
    # 在图像上添加白色字体的计数结果
    cv2.putText(density_map_bgr, f"GT Count: {gt_count:.3f}", (w-1000, h-100), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(density_map_bgr, f"Pred Count: {pred_count:.3f}", (w-1000, h-200), font, font_scale, (255, 255, 255), thickness)
    
    # 将密度图叠加到图像上
    alpha = 0.5  # 透明度
    overlay = cv2.addWeighted(image, 1 - alpha, density_map_bgr, alpha, 0)

    # 保存叠加后的图像
    output_path = os.path.join(save_folder, 'overlay_' + image_name)
    cv2.imwrite(output_path, overlay)