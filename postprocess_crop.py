import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import json
import torch
import math

import logging

# 配置日志
logging.basicConfig(filename='./logs/log_eval_loca_crop_60_overlay.txt', level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s')

original_dir = "/opt/data/private/data/blueberry2"
original_train_val_test_path = os.path.join(original_dir, "train_val_test.json")
pred_dir = "./results/predicts/predict_loca_crop_60"
save_folder = "./results/plots/plot_prediction_loca_crop_60"

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 不存在，已创建成功。")
    else:
        print(f"文件夹 '{folder_path}' 已存在。")

create_folder_if_not_exists(save_folder)

ae = 0
se = 0
# 读取JSON文件
with open(original_train_val_test_path, 'r') as f:
    data = json.load(f)

# 获取测试集的文件名列表
test_imagenames = data['test']
val_imagenames = data['val']

split = "test"
imagenames = test_imagenames

# 遍历测试集文件名列表
for imagename in imagenames:
    # 在images文件夹中查找包含 imagename 的文件名
    pred_imagenames = []
    for pred_imagename in os.listdir(pred_dir):
        if imagename[:-4] in pred_imagename:
            pred_imagenames.append(pred_imagename)
    # 打印符合条件的文件名列表
    # print(pred_imagenames)
    # 在images文件夹中查找包含 imagename 的文件名
    pred_imagenames = []
    for pred_imagename in os.listdir(pred_dir):
        if imagename[:-4] in pred_imagename:
            pred_imagenames.append(pred_imagename)
    # 打印符合条件的文件名列表

    # 读取原图像
    original_image_path = os.path.join(original_dir, "images", imagename)
    original_image = cv2.imread(original_image_path)
    # 获取图像的尺寸
    width, height = original_image.shape[1], original_image.shape[0]
    # 定义有效尺寸
    crop_size = (1024, 800)
    new_h = crop_size[0] * int(height / crop_size[0])
    new_w = crop_size[1] * int(width / crop_size[1])
    # 定义裁剪尺寸

    density_map_bgr = np.zeros((new_h, new_w, 3), dtype=np.float32)

    pred_count = 0
    for pred_imagename in pred_imagenames:
        # 加载密度图
        density_map_path = os.path.join(pred_dir, pred_imagename)
        density_map = np.load(density_map_path)
        # 将密度地图中的负数值设为0
        density_map[density_map < 0] = 0
        # 把密度图放缩到crop大小
        # 把密度图放缩到crop大小
        original_sum = density_map.sum()
        print(original_sum)
        density_map_resized = cv2.resize(density_map, (crop_size[1], crop_size[0]))
        density_map_resized = density_map_resized / density_map_resized.sum() * original_sum
        # density_map_resized = cv2.resize(density_map, (crop_size[1], crop_size[0]))
        # 切割密度图
        parts = pred_imagename.split('_')
        imagename = '_'.join(parts[:-5])
        row = int(parts[-5])
        col = int(parts[-4])
        h_plus = float(parts[-3])
        h = int(h_plus / (h_plus + 1024) * 1024)
        density_map1 = density_map_resized[:-h, :]
        # 把密度图放缩到crop的大小
        # density_map1_resized = cv2.resize(density_map1, (crop_size[1], crop_size[0]))
        original_sum = density_map1.sum()
        print(original_sum)
        density_map1_resized = cv2.resize(density_map1, (crop_size[1], crop_size[0]))
        if original_sum > 0:     
            density_map1_resized = density_map1_resized / density_map1_resized.sum() * original_sum
        else:
            density_map1_resized = density_map1_resized * original_sum           

        print(np.sum(density_map1_resized))
        print("\n")
        pred_count += np.sum(density_map1_resized)

        # 拼接赋值，对有效尺寸进行拼接
        density_map_bgr[row * crop_size[0]:(row+1) * crop_size[0], col * crop_size[1]:(col+1) * crop_size[1], 2] = density_map1_resized

    # 归一化，映射到255
    min_pixel = density_map_bgr[:, :, 2].min()
    max_pixel = density_map_bgr[:, :, 2].max()
    density_map_normalized = (density_map_bgr[:, :, 2] - min_pixel) / (max_pixel - min_pixel)
    density_map_bgr[:, :, 2] = density_map_normalized * 255
    # 获取gt：到density_map中获取
    original_density_map = torch.from_numpy(np.load(os.path.join(
        original_dir,
        'gt_density_map_adaptive_512_512_blueberry',
        imagename + '.npy',
    ))).unsqueeze(0)
    gt_count = float(original_density_map.flatten(1).sum(dim=1))

    # 设置字体和文本大小
    h = original_image.shape[0]
    w = original_image.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 14

    # 将密度图叠加到图像上
    alpha = 0.5  # 透明度
    density_map_bgr_resized = cv2.resize(density_map_bgr, (width, height))
    density_map_bgr_resized = density_map_bgr_resized.astype(original_image.dtype)

    # 保存叠加后的密度图
    
    # 在图像上添加白色字体的计数结果
    cv2.putText(density_map_bgr_resized, f"GT Count: {gt_count:.3f}", (w-1000, h-100), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(density_map_bgr_resized, f"Pred Count: {pred_count:.3f}", (w-1000, h-200), font, font_scale, (255, 255, 255), thickness)
    overlay = cv2.addWeighted(original_image, 1 - alpha, density_map_bgr_resized, alpha, 0)

    # 保存叠加后的图像
    output_path = os.path.join(save_folder, 'overlay_' + imagename + ".jpg")
    cv2.imwrite(output_path, overlay)
    print(f"{imagename} 拼接完成...")

    # 计算评价指标
    ae += abs(gt_count - pred_count)
    se += (gt_count - pred_count) ** 2

logging.info(
    f"{split} set, "+
    f"MAE: {ae / len(imagenames):.2f}, " + 
    f"RMSE: {math.sqrt(se / len(imagenames)):.2f}"
)

    