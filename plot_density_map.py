import numpy as np
import cv2

def visualize_density_map(density_map_path, save_path, image_path):
    # 加载.npy文件(512, 512)
    dmap = np.load(density_map_path)

    # 和image叠加
    image = cv2.imread(image_path)
    height = image.shape[0]
    width = image.shape[1]

    density_map_bgr = np.zeros((dmap.shape[0], dmap.shape[1], 3), dtype=np.float32)
    # 归一化，映射到255
    min_pixel = dmap.min()
    max_pixel = dmap.max()
    density_map_normalized = (dmap - min_pixel) / (max_pixel - min_pixel)
    density_map_bgr[:, :, 2] = density_map_normalized * 255

    # 将密度图叠加到图像上
    alpha = 0.5  # 透明度
    density_map_bgr_resized = cv2.resize(density_map_bgr, (width, height))
    density_map_bgr_resized = density_map_bgr_resized.astype(image.dtype)

    alpha = 0.5
    overlay = cv2.addWeighted(image, 1 - alpha, density_map_bgr_resized, alpha, 0)

    cv2.imwrite(save_path, overlay)


# 密度图文件的路径
# density_map_path = '/opt/data/private/data/blueberry/gt_density_map_adaptive_1024_1024_blueberry/IMG_20230608_092156.npy'
# save_path = '/opt/data/private/data/blueberry/gt_density_map_adaptive_1024_1024_blueberry_plot/IMG_20230608_092156.png'
density_map_path = './results/predicts/predict_loca_sfe_crop_padding_128/mmexport1711167027826_2_1_734_9.168_15.000.npy'
image_path = '/opt/data/private/data/blueberry2_crop_padding_128/images/mmexport1711167027826_2_1_734.jpg'
save_path = './test/mmexport1711167027826_734_2_1.png'

# 调用可视化函数
visualize_density_map(density_map_path, save_path, image_path)
