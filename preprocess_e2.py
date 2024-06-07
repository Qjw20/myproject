import cv2
import numpy as np
import os


image_dir = "/opt/data/private/data/blueberry2_crop_padding_32/images"
save_dir = "/opt/data/private/data/blueberry2_crop_padding_32/images_enhancement1.5_aftercrop"

# 观察hsv空间上的结果
# 遍历文件夹中的所有文件
# for imagename in os.listdir(image_dir):
#     # 获取文件的完整路径
#     image_path = os.path.join(image_dir, imagename)
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)   

#     # # 将彩色图像转换为YUV色彩空间
#     # image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
#     # # 应用直方图均衡化到Y通道
#     # image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
#     # # 将图像转换回BGR色彩空间
#     # equalized_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
       
#     # 先均衡化再提高饱和度
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

#     # 分割HSV的三个通道
#     hue, saturation, value = cv2.split(hsv_image)

#     # 增加饱和度，使对象和背景更明显
#     saturation_enhanced = cv2.add(saturation, -100)  # 加100，饱和度范围0-255
#     saturation_enhanced = np.clip(saturation_enhanced, 0, 255)  # 防止超出范围

#     # 重新组合HSV通道
#     image_hsv_enhanced = cv2.merge([hue, saturation_enhanced, value])

#     # 转换回RGB以显示
#     image_rgb_enhanced = cv2.cvtColor(image_hsv_enhanced, cv2.COLOR_HSV2RGB)

#     save_path = os.path.join(save_dir, imagename)
#     cv2.imwrite(save_path, image_rgb_enhanced)
#     print("save ", imagename)



# 遍历文件夹中的所有文件
for imagename in os.listdir(image_dir):
    # 获取文件的完整路径
    image_path = os.path.join(image_dir, imagename)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # # 将彩色图像转换为YUV色彩空间
    # image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # # 应用直方图均衡化到Y通道
    # image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
    # # 将图像转换回BGR色彩空间
    # equalized_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    # 先均衡化再提高饱和度
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # 随机生成饱和度调整值
    saturation_factor = 1.5
    # hue_factor = 4.0
    # 调整色调和饱和度
    # hsv_image[:,:,0] = np.clip(hsv_image[:,:,0] * hue_factor, 0, 179)  # 色调
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)    # 饱和度
    # 将图像从HSV色彩空间转换回RGB色彩空间
    enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    save_path = os.path.join(save_dir, imagename)
    cv2.imwrite(save_path, enhanced_image)
    print("save ", imagename)


# 对比度增强
# 随机生成对比度缩放因子
# a = 1.1
# b = 2.0
# for alpha in np.arange(a, b, 0.1):
#     # 调整图像的对比度
#     enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
#     save_path = os.path.join(save_dir, "duibidu({:.1f}).jpg".format(alpha))
#     cv2.imwrite(save_path, enhanced_image)


# # 亮度增强
# 随机生成亮度调整值
# a = -20.0
# b = 20.0
# for beta in np.arange(a, b, 4.0):
#     # 调整图像的亮度
#     enhanced_image = cv2.convertScaleAbs(image, alpha=1, beta=beta)
#     save_path = os.path.join(save_dir, "liangdu({:.1f}).jpg".format(beta))
#     cv2.imwrite(save_path, enhanced_image)


# # 饱和度增强    
# a = -10.0
# b = 12.0
# for saturation_factor in np.arange(a, b, 1.0):
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#     # 随机生成饱和度调整值
#     saturation_factor = np.random.uniform(a, b)
#     # 调整图像的饱和度
#     hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
#     # 将图像从HSV色彩空间转换回RGB色彩空间
#     enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
#     save_path = os.path.join(save_dir, "baohedu({:.1f}).jpg".format(saturation_factor))
#     cv2.imwrite(save_path, enhanced_image)


# 应用直方图均衡化
# 读取彩色图像
# image = cv2.imread(image_path)
# # 将彩色图像转换为YUV色彩空间
# image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
# # 应用直方图均衡化到Y通道
# image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
# # 将图像转换回BGR色彩空间
# equalized_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
# # 先均衡化再灰度变换
# def map_grayscale_to_color_channels(gray_image):
#     # 定义映射函数，根据像素的灰度值映射到不同的彩色通道
#     red_channel = gray_image.copy()
#     green_channel = gray_image.copy()
#     blue_channel = gray_image.copy()

#     # 在这里定义你的映射函数
#     # 这里简单地示范了将灰度值分成三个区间映射到RGB三个通道上
#     red_channel[gray_image < 230] = 0  # 灰度值小于85的像素映射到红色通道 灰绿色
#     green_channel[(gray_image >= 200) & (gray_image < 200)] = 0  # 灰度值在85到170之间的像素映射到绿色通道 紫色
#     blue_channel[gray_image >= 230] = 0  # 灰度值大于等于170的像素映射到蓝色通道 亮黄色

#     # 将三个通道合成为彩色图像
#     colored_image = cv2.merge([blue_channel, green_channel, red_channel])

#     return colored_image

# # 转换为灰度图像
# gray_image = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2GRAY)
# # 映射灰度图像到彩色通道
# colored_image = map_grayscale_to_color_channels(gray_image)
# # 保存增强后的图像
# save_path = os.path.join(save_dir, "zhifangtu_huidubianhuan.jpg")
# cv2.imwrite(save_path, colored_image)
# # 先均衡化再提高饱和度
# hsv_image = cv2.cvtColor(equalized_image, cv2.COLOR_RGB2HSV)
# # 随机生成饱和度调整值
# saturation_factor = 2.0
# hue_factor = 4.0
# # 调整色调和饱和度
# hsv_image[:,:,0] = np.clip(hsv_image[:,:,0] * hue_factor, 0, 179)  # 色调
# # hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)    # 饱和度
# # 将图像从HSV色彩空间转换回RGB色彩空间
# enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
# save_path = os.path.join(save_dir, "zhifangtu_sediao({:.1f}).jpg".format(hue_factor))
# cv2.imwrite(save_path, enhanced_image)


# 边缘增强
# 读取彩色图像
# image = cv2.imread(image_path)
# # 将彩色图像转换为灰度图像
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# for a in range(100, 200, 20):
#     for b in range(200, 100, -20):
#         if a < b:
#             # 应用Canny边缘检测算法
#             edges = cv2.Canny(gray_image, a, b)
#             # 将边缘图像转换为彩色图像
#             edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#             # 将边缘图像与原始彩色图像叠加
#             enhanced_image = cv2.addWeighted(image, 1, edges_colored, 1, 0)
#             # 保存增强后的图像
#             save_path = os.path.join(save_dir, "bianyuan({:.1f}, {:.1f}).jpg".format(a, b))
#             cv2.imwrite(save_path, enhanced_image)


# 灰度分层法
# def map_color_to_blue_red(gray_image):
#     # 灰度图映射到深蓝色-红色区间
#     colormap = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
#     return colormap

# # 读取彩色图像
# color_image = cv2.imread(image_path)
# # 转换为灰度图像
# gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
# # 将灰度图像映射到深蓝色-红色区间
# enhanced_image = map_color_to_blue_red(gray_image)
# # 保存增强后的图像
# save_path = os.path.join(save_dir, "huidu.jpg")
# cv2.imwrite(save_path, gray_image)
# # 保存增强后的图像
# save_path = os.path.join(save_dir, "huidufenceng.jpg")
# cv2.imwrite(save_path, enhanced_image)


# 灰度变换法
# def map_grayscale_to_color_channels(gray_image):
#     # 定义映射函数，根据像素的灰度值映射到不同的彩色通道
#     red_channel = gray_image.copy()
#     green_channel = gray_image.copy()
#     blue_channel = gray_image.copy()

#     # 在这里定义你的映射函数
#     # 这里简单地示范了将灰度值分成三个区间映射到RGB三个通道上
#     red_channel[gray_image < 120] = 0  # 灰度值小于85的像素映射到红色通道 灰绿色
#     green_channel[(gray_image >= 120) & (gray_image < 180)] = 0  # 灰度值在85到170之间的像素映射到绿色通道 紫色
#     blue_channel[gray_image >= 180] = 0  # 灰度值大于等于170的像素映射到蓝色通道 亮黄色

#     # 将三个通道合成为彩色图像
#     colored_image = cv2.merge([blue_channel, green_channel, red_channel])

#     return colored_image

# # 读取彩色图像
# color_image = cv2.imread(image_path)
# # 转换为灰度图像
# gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
# # 映射灰度图像到彩色通道
# colored_image = map_grayscale_to_color_channels(gray_image)
# # 保存增强后的图像
# save_path = os.path.join(save_dir, "huidubianhuan.jpg")
# cv2.imwrite(save_path, colored_image)






