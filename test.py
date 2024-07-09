from PIL import Image, ImageOps

image_path = "./test/overlay_IMG_20230608_091558.jpg"
save_path ="./test/padding_test.jpg"

# 打开图像
image = Image.open(image_path)

# 定义要添加的边框大小（左、上、右、下）
border_size = (256, 256, 256, 256)

# 添加黑边
image_with_border = ImageOps.expand(image, border=border_size, fill='black')
print(image_with_border.size)
# 显示图像
image_with_border.save(save_path)
# 如果需要保存图像
# image_with_border.save('image_with_border.png')