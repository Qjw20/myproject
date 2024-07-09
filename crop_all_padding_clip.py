import os
import json
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import torchvision.transforms.functional as TF
 


# 只裁剪train和val
# 源文件夹和目标文件夹路径 
source_folder = "/opt/data/private/data/blueberry2"
target_folder = "/opt/data/private/data/blueberry2_crop_padding_32_clipcount"


image_source_folder = os.path.join(source_folder, "images")
image_target_folder = os.path.join(target_folder, "images")
output_json_path = os.path.join(target_folder, "annotation.json")
images_class_file_path = os.path.join(target_folder, "images_class.txt")

# 创建目标文件夹
os.makedirs(target_folder, exist_ok=True)
os.makedirs(image_target_folder, exist_ok=True)

# 裁剪图像
# 读取 train_val_test.json 文件
train_val_test_path = os.path.join(source_folder, "train_val_test.json")
with open(train_val_test_path, "r") as f:
    data = json.load(f)

# 读取annotation.json
annotation_path = os.path.join(source_folder, "annotation.json")
with open(annotation_path, "r") as f:
    annotations = json.load(f)

# 获取 train 和 val 部分对应的图像名称列表
train_images = data["train"]
val_images = data["val"]
test_images = data["test"]

train_val_test_images = train_images + val_images + test_images
data = {}
new_train_val_test_images = {}
new_train_images = []
new_val_images = []
new_test_images = []
contents = []


# 对 train 部分的图像执行裁剪操作
for image_name in train_val_test_images:
    annotation = annotations[image_name]
    points = annotation["points"]
    box_examples_coordinates = annotation["box_examples_coordinates"]

    image_path = os.path.join(image_source_folder, image_name)
    output_json_path = os.path.join(target_folder, "annotation.json")
    output_train_val_json_path = os.path.join(target_folder, "train_val_test.json")

    print(image_path, "正在裁剪...")

    # 打开图像
    image = Image.open(image_path)
    # 获取图像的尺寸
    width, height = image.size
    # 定义裁剪尺寸
    crop_size_before_padding = (1024, 800)
    new_h_before_padding = crop_size_before_padding[0] * int(height / crop_size_before_padding[0])
    new_w_before_padding = crop_size_before_padding[1] * int(width / crop_size_before_padding[1])
    # resize image
    resized_image_before_padding = transforms.Resize((new_h_before_padding, new_w_before_padding))(image)

    # padding，更新resized_image new_h new_w 裁剪尺寸 滑动窗口步长(就是crop_size_before_padding)
    padding_length = 32
    border_size = (padding_length, padding_length, padding_length, padding_length)

    resized_image = ImageOps.expand(resized_image_before_padding, border=border_size, fill='black')
    new_h = new_h_before_padding + padding_length * 2
    new_w = new_w_before_padding + padding_length * 2
    crop_size = (crop_size_before_padding[0] + padding_length * 2, crop_size_before_padding[1] + padding_length * 2)

    # point box的放缩原则：先按照new_h_before_padding new_w_before_padding防缩，再都加上padding的大小
    # resize points
    w_scale = new_w_before_padding / width
    h_scale = new_h_before_padding / height

    img_res = np.array([w_scale, h_scale], dtype=np.float32)    
    points = np.array(points, dtype=np.float32)
    points = points * img_res + padding_length
    
    # 计算裁剪后子图的数量
    num_cols = new_w_before_padding // crop_size_before_padding[1]
    num_rows = new_h_before_padding // crop_size_before_padding[0]

    # 裁剪示例框
    exemplar_boxes_tensors = list()
    cnt = 0
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image.load()
    TTensor = transforms.Compose([
        transforms.ToTensor(),
    ])
    ToPILImage = transforms.ToPILImage()
    image = TTensor(image)
    for box in box_examples_coordinates:
        cnt += 1
        if cnt > 3:
            break
        x1 = int(box[0][0])
        y1 = int(box[0][1])
        x2 = int(box[2][0])
        y2 = int(box[2][1])
        bbox = image[:, y1:y2 + 1, x1:x2 + 1]
        exemplar_boxes_tensors.append(bbox)
    
    # 裁剪并保存子图
    for i in range(num_rows):
        for j in range(num_cols):
            box = (j * crop_size_before_padding[1], i * crop_size_before_padding[0], j * crop_size_before_padding[1] + crop_size[1], i * crop_size_before_padding[0] + crop_size[0])
            cropped_image = resized_image.crop(box)

            # 拼接示例框到cropped_image
            # 计算新图的高度和宽度
            cropped_image_tensor = TTensor(cropped_image)
            # c h w
            h_plus = max(box.shape[1] for box in exemplar_boxes_tensors)
            h_new = cropped_image_tensor.shape[1] + max(box.shape[1] for box in exemplar_boxes_tensors)
            w_new = max(cropped_image_tensor.shape[2], sum(box.shape[2] for box in exemplar_boxes_tensors))

            # 构建新图的张量
            new_image = torch.zeros((3, h_new, w_new), dtype=torch.float32)
            # 将 cropped_image 拼接到新图上
            cropped_image_h, cropped_image_w = cropped_image_tensor.shape[1], cropped_image_tensor.shape[2]
            new_image[:, :cropped_image_h, :cropped_image_w] = cropped_image_tensor
            # 将 exemplar_boxes 拼接到新图上
            h_offset = cropped_image_h
            w_offset = 0
            exemplar_points = []
            exemplar_boxes = []
            for box in exemplar_boxes_tensors:
                box_h, box_w = box.shape[1], box.shape[2]
                # new_image[h_offset:h_offset+box_h, :box_w] = box
                # h_offset += box_h
                new_image[:, h_offset:h_offset+box_h, w_offset:w_offset+box_w] = box
                x_cen = w_offset + box_w/2
                y_cen = h_offset + box_h/2
                x_min = w_offset
                x_max = w_offset+box_w
                y_min = h_offset
                y_max = h_offset+box_h
                exemplar_points.append([x_cen, y_cen])
                exemplar_boxes.append([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])
                w_offset += box_w

            # 在points中添加示例框坐标
            # 裁剪points   
            cropped_points = torch.tensor(points) - torch.as_tensor([[j * crop_size_before_padding[1], i * crop_size_before_padding[0]]])
            keep = (cropped_points[:, 0] <= crop_size[1]) & (cropped_points[:, 1] <= crop_size[0]) & (cropped_points[:, 0] >= 0) & (cropped_points[:, 1] >= 0)
            cropped_points = cropped_points[keep]
            cropped_points = cropped_points.tolist() + exemplar_points
            
            # 拼接完再resize（记录scale）
            new_image = TF.to_pil_image(new_image)
            new_image = transforms.Resize((384, 384))(new_image)
            cropped_image_name = image_name[:-4] + f"_{i}_{j}_{h_plus}.jpg"
            save_path = os.path.join(image_target_folder, cropped_image_name)
            new_image.save(save_path)

            # 创建属于这个cropped_image的annotation
            annotation = {}
            annotation['H'] = 384
            annotation['W'] = 384
            annotation['img_path'] = save_path

            # resize points和示例框
            w_scale = 384 / w_new
            h_scale = 384 / h_new
            img_res = np.array([w_scale, h_scale], dtype=np.float32)    
            cropped_points = np.array(cropped_points, dtype=np.float32)
            cropped_points = cropped_points * img_res
            annotation['points'] = cropped_points.tolist()

            img_res = np.array([[w_scale, h_scale], [w_scale, h_scale], [w_scale, h_scale], [w_scale, h_scale]], dtype=np.float32)    
            exemplar_boxes = np.array(exemplar_boxes, dtype=np.float32)
            exemplar_boxes = exemplar_boxes * img_res
            annotation['box_examples_coordinates'] = exemplar_boxes.tolist()         
        
            data[cropped_image_name] = annotation
            # 创建新的train_val_test.json
            if image_name in train_images:
                new_train_images.append(cropped_image_name)
            elif image_name in val_images:
                new_val_images.append(cropped_image_name)
            else:
                new_test_images.append(cropped_image_name)

            # images_classes.txt
            contents.append(cropped_image_name + " blueberry\n")

            # # 保存裁剪图像
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1)
            # ax.imshow(new_image)
            # # print("bboxes", bboxes)
            # # 遍历注释并绘制伪标签框
            # for point in annotation['points']:
            # # 创建伪标签框
            #     ax.scatter(point[0], point[1], color='red', marker='o', s=5)

            # for index, ex_rect in enumerate(annotation['box_examples_coordinates']):
            # # 创建示例框
            #     import matplotlib.patches as patches
            #     # bbox[0]左上角x  bbox[1]左上角y  bbox[2]边界框宽w  bbox[3]边界框高h
            #     w = ex_rect[2][0] - ex_rect[0][0]
            #     h = ex_rect[3][1] - ex_rect[1][1]
            #     rect = patches.Rectangle((ex_rect[1][0], ex_rect[1][1]), w, h, linewidth=1, edgecolor='orange', facecolor='none')
            #     # w = ex_rect[2] - ex_rect[0]
            #     # h = ex_rect[3] - ex_rect[1]
            #     # rect = patches.Rectangle((ex_rect[0], ex_rect[1]), w, h, linewidth=1, edgecolor='orange', facecolor='none')
            #     # 将矩形框添加到轴上
            #     ax.add_patch(rect)
            # output_dir = "./crop_outputs"
            # output_path = f"{output_dir}/{cropped_image_name}"
            # plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0, dpi=300)

# 保存更新后的标注信息为新的 annotation.json 文件
with open(output_json_path, "w") as f:
    json.dump(data, f, indent=2)

new_train_val_test_images['train'] = new_train_images
new_train_val_test_images['val'] = new_val_images
new_train_val_test_images['test'] = new_test_images

# 保存更新后的标注信息为新的 annotation.json 文件
with open(output_train_val_json_path, "w") as f:
    json.dump(new_train_val_test_images, f, indent=2)

# 将内容写入文件
with open(images_class_file_path, "w") as file:
    for content in contents:
        file.write(content)