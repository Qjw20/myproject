import os
import json
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import torchvision.transforms.functional as TF
 


# 只裁剪train和val
# 源文件夹和目标文件夹路径 
source_folder = "/opt/data/private/data/blueberry"
target_folder = "/opt/data/private/data/blueberry_crop_padding_64"

image_source_folder = os.path.join(source_folder, "images")

# 裁剪图像
# instances_val.json
instances_val_path = os.path.join(source_folder, "instances_val.json")
with open(instances_val_path, "r") as f:
    instances_val = json.load(f)

# instances_val.json
instances_test_path = os.path.join(source_folder, "instances_test.json")
with open(instances_test_path, "r") as f:
    instances_test = json.load(f)

# 读取annotation.json
annotation_path = os.path.join(source_folder, "annotation.json")
with open(annotation_path, "r") as f:
    annotations = json.load(f)

# 获取 train 和 val 部分对应的图像名称列表
val_images = [instance_val["file_name"] for instance_val in instances_val["images"]]
val_ids = [instance_val["id"] for instance_val in instances_val["images"]]
test_images = [instance_test["file_name"] for instance_test in instances_test["images"]]
test_ids = [instance_test["id"] for instance_test in instances_test["images"]]

val_annotations = instances_val["annotations"]
test_annotations = instances_test["annotations"]

new_val_annotations = []
new_image_annotations = []
new_id = 1
new_image_id = 1

# 对 val 部分的图像执行裁剪操作
for image_name, image_id in zip(val_images, val_ids):
    annotation = annotations[image_name]
    box_examples_coordinates = annotation["box_examples_coordinates"]
    bboxes = [val_annotation["bbox"] for val_annotation in val_annotations if val_annotation["image_id"] == image_id]
    # bboxes->xyxy
    xyxy_boxes = []
    for bbox in bboxes:
        x1, y1, w, h = bbox
        xyxy_boxes.append([x1, y1, x1 + w, y1 + h])

    image_path = os.path.join(image_source_folder, image_name)
    output_json_path = os.path.join(target_folder, "instances_val.json")

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
    padding_length = 128
    border_size = (padding_length, padding_length, padding_length, padding_length)

    resized_image = ImageOps.expand(resized_image_before_padding, border=border_size, fill='black')
    new_h = new_h_before_padding + padding_length * 2
    new_w = new_w_before_padding + padding_length * 2
    crop_size = (crop_size_before_padding[0] + padding_length * 2, crop_size_before_padding[1] + padding_length * 2)
    
    w_scale = new_w_before_padding / width
    h_scale = new_h_before_padding / height
    # 缩放bbox
    img_res = np.array([[w_scale, h_scale, w_scale, h_scale]], dtype=np.float32)    
    xyxy_boxes = np.array(xyxy_boxes, dtype=np.float32)
    xyxy_boxes = xyxy_boxes * img_res + padding_length
    xyxy_boxes = xyxy_boxes.tolist()  

    # 计算裁剪后子图的数量
    num_cols = new_w_before_padding // crop_size_before_padding[1]
    num_rows = new_h_before_padding // crop_size_before_padding[0]

    # 裁剪示例框，resize示例框
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
            cropped_image_name = image_name[:-4] + f"_{i}_{j}_{h_plus}.jpg"
            # 构建新图的张量
            new_image = torch.zeros((3, h_new, w_new), dtype=torch.float32)
            # 将 cropped_image 拼接到新图上
            cropped_image_h, cropped_image_w = cropped_image_tensor.shape[1], cropped_image_tensor.shape[2]
            new_image[:, :cropped_image_h, :cropped_image_w] = cropped_image_tensor
            # 将 exemplar_boxes 拼接到新图上
            h_offset = cropped_image_h
            w_offset = 0
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
                exemplar_boxes.append([x_min, y_min, x_max, y_max])
                w_offset += box_w

            # 在bbox中添加示例框坐标
            # 裁剪bbox            
            cropped_bbox = torch.tensor(xyxy_boxes) - torch.as_tensor([[j * crop_size_before_padding[1], i * crop_size_before_padding[0], j * crop_size_before_padding[1], i * crop_size_before_padding[0]]])
            # 限制横坐标在 0 到 512 之间
            cropped_bbox[:, 0] = cropped_bbox[:, 0].clamp(min=0, max=crop_size[1])
            cropped_bbox[:, 2] = cropped_bbox[:, 2].clamp(min=0, max=crop_size[1])
            # 限制纵坐标在 0 到 384 之间
            cropped_bbox[:, 1] = cropped_bbox[:, 1].clamp(min=0, max=crop_size[0])
            cropped_bbox[:, 3] = cropped_bbox[:, 3].clamp(min=0, max=crop_size[0])
            # remove elements for which the boxes or masks that have zero area
            keep = (cropped_bbox[:, 0] < cropped_bbox[:, 2]) & (cropped_bbox[:, 1] < cropped_bbox[:, 3])    
            cropped_bbox = cropped_bbox[keep]
            bboxes = cropped_bbox.tolist() + exemplar_boxes
            
            # 拼接完再resize（记录scale）
            new_image = TF.to_pil_image(new_image)
            new_image = transforms.Resize((crop_size[0], crop_size[1]))(new_image)

            # resize boxes
            w_scale = crop_size[1] / w_new
            h_scale = crop_size[0] / h_new
            img_res = np.array([[w_scale, h_scale, w_scale, h_scale]], dtype=np.float32)    
            bboxes = np.array(bboxes, dtype=np.float32)
            bboxes = bboxes * img_res
            bboxes = bboxes.tolist()  

            # 创建属于这个cropped_image的annotation
            for box in bboxes:
                w = box[2] - box[0]
                h = box[3] - box[1]
                x_min = box[0]
                y_min = box[1]
                annotation = {}
                annotation['iscrowd'] = 0
                annotation['image_id'] = new_image_id
                annotation['bbox'] = [x_min, y_min, w, h]
                annotation['segmentation'] = []
                annotation['ignore'] = 0
                annotation['category_id'] = 1
                annotation['id'] = new_id 
                annotation['area'] = w * h 
                new_val_annotations.append(annotation)
                new_id += 1

            # 创建属于这个cropped_image的image
            image_annotation = {}
            image_annotation["height"] = crop_size[0]
            image_annotation["width"] = crop_size[1]
            image_annotation["id"] = new_image_id
            image_annotation["file_name"] = cropped_image_name
            new_image_annotations.append(image_annotation)
            new_image_id += 1

            # 保存裁剪图像
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1)
            # ax.imshow(new_image)
            # # 遍历注释并绘制标签框
            # for box in bboxes:
            # # 创建框
            #     import matplotlib.patches as patches
            #     # bbox[0]左上角x  bbox[1]左上角y  bbox[2]边界框宽w  bbox[3]边界框高h
            #     w = box[2] - box[0]
            #     h = box[3] - box[1]
            #     x_min = box[0]
            #     y_min = box[1]
            #     rect = patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor='orange', facecolor='none')
            #     # 将矩形框添加到轴上
            #     ax.add_patch(rect)
            # output_dir = "./crop_outputs_2"
            # output_path = f"{output_dir}/{cropped_image_name}"
            # plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0, dpi=300)

# 对 val 部分的图像执行裁剪操作
# for image_name, image_id in zip(test_images, test_ids):
#     annotation = annotations[image_name]
#     box_examples_coordinates = annotation["box_examples_coordinates"]
#     bboxes = [test_annotation["bbox"] for test_annotation in test_annotations if test_annotation["image_id"] == image_id]
#     # bboxes->xyxy
#     xyxy_boxes = []
#     for bbox in bboxes:
#         x1, y1, w, h = bbox
#         xyxy_boxes.append([x1, y1, x1 + w, y1 + h])

#     image_path = os.path.join(image_source_folder, image_name)
#     output_json_path = os.path.join(target_folder, "instances_test.json")

#     print(image_path, "正在裁剪...")

#     # 打开图像
#     image = Image.open(image_path)
#     # 获取图像的尺寸
#     width, height = image.size
    
#     # 定义裁剪尺寸
#     crop_size_before_padding = (1024, 800)
#     new_h_before_padding = crop_size_before_padding[0] * int(height / crop_size_before_padding[0])
#     new_w_before_padding = crop_size_before_padding[1] * int(width / crop_size_before_padding[1])
#     # resize image
#     resized_image_before_padding = transforms.Resize((new_h_before_padding, new_w_before_padding))(image)

#     # padding，更新resized_image new_h new_w 裁剪尺寸 滑动窗口步长(就是crop_size_before_padding)
#     padding_length = 64
#     border_size = (padding_length, padding_length, padding_length, padding_length)

#     resized_image = ImageOps.expand(resized_image_before_padding, border=border_size, fill='black')
#     new_h = new_h_before_padding + padding_length * 2
#     new_w = new_w_before_padding + padding_length * 2
#     crop_size = (crop_size_before_padding[0] + padding_length * 2, crop_size_before_padding[1] + padding_length * 2)
    
#     w_scale = new_w_before_padding / width
#     h_scale = new_h_before_padding / height
#     # 缩放bbox
#     img_res = np.array([[w_scale, h_scale, w_scale, h_scale]], dtype=np.float32)    
#     xyxy_boxes = np.array(xyxy_boxes, dtype=np.float32)
#     xyxy_boxes = xyxy_boxes * img_res + padding_length
#     xyxy_boxes = xyxy_boxes.tolist()  

#     # 计算裁剪后子图的数量
#     num_cols = new_w_before_padding // crop_size_before_padding[1]
#     num_rows = new_h_before_padding // crop_size_before_padding[0]

#     # 裁剪示例框
#     exemplar_boxes_tensors = list()
#     cnt = 0
#     if image.mode == "RGBA":
#         image = image.convert("RGB")
#     image.load()
#     TTensor = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     ToPILImage = transforms.ToPILImage()
#     image = TTensor(image)
#     for box in box_examples_coordinates:
#         cnt += 1
#         if cnt > 3:
#             break
#         x1 = int(box[0][0])
#         y1 = int(box[0][1])
#         x2 = int(box[2][0])
#         y2 = int(box[2][1])
#         bbox = image[:, y1:y2 + 1, x1:x2 + 1]
#         exemplar_boxes_tensors.append(bbox)
    
#     # 裁剪并保存子图
#     for i in range(num_rows):
#         for j in range(num_cols):
#             box = (j * crop_size_before_padding[1], i * crop_size_before_padding[0], j * crop_size_before_padding[1] + crop_size[1], i * crop_size_before_padding[0] + crop_size[0])
#             cropped_image = resized_image.crop(box)

#             # 拼接示例框到cropped_image
#             # 计算新图的高度和宽度
#             cropped_image_tensor = TTensor(cropped_image)
#             # c h w
#             h_plus = max(box.shape[1] for box in exemplar_boxes_tensors)
#             h_new = cropped_image_tensor.shape[1] + max(box.shape[1] for box in exemplar_boxes_tensors)
#             w_new = max(cropped_image_tensor.shape[2], sum(box.shape[2] for box in exemplar_boxes_tensors))
#             cropped_image_name = image_name[:-4] + f"_{i}_{j}_{h_plus}.jpg"
#             # 构建新图的张量
#             new_image = torch.zeros((3, h_new, w_new), dtype=torch.float32)
#             # 将 cropped_image 拼接到新图上
#             cropped_image_h, cropped_image_w = cropped_image_tensor.shape[1], cropped_image_tensor.shape[2]
#             new_image[:, :cropped_image_h, :cropped_image_w] = cropped_image_tensor
#             # 将 exemplar_boxes 拼接到新图上
#             h_offset = cropped_image_h
#             w_offset = 0
#             exemplar_boxes = []
#             for box in exemplar_boxes_tensors:
#                 box_h, box_w = box.shape[1], box.shape[2]
#                 # new_image[h_offset:h_offset+box_h, :box_w] = box
#                 # h_offset += box_h
#                 new_image[:, h_offset:h_offset+box_h, w_offset:w_offset+box_w] = box
#                 x_cen = w_offset + box_w/2
#                 y_cen = h_offset + box_h/2
#                 x_min = w_offset
#                 x_max = w_offset+box_w
#                 y_min = h_offset
#                 y_max = h_offset+box_h
#                 exemplar_boxes.append([x_min, y_min, x_max, y_max])
#                 w_offset += box_w

#             # 在bbox中添加示例框坐标
#             # 裁剪bbox            
#             cropped_bbox = torch.tensor(xyxy_boxes) - torch.as_tensor([[j * crop_size_before_padding[1], i * crop_size_before_padding[0], j * crop_size_before_padding[1], i * crop_size_before_padding[0]]])
#             # 限制横坐标在 0 到 512 之间
#             cropped_bbox[:, 0] = cropped_bbox[:, 0].clamp(min=0, max=crop_size[1])
#             cropped_bbox[:, 2] = cropped_bbox[:, 2].clamp(min=0, max=crop_size[1])
#             # 限制纵坐标在 0 到 384 之间
#             cropped_bbox[:, 1] = cropped_bbox[:, 1].clamp(min=0, max=crop_size[0])
#             cropped_bbox[:, 3] = cropped_bbox[:, 3].clamp(min=0, max=crop_size[0])
#             # remove elements for which the boxes or masks that have zero area
#             keep = (cropped_bbox[:, 0] < cropped_bbox[:, 2]) & (cropped_bbox[:, 1] < cropped_bbox[:, 3])    
#             cropped_bbox = cropped_bbox[keep]
#             bboxes = cropped_bbox.tolist() + exemplar_boxes
            
#             # 拼接完再resize（记录scale）
#             new_image = TF.to_pil_image(new_image)
#             new_image = transforms.Resize((crop_size[0], crop_size[1]))(new_image)

#             # resize boxes
#             w_scale = crop_size[1] / w_new
#             h_scale = crop_size[0] / h_new
#             img_res = np.array([[w_scale, h_scale, w_scale, h_scale]], dtype=np.float32)    
#             bboxes = np.array(bboxes, dtype=np.float32)
#             bboxes = bboxes * img_res
#             bboxes = bboxes.tolist()  

#             # 创建属于这个cropped_image的annotation
#             for box in bboxes:
#                 w = box[2] - box[0]
#                 h = box[3] - box[1]
#                 x_min = box[0]
#                 y_min = box[1]
#                 annotation = {}
#                 annotation['iscrowd'] = 0
#                 annotation['image_id'] = new_image_id
#                 annotation['bbox'] = [x_min, y_min, w, h]
#                 annotation['segmentation'] = []
#                 annotation['ignore'] = 0
#                 annotation['category_id'] = 1
#                 annotation['id'] = new_id 
#                 annotation['area'] = w * h 
#                 new_val_annotations.append(annotation)
#                 new_id += 1

#             # 创建属于这个cropped_image的image
#             image_annotation = {}
#             image_annotation["height"] = crop_size[0]
#             image_annotation["width"] = crop_size[1]
#             image_annotation["id"] = new_image_id
#             image_annotation["file_name"] = cropped_image_name
#             new_image_annotations.append(image_annotation)
#             new_image_id += 1

#             # 保存裁剪图像
#             # import matplotlib.pyplot as plt
#             # fig, ax = plt.subplots(1)
#             # ax.imshow(new_image)
#             # # 遍历注释并绘制标签框
#             # for box in bboxes:
#             # # 创建框
#             #     import matplotlib.patches as patches
#             #     # bbox[0]左上角x  bbox[1]左上角y  bbox[2]边界框宽w  bbox[3]边界框高h
#             #     w = box[2] - box[0]
#             #     h = box[3] - box[1]
#             #     x_min = box[0]
#             #     y_min = box[1]
#             #     rect = patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor='orange', facecolor='none')
#             #     # 将矩形框添加到轴上
#             #     ax.add_patch(rect)
#             # output_dir = "./crop_outputs_2"
#             # output_path = f"{output_dir}/{cropped_image_name}"
#             # plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0, dpi=300)

data = {}
data["images"] = new_image_annotations
data["annotations"] = new_val_annotations
# 保存更新后的标注信息为新的 instance.json 文件
with open(output_json_path, "w") as f:
    json.dump(data, f, indent=2)
print(output_json_path + " 保存成功...")


