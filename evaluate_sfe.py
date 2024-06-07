from models.loca_sfe import build_model
from utils.data import FSC147Dataset
from utils.arg_parser import get_argparser

import argparse
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
import time

import logging
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image
import cv2


images_num = 0


@torch.no_grad()
def evaluate(args):
    # 配置日志
    logging.basicConfig(filename=args.log_path, level=logging.INFO,
                        format='%(asctime)s [%(levelname)s]: %(message)s')

    if 'SLURM_PROCID' in os.environ:
        world_size = int(os.environ['SLURM_NTASKS'])
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        print("Running on SLURM", world_size, rank, gpu)
    else:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        gpu = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank
    )

    model = DistributedDataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )
    state_dict = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))['model']
    state_dict = {k if 'module.' in k else 'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    total_val_inference_time = 0
    total_test_inference_time = 0
    total_val_images_num = 0
    total_test_images_num = 0
    for split in ['val', 'test']:
        test = FSC147Dataset(
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
        )
        test_loader = DataLoader(
            test,
            sampler=DistributedSampler(test),
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers
        )
        if split == 'val':
            total_val_images_num = len(test_loader)
        else:
            total_test_images_num = len(test_loader)
        ae = torch.tensor(0.0).to(device)
        se = torch.tensor(0.0).to(device)
        model.eval()

        # # start
        # layers = {
        #     'ope': model.module.ope
        # }

        # def save_feature_map_as_image(feature_map, save_path, imagename):
        #     """
        #     将特征图保存为图像文件。
            
        #     参数:
        #     - feature_map: torch.Tensor, 单个特征图
        #     - save_path: str, 保存图像文件的路径
        #     """
        #     feature_map = feature_map.cpu().numpy()
        #     feature_map = np.squeeze(feature_map)  # 去掉维度为1的维度

        #     # 假设 feature_map 的形状是 (C, H, W)
        #     if len(feature_map.shape) == 3:
        #         C, H, W = feature_map.shape
        #         for i in range(C):
        #             channel_image = feature_map[i]
        #             # Normalize to 0-255 and convert to uint8
        #             normalized_image = cv2.normalize(channel_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #             # Apply color map
        #             colored_image = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)
        #             image_save_path = os.path.join(save_path, f"{imagename[0][:-4]}_channel_{i}.png")
        #             cv2.imwrite(image_save_path, colored_image)
        #     elif len(feature_map.shape) == 2:
        #         # Handle the case where feature_map is already 2D
        #         normalized_image = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #         colored_image = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)
        #         image_save_path = os.path.join(save_path, f"{imagename[0][:-4]}.png")
        #         cv2.imwrite(image_save_path, colored_image)

        # def get_features(model, layers, data_loader):
        #     features = {}
            
        #     def hook(module, input, output, layer_name):
        #         if layer_name not in features:
        #             features[layer_name] = []
        #         features[layer_name].append(output.detach().cpu())
        #         print(f"Hook triggered for layer: {layer_name}, output shape: {output.shape}")


        #     hooks = []
        #     for layer_name, layer in layers.items():
        #         hooks.append(layer.register_forward_hook(lambda module, input, output, layer_name=layer_name: hook(module, input, output, layer_name)))
        #         print(f"Hook registered for layer: {layer_name}")

        #     model.eval()  # 确保模型在评估模式下
            
        #     with torch.no_grad():  # 禁用梯度计算
        #         for batch_idx, batch in enumerate(data_loader):
        #             images, bboxes, _, imagename = batch  # 假设数据加载器返回 (图像, 标签) 元组
        #             images = images.to(next(model.parameters()).device)  # 将图像移动到模型的设备上
        #             print(f"Processing batch with images shape: {images.shape}")
        #             model(images, bboxes)  # 前向传播

        #             # 保存每个 batch 的特征图
        #             for layer_name, layer_outputs in features.items():
        #                 for idx, feature_map in enumerate(layer_outputs):
        #                     # 二进制文件
        #                     # torch.save(feature_map, save_path)
        #                     save_feature_map_as_image(feature_map, args.save_path, imagename)

        #     for hook in hooks:
        #         hook.remove()
            
        #     # 将每一层的特征图列表转换为一个张量
        #     for layer_name in features:
        #         features[layer_name] = torch.cat(features[layer_name], dim=0)
            
        #     return features

        # # 假设 model, test_loader 和 layers 已定义
        # features = get_features(model, layers, test_loader)

        # # 现在 features 字典包含了每一层的特征图
        # for layer_name, feature in features.items():
        #     print(f"Layer: {layer_name}, Feature shape: {feature.shape}")
        
        # # end


        global images_num
        images_num = len(test_loader)
        for i, (img, bboxes, density_map, imgname) in enumerate(test_loader):
            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)
            start_time = time.time()
            out, _ = model(img, bboxes)
            end_time = time.time()
            # 密度图里不应该出现负数
            out[out < 0] = 0
            # 计算推理时间
            inference_time = end_time - start_time
            # 计算帧率（FPS）
            fps = 1 / inference_time
            logging.info(
                "inference time: %.4f s"
                "FPS: %.4f",
                 inference_time, fps
            )
            if split == 'val':
                total_val_inference_time += inference_time
            else:
                total_test_inference_time += inference_time
            ae += torch.abs(
                density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ).sum()
            se += ((
                density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ) ** 2).sum()

            out_np = out.squeeze().cpu().numpy()
            pred_count = out.flatten(1).sum(dim=1)
            gt_count = density_map.flatten(1).sum(dim=1)
            save_dir = args.save_path
            save_path = os.path.join(save_dir, imgname[0][:-4] + "_{:.3f}_{:.3f}".format(float(pred_count), float(gt_count)) + ".npy")
                    
            np.save(save_path, out_np)
            np.save(save_path, out_np)

        dist.all_reduce_multigpu([ae])
        dist.all_reduce_multigpu([se])

        if rank == 0:
            logging.info(
                f"{split.capitalize()} set, "+
                f"MAE: {ae.item() / len(test):.2f}, " + 
                f"RMSE: {torch.sqrt(se / len(test)).item():.2f}"
            )

        # 检查 GPU 是否可用
        if torch.cuda.is_available():
            # 获取当前设备上的 GPU 数量
            gpu_count = torch.cuda.device_count()
            # 遍历每个 GPU，输出显存信息
            for gpu_id in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(gpu_id)
                gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3)  # 转换为 MB
                # 获取最大显存占用（在程序执行期间的峰值，以字节为单位）
                max_memory = torch.cuda.max_memory_allocated(device)
                gpu_stats = f"GPU {gpu_id}: {gpu_name}, {max_memory / (1024 ** 3):.2f} / {gpu_memory:.2f} GB"
                print(gpu_stats)
        else:
            gpu_stats = "no GPU"
            print(gpu_stats)
        logging.info(gpu_stats)

    dist.destroy_process_group()
    return total_val_inference_time, total_test_inference_time, total_val_images_num, total_test_images_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LOCA', parents=[get_argparser()])
    args = parser.parse_args()
    def create_folder_if_not_exists(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"文件夹 '{folder_path}' 不存在，已创建成功。")
        else:
            print(f"文件夹 '{folder_path}' 已存在。")

    create_folder_if_not_exists(args.save_path)
    # evaluate(args)

    total_val_inference_time, total_test_inference_time, total_val_images_num, total_test_images_num = evaluate(args)

    # 计算推理时间
    val_inference_time_average = total_val_inference_time / total_val_images_num
    test_inference_time_average = total_test_inference_time / total_test_images_num
 
    # 计算帧率（FPS）
    fps_val = 1 / val_inference_time_average
    fps_test = 1 / test_inference_time_average
    logging.info(
        "Val: total images: %d "
        "total inference time: %.4f s"
        "average inference time: %.4f s"
        "average FPS: %.4f"
        "Test: total images: %d "
        "total inference time: %.4f s"
        "average inference time: %.4f s"
        "average FPS: %.4f",
        total_val_images_num, total_val_inference_time, val_inference_time_average, fps_val,
        total_test_images_num, total_test_inference_time, test_inference_time_average, fps_test,
    )