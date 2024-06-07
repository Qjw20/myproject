from models.loca import build_model
from utils.data_pyramid import FSC147Dataset
from utils.arg_parser import get_argparser

import argparse
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import time

import logging
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

import numpy as np

# 配置日志
logging.basicConfig(filename='log_eval_pyramid.txt', level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s')

images_num = 0

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义CNN模型结构
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x

class PyramidPooling(nn.Module):
    def __init__(self, scales, fusion_method='weighted'):
        super(PyramidPooling, self).__init__()
        self.scales = scales
        self.fusion_method = fusion_method

        # 初始化融合权重
        self.weights = nn.Parameter(torch.ones(len(scales)))

    def forward(self, x):
        feature_maps = []
        for scale in self.scales:
            scaled_x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            pooled_x = F.avg_pool2d(scaled_x, kernel_size=scaled_x.size()[2:])
            feature_maps.append(pooled_x)

        # 对特征图进行融合
        if self.fusion_method == 'weighted':
            fused_feature_map = self.weighted_fusion(feature_maps)
        elif self.fusion_method == 'concatenate':
            fused_feature_map = torch.cat(feature_maps, dim=1)
        else:
            raise ValueError("Invalid fusion method")

        return fused_feature_map

    def weighted_fusion(self, feature_maps):
        # 将特征图按权重加权求和
        weighted_sum = sum(w * f for w, f in zip(self.weights, feature_maps))
        return weighted_sum

@torch.no_grad()
def evaluate(args):

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
        global images_num
        images_num = len(test_loader)
        for i, (img, bboxes, density_map, imgname) in enumerate(test_loader):
            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)
            start_time = time.time()
            out, _ = model(img, bboxes)
            end_time = time.time()
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
            save_dir = "./predict_crop_e2"
            save_path = os.path.join(save_dir, imgname[0][:-4] + "_{:.3f}_{:.3f}".format(float(pred_count), float(gt_count)) + ".npy")
            np.save(save_path, out_np)

        dist.all_reduce_multigpu([ae])
        dist.all_reduce_multigpu([se])

        if rank == 0:
            logging.info(
                f"{split.capitalize()} set, "+
                f"MAE: {ae.item() / len(test):.2f}, " + 
                f"RMSE: {torch.sqrt(se / len(test)).item():.2f}"
            )

    dist.destroy_process_group()
    return total_val_inference_time, total_test_inference_time, total_val_images_num, total_test_images_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LOCA', parents=[get_argparser()])
    args = parser.parse_args()
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