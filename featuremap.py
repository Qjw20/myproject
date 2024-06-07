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


def get_features(model, layers, data_loader):
    features = {}
    
    def hook(module, input, output, layer_name):
        if layer_name not in features:
            features[layer_name] = []
        features[layer_name].append(output.detach().cpu())

    hooks = []
    for layer_name, layer in layers.items():
        hooks.append(layer.register_forward_hook(lambda module, input, output, layer_name=layer_name: hook(module, input, output, layer_name)))
    
    model.eval()  # 确保模型在评估模式下
    
    with torch.no_grad():  # 禁用梯度计算
        for batch in data_loader:
            images, _ = batch  # 假设数据加载器返回 (图像, 标签) 元组
            images = images.to(next(model.parameters()).device)  # 将图像移动到模型的设备上
            model(images)  # 前向传播
            
    for hook in hooks:
        hook.remove()
    
    # 将每一层的特征图列表转换为一个张量
    for layer_name in features:
        features[layer_name] = torch.cat(features[layer_name], dim=0)
    
    return features


if __name__ == '__main__':


    parser = argparse.ArgumentParser('LOCA', parents=[get_argparser()])
    args = parser.parse_args()

    split = "test"
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

    model = DistributedDataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )
    state_dict = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))['model']
    state_dict = {k if 'module.' in k else 'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # 定义包含感兴趣层的字典
    layers = {
        'soo': model.ope.shape_or_objectness
    }

    # 假设 model, test_loader 和 layers 已定义
    features = get_features(model, layers, test_loader)

    # 现在 features 字典包含了每一层的特征图
    for layer_name, feature in features.items():
        print(f"Layer: {layer_name}, Feature shape: {feature.shape}")


