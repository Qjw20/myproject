from models.loca import build_model
from utils.data_pyramid import FSC147Dataset
from utils.arg_parser import get_argparser
from utils.losses import ObjectNormalizedL2Loss

from time import perf_counter
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random
from torchvision import transforms as T

import logging

torch.backends.cudnn.benchmark = True

# 配置日志
logging.basicConfig(filename='log_pyramid.txt', level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s')

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
writer = SummaryWriter()

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 输入通道数：3，输出通道数：8，卷积核大小：3x3，步长：1，填充：1
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        # 输入通道数：8，输出通道数：8，卷积核大小：3x3，步长：1，填充：1
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        # 输入通道数：16，输出通道数：8，卷积核大小：3x3，步长：1，填充：1
        self.conv3 = nn.Conv2d(8, 3, kernel_size=3, padding=1)

        # 最大池化层，池化核大小：2x2，步长：2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 输入形状：[3, 3072, 3072]
        x = F.relu(self.conv1(x))
        # 输出形状：[8, 3072, 3072]
        x = self.pool(x)
        # 输出形状：[8, 1536, 1536]
        x = F.relu(self.conv2(x))
        # 输出形状：[8, 1536, 1536]
        x = self.pool(x)
        # 输出形状：[8, 768, 768]
        x = F.relu(self.conv3(x))
        # 输出形状：[3, 768, 768]
        x = self.pool(x)
        # 输出形状：[3, 384, 384]
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

def train(args):

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

    backbone_params = dict()
    non_backbone_params = dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in n:
            backbone_params[n] = p
        else:
            non_backbone_params[n] = p

    optimizer = torch.optim.AdamW(
        [
            {'params': non_backbone_params.values()},
            {'params': backbone_params.values(), 'lr': args.backbone_lr}
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.25)
    if args.resume_training:
        checkpoint = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best = checkpoint['best_val_ae']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0
        best = 10000000000000

    criterion = ObjectNormalizedL2Loss()
    train = FSC147Dataset(
        args.data_path,
        args.image_size,
        split='train',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot
    )
    val = FSC147Dataset(
        args.data_path,
        args.image_size,
        split='val',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p
    )
    train_loader = DataLoader(
        train,
        sampler=DistributedSampler(train),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val,
        sampler=DistributedSampler(val),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers
    )

    print(rank)

    scales = [0.25, 0.5, 0.75, 1.0]
    cnn = CNN().to(device)
    pyramid_pooling = PyramidPooling(scales).to(device)
    
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if rank == 0:
            start = perf_counter()
        train_loss = torch.tensor(0.0).to(device)
        val_loss = torch.tensor(0.0).to(device)
        aux_train_loss = torch.tensor(0.0).to(device)
        aux_val_loss = torch.tensor(0.0).to(device)
        train_ae = torch.tensor(0.0).to(device)
        val_ae = torch.tensor(0.0).to(device)

        train_loader.sampler.set_epoch(epoch)
        model.train()
        
        for img, bboxes, density_map, imgname in train_loader:
            # img大小都是3072✖️3072
            # result_feature_maps = []            
            # with torch.no_grad():
            #     for img1 in img:                
            #         # 使用CNN提取特征
            #         img1 = img1.unsqueeze(0).to(device)
            #         features = cnn(img1)
            #         print(features.shape)
            #         result_feature_maps.append(features)
            #         # 应用金字塔池化
            #         # features = features.to(device)
            #         # output = pyramid_pooling(features)
            #         # 输出的特征图形状为[1, 3, 512, 512]                   
            #         # print("output:", output.shape)
            #         # result_feature_maps.append(output)
            #         # 检查CUDA是否可用
            #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #         # 打印当前显存的占用情况
            #         print("6.当前显存占用：", torch.cuda.memory_allocated(device) / 1024**3, "GB")
            # result_feature_maps = torch.cat(result_feature_maps, dim=0)
            # img = result_feature_maps
            # # 打印最终特征图数组的形状
            # print("Final feature maps array shape of train set:", img.shape)
            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)
            # density_map形状
            # resize_py = T.Resize((384, 384))
            # original_sum = density_map.sum()
            # density_map = resize_py(density_map)
            # density_map = density_map / density_map.sum() * original_sum
            
            optimizer.zero_grad()
            out, aux_out = model(img, bboxes)

            # obtain the number of objects in batch
            with torch.no_grad():
                num_objects = density_map.sum()
                dist.all_reduce_multigpu([num_objects])

            main_loss = criterion(out, density_map, num_objects)
            aux_loss = sum([
                args.aux_weight * criterion(aux, density_map, num_objects) for aux in aux_out
            ])
            loss = main_loss + aux_loss
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            train_loss += main_loss * img.size(0)
            aux_train_loss += aux_loss * img.size(0)
            train_ae += torch.abs(
                density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ).sum()

        model.eval()
        with torch.no_grad():
            for img, bboxes, density_map, imgname in val_loader:
                img = img.to(device)
                bboxes = bboxes.to(device)
                density_map = density_map.to(device)
                out, aux_out = model(img, bboxes)
                with torch.no_grad():
                    num_objects = density_map.sum()
                    dist.all_reduce_multigpu([num_objects])

                main_loss = criterion(out, density_map, num_objects)
                aux_loss = sum([
                    args.aux_weight * criterion(aux, density_map, num_objects) for aux in aux_out
                ])
                loss = main_loss + aux_loss

                val_loss += main_loss * img.size(0)
                aux_val_loss += aux_loss * img.size(0)
                val_ae += torch.abs(
                    density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
                ).sum()
        dist.all_reduce_multigpu([train_loss])
        dist.all_reduce_multigpu([val_loss])
        dist.all_reduce_multigpu([aux_train_loss])
        dist.all_reduce_multigpu([aux_val_loss])
        dist.all_reduce_multigpu([train_ae])
        dist.all_reduce_multigpu([val_ae])

        scheduler.step()

        if rank == 0:
            end = perf_counter()
            best_epoch = False
            if val_ae.item() / len(val) < best:
                best = val_ae.item() / len(val)
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_ae': val_ae.item() / len(val)
                }
                torch.save(
                    checkpoint,
                    os.path.join(args.model_path, f'{args.model_name}.pt')
                )
                best_epoch = True

            print(
                f"Epoch: {epoch}",
                f"Train loss: {train_loss.item():.3f}",
                f"Aux train loss: {aux_train_loss.item():.3f}",
                f"Val loss: {val_loss.item():.3f}",
                f"Aux val loss: {aux_val_loss.item():.3f}",
                f"Train MAE: {train_ae.item() / len(train):.3f}",
                f"Val MAE: {val_ae.item() / len(val):.3f}",
                f"Epoch time: {end - start:.3f} seconds",
                'best' if best_epoch else ''
            )

            logging.info(
                "Epoch: %d Train loss: %.3f Aux train loss: %.3f Val loss: %.3f Aux val loss: %.3f "
                "Train MAE: %.3f Val MAE: %.3f Epoch time: %.3f seconds %s",
                epoch, train_loss.item(), aux_train_loss.item(), val_loss.item(),
                aux_val_loss.item(), train_ae.item() / len(train), val_ae.item() / len(val),
                end - start, 'best' if best_epoch else ''
            )

            writer.add_scalar('train loss', train_loss.item(), epoch)
            writer.add_scalar('val loss', val_loss.item(), epoch)
            writer.add_scalar('train mae', train_ae.item() / len(train), epoch)
            writer.add_scalar('val mae', val_ae.item() / len(val), epoch)


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

    writer.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LOCA', parents=[get_argparser()])
    args = parser.parse_args()
    train(args)
