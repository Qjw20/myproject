from models.loca_fb import build_model
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