#!/bin/bash

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 evaluate.py \
    --model_name loca_zero_shot_blueberry \
    --data_path /opt/data/private/data/blueberry \
    --model_path ./pretrained \
    --backbone resnet50 \
    --swav_backbone \
    --reduction 8 \
    --image_size 512 \
    --num_enc_layers 3 \
    --num_ope_iterative_steps 3 \
    --emb_dim 256 \
    --num_heads 8 \
    --kernel_dim 3 \
    --num_objects 3 \
    --pre_norm \
    --zero_shot
