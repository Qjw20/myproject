 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_sfe.py \
    --model_name loca_sfe2_few_shot_blueberry2_crop_padding_32_lut112 \
    --data_path /opt/data/private/data/blueberry2_crop_padding_32 \
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
    --epochs 60 \
    --lr 1e-4 \
    --backbone_lr 0 \
    --lr_drop 300 \
    --weight_decay 1e-4 \
    --batch_size 4 \
    --dropout 0.1 \
    --num_workers 8 \
    --max_grad_norm 0.1 \
    --aux_weight 0.3 \
    --tiling_p 0.5 \
    --save_path ./logs/train/log_loca_sfe2_crop_padding_32_lut112.txt \
    --pre_norm
