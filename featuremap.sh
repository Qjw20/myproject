CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 featuremap.py \
    --model_name loca_sfe2_few_shot_blueberry2_crop_padding_32_lut1.4 \
    --data_path /opt/data/private/data/blueberry2_crop_padding_32 \
    --model_path ./pretrained \
    --backbone resnet50 \
    --swav_backbone \
    --batch_size 1 \
    --reduction 8 \
    --image_size 512 \
    --num_enc_layers 3 \
    --num_ope_iterative_steps 3 \
    --emb_dim 256 \
    --num_heads 8 \
    --kernel_dim 3 \
    --num_objects 3 \
    --pre_norm \
    --log_path ./logs/eval/log_eval_loca_sfe2_crop_padding_32_lut1.4.txt \
    --save_path ./results/predicts/predict_loca_sfe2_crop_padding_32_lut1.4

