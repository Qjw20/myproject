#!/bin/bash
# 执行训练脚本
# ./train_few_shot.sh
# 执行评估脚本
./eval_few_shot.sh
# 执行后处理脚本
python postprocess_crop_padding.py
