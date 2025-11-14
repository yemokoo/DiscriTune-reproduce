#!/bin/bash
echo "DiscriTune training - COCO kaparthy train dataset"
export CUDA_VISIBLE_DEVICES=3
python discritune_train.py \
    --clipcap_checkpoint checkpoints/coco_weights.pt \
    --data_dir data/coco \
    --image_list_file data/coco/karpathy_train.txt \
    --dataset_name coco \
    --num_epochs 20 \
    --batch_size 100 \
    --learning_rate 1e-7 \
    --baseline_momentum 0.9 \
    --num_workers 4 \
    --output_dir outputs/discritune_coco \
    --save_every 5
echo "Training completed!"