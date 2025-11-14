#!/bin/bash

# COCO Test Set Evaluation Script
# Uses Beam Search (beam_size=5) for better caption quality

echo "Starting DiscriTune COCO Evaluation..."
echo "Decoding: Beam Search (beam_size=5)"
export CUDA_VISIBLE_DEVICES=0

python evaluation.py \
    --checkpoint ../outputs/discritune_coco_kaparthy/discritune_coco_epoch_5.pt \
    --prefix_length 10 \
    --data_dir ../data/nocaps \
    --image_list_file /home/semlab/SEM/yemo/CLIP/DiscriTune-reproduce/discritune_training/data/nocaps/nocaps_near_test.txt \
    --batch_size 100 \
    --beam_size 5 \
    --num_workers 4 \
    --output_dir ./eval_results_nocaps_near \
    --save_detailed \
    --output_file_name table1-DiscriTune-COCO-NoCaps-Near-kaparthy-5.json


