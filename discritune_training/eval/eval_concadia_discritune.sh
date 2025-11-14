#!/bin/bash

# Concadia Test Set Evaluation Script - DiscriTune Epoch 10
# Uses Beam Search (beam_size=5) for better caption quality

echo "Starting DiscriTune Evaluation on Concadia (Epoch 10)..."
echo "Model: DiscriTune (Trained)"
echo "Decoding: Beam Search (beam_size=5)"
export CUDA_VISIBLE_DEVICES=0

python evaluation.py \
    --checkpoint ../outputs/discritune_coco_kaparthy/discritune_coco_epoch_5.pt \
    --prefix_length 10 \
    --data_dir ../data/concadia \
    --image_list_file ../data/concadia/test_images.txt \
    --batch_size 100 \
    --beam_size 5 \
    --num_workers 4 \
    --output_dir ./eval_results_concadia \
    --save_detailed \
    --output_file_name table1-DiscriTune-COCO-Concadia-kaparthy_5.json

