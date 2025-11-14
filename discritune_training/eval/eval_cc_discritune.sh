#!/bin/bash

# Conceptual Captions Test Set Evaluation Script - DiscriTune Epoch 10
# Uses Beam Search (beam_size=5) for better caption quality

echo "Starting DiscriTune Evaluation on Conceptual Captions (Epoch 10)..."
echo "Model: DiscriTune (Trained)"
echo "Decoding: Beam Search (beam_size=5)"
export CUDA_VISIBLE_DEVICES=0

python evaluation.py \
    --checkpoint ../outputs/discritune_coco_kaparthy/discritune_coco_epoch_5.pt \
    --prefix_length 10 \
    --data_dir ../data/conceptual_captions \
    --image_list_file ../data/conceptual_captions/conceptual_captions_test_valid.txt \
    --batch_size 100 \
    --beam_size 5 \
    --num_workers 4 \
    --output_dir ./eval_results_cc \
    --save_detailed \
    --output_file_name table1-DiscriTune-COCO-CC-kaparthy_5.json


