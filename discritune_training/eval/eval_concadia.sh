#!/bin/bash

# Concadia Test Set Evaluation Script - ClipCap Baseline
# Uses Beam Search (beam_size=5) for better caption quality

echo "Starting ClipCap Evaluation on Concadia..."
echo "Model: ClipCap (Pre-trained baseline)"
echo "Decoding: Beam Search (beam_size=5)"
export CUDA_VISIBLE_DEVICES=0

python evaluation.py \
    --checkpoint ../checkpoints/coco_weights.pt \
    --prefix_length 10 \
    --data_dir ../data/concadia \
    --image_list_file ../data/concadia/test_images.txt \
    --batch_size 100 \
    --beam_size 5 \
    --num_workers 4 \
    --output_dir ./eval_results_concadia \
    --save_detailed \
    --output_file_name table1-ClipCap-COCO-Concadia.json

echo ""
echo "ClipCap Evaluation on Concadia completed!"
echo "Results saved to eval_results_concadia/table1-ClipCap-COCO-Concadia.json"
