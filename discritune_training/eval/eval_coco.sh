#!/bin/bash

# COCO Test Set Evaluation Script
# Uses Beam Search (beam_size=5) for better caption quality

echo "Starting DiscriTune COCO Evaluation..."
echo "Decoding: Beam Search (beam_size=5)"
export CUDA_VISIBLE_DEVICES=0

python evaluation.py \
    --checkpoint /home/semlab/SEM/yemo/CLIP/DiscriTune-reproduce/discritune_training/checkpoints/coco_weights.pt \
    --prefix_length 10 \
    --data_dir ../data/coco \
    --image_list_file /home/semlab/SEM/yemo/CLIP/DiscriTune-reproduce/discritune_training/data/coco/karpathy_test.txt \
    --batch_size 100 \
    --beam_size 5 \
    --num_workers 4 \
    --output_dir ./eval_results_coco \
    --save_detailed \
    --output_file_name table1-ClipCap-COCO-COCO.json

echo ""
echo "Evaluation completed!"
echo "Results saved to eval_results_coco/"
