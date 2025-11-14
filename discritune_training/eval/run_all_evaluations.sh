#!/bin/bash

# Run all evaluations - Simple and modular format
# Add or remove tasks easily by modifying the scripts array

echo "=========================================="
echo "Running All Evaluations"
echo "=========================================="
echo ""

# Define evaluation scripts
scripts=(
    #"eval_coco.sh"
    #"eval_coco_discritune.sh"
    #"eval_concadia.sh"
    "eval_concadia_discritune.sh"
    #"eval_cc.sh"
    "eval_cc_discritune.sh"
    #"eval_flickr.sh"
    "eval_flickr_discritune.sh"
    #"eval_nocaps_near.sh"
    "eval_nocaps_near_discritune.sh"
    #"eval_nocaps_out.sh"
    "eval_nocaps_out_discritune.sh"
)

# Run all scripts
total=${#scripts[@]}
for i in "${!scripts[@]}"; do
    script="${scripts[$i]}"
    echo ""
    echo "[$((i+1))/$total] Running $script..."
    echo "=========================================="
    bash "$script"
done

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
echo ""
echo "View results:"
echo "  python compare_results.py"
echo ""
