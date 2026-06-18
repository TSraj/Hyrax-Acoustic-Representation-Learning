#!/bin/bash
# Phase 2 - Stage 2: Run all model×dataset combinations
# This script runs zero-shot evaluation for all 5 models × 7 datasets = 35 combinations

# Exit on error
set -e

# Read models from config.yaml
MODELS=($(python -c "import yaml; config=yaml.safe_load(open('config/config.yaml')); print(' '.join(config['phase2']['models']))"))

# Datasets to evaluate
DATASETS=(
    "anuraset"
    "bengalese_finch"
    "macaque"
    "marmoset"
    "picidae"
    "wetlands_bird"
    "zebra_finch"
)

echo "============================================"
echo "Phase 2 - Stage 2: Zero-Shot Evaluation"
echo "============================================"
echo "Models: ${#MODELS[@]}"
echo "Datasets: ${#DATASETS[@]}"
echo "Total combinations: $((${#MODELS[@]} * ${#DATASETS[@]}))"
echo ""

# Counter
count=0
total=$((${#MODELS[@]} * ${#DATASETS[@]}))

# Run all combinations
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        count=$((count + 1))
        echo ""
        echo "[$count/$total] Running: $model × $dataset"
        echo "----------------------------------------"

        python scripts/phase2_02_zero_shot_per_dataset.py \
            --model "$model" \
            --dataset "$dataset"

        echo "✓ Complete: $model × $dataset"
    done
done

echo ""
echo "============================================"
echo "All combinations complete!"
echo "============================================"
echo "Results saved to: outputs/phase2/zero_shot/per_dataset/"
