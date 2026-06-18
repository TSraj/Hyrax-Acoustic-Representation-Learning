#!/bin/bash
# Phase 2 - Stage 3: Run pooled evaluation for all 5 models

set -e

# Read models from config.yaml
MODELS=($(python -c "import yaml; config=yaml.safe_load(open('config/config.yaml')); print(' '.join(config['phase2']['models']))"))

echo "============================================"
echo "Phase 2 - Stage 3: Pooled Zero-Shot"
echo "============================================"
echo "Models: ${#MODELS[@]}"
echo "Testing: All datasets combined (~100+ classes)"
echo ""

count=0
total=${#MODELS[@]}

for model in "${MODELS[@]}"; do
    count=$((count + 1))
    echo ""
    echo "[$count/$total] Running pooled evaluation: $model"
    echo "----------------------------------------"

    python scripts/phase2_03_zero_shot_pooled.py --model "$model"

    echo "✓ Complete: $model"
done

echo ""
echo "============================================"
echo "All pooled evaluations complete!"
echo "============================================"
echo "Results saved to: outputs/phase2/zero_shot/pooled/"
