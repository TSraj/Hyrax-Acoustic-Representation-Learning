#!/bin/bash
#SBATCH --job-name=phase3_part1
#SBATCH --output=logs/phase3_part1_%j.out
#SBATCH --error=logs/phase3_part1_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Phase 3 Part 1: Extract + Manifests + Zero-Shot Evaluation
# Steps 1-3 only

set -e  # Exit on error

echo "========================================"
echo "PHASE 3 - PART 1: ZERO-SHOT EVALUATION"
echo "Started: $(date)"
echo "========================================"

# Activate environment
source venv/bin/activate

# Create log directory
mkdir -p logs

# Models to evaluate
MODELS=("wav2vec2_base" "wav2vec2_base_960h" "hubert_base" "xls_r" "wavlm" "ecapa_tdnn")
TASKS=("species_id" "hyrax_id")

echo ""
echo "========================================"
echo "STEP 1 & 2: Extract Bouts + Create Manifests"
echo "========================================"
python scripts/phase3_01_extract_hyrax_bouts.py
python scripts/phase3_02_create_manifests.py

echo ""
echo "========================================"
echo "STEP 3: Zero-Shot Evaluation (All 6 Models × 2 Tasks = 12 runs)"
echo "========================================"

for task in "${TASKS[@]}"; do
    echo ""
    echo "--- Task: $task ---"
    for model in "${MODELS[@]}"; do
        echo "Running $model on $task..."
        python scripts/phase3_03_zero_shot_evaluation.py \
            --model "$model" \
            --task "$task"
    done
done

echo ""
echo "========================================"
echo "PART 1 COMPLETE"
echo "Finished: $(date)"
echo "========================================"
echo ""
echo "Zero-shot results saved to: outputs/phase3/zero_shot/"
echo ""
echo "Ready for Part 2: Model Selection + Fine-Tuning"
