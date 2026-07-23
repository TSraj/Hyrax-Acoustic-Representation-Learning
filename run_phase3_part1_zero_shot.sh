#!/bin/bash
#SBATCH --job-name=phase3_part1
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --output=logs/phase3_part1_%j.out
#SBATCH --error=logs/phase3_part1_%j.err

# Phase 3 Part 1: Extract + Manifests + Zero-Shot Evaluation
# Steps 1-3 only

set -e  # Exit on error

PROJECT_DIR="/home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning"
cd "$PROJECT_DIR"

echo "========================================"
echo "PHASE 3 - PART 1: ZERO-SHOT EVALUATION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================"

# Load modules
module load cuda/11.8.0
module load python/3.12-conda
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
