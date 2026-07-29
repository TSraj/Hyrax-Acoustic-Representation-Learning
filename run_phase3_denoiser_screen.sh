#!/bin/bash
#SBATCH --job-name=phase3_denoiser_screen
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --output=logs/phase3_denoiser_screen_%j.out
#SBATCH --error=logs/phase3_denoiser_screen_%j.err

# Phase 3 - Denoiser Screen
#
# Screening experiment (NOT the full pipeline): decide which audio signal
# version to fine-tune on.
#
#   Versions : Original (Audio/), BIODA (BIODA/denoised/), ACA (ACA/)
#   Model    : XLS-R only, frozen encoder + mean-pool + linear head
#   Task     : 8-individual hyrax ID (R3, Q7, P1, P8, Kashtan, O7, M9, U7)
#   Protocols: within-session (random 80/20 bouts) + session-holdout
#   Windowing: 5.0s / 2.5s stride
#
# Everything except the source audio folder is held fixed: same GTLabels
# boundaries, same bouts, same junk-session exclusions, same held-out sessions,
# same windowing, same classifier, same seed (42).

set -e

PROJECT_DIR="/home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning"
cd "$PROJECT_DIR"

echo "========================================"
echo "PHASE 3 - DENOISER SCREEN"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================"

module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate

mkdir -p logs

SCREEN_ROOT="outputs/phase3/denoiser_screen"
MODEL="xls_r"
VERSIONS=("original" "bioda" "aca")
TASKS=("hyrax_id_within_session" "hyrax_id_session_holdout")

echo ""
echo "========================================"
echo "STEP 1: Build manifests (one per audio version)"
echo "========================================"
for version in "${VERSIONS[@]}"; do
    echo ""
    echo "--- Manifests: $version ---"
    python scripts/phase3_02_create_manifests.py \
        --audio-source "$version" \
        --tasks session_screen \
        --output-dir "$SCREEN_ROOT/manifests/$version"
done

echo ""
echo "========================================"
echo "STEP 2: Verify all three versions use identical data"
echo "========================================"
# Hard gate: if bouts/individuals/sessions differ, the comparison is not clean.
python scripts/phase3_09_denoiser_screen.py --screen-root "$SCREEN_ROOT" --check-only

echo ""
echo "========================================"
echo "STEP 3: Zero-shot evaluation (3 versions x 2 protocols = 6 runs)"
echo "========================================"
for version in "${VERSIONS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo ""
        echo "--- $version / $task ---"
        python scripts/phase3_03_zero_shot_evaluation.py \
            --model "$MODEL" \
            --task "$task" \
            --manifest-dir "$SCREEN_ROOT/manifests/$version" \
            --output-dir "$SCREEN_ROOT/results/$version/$task/$MODEL" \
            --log-tag "screen_$version"
    done
done

echo ""
echo "========================================"
echo "STEP 4: Aggregate comparison"
echo "========================================"
python scripts/phase3_09_denoiser_screen.py --screen-root "$SCREEN_ROOT" --model "$MODEL"

echo ""
echo "========================================"
echo "DENOISER SCREEN COMPLETE"
echo "Finished: $(date)"
echo "========================================"
echo ""
echo "Summary: $SCREEN_ROOT/summary/"
echo "  - denoiser_screen_results.csv"
echo "  - denoiser_screen_report.md"
echo "  - denoiser_screen_comparison.png"
