#!/bin/bash
#SBATCH --job-name=phase3_lora_sweep
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --array=0-15%4
#SBATCH --output=logs/phase3_lora_sweep_%A_%a.out
#SBATCH --error=logs/phase3_lora_sweep_%A_%a.err

# Phase 3 - LoRA Fine-Tuning Sweep (16 runs, one GPU each)
#
#   models    : xls_r (multilingual) x hubert_base (monolingual)      = 2
#   tasks     : species_id x hyrax_id_session_holdout_ft              = 2
#   fractions : 10% / 25% / 50% / 100% of training windows            = 4
#   total     : 2 x 2 x 4 = 16 array tasks
#
# --array=0-15%4 runs at most 4 concurrently, to stay inside TinyGPU's per-user
# GPU limit. Raise or drop the %4 if your allocation allows more.
#
# Config is exactly the one validated in the single XLS-R run:
#   LoRA r=16 alpha=32 dropout=0.05 on q/k/v/out_proj, all layers
#   frozen CNN extractor, LayerDrop 0.0
#   Dropout(0.3) -> Linear head
#   AdamW: adapters 1e-4, head 1e-3
#   ReduceLROnPlateau(mode=max, factor=0.5, patience=3) on val macro-F1
#   batch 8, 5s/2.5s windowing, window-inverse class weights, seed 42
# and it carries all four bug fixes from phase3_10_lora_fine_tuning.py.
#
# WALL CLOCK: every epoch writes outputs/.../checkpoint.pt (adapters + head +
# optimizer + scheduler + history, ~12 MB). A task killed at 24h resumes from
# the last completed epoch on resubmission - just resubmit the same array, or
# only the unfinished indices:
#   sbatch --array=3,7,11 run_phase3_lora_sweep.sh
# Use --no-resume to force a task to start over.
#
# Submit AFTER the cache prep job. Use run_phase3_lora_submit.sh, which checks
# that the prep job actually got a job id first - if that submission fails, an
# empty $JID turns into "Job dependency problem" here and both jobs are lost.

set -e

PROJECT_DIR="/home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning"
cd "$PROJECT_DIR"

module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate
mkdir -p logs

CACHE_DIR="outputs/phase3/window_cache"
SWEEP_ROOT="outputs/phase3/lora_sweep"

MODELS=("xls_r" "hubert_base")
TASKS=("hyrax_session_holdout" "species_id")
FRACTIONS=("0.10" "0.25" "0.50" "1.00")

# Decode the array index -> (model, task, fraction)
IDX=$SLURM_ARRAY_TASK_ID
N_FRAC=${#FRACTIONS[@]}
N_TASK=${#TASKS[@]}

FRAC_I=$(( IDX % N_FRAC ))
TASK_I=$(( (IDX / N_FRAC) % N_TASK ))
MODEL_I=$(( IDX / (N_FRAC * N_TASK) ))

MODEL=${MODELS[$MODEL_I]}
TASK=${TASKS[$TASK_I]}
FRACTION=${FRACTIONS[$FRAC_I]}

# Per-task manifest, zero-shot macro-F1 baseline, and window cap
if [ "$TASK" = "species_id" ]; then
    MANIFEST="outputs/phase3/manifests/species_id.json"
    MAX_WIN="--max-windows-per-file 1"
    if [ "$MODEL" = "xls_r" ]; then BASELINE_F1=0.7194; else BASELINE_F1=0.8635; fi
else
    MANIFEST="outputs/phase3/denoiser_screen/manifests/bioda/hyrax_id_session_holdout_ft.json"
    MAX_WIN=""
    if [ "$MODEL" = "xls_r" ]; then BASELINE_F1=0.1017; else BASELINE_F1=0.1735; fi
fi

FRAC_TAG=$(printf "%.0f" "$(echo "$FRACTION * 100" | bc)")
OUT_DIR="$SWEEP_ROOT/$TASK/$MODEL/frac${FRAC_TAG}"

echo "========================================"
echo "PHASE 3 - LoRA SWEEP"
echo "Array task : $IDX / 15"
echo "Job        : $SLURM_ARRAY_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Model      : $MODEL"
echo "Task       : $TASK"
echo "Fraction   : $FRACTION (${FRAC_TAG}%)"
echo "Manifest   : $MANIFEST"
echo "Baseline F1: $BASELINE_F1"
echo "Output     : $OUT_DIR"
echo "Started    : $(date)"
echo "========================================"

mkdir -p "$OUT_DIR"

python scripts/phase3_10_lora_fine_tuning.py \
    --model "$MODEL" \
    --manifest "$MANIFEST" \
    --output-dir "$OUT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --data-fraction "$FRACTION" \
    --baseline-f1 "$BASELINE_F1" \
    $MAX_WIN \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --layerdrop 0.0 \
    --head-dropout 0.3 \
    --encoder-lr 1e-4 --head-lr 1e-3 \
    --batch-size 8 \
    --max-epochs 20 \
    --patience 6 --plateau-patience 3 \
    --class-weights window_inverse \
    --seed 42

echo ""
echo "========================================"
echo "Array task $IDX complete: $(date)"
echo "Results: $OUT_DIR/lora_fine_tuning_results.json"
echo "========================================"
