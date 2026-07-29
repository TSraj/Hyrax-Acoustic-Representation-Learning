#!/bin/bash
#SBATCH --job-name=phase3_lora_seeds
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --array=0-31%4
#SBATCH --output=logs/phase3_lora_seeds_%A_%a.out
#SBATCH --error=logs/phase3_lora_seeds_%A_%a.err

# Phase 3 - LoRA multi-seed runs (hyrax task only)
#
# The single-seed sweep left XLS-R looking unstable on hyrax (macro-F1
# 0.084 -> 0.309 -> 0.237 -> 0.245, non-monotonic), which cannot be separated
# from run-to-run variance with n=1. These runs add 4 more seeds so the
# monolingual-vs-multilingual comparison can be stated as mean +/- std.
#
#   models    xls_r x hubert_base                        = 2
#   fractions 10% / 25% / 50% / 100%                     = 4
#   seeds     43 / 44 / 45 / 46  (42 already exists)     = 4
#   total     2 x 4 x 4 = 32 array tasks
#
# Species ID is saturated (~0.97-0.98 macro-F1 by 50% data for both models) and
# is deliberately NOT re-run; it stays single-seed.
#
# Seed 42 is NOT re-run. The existing 8 runs are reused as-is - rerunning would
# not reproduce them anyway, since cuDNN kernels are nondeterministic on GPU.
#
# Training logic and config are unchanged from the validated run. Only --seed
# and the output directory differ. The window cache is shared: its key is a hash
# of the window params and file list and does not include the seed, so all runs
# read the same float16 cache built from the same BIODA manifests.
#
# Results go to .../frac<NN>/seed<S>/ rather than .../frac<NN>/. The seed MUST
# be in the path: checkpoint.pt is written per output directory, so seeds
# sharing a directory would resume each other's checkpoints.
#
# Hyrax runs are small - the single-seed sweep took 2.6-13.5 min each on a
# V100 - so 4h is generous and every task checkpoints per epoch regardless.
# Resubmit unfinished indices to resume:  sbatch --array=5,9 run_phase3_lora_seeds.sh
#
# SUBMIT:
#   sbatch run_phase3_lora_seeds.sh

set -e

PROJECT_DIR="/home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning"
cd "$PROJECT_DIR"

module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate
mkdir -p logs

CACHE_DIR="outputs/phase3/window_cache"
SWEEP_ROOT="outputs/phase3/lora_sweep"
MANIFEST="outputs/phase3/denoiser_screen/manifests/bioda/hyrax_id_session_holdout_ft.json"
TASK="hyrax_session_holdout"

MODELS=("xls_r" "hubert_base")
FRACTIONS=("0.10" "0.25" "0.50" "1.00")
SEEDS=("43" "44" "45" "46")

IDX=$SLURM_ARRAY_TASK_ID
N_SEED=${#SEEDS[@]}
N_FRAC=${#FRACTIONS[@]}

SEED_I=$(( IDX % N_SEED ))
FRAC_I=$(( (IDX / N_SEED) % N_FRAC ))
MODEL_I=$(( IDX / (N_SEED * N_FRAC) ))

MODEL=${MODELS[$MODEL_I]}
FRACTION=${FRACTIONS[$FRAC_I]}
SEED=${SEEDS[$SEED_I]}

# Zero-shot macro-F1 baseline for the curve annotation
if [ "$MODEL" = "xls_r" ]; then BASELINE_F1=0.1017; else BASELINE_F1=0.1735; fi

FRAC_TAG=$(printf "%.0f" "$(echo "$FRACTION * 100" | bc)")
OUT_DIR="$SWEEP_ROOT/$TASK/$MODEL/frac${FRAC_TAG}/seed${SEED}"

echo "========================================"
echo "PHASE 3 - LoRA MULTI-SEED (hyrax)"
echo "Array task : $IDX / 31"
echo "Job        : $SLURM_ARRAY_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Model      : $MODEL"
echo "Fraction   : $FRACTION (${FRAC_TAG}%)"
echo "Seed       : $SEED"
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
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --layerdrop 0.0 \
    --head-dropout 0.3 \
    --encoder-lr 1e-4 --head-lr 1e-3 \
    --batch-size 8 \
    --max-epochs 20 \
    --patience 6 --plateau-patience 3 \
    --class-weights window_inverse \
    --seed "$SEED"

echo ""
echo "========================================"
echo "Array task $IDX complete: $(date)"
echo "Results: $OUT_DIR/lora_fine_tuning_results.json"
echo "========================================"
