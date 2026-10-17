#!/bin/bash
#SBATCH --job-name=lora_cv_eval
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/lora_cv_eval_%A_%a.out
#SBATCH --error=logs/lora_cv_eval_%A_%a.err

# Phase 3 - Step 43, FULL RUN v2: LoRA fine-tuning on hyrax ID, 5 folds,
# ENCODER IN EVAL MODE (--encoder-eval).
#
# WHY A v2. The first full run (run_phase3_lora_hyrax_cv.sh ->
# outputs/phase3/lora_hyrax_cv/) trained with the encoder in train mode. For
# WavLM Large that turned on dropout and SpecAugment, which the frozen probe
# never sees, and fine-tuning failed (0.18 pooled test UAR). The diagnostic
# (run_phase3_lora_wavlm_diag.sh, round 3, fold 0) fixed it with
# --encoder-eval: test UAR 0.106 -> 0.303. Both encoders are re-run with it,
# so they are trained identically. Everything else is unchanged.
#
# EPOCHS. Validation UAR was still rising at epoch 8 in the diagnostic, so the
# cap goes from 20 to 30 and early-stopping patience from 4 to 6. The best
# validation epoch is still the one kept.
#
# Writes ONLY to outputs/phase3/lora_hyrax_cv_evalmode/ -- the v1 results in
# outputs/phase3/lora_hyrax_cv/ are never read or written.
#
#   task 0  wavlm_large      (best speech encoder)
#   task 1  aves2_eat_all    (better AVES encoder)
#
# Each task trains all 5 folds of the 5-fold-by-recording split, repetition 0,
# then pools them and writes the frozen-vs-fine-tuned comparison:
#   outputs/phase3/lora_hyrax_cv_evalmode/<model>/fold_<k>/{checkpoint.pth,result.json,probs.npz}
#   outputs/phase3/lora_hyrax_cv_evalmode/<model>/{summary.json,layer_comparison.csv}
#
# TIMING (measured in the diagnostic, V100): a WavLM Large epoch is ~4 min one
# bout per pass, so 30 epochs is ~2 h per fold worst case; with early
# stopping, expect roughly 6-10 h for five folds. AVES ran ~1-1.5 h per fold
# in v1. Both fit in 24 h, and the job resumes if killed.
#
# RESUMABLE: a finished fold is skipped, and an unfinished one resumes from
# its last completed epoch, so resubmitting after a kill loses at most one
# epoch:  sbatch --array=<task> run_phase3_lora_hyrax_cv_evalmode.sh
#
# The frozen comparison reads the layer sweep (run_phase3_cv_layers.sh):
#   outputs/phase3/cv_original/layers/cv_layers_by_file_5fold_<model>.json

set -euo pipefail

MODELS=(wavlm_large aves2_eat_all)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "FATAL: no SLURM_ARRAY_TASK_ID. Submit with sbatch."
    exit 1
fi
if (( SLURM_ARRAY_TASK_ID >= ${#MODELS[@]} )); then
    echo "FATAL: task $SLURM_ARRAY_TASK_ID is outside the ${#MODELS[@]}-task grid."
    exit 1
fi
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$PROJECT_DIR"

MANIFEST="outputs/phase3/manifests_cv_original/cv_by_file_5fold.json"
OUTPUT_DIR="outputs/phase3/lora_hyrax_cv_evalmode"
FROZEN="outputs/phase3/cv_original/layers/cv_layers_by_file_5fold_${MODEL}.json"

# ---------------------------------------------------------------- caches
if [[ -z "${WORK:-}" ]]; then
    echo "FATAL: \$WORK is not set, so model caches would land on HOME."
    exit 1
fi
export HF_HOME=${HF_HOME:-$WORK/hf_cache/huggingface}
export ESP_CACHE_HOME=${ESP_CACHE_HOME:-$WORK/hf_cache/esp}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

echo "=============================================================="
echo "LoRA HYRAX-ID FINE-TUNING v2 (encoder eval mode) - $MODEL, 5 folds"
echo "=============================================================="
echo "task      : ${SLURM_ARRAY_TASK_ID} / $((${#MODELS[@]} - 1))"
echo "manifest  : $MANIFEST"
echo "output    : $OUTPUT_DIR/$MODEL"
echo "node      : $(hostname)"
echo "started   : $(date)"

[[ -d "Data" ]] || { echo "FATAL: Data/ not found"; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "FATAL: manifest missing: $MANIFEST"; exit 1; }
# checked up front so a 5-hour run cannot fail at its last step
[[ -f "$FROZEN" ]] || { echo "FATAL: frozen layer sweep missing: $FROZEN"; exit 1; }

mkdir -p logs "$OUTPUT_DIR"
module load cuda 2>/dev/null || true

# same split as run_phase3_cv.sh: AVES needs the avex environment
if [[ "$MODEL" == aves2_* ]]; then
    AVEX_VENV=${AVEX_VENV:-$WORK/venv_avex}
    [[ -d "$AVEX_VENV" ]] || { echo "FATAL: avex venv missing: $AVEX_VENV"; exit 1; }
    echo "venv      : $AVEX_VENV (avex)"
    source "$AVEX_VENV/bin/activate"
else
    echo "venv      : venv (main)"
    source venv/bin/activate
fi
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch; print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/phase3_43_lora_hyrax_cv.py \
    --model "$MODEL" \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --fold 0 1 2 3 4 \
    --encoder-eval \
    --max-epochs 30 \
    --patience 6 \
    --num-workers 3

python scripts/phase3_43_lora_hyrax_cv.py \
    --model "$MODEL" \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --encoder-eval \
    --summarize

echo ""
echo "finished  : $(date)"
echo "result    : $OUTPUT_DIR/$MODEL/summary.json"
