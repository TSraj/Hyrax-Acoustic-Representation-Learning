#!/bin/bash
#SBATCH --job-name=wavlm_diag
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/wavlm_diag_%j.out
#SBATCH --error=logs/wavlm_diag_%j.err

# Phase 3 - Step 43 DIAGNOSTIC: why WavLM Large fine-tuning failed.
#
# In the full run (run_phase3_lora_hyrax_cv.sh) WavLM Large's training loss
# fell while validation UAR stayed at 0.10-0.24, two folds selected epoch 1,
# and end-to-end test UAR was 0.18 -- below even the FROZEN last layer (0.38).
# AVES 2 EAT, same script, trained normally. Three variants, fold 0, 8 epochs
# each, no early stopping, so the whole validation curve is visible:
#
#   A  head only        encoder fully frozen, only the linear head trains.
#                       Must climb toward the frozen last-layer level (~0.38).
#                       If it does NOT, the WavLM training path itself is
#                       broken and no learning-rate change can help.
#   B  low LR           LoRA 2e-5, front end 2e-6 (5x lower than the run)
#   C  low LR, frozen   LoRA 2e-5, front end frozen -- fewer trainable parts
#      front end
#
# Writes ONLY to outputs/phase3/lora_wavlm_diag/ -- the full-run results in
# outputs/phase3/lora_hyrax_cv/ are never read or written.
#
# USAGE
#     sbatch run_phase3_lora_wavlm_diag.sh

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$PROJECT_DIR"

MANIFEST="outputs/phase3/manifests_cv_original/cv_by_file_5fold.json"
OUT="outputs/phase3/lora_wavlm_diag"

if [[ -z "${WORK:-}" ]]; then
    echo "FATAL: \$WORK is not set, so model caches would land on HOME."
    exit 1
fi
export HF_HOME=${HF_HOME:-$WORK/hf_cache/huggingface}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

echo "=============================================================="
echo "WavLM Large DIAGNOSTIC - fold 0, 8 epochs, 3 variants"
echo "=============================================================="
echo "output    : $OUT  (separate from lora_hyrax_cv)"
echo "node      : $(hostname)"
echo "started   : $(date)"

[[ -d "Data" ]] || { echo "FATAL: Data/ not found"; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "FATAL: manifest missing: $MANIFEST"; exit 1; }
mkdir -p logs "$OUT"
module load cuda 2>/dev/null || true
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch; print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}')"

run_variant() {
    local name=$1; shift
    echo ""
    echo "=============================================================="
    echo "VARIANT $name"
    echo "=============================================================="
    python scripts/phase3_43_lora_hyrax_cv.py \
        --model wavlm_large \
        --manifest "$MANIFEST" \
        --output-dir "$OUT/$name" \
        --fold 0 \
        --max-epochs 8 \
        --patience 99 \
        --skip-probe \
        --num-workers 3 \
        "$@"
}

run_variant A_head_only         --no-lora --freeze-front
run_variant B_low_lr            --lr-lora 2e-5 --lr-front 2e-6
run_variant C_low_lr_frozen_front --lr-lora 2e-5 --freeze-front

echo ""
echo "finished  : $(date)"
echo "reference : full run fold 0 had val UAR 0.12-0.15, test UAR 0.106;"
echo "            frozen last-layer probe UAR is ~0.38"
