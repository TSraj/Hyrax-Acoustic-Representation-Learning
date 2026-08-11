#!/bin/bash
#SBATCH --job-name=hyrax_layer_probe
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0-3%1
#SBATCH --output=logs/hyrax_layer_probe_%A_%a.out
#SBATCH --error=logs/hyrax_layer_probe_%A_%a.err

# Phase 3 - Step 24: per-layer hyrax probe, base vs species-adapted.
#
# THE MEASUREMENT THIS PROJECT HAS BEEN TRYING TO MAKE. Four cells:
#
#   0  hubert_base  base       frozen pretrained encoder
#   1  hubert_base  adapted    step-23 checkpoint, adapted on 7 species
#   2  xls_r        base
#   3  xls_r        adapted
#
# Each extracts mean-pooled embeddings from EVERY layer of the hyrax
# session-holdout set (5 s windows, 2.5 s stride -- the phase3_03 regime for
# hyrax tasks, which the corrected baselines were measured under), then trains a
# converged linear probe per layer over 5 seeds.
#
# Forward-only: no training, no gradients. Roughly 20-40 min per cell.
#
# Sequential (%1) so the four cells cannot contend for one GPU, which is what
# broke the step-23 submission.
#
# OUTPUT: outputs/phase3/hyrax_layer_probe/layer_probe_<model>_<condition>.json
# plus a cached .npz of embeddings per cell, so re-probing costs no GPU time.
#
# Figures are made afterwards by phase3_25_layer_figures.py, which is CPU-only
# and can run anywhere.

set -euo pipefail

MODELS=(hubert_base hubert_base xls_r xls_r)
CONDITIONS=(base adapted base adapted)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

PROJECT_DIR=${PROJECT_DIR:-$SLURM_SUBMIT_DIR}
cd "$PROJECT_DIR"

MANIFEST="outputs/phase3/manifests/hyrax_id_session_holdout.json"
CKPT="outputs/phase3/species7_finetune/${MODEL}/checkpoints/best_model.pth"
OUTPUT_DIR="outputs/phase3/hyrax_layer_probe"

echo "=============================================================="
echo "PHASE 3 STEP 24 - per-layer hyrax probe"
echo "=============================================================="
echo "cell         : $SLURM_ARRAY_TASK_ID  ->  $MODEL / $CONDITION"
echo "node         : $(hostname)"
echo "started      : $(date)"

[[ -f "$MANIFEST" ]] || { echo "FATAL: manifest missing: $MANIFEST"; exit 1; }
[[ -d "Data" ]]      || { echo "FATAL: Data/ not found"; exit 1; }

CKPT_ARG=()
if [[ "$CONDITION" == "adapted" ]]; then
    [[ -f "$CKPT" ]] || { echo "FATAL: adapted checkpoint missing: $CKPT"; exit 1; }
    CKPT_ARG=(--checkpoint "$CKPT")
    echo "checkpoint   : $CKPT"
fi

mkdir -p logs "$OUTPUT_DIR"

module load cuda 2>/dev/null || true
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}')"

python scripts/phase3_24_hyrax_layer_probe.py \
    --model "$MODEL" \
    --condition "$CONDITION" \
    "${CKPT_ARG[@]}" \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --probe-seeds 5 \
    --probe-steps 5000 \
    --probe-patience 500

echo ""
echo "finished     : $(date)"
echo "result       : $OUTPUT_DIR/layer_probe_${MODEL}_${CONDITION}.json"
