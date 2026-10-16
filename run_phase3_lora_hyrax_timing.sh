#!/bin/bash
#SBATCH --job-name=lora_timing
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/lora_timing_%j.out
#SBATCH --error=logs/lora_timing_%j.err

# Phase 3 - Step 43, TIMING TEST: one encoder, fold 0, one training epoch.
#
# Runs scripts/phase3_43_lora_hyrax_cv.py --timing, which
#   1. builds the encoder with LoRA on every block and the front end trainable
#   2. runs the gradient gate (every adapter, the front end and the head must
#      receive gradient -- the job stops here if not)
#   3. trains ONE full epoch of fold 0 and times it
#   4. times a validation pass, an embedding pass and a 5-seed one-layer probe
#   5. writes outputs/phase3/lora_hyrax_cv/<model>/timing.json with a
#      full-run estimate (5 folds, worst case and early-stop case)
# Nothing it writes is a result; the full run is a separate job.
#
# USAGE
#     sbatch run_phase3_lora_hyrax_timing.sh                          # WavLM Large
#     sbatch --export=ALL,MODEL=aves2_eat_all run_phase3_lora_hyrax_timing.sh

set -euo pipefail

MODEL=${MODEL:-wavlm_large}
case "$MODEL" in
    wavlm_large|aves2_eat_all) ;;
    *) echo "FATAL: MODEL must be wavlm_large or aves2_eat_all, got '$MODEL'"; exit 1 ;;
esac

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$PROJECT_DIR"

MANIFEST="outputs/phase3/manifests_cv_original/cv_by_file_5fold.json"
OUTPUT_DIR="outputs/phase3/lora_hyrax_cv"

# ---------------------------------------------------------------- caches
if [[ -z "${WORK:-}" ]]; then
    echo "FATAL: \$WORK is not set, so model caches would land on HOME."
    exit 1
fi
export HF_HOME=${HF_HOME:-$WORK/hf_cache/huggingface}
export ESP_CACHE_HOME=${ESP_CACHE_HOME:-$WORK/hf_cache/esp}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

echo "=============================================================="
echo "LoRA HYRAX-ID TIMING TEST - $MODEL, fold 0, one epoch"
echo "=============================================================="
echo "manifest  : $MANIFEST"
echo "output    : $OUTPUT_DIR/$MODEL/timing.json"
echo "node      : $(hostname)"
echo "started   : $(date)"

[[ -d "Data" ]] || { echo "FATAL: Data/ not found"; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "FATAL: manifest missing: $MANIFEST"; exit 1; }

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
    --timing \
    --num-workers 3

echo ""
echo "finished  : $(date)"
echo "result    : $OUTPUT_DIR/$MODEL/timing.json"
