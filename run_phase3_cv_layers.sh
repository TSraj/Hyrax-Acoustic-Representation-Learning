#!/bin/bash
#SBATCH --job-name=cv_layers
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-29%4
#SBATCH --output=logs/cv_layers_%A_%a.out
#SBATCH --error=logs/cv_layers_%A_%a.err

# Phase 3: frozen 5-fold cross-validation at EVERY layer, 15 encoders.
#
# Same grid and environment as run_phase3_cv.sh (15 models x 2 audio = 30
# tasks), but each task runs scripts/phase3_41_cv_layer_sweep.py on the
# 5-fold-by-recording manifest only, and scores every layer instead of the
# inner-CV-chosen one. Output feeds the layer plot (phase3_42).
#
# PREREQUISITE: run_phase3_cv.sh has already run, so the embeddings are in
#   outputs/phase3/cv_<audio>/emb_cache/
# Nothing is re-embedded; only linear probes are fitted. A missing cache is
# extracted rather than failing, which costs GPU time -- the log says so.
#
# USAGE
#     ./run_phase3_cv_layers.sh --list     # the task -> cell mapping
#     sbatch run_phase3_cv_layers.sh       # all 30
#
# RESUMABLE: a finished model is skipped, and inside a model each finished
# layer is kept in a .partial.json, so resubmitting after a walltime kill
# loses at most one layer.
#
# WALLTIME: cost scales with layers. Estimated from the phase3_37 timings:
# ~1.5 h for a 13-layer model, ~3 h for 25 layers, ~6.5 h for xls_r_1b
# (49 layers) -- all under the 24 h cap.
#
# OUTPUT
#   outputs/phase3/cv_<audio>/layers/cv_layers_by_file_5fold_<model>.json

set -euo pipefail

# ---------------------------------------------------------------- the grid
MODELS=(
    wav2vec2_base
    wav2vec2_base_960h
    hubert_base
    hubert_large
    wavlm
    wavlm_large
    data2vec_base
    unispeech_sat
    mhubert_147
    mms_300m
    xls_r
    xls_r_1b
    ecapa_tdnn
    aves2_eat_bio
    aves2_eat_all
)
AUDIOS=(bioda original)

N_MODELS=${#MODELS[@]}
N_AUDIO=${#AUDIOS[@]}
N_CELLS=$((N_MODELS * N_AUDIO))

# model varies slowest, so a truncated submission covers whole models
cell_for() {
    local idx=$1
    CELL_MODEL=${MODELS[$((idx / N_AUDIO))]}
    CELL_AUDIO=${AUDIOS[$((idx % N_AUDIO))]}
}

if [[ "${1:-}" == "--list" ]]; then
    echo "$N_CELLS tasks (${N_MODELS} models x ${N_AUDIO} audio versions)"
    printf "%5s  %-20s %s\n" "task" "model" "audio"
    for i in $(seq 0 $((N_CELLS - 1))); do
        cell_for "$i"
        printf "%5d  %-20s %s\n" "$i" "$CELL_MODEL" "$CELL_AUDIO"
    done
    exit 0
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "FATAL: no SLURM_ARRAY_TASK_ID. Submit with sbatch, or pass --list."
    exit 1
fi
if (( SLURM_ARRAY_TASK_ID >= N_CELLS )); then
    echo "FATAL: task $SLURM_ARRAY_TASK_ID is outside the $N_CELLS-task grid."
    exit 1
fi

cell_for "$SLURM_ARRAY_TASK_ID"
MODEL=$CELL_MODEL
AUDIO=$CELL_AUDIO

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$PROJECT_DIR"

# ------------------------------------------------------------- paths
if [[ "$AUDIO" == "bioda" ]]; then
    MANIFEST_DIR="outputs/phase3/manifests_cv"
    EXPECT_PATH="BIODA/denoised"
else
    MANIFEST_DIR="outputs/phase3/manifests_cv_original"
    EXPECT_PATH="Audio"
fi
MANIFEST="$MANIFEST_DIR/cv_by_file_5fold.json"
CACHE_DIR="outputs/phase3/cv_${AUDIO}/emb_cache"
OUTPUT_DIR="outputs/phase3/cv_${AUDIO}/layers"

# ---------------------------------------------------------------- caches
if [[ -z "${WORK:-}" ]]; then
    echo "FATAL: \$WORK is not set, so model caches would land on HOME."
    exit 1
fi
export HF_HOME=${HF_HOME:-$WORK/hf_cache/huggingface}
export ESP_CACHE_HOME=${ESP_CACHE_HOME:-$WORK/hf_cache/esp}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

echo "=============================================================="
echo "CV LAYER SWEEP - every layer, 5-fold by recording"
echo "=============================================================="
echo "task      : ${SLURM_ARRAY_TASK_ID} / $((N_CELLS - 1))"
echo "cell      : $MODEL | $AUDIO"
echo "manifest  : $MANIFEST"
echo "cache     : $CACHE_DIR"
echo "output    : $OUTPUT_DIR"
echo "node      : $(hostname)"
echo "started   : $(date)"

[[ -d "Data" ]] || { echo "FATAL: Data/ not found"; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "FATAL: manifest missing: $MANIFEST"; exit 1; }

# the wrong-audio check from run_phase3_cv.sh
FIRST_FILE=$(python -c "
import json
m = json.load(open('$MANIFEST'))
print(m['items'][0]['file'])
")
case "$AUDIO:$FIRST_FILE" in
    bioda:*/BIODA/denoised/*) ;;
    original:*/Audio/*)       ;;
    *)
        echo "FATAL: audio '$AUDIO' does not match the manifest's paths."
        echo "       expected $EXPECT_PATH, first file: $FIRST_FILE"
        exit 1 ;;
esac
echo "verified  : manifest points at '$AUDIO' audio"

if ls "$CACHE_DIR/${MODEL}_"*.npz >/dev/null 2>&1; then
    echo "embedding : cached"
else
    echo "WARNING   : no cached embeddings for $MODEL -- will extract (slower)"
fi

mkdir -p logs "$OUTPUT_DIR" "$CACHE_DIR"

module load cuda 2>/dev/null || true

# ---------------------------------------------------------------- venv
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

python scripts/phase3_41_cv_layer_sweep.py \
    --model "$MODEL" \
    --manifest "$MANIFEST" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --probe-seeds 5 \
    --probe-steps 5000 \
    --probe-patience 500

echo ""
echo "finished  : $(date)"
echo "result    : $OUTPUT_DIR/cv_layers_by_file_5fold_${MODEL}.json"
