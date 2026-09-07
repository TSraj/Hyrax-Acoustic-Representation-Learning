#!/bin/bash
#SBATCH --job-name=audio_comparison
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-59%4
#SBATCH --output=logs/audio_comparison_%A_%a.out
#SBATCH --error=logs/audio_comparison_%A_%a.err

# Phase 3: ORIGINAL vs BIODA-DENOISED audio, 15 encoders, frozen zero-shot.
#
# THE QUESTION
# ------------
# The supervisor reports that original recordings score better than the
# BIODA-denoised ones. An older comparison here favoured BIODA (0.340 vs
# 0.291), but it ran on 5 s windows cut from CONCATENATED bouts and is
# superseded. This settles it on the real unit: one ground-truth bout, sliced
# at its own start/end.
#
# 60 CELLS = 15 models x 2 audio versions x 2 splits.
#
# Every cell is frozen (--condition base). No adaptation, no checkpoints: this
# is the zero-shot grid, and mixing an adapted cell in would put a fine-tuned
# encoder in the same column as a frozen one.
#
# The two audio versions differ ONLY in the file path. The manifests are built
# from the same GTLabels, so same bouts, same individuals, same splits, same
# class weights. Any delta is the denoiser and nothing else.
#
# The two SPLITS are different tasks -- 8 individuals at chance 0.125 versus 10
# at chance 0.100 -- so original-vs-BIODA is read WITHIN a split. Never compare
# a session-holdout number against a by-file one.
#
# USAGE
# -----
#     sbatch run_phase3_audio_comparison.sh              # all 60
#     sbatch --array=0-3 run_phase3_audio_comparison.sh  # just the first 4
#     ./run_phase3_audio_comparison.sh --list            # print the mapping
#
# Run --list first. It prints which task id is which cell without submitting
# anything, so a partial submission can be aimed deliberately rather than
# guessed at.
#
# RESUMABLE: phase3_24 skips a cell whose result JSON already exists, and the
# embedding cache is per split, so a cell killed mid-extraction resumes from
# whichever split finished. Re-running the whole array after a partial failure
# costs only the missing cells.
#
# OUTPUT
#   outputs/phase3/hyrax_probe_bout_<audio>_<split>/layer_probe_<model>_base.json
#
# Figures and the comparison table are made afterwards by
# phase3_34_evaluation_report.py, which is CPU-only and runs anywhere.

set -euo pipefail

# ---------------------------------------------------------------- the grid
# 15 encoders. aves2_* load through avex, everything else through HuggingFace.
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
SPLITS=(session_holdout by_file)

N_MODELS=${#MODELS[@]}
N_AUDIO=${#AUDIOS[@]}
N_SPLITS=${#SPLITS[@]}
N_CELLS=$((N_MODELS * N_AUDIO * N_SPLITS))
PER_MODEL=$((N_AUDIO * N_SPLITS))

# task id -> (model, audio, split). Model varies slowest so a truncated
# submission covers whole models rather than half of several.
cell_for() {
    local idx=$1
    CELL_MODEL=${MODELS[$((idx / PER_MODEL))]}
    CELL_AUDIO=${AUDIOS[$(((idx % PER_MODEL) / N_SPLITS))]}
    CELL_SPLIT=${SPLITS[$((idx % N_SPLITS))]}
}

if [[ "${1:-}" == "--list" ]]; then
    echo "$N_CELLS cells (${N_MODELS} models x ${N_AUDIO} audio x ${N_SPLITS} splits)"
    printf "%5s  %-20s %-10s %s\n" "task" "model" "audio" "split"
    for i in $(seq 0 $((N_CELLS - 1))); do
        cell_for "$i"
        printf "%5d  %-20s %-10s %s\n" "$i" "$CELL_MODEL" "$CELL_AUDIO" "$CELL_SPLIT"
    done
    exit 0
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "FATAL: no SLURM_ARRAY_TASK_ID. Submit with sbatch, or pass --list."
    exit 1
fi
if (( SLURM_ARRAY_TASK_ID >= N_CELLS )); then
    echo "FATAL: task $SLURM_ARRAY_TASK_ID is outside the $N_CELLS-cell grid."
    exit 1
fi

cell_for "$SLURM_ARRAY_TASK_ID"
MODEL=$CELL_MODEL
AUDIO=$CELL_AUDIO
SPLIT=$CELL_SPLIT

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$PROJECT_DIR"

# ------------------------------------------------------------- manifest dir
# bioda    -> outputs/phase3/manifests_bout           (paths in BIODA/denoised)
# original -> outputs/phase3/manifests_bout_original  (paths in Audio)
if [[ "$AUDIO" == "bioda" ]]; then
    MANIFEST_DIR="outputs/phase3/manifests_bout"
else
    MANIFEST_DIR="outputs/phase3/manifests_bout_original"
fi
MANIFEST="$MANIFEST_DIR/hyrax_bout_${SPLIT}.json"
OUTPUT_DIR="outputs/phase3/hyrax_probe_bout_${AUDIO}_${SPLIT}"

# ---------------------------------------------------------------- caches
# Slurm does not reliably source .bashrc, so an interactive HF_HOME does not
# survive into the job. Set the caches explicitly rather than silently filling
# a quota-limited HOME with tens of GB.
if [[ -z "${WORK:-}" ]]; then
    echo "FATAL: \$WORK is not set, so model caches would land on HOME."
    exit 1
fi
export HF_HOME=${HF_HOME:-$WORK/hf_cache/huggingface}
export ESP_CACHE_HOME=${ESP_CACHE_HOME:-$WORK/hf_cache/esp}
# HuggingFace serves large files through Xet, which this cluster's compute
# nodes cannot reach: small files download fine, then weights fail with "CAS
# Client Error".
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
# NOT set offline on purpose. Every previous job in this project ran online,
# and the cache turns out to be incomplete in a way that only offline mode
# exposes: for several repos the weights and the config were fetched at
# DIFFERENT revisions, so refs/main resolves to a snapshot holding config.json
# and no weights. Online, transformers just fetches the missing file and the
# run proceeds -- which is why this never surfaced before. Forcing offline
# turned a working setup into a hard failure, so it stays off.
#
# Do not "fix" this by pinning revisions or rebuilding the cache: the models
# are already downloaded, and the only thing offline mode bought was failing
# fast on a network stall that has never actually happened here.

echo "=============================================================="
echo "AUDIO COMPARISON - frozen zero-shot"
echo "=============================================================="
echo "task      : ${SLURM_ARRAY_TASK_ID} / $((N_CELLS - 1))"
echo "cell      : $MODEL | $AUDIO | $SPLIT"
echo "manifest  : $MANIFEST"
echo "output    : $OUTPUT_DIR"
echo "node      : $(hostname)"
echo "started   : $(date)"

[[ -f "$MANIFEST" ]] || {
    echo "FATAL: manifest missing: $MANIFEST"
    if [[ "$AUDIO" == "original" ]]; then
        echo "       build it first:"
        echo "         python scripts/phase3_27_bout_manifests.py \\"
        echo "           --audio-subdir Audio \\"
        echo "           --output-dir outputs/phase3/manifests_bout_original"
    fi
    exit 1
}
[[ -d "Data" ]] || { echo "FATAL: Data/ not found"; exit 1; }

# Guard the mistake this whole experiment turns on: an ORIGINAL cell whose
# manifest actually points at denoised audio, or the reverse, would produce a
# confident number under the wrong label. Check the manifest's own paths.
FIRST_FILE=$(python -c "
import json,sys
m=json.load(open('$MANIFEST'))
print(m['splits']['train'][0]['file'])
")
case "$AUDIO:$FIRST_FILE" in
    bioda:*/BIODA/denoised/*) ;;
    original:*/Audio/*)       ;;
    *)
        echo "FATAL: audio '$AUDIO' does not match the manifest's paths."
        echo "       first train file: $FIRST_FILE"
        exit 1
        ;;
esac
echo "verified  : manifest paths match '$AUDIO'"

mkdir -p logs "$OUTPUT_DIR"

module load cuda 2>/dev/null || true

# ---------------------------------------------------------------- venv
# avex needs torch>=2.5, which the main venv does not have, and upgrading it
# would put every published number at risk of moving for unrelated reasons.
# The AVES models therefore run from a separate environment.
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

python scripts/phase3_24_hyrax_layer_probe.py \
    --model "$MODEL" \
    --condition base \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --probe-seeds 5 \
    --probe-steps 5000 \
    --probe-patience 500

echo ""
echo "finished  : $(date)"
echo "result    : $OUTPUT_DIR/layer_probe_${MODEL}_base.json"
