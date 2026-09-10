#!/bin/bash
#SBATCH --job-name=cv_probe
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-29%4
#SBATCH --output=logs/cv_probe_%A_%a.out
#SBATCH --error=logs/cv_probe_%A_%a.err

# Phase 3: CROSS-VALIDATED probing, all 18 individuals, 15 encoders.
#
# WHAT CHANGED FROM run_phase3_audio_comparison.sh
# ------------------------------------------------
# That job was 60 tasks: 15 models x 2 audio x 2 SPLITS, and every task
# extracted its own embeddings. Here the split is not fixed -- the folds are
# built inside the job -- so the split axis leaves the task grid:
#
#     30 TASKS = 15 models x 2 audio versions
#
# Each task embeds all 4141 bouts ONCE and then runs BOTH protocols
# (leave-one-session-out and 5-fold by recording) off that one cache. Same
# total work as before, half the extraction.
#
# WHAT IT ANSWERS
# ---------------
# The supervisor asked for five things, all of them here:
#   1. cross-validation instead of a single fixed partition
#   2. all 18 individuals, not the 8/10 with the most data
#   3. the layer chosen on inner folds, never on test
#   4. all 15 encoders, both audio versions
#   5. fold-averaged scores plus a per-individual breakdown
#
# PREREQUISITE -- BUILD THE MANIFESTS FIRST
# -----------------------------------------
# Once, on the login node (CPU, a few seconds). Do NOT put this in the array:
# 30 tasks writing the same file concurrently would corrupt it.
#
#   python scripts/phase3_36_cv_manifests.py \
#       --audio-subdir BIODA/denoised --output-dir outputs/phase3/manifests_cv
#   python scripts/phase3_36_cv_manifests.py \
#       --audio-subdir Audio --output-dir outputs/phase3/manifests_cv_original
#
# The job refuses to start if they are missing.
#
# USAGE
# -----
#     ./run_phase3_cv.sh --list             # the task -> cell mapping
#     sbatch run_phase3_cv.sh               # all 30
#     sbatch --array=0-1 run_phase3_cv.sh   # one model, both audio versions
#
# RESUMABLE: a protocol whose result JSON exists is skipped, and the embedding
# cache is keyed on the bout list, so a task killed by the walltime re-runs
# without re-embedding anything. Re-submitting the whole array after a partial
# failure costs only the missing cells.
#
# WALLTIME: xls_r_1b is the outlier -- 49 layers x 3 inner folds x (15 + 25)
# outer folds is roughly 6000 probe fits per audio version. Everything with 13
# layers is several times faster. If the cluster refuses 24h, drop to 12h and
# resubmit; resumability makes that safe.
#
# OUTPUT
#   outputs/phase3/cv_<audio>/cv_<protocol>_<model>.json
#   outputs/phase3/cv_<audio>/emb_cache/<model>_<hash>.npz

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
    echo "each task runs BOTH protocols off one embedding cache"
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

# ------------------------------------------------------------- manifest dir
if [[ "$AUDIO" == "bioda" ]]; then
    MANIFEST_DIR="outputs/phase3/manifests_cv"
    EXPECT_PATH="BIODA/denoised"
else
    MANIFEST_DIR="outputs/phase3/manifests_cv_original"
    EXPECT_PATH="Audio"
fi
OUTPUT_DIR="outputs/phase3/cv_${AUDIO}"
# ONE cache per audio version, shared by both protocols: they hold the same
# bouts in the same order and differ only in fold ids
CACHE_DIR="$OUTPUT_DIR/emb_cache"

PROTOCOLS=(cv_session_loso cv_by_file_5fold)

# ---------------------------------------------------------------- caches
if [[ -z "${WORK:-}" ]]; then
    echo "FATAL: \$WORK is not set, so model caches would land on HOME."
    exit 1
fi
export HF_HOME=${HF_HOME:-$WORK/hf_cache/huggingface}
export ESP_CACHE_HOME=${ESP_CACHE_HOME:-$WORK/hf_cache/esp}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
# NOT set offline, for the reason documented in
# run_phase3_audio_comparison.sh: several repos have weights and config cached
# at different revisions, and offline mode turns a working setup into a hard
# failure.

echo "=============================================================="
echo "CROSS-VALIDATED PROBE - all 18 individuals"
echo "=============================================================="
echo "task      : ${SLURM_ARRAY_TASK_ID} / $((N_CELLS - 1))"
echo "cell      : $MODEL | $AUDIO"
echo "manifests : $MANIFEST_DIR"
echo "output    : $OUTPUT_DIR"
echo "node      : $(hostname)"
echo "started   : $(date)"

[[ -d "Data" ]] || { echo "FATAL: Data/ not found"; exit 1; }

for p in "${PROTOCOLS[@]}"; do
    [[ -f "$MANIFEST_DIR/$p.json" ]] || {
        echo "FATAL: manifest missing: $MANIFEST_DIR/$p.json"
        echo "       build both manifest sets ONCE on the login node:"
        echo "         python scripts/phase3_36_cv_manifests.py \\"
        echo "           --audio-subdir BIODA/denoised \\"
        echo "           --output-dir outputs/phase3/manifests_cv"
        echo "         python scripts/phase3_36_cv_manifests.py \\"
        echo "           --audio-subdir Audio \\"
        echo "           --output-dir outputs/phase3/manifests_cv_original"
        exit 1
    }
done

# The mistake this whole comparison turns on: an ORIGINAL cell whose manifest
# actually points at denoised audio, or the reverse, would produce a confident
# number under the wrong label. Check the manifest's own paths, and check that
# the two protocols agree with each other.
for p in "${PROTOCOLS[@]}"; do
    FIRST_FILE=$(python -c "
import json
m = json.load(open('$MANIFEST_DIR/$p.json'))
print(m['items'][0]['file'])
")
    case "$AUDIO:$FIRST_FILE" in
        bioda:*/BIODA/denoised/*) ;;
        original:*/Audio/*)       ;;
        *)
            echo "FATAL: audio '$AUDIO' does not match $p's paths."
            echo "       expected $EXPECT_PATH, first file: $FIRST_FILE"
            exit 1 ;;
    esac
done
echo "verified  : both manifests point at '$AUDIO' audio"

mkdir -p logs "$OUTPUT_DIR" "$CACHE_DIR"

module load cuda 2>/dev/null || true

# ---------------------------------------------------------------- venv
# avex needs torch>=2.5, which the main venv does not have, and upgrading it
# would put every already-published number at risk of moving for unrelated
# reasons. The AVES models therefore run from a separate environment.
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

# Both protocols, in one task, off one embedding cache. session_loso runs
# first because it is the cheaper of the two (15 folds x 1 repeat versus
# 5 x 5), so a task that dies on the walltime still leaves a complete result.
for p in "${PROTOCOLS[@]}"; do
    echo ""
    echo "--------------------------------------------------------------"
    echo "protocol  : $p"
    echo "--------------------------------------------------------------"
    python scripts/phase3_37_cv_probe.py \
        --model "$MODEL" \
        --manifest "$MANIFEST_DIR/$p.json" \
        --output-dir "$OUTPUT_DIR" \
        --cache-dir "$CACHE_DIR" \
        --probe-seeds 5 \
        --probe-steps 5000 \
        --selection-steps 2000 \
        --probe-patience 500 \
        --inner-folds 3
done

echo ""
echo "finished  : $(date)"
for p in "${PROTOCOLS[@]}"; do
    echo "result    : $OUTPUT_DIR/${p}_${MODEL}.json"
done
