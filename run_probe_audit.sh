#!/bin/bash
#SBATCH --job-name=probe_audit
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --output=logs/probe_audit_%A_%a.out
#SBATCH --error=logs/probe_audit_%A_%a.err
#SBATCH --array=0-11%2

# Audit of phase3_03 probe undertraining.
#
# phase3_03 trains its linear probe with FULL-BATCH gradient descent - one
# optimizer step per "epoch" - and its no-val branch runs 50 epochs, i.e. 50
# gradient steps, keeping the final state. On the hyrax tasks that leaves the
# probe unfit: the published runs report TRAIN macro-F1 of 0.08-0.53 on an
# 8-class task whose chance level is 0.125.
#
# ALREADY DONE LOCALLY (do not re-run): the 6-model session-holdout audit.
# 50-step replication matched published to within 0.002 for 5 of 6 models, and
# the corrected numbers moved +0.05 to +0.26, reordering the model ranking.
#
# THIS JOB covers the two remaining pieces, both of which need encoder forward
# passes and so belong on the GPU:
#
#   tasks 0-5   denoiser screen: xls_r x {original, bioda, aca} x
#               {within_session, session_holdout}
#   tasks 6-11  7-way species baselines, all 6 models. Heavier than the hyrax
#               cells: phase3_03 does NOT window species, so it is one embedding
#               per file over 16373 files with 30s truncation.
#
# Features are re-extracted rather than loaded, because phase3_03 never saved
# its embeddings. Extraction is cached to --emb-cache, so the probe trajectory
# (50/100/.../5000 steps) then costs nothing.
#
# READ-ONLY with respect to published results. Everything lands under
# outputs/phase3/probe_audit/.
#
# SUBMIT:
#   sbatch run_probe_audit.sh
#   bash run_probe_audit.sh --list

set -e

# "<label> <model> <manifest> <label_key> <published_f1>"
JOBS=(
  "screen_original_within  xls_r  outputs/phase3/denoiser_screen/manifests/original/hyrax_id_within_session.json   individual 0.1043"
  "screen_original_holdout xls_r  outputs/phase3/denoiser_screen/manifests/original/hyrax_id_session_holdout.json  individual 0.1123"
  "screen_bioda_within     xls_r  outputs/phase3/denoiser_screen/manifests/bioda/hyrax_id_within_session.json      individual 0.1125"
  "screen_bioda_holdout    xls_r  outputs/phase3/denoiser_screen/manifests/bioda/hyrax_id_session_holdout.json     individual 0.1036"
  "screen_aca_within       xls_r  outputs/phase3/denoiser_screen/manifests/aca/hyrax_id_within_session.json        individual 0.0674"
  "screen_aca_holdout      xls_r  outputs/phase3/denoiser_screen/manifests/aca/hyrax_id_session_holdout.json       individual 0.0664"
  "species7_hubert         hubert_base         outputs/phase3/manifests_species7/species_id.json species 0.8736"
  "species7_xlsr           xls_r               outputs/phase3/manifests_species7/species_id.json species 0.8051"
  "species7_wavlm          wavlm               outputs/phase3/manifests_species7/species_id.json species 0.7603"
  "species7_w2v2base       wav2vec2_base       outputs/phase3/manifests_species7/species_id.json species 0.7971"
  "species7_w2v2960h       wav2vec2_base_960h  outputs/phase3/manifests_species7/species_id.json species 0.5378"
  "species7_ecapa          ecapa_tdnn          outputs/phase3/manifests_species7/species_id.json species 0.7708"
)
N_JOBS=${#JOBS[@]}

if [ "${1:-}" = "--list" ]; then
    echo "Job table: $N_JOBS tasks  ->  --array=0-$((N_JOBS - 1))%2"
    printf "%5s  %-24s %-20s %-8s %s\n" "index" "label" "model" "labelkey" "published"
    for i in "${!JOBS[@]}"; do
        # read -r, NOT `set -- $spec`: zsh does not word-split the latter, which
        # silently collapsed every argument into $1 on a previous attempt.
        read -r label model manifest label_key pub <<< "${JOBS[$i]}"
        printf "%5d  %-24s %-20s %-8s %s\n" "$i" "$label" "$model" "$label_key" "$pub"
    done
    exit 0
fi

PROJECT_DIR="/home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning"
cd "$PROJECT_DIR"

export HF_HOME=$WORK/hf_cache/huggingface

module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate
mkdir -p logs

OUT_ROOT="outputs/phase3/probe_audit"
EMB_CACHE="$OUT_ROOT/emb_cache"

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "$N_JOBS" ]; then
    echo "ERROR: array index $IDX outside job table (0-$((N_JOBS - 1)))."
    exit 1
fi

read -r LABEL MODEL MANIFEST LABEL_KEY PUB <<< "${JOBS[$IDX]}"

echo "========================================"
echo "PROBE AUDIT"
echo "Array task : $IDX / $((N_JOBS - 1))"
echo "Label      : $LABEL"
echo "Model      : $MODEL"
echo "Manifest   : $MANIFEST"
echo "Label key  : $LABEL_KEY"
echo "Published  : $PUB"
echo "Output     : $OUT_ROOT"
echo "HF_HOME    : $HF_HOME"
echo "Started    : $(date)"
echo "========================================"

# Fail loudly, not silently. The previous local attempt used `cmd && echo done`,
# so six failing cells produced no output at all and looked like a clean run.
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST"
    exit 1
fi

RESULT="$OUT_ROOT/probe_audit_${MODEL}_${LABEL}.json"
if [ -f "$RESULT" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "SKIP: $RESULT exists (FORCE=1 to re-run)"
    exit 0
fi

mkdir -p "$OUT_ROOT" "$EMB_CACHE"

python scripts/phase3_20_probe_audit.py \
    --model "$MODEL" \
    --manifest "$MANIFEST" \
    --label-key "$LABEL_KEY" \
    --published-f1 "$PUB" \
    --output-dir "$OUT_ROOT" \
    --emb-cache "$EMB_CACHE" \
    --tag "$LABEL"

echo ""
echo "========================================"
echo "Task $IDX ($LABEL) complete: $(date)"
echo "Result: $RESULT"
echo "========================================"
