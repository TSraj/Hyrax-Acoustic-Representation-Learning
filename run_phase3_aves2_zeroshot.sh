#!/bin/bash
#SBATCH --job-name=aves2_zeroshot
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-3%1
#SBATCH --output=logs/aves2_zeroshot_%A_%a.out
#SBATCH --error=logs/aves2_zeroshot_%A_%a.err

# AVES 2 (EAT, bio-pretrained) -- ZERO-SHOT ONLY.
#
# Frozen encoder + trained linear probe. No fine-tuning, no adapted cells, no
# checkpoints. phase3_28 refuses a --checkpoint outright so this cannot drift.
#
#   0  hyrax bouts, session-holdout   8 individuals, chance 0.125
#   1  hyrax bouts, by-file          10 individuals, chance 0.100
#   2  species 7-class                hyrax excluded, chance 0.143
#   3  hyrax session-holdout, TILED   sensitivity check ONLY (see below)
#
# RESULTS ARE WRITTEN TO A SEPARATE TREE AND MERGE WITH NOTHING:
#
#   outputs/phase3/aves2_zeroshot/<condition>/layer_probe_aves2_eat_bio_base.json
#
# No existing results file, figure or cache is read, written or overwritten.
#
# WHY CELL 3 IS SEPARATE AND SECONDARY
# ------------------------------------
# EAT sees a fixed 10.24 s canvas, and a hyrax bout has a median duration near
# 1 s, so ~86% of the canvas is padding. Cells 0-2 mask that padding out of the
# mean pooling. Cell 3 instead TILES the bout to fill the canvas, which scores
# higher on a separability proxy -- but a tiled 1.4 s bout is a repeated call,
# a different stimulus from the one the bout manifests define. It is reported
# as a sensitivity check, never as the headline number, and lands in its own
# file (..._tile.json) so it cannot be mistaken for one.
#
# PREREQUISITE: run_phase3_aves2_predownload.sh must have completed. This job
# does not download anything; if the caches are cold it will fail fast rather
# than spend GPU time on the network.
#
# Sequential (%1) so cells cannot contend for one GPU.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$PROJECT_DIR"

# Slurm does not reliably source .bashrc. Both caches must be redirected off a
# quota-limited HOME -- and ESP_CACHE_HOME is NOT covered by HF_HOME: avex
# resolves the AVES weights to Path.home()/".cache"/"esp" unless it is set.
if [[ -z "${WORK:-}" ]]; then
    echo "FATAL: \$WORK is not set, so the model caches would land on HOME."
    exit 1
fi
export HF_HOME=${HF_HOME:-$WORK/hf_cache/huggingface}
export ESP_CACHE_HOME=${ESP_CACHE_HOME:-$WORK/hf_cache/esp}

MODEL=aves2_eat_bio
OUT_ROOT="outputs/phase3/aves2_zeroshot"

MANIFESTS=(
    "outputs/phase3/manifests_bout/hyrax_bout_session_holdout.json"
    "outputs/phase3/manifests_bout/hyrax_bout_by_file.json"
    "outputs/phase3/manifests_species7/species_id.json"
    "outputs/phase3/manifests_bout/hyrax_bout_session_holdout.json"
)
TAGS=(hyrax_bout_session_holdout hyrax_bout_by_file species7 hyrax_bout_session_holdout_tile)
SCRIPTS=(hyrax hyrax species hyrax)
PAD_MODES=(zero zero zero tile)

I=$SLURM_ARRAY_TASK_ID
MANIFEST=${MANIFESTS[$I]}
TAG=${TAGS[$I]}
KIND=${SCRIPTS[$I]}
PAD_MODE=${PAD_MODES[$I]}
OUTPUT_DIR="$OUT_ROOT/$TAG"

echo "=============================================================="
echo "AVES 2 ZERO-SHOT - cell $I"
echo "=============================================================="
echo "model      : $MODEL  (frozen, zero-shot)"
echo "task       : $TAG"
echo "manifest   : $MANIFEST"
echo "pad mode   : $PAD_MODE"
echo "output     : $OUTPUT_DIR"
echo "node       : $(hostname)"
echo "started    : $(date)"
echo ""

[[ -f "$MANIFEST" ]] || { echo "FATAL: manifest missing: $MANIFEST"; exit 1; }
[[ -d "Data" ]]      || { echo "FATAL: Data/ not found"; exit 1; }

mkdir -p logs "$OUTPUT_DIR"

module load cuda 2>/dev/null || true

# AVES runs in its own venv: avex needs torch>=2.5 and the project venv is
# older, which we deliberately do not upgrade. Built by the pre-download job.
AVEX_VENV=${AVEX_VENV:-$WORK/venv_avex}
if [[ ! -x "$AVEX_VENV/bin/python" ]]; then
    echo "FATAL: AVES venv missing at $AVEX_VENV"
    echo "       Run run_phase3_aves2_predownload.sh first."
    exit 1
fi
source "$AVEX_VENV/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}')"

# Fail fast on a cold cache rather than discovering it mid-extraction. Also
# guards against a compute node with no outbound network, where a download
# would hang instead of erroring.
# Check the caches the ENV VARS point at -- not Path.home(), which would look in
# the wrong place entirely once the caches are redirected to $WORK.
python - <<'PYEOF'
import os
import sys
from pathlib import Path

hf = Path(os.environ["HF_HOME"])
esp_root = Path(os.environ["ESP_CACHE_HOME"])
eat = list(hf.glob("hub/models--worstchan--EAT-base_epoch30_pretrain"))
esp = list(esp_root.glob("esp-aves2-eat-bio-*.safetensors"))

print(f"HF_HOME        : {hf}")
print(f"ESP_CACHE_HOME : {esp_root}")
if not eat or not esp:
    print("FATAL: AVES caches are cold.", file=sys.stderr)
    print(f"  EAT backbone cached : {bool(eat)}", file=sys.stderr)
    print(f"  AVES weights cached : {bool(esp)}", file=sys.stderr)
    print("  Run run_phase3_aves2_predownload.sh first.", file=sys.stderr)
    sys.exit(1)
for p in eat + esp:
    if "/home/hpc/" in str(p.resolve()):
        print(f"FATAL: cache resolves onto quota-limited HOME: {p.resolve()}",
              file=sys.stderr)
        sys.exit(1)
print("AVES caches warm, and off /home/hpc")
PYEOF

if [[ "$KIND" == "species" ]]; then
    python scripts/phase3_29_species_layer_probe.py \
        --model "$MODEL" \
        --manifest "$MANIFEST" \
        --output-dir "$OUTPUT_DIR" \
        --probe-seeds 5 \
        --probe-steps 5000 \
        --probe-patience 500 \
        --pad-mode "$PAD_MODE" \
        --batch-size 16
else
    python scripts/phase3_24_hyrax_layer_probe.py \
        --model "$MODEL" \
        --condition base \
        --manifest "$MANIFEST" \
        --output-dir "$OUTPUT_DIR" \
        --probe-seeds 5 \
        --probe-steps 5000 \
        --probe-patience 500 \
        --pad-mode "$PAD_MODE" \
        --batch-size 16
fi

echo ""
echo "finished   : $(date)"
echo "result     : $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || true
