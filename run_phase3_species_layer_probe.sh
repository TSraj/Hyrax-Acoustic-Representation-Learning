#!/bin/bash
#SBATCH --job-name=species_layers
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-5%1
#SBATCH --output=logs/species_layers_%A_%a.out
#SBATCH --error=logs/species_layers_%A_%a.err

# Phase 3 - Step 24 applied to the SPECIES task: which layer carries species
# identity, frozen and after LoRA adaptation?
#
# The hyrax side already has this: identity peaks at layer 3 and falls away with
# depth. Species was only ever measured at the FINAL layer, because those numbers
# came from phase3_20_probe_audit -- a script written to test probe
# undertraining, not to sweep layers. The layer-sweep machinery arrived later and
# was only ever pointed at hyrax. A gap of sequence, not of intent.
#
# Filling it makes the central claim mechanistic rather than asserted: if species
# peaks LATE and individual identity peaks EARLY, the two tasks demonstrably live
# in different parts of the network.
#
# SIX CELLS:
#   0  hubert_base         base      frozen pretrained encoder
#   1  hubert_base         adapted   LoRA checkpoint from step 29
#   2  xls_r               base
#   3  xls_r               adapted
#   4  wavlm               base      second monolingual candidate
#   5  wav2vec2_base       base      second monolingual candidate
#
# WavLM and wav2vec2 are zero-shot ONLY -- neither was fine-tuned, so there is
# no adapted cell for them. Both are included because the runner-up monolingual
# depends on the task: WavLM leads on species (0.948 vs 0.935) while wav2vec2
# leads on hyrax session-holdout (0.390 vs 0.373).
#
# Partial fine-tuning is deliberately not included -- only zero-shot and LoRA
# were asked for.
#
# MANIFEST: manifests_species7/species_id.json -- 7 classes, HYRAX EXCLUDED, the
# same manifest the encoders were adapted on. The probe detects the species label
# key and switches to ONE embedding per FILE truncated to 30 s, matching
# phase3_03:217. It must not window: the published 0.969 / 0.962 were measured
# per file, and windowing changes the dataset size about fourfold.
#
# COST: 18,162 files, roughly 34 hours of audio -- about 26x the bout set.
# Expect ~1 h per base-sized cell (HuBERT, WavLM, wav2vec2) and ~3 h per XLS-R
# cell, so ~10 h across the six. The 24 h limit is PER ARRAY TASK, not for the
# set, so every cell has roughly 8x headroom; %1 only serialises them. Each cell
# caches its embeddings, so re-running one is cheap.

set -euo pipefail

MODELS=(hubert_base hubert_base xls_r xls_r wavlm wav2vec2_base)
CONDITIONS=(base adapted base adapted base base)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

PROJECT_DIR=${PROJECT_DIR:-$SLURM_SUBMIT_DIR}
cd "$PROJECT_DIR"

MANIFEST=${MANIFEST:-outputs/phase3/manifests_species7/species_id.json}
EXPERIMENT=${EXPERIMENT:-ft_lora}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/phase3/species_layer_probe_lora}
CKPT="outputs/phase3/${EXPERIMENT}/${MODEL}/checkpoints/best_model.pth"

echo "=============================================================="
echo "SPECIES LAYER PROBE - which layer carries species identity?"
echo "=============================================================="
echo "cell         : $SLURM_ARRAY_TASK_ID  ->  $MODEL / $CONDITION"
echo "manifest     : $MANIFEST"
echo "output       : $OUTPUT_DIR"
echo "node         : $(hostname)"
echo "started      : $(date)"

[[ -f "$MANIFEST" ]] || { echo "FATAL: manifest missing: $MANIFEST"; exit 1; }
[[ -d "Data" ]]      || { echo "FATAL: Data/ not found"; exit 1; }

python - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["num_classes"] == 7, f"expected 7 classes, got {m['num_classes']}"
assert "hyrax" not in m["species"], "hyrax present in the label space"
print(f"preflight OK: 7 classes, hyrax excluded, "
      f"{m['split_counts']['train']}/{m['split_counts']['test']} train/test files")
PY

CKPT_ARG=()
if [[ "$CONDITION" == "adapted" ]]; then
    [[ -f "$CKPT" ]] || { echo "FATAL: LoRA checkpoint missing: $CKPT"; exit 1; }
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
python - "$OUTPUT_DIR/layer_probe_${MODEL}_${CONDITION}.json" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
if p.exists():
    j = json.load(open(p))
    d = j["layers"][str(j["best_layer"])]
    print(f"RESULT {j['model']} / {j['condition']}: best layer {j['best_layer']} of "
          f"{j['n_layers'] - 1}  F1 {d['f1_macro_mean']:.4f}  acc {d['accuracy_mean']:.4f}")
PY
