#!/bin/bash
#SBATCH --job-name=species7_ft
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-1%1
#SBATCH --output=logs/species7_ft_%A_%a.out
#SBATCH --error=logs/species7_ft_%A_%a.err

# Phase 3 - Step 23: staged 7-species adaptation by REAL fine-tuning.
#
# Replaces the LoRA path (run_staged_lora_species7.sh), which could not move
# the layer that matters: hyrax peaks at hidden_states[0], the CNN front-end,
# and LoRA left it frozen -- so the measured adaptation delta was 0.000 by
# construction, not by experiment.
#
# Here the conv feature extractor + feature_projection are UNFROZEN at a lower
# LR (1e-5) alongside transformer blocks 0-3 (1e-4). Head at 1e-3.
#
# SCOPE: 2 runs -- {hubert_base, xls_r} x 7-class species x seed 42.
# Hyrax is EXCLUDED from the manifest and never seen during adaptation.
#
# OUTPUT per model:
#   outputs/phase3/species7_finetune/<model>/checkpoints/best_model.pth
#   outputs/phase3/species7_finetune/<model>/adaptation_summary.json
# The checkpoint is the input to the per-layer hyrax probe (step 3).
#
# NOTE ON DATA REGIME: this is the phase2_05 regime -- one sample per FILE,
# truncated to 30 s, loaded on the fly. It does NOT use window_cache_species7,
# which belongs to the LoRA path (5 s/2.5 s windows). Nothing to preflight
# there; the audio files themselves must be present under Data/.

set -euo pipefail

MODELS=(hubert_base xls_r)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

PROJECT_DIR=${PROJECT_DIR:-$SLURM_SUBMIT_DIR}
cd "$PROJECT_DIR"

MANIFEST="outputs/phase3/manifests_species7/species_id.json"
OUTPUT_DIR="outputs/phase3/species7_finetune/${MODEL}"

# ---------------------------------------------------------------- preflight
echo "=============================================================="
echo "PHASE 3 STEP 23 - staged 7-species adaptation (real fine-tuning)"
echo "=============================================================="
echo "model        : $MODEL"
echo "task id      : $SLURM_ARRAY_TASK_ID"
echo "node         : $(hostname)"
echo "started      : $(date)"
echo "project dir  : $PROJECT_DIR"

[[ -f "$MANIFEST" ]] || { echo "FATAL: manifest missing: $MANIFEST"; exit 1; }
[[ -d "Data" ]]      || { echo "FATAL: Data/ not found"; exit 1; }
[[ -f "scripts/phase3_23_species7_finetune.py" ]] || {
    echo "FATAL: training script missing"; exit 1; }

python - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["num_classes"] == 7, f"expected 7 classes, got {m['num_classes']}"
assert "hyrax" in m.get("excluded_species", []), "hyrax is NOT excluded -- wrong manifest"
assert "hyrax" not in m["species"], "hyrax present in label space"
print(f"preflight OK: {m['num_classes']} classes, hyrax excluded, "
      f"{m['split_counts']['train']}/{m['split_counts']['val']}/{m['split_counts']['test']} files")
PY

mkdir -p logs "$OUTPUT_DIR"

# ---------------------------------------------------------------- env
module load cuda 2>/dev/null || true
source venv/bin/activate

# reduces fragmentation, which is what turned a tight fit into an OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "python       : $(which python)"
python -c "import torch; print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}  dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# ---------------------------------------------------------------- gate 1
# Refuse to burn a GPU hour if gradients do not reach the conv stack.
echo ""
echo "--- gradient gate ---"
python scripts/phase3_23_species7_finetune.py \
    --model "$MODEL" \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --check-grads \
    --batch-size 2 \
    --max-duration 5 \
    --no-cudnn || { echo "FATAL: gradient gate FAILED for $MODEL"; exit 1; }

# ---------------------------------------------------------------- train
echo ""
echo "--- adaptation ---"
# No srun: it launched one task PER allocated task slot, so two processes
# landed on the same GPU and fought over its 32 GB until one OOM'd. This is a
# single-process job; run python directly.
#
# --grad-checkpoint is REQUIRED, not an optimisation. Gradients must reach
# blocks 0-3 and the conv stack, so activations for EVERY layer above them
# (all 24 in XLS-R) stay live for the backward pass. That is strictly more
# memory than phase2_05 ever needed, where the conv stack was frozen.
python scripts/phase3_23_species7_finetune.py \
    --model "$MODEL" \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --num-layers 4 \
    --layerdrop 0.0 \
    --lr-conv 1e-5 \
    --lr-backbone 1e-4 \
    --lr-head 1e-3 \
    --batch-size "${BATCH_SIZE:-8}" \
    --max-epochs 16 \
    --patience 5 \
    --max-duration "${MAX_DURATION:-30}" \
    --num-workers 4 \
    --seed 42 \
    --grad-checkpoint \
    --min-species-f1 0.90 \
    --no-cudnn

echo ""
echo "finished     : $(date)"
echo "checkpoint   : $OUTPUT_DIR/checkpoints/best_model.pth"
python - "$OUTPUT_DIR/adaptation_summary.json" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
if p.exists():
    s = json.load(open(p))
    print(f"RESULT {s['model']}: test macro-F1 {s['test_f1_macro']:.4f} "
          f"acc {s['test_accuracy']:.4f} (best epoch {s['best_epoch']}/{len(s['history'])})")
PY
