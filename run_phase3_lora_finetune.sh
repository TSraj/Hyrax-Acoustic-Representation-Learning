#!/bin/bash
#SBATCH --job-name=lora_species7
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/lora_species7_%j.out
#SBATCH --error=logs/lora_species7_%j.err

# Phase 3 - Step 29: staged 7-species adaptation by LoRA.
#
# The METHOD comparison against partial fine-tuning (step 23). Everything except
# the parameterisation is identical by construction: phase3_29 imports
# phase3_23's dataset and collate, so the training regime cannot drift.
#
#   partial fine-tune  blocks 0-3 weights updated directly   ~28M params
#   LoRA               blocks 0-3 attention via rank-16 adapters   ~0.4M params
#
# Both unfreeze the CNN front-end at 1e-5, because LoRA only touches attention
# projections and hyrax identity peaks at hidden_states[0], which the conv stack
# produces. Without that, layer 0 could not move and the result would be a
# foregone conclusion -- the defect that invalidated the original LoRA run.
#
# Hyrax is EXCLUDED from training. It appears only at evaluation, via the
# bout-level probe.
#
# RESUME AND CHAINING
# -------------------
# The cluster caps jobs at 24 h and XLS-R needed ~28 h last time, dying at epoch
# 14 of 16. The trainer now writes full resume state every epoch and a DONE
# marker when finished, so submit a CHAIN and each link continues where the last
# stopped. A link that finds DONE exits in seconds, so over-provisioning is free.
#
#   ./submit_lora_chain.sh hubert_base 2
#   ./submit_lora_chain.sh xls_r 3
#
# Or by hand:
#   J1=$(sbatch --parsable run_phase3_lora_finetune.sh xls_r)
#   J2=$(sbatch --parsable --dependency=afterany:$J1 run_phase3_lora_finetune.sh xls_r)
#
# OUTPUT per model:
#   outputs/phase3/ft_lora/<model>/checkpoints/best_model.pth   (MERGED backbone)
#   outputs/phase3/ft_lora/<model>/checkpoints/resume.pth
#   outputs/phase3/ft_lora/<model>/adaptation_summary.json
#   outputs/phase3/ft_lora/<model>/DONE

set -euo pipefail

# One model per job, passed as argument 1. The two models are independent, so
# they run as two PARALLEL chains on separate GPUs rather than one sequential
# array -- roughly halving wall-clock, which matters against a deadline. (The
# earlier OOM came from srun launching two tasks onto one GPU, not from running
# two jobs at once.)
MODEL=${1:-${MODEL:-}}
if [[ -z "$MODEL" ]]; then
    echo "FATAL: no model given."
    echo "  usage: sbatch $0 <hubert_base|xls_r>"
    exit 1
fi

PROJECT_DIR=${PROJECT_DIR:-$SLURM_SUBMIT_DIR}
cd "$PROJECT_DIR"

MANIFEST=${MANIFEST:-outputs/phase3/manifests_species7/species_id.json}
EXPERIMENT=${EXPERIMENT:-ft_lora}
OUTPUT_DIR="outputs/phase3/${EXPERIMENT}/${MODEL}"

echo "=============================================================="
echo "PHASE 3 STEP 29 - LoRA adaptation on 7 species"
echo "=============================================================="
echo "model        : $MODEL"
echo "job id       : ${SLURM_JOB_ID:-none}"
echo "manifest     : $MANIFEST"
echo "output       : $OUTPUT_DIR"
echo "node         : $(hostname)"
echo "started      : $(date)"

# ---------------------------------------------------------------- short-circuit
if [[ -f "$OUTPUT_DIR/DONE" ]]; then
    echo ""
    echo "DONE marker already present:"
    cat "$OUTPUT_DIR/DONE"
    echo "nothing to do -- this link of the chain exits immediately."
    exit 0
fi

if [[ -f "$OUTPUT_DIR/checkpoints/resume.pth" ]]; then
    echo "resume state found -- continuing a previous run"
else
    echo "no resume state -- starting from scratch"
fi

# ---------------------------------------------------------------- preflight
[[ -f "$MANIFEST" ]] || { echo "FATAL: manifest missing: $MANIFEST"; exit 1; }
[[ -d "Data" ]]      || { echo "FATAL: Data/ not found"; exit 1; }
[[ -f "scripts/phase3_29_lora_finetune.py" ]] || {
    echo "FATAL: trainer missing"; exit 1; }

python - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["num_classes"] == 7, f"expected 7 classes, got {m['num_classes']}"
assert "hyrax" in m.get("excluded_species", []), "hyrax is NOT excluded -- wrong manifest"
assert "hyrax" not in m["species"], "hyrax present in label space"
print(f"preflight OK: 7 classes, hyrax excluded, "
      f"{m['split_counts']['train']}/{m['split_counts']['val']}/{m['split_counts']['test']} files")
PY

mkdir -p logs "$OUTPUT_DIR"

# ---------------------------------------------------------------- env
module load cuda 2>/dev/null || true
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "python       : $(which python)"
python -c "import torch, peft; print(f'torch {torch.__version__}  peft {peft.__version__}  cuda={torch.cuda.is_available()}')"

# ---------------------------------------------------------------- gate
# Refuse to spend a GPU day if the adapters or the conv stack get no gradient.
echo ""
echo "--- gradient gate ---"
python scripts/phase3_29_lora_finetune.py \
    --model "$MODEL" \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --check-grads \
    --batch-size 2 \
    --max-duration 5 \
    --force \
    ${NO_CUDNN:+--no-cudnn} || { echo "FATAL: gradient gate FAILED for $MODEL"; exit 1; }

# ---------------------------------------------------------------- train
echo ""
echo "--- LoRA adaptation ---"
# Config matches phase3_23 exactly except for the parameterisation: same batch,
# epochs, patience, duration, seed, layerdrop and pooling. --lora-layers 4
# matches the 4 blocks phase3_23 unfroze, so this compares METHOD, not scope.
#
# NO_CUDNN is opt-in here. It was inherited from phase2_05 as a V100 workaround
# and badly slows convolutions -- which now matters, since the conv stack trains.
# Leave it unset unless cuDNN actually misbehaves.
python scripts/phase3_29_lora_finetune.py \
    --model "$MODEL" \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --lora-layers 4 \
    --layerdrop 0.0 \
    --lr-lora 1e-4 \
    --lr-conv 1e-5 \
    --lr-head 1e-3 \
    --batch-size "${BATCH_SIZE:-8}" \
    --max-epochs 16 \
    --patience 5 \
    --max-duration "${MAX_DURATION:-30}" \
    --num-workers 4 \
    --seed 42 \
    --grad-checkpoint \
    --min-species-f1 0.90 \
    ${NO_CUDNN:+--no-cudnn}

echo ""
echo "finished     : $(date)"
python - "$OUTPUT_DIR/adaptation_summary.json" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
if p.exists():
    s = json.load(open(p))
    print(f"RESULT {s['model']} (lora): test macro-F1 {s['test_f1_macro']:.4f} "
          f"acc {s['test_accuracy']:.4f} (best epoch {s['best_epoch']}/{len(s['history'])})")
else:
    print("no summary yet -- run did not finish; the next link in the chain will resume")
PY
