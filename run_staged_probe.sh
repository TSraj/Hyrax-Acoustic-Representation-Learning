#!/bin/bash
#SBATCH --job-name=staged_probe
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/staged_probe_%j.out
#SBATCH --error=logs/staged_probe_%j.err

# Phase C - frozen linear probe on hyrax, over every layer and two pooling
# variants, for BOTH the un-adapted base encoder and the species-adapted one.
#
# FOUR SWEEPS, NOT TWO. The published frozen numbers (HuBERT 0.1735, XLS-R
# 0.1017) come from phase3_03, which uses last_hidden_state - the FINAL layer
# only. They are not best-layer numbers. Comparing an adapted best-of-13-or-25
# against a base final-layer number would hand the adapted encoder a
# max-over-layers selection advantage the base never had. So the base encoder
# gets the identical sweep, and the published values are kept as a separate
# reference column.
#
# SELECTION IS ON VAL. 26 cells for HuBERT, 50 for XLS-R. Picking the best by
# test score across that many candidates, on 409 test windows over 8 classes,
# would be badly optimistic. phase3_18 selects on val macro-F1 and reports that
# cell's test score; phase3_19 additionally prints the test-oracle so the size
# of the selection gap stays visible.
#
# MANIFEST. The _ft session-holdout manifest, because it is the only variant
# with a val split and val-based selection requires one. Its TEST split is
# identical to the plain manifest's (409 windows, same held-out sessions), so
# the test set matches both reference points. Its train is smaller (1011 vs
# ~1353 windows), which is why the base final-layer cell will land near but not
# exactly on the published number.
#
# WINDOWS come from the existing hyrax cache, byte-identical to what the LoRA
# runs trained on (keys 7ef9e0a1822d / a1741a737762 / 9a2edf1d1b2e). The probe
# is cache-only and will fail rather than silently re-decode audio, since
# differently-cut windows would break the comparison.
#
# COST. Forward-only over 1748 windows (1011/328/409) = 219 batches, one pass
# per split yielding all layers at once. Minutes per sweep, not hours. 2h wall
# is already generous; this could run interactively.
#
# ADAPTER SELECTION. Set ADAPTER_SUFFIX to point at a different adapter run,
# e.g. the 30-epoch HuBERT re-run:
#   ADAPTER_SUFFIX=_e30 sbatch run_staged_probe.sh
# Applies to hubert_base only, since xls_r has no _e30 variant; the script
# falls back to seed42 for any model whose suffixed adapter is absent.
#
# SUBMIT:
#   sbatch run_staged_probe.sh
#   bash run_staged_probe.sh --list

set -e

MODELS=("hubert_base" "xls_r")
ADAPTER_SUFFIX="${ADAPTER_SUFFIX:-}"

STAGED_ROOT="outputs/phase3/staged_lora/species7"
PROBE_ROOT="outputs/phase3/staged_lora/probe"
MANIFEST="outputs/phase3/denoiser_screen/manifests/bioda/hyrax_id_session_holdout_ft.json"
CACHE_DIR="outputs/phase3/window_cache"

if [ "${1:-}" = "--list" ]; then
    echo "Sweeps (4 = 2 models x {base, adapted}), ADAPTER_SUFFIX='${ADAPTER_SUFFIX}'"
    for m in "${MODELS[@]}"; do
        echo "  $m base    -> $PROBE_ROOT/$m/base"
        echo "  $m adapted -> $PROBE_ROOT/$m/adapted"
        echo "     adapter: $STAGED_ROOT/$m/seed42${ADAPTER_SUFFIX}/adapter"
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

echo "========================================"
echo "PHASE C - STAGED FROZEN PROBE"
echo "Job        : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURM_NODELIST:-local}"
echo "Manifest   : $MANIFEST"
echo "Cache      : $CACHE_DIR"
echo "Adapters   : $STAGED_ROOT/<model>/seed42${ADAPTER_SUFFIX}/adapter"
echo "Output     : $PROBE_ROOT"
echo "HF_HOME    : $HF_HOME"
echo "Started    : $(date)"
echo "========================================"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST"
    exit 1
fi

for MODEL in "${MODELS[@]}"; do
    # ---- base (un-adapted) sweep: no --adapter-dir
    OUT="$PROBE_ROOT/$MODEL/base"
    if [ -f "$OUT/staged_probe_results.json" ] && [ "${FORCE:-0}" != "1" ]; then
        echo ""
        echo "SKIP $MODEL base: $OUT/staged_probe_results.json exists (FORCE=1 to re-run)"
    else
        echo ""
        echo "=== $MODEL | BASE (un-adapted) ==="
        mkdir -p "$OUT"
        python scripts/phase3_18_staged_probe.py \
            --model "$MODEL" \
            --manifest "$MANIFEST" \
            --cache-dir "$CACHE_DIR" \
            --output-dir "$OUT"
    fi

    # ---- adapted sweep
    ADAPTER="$STAGED_ROOT/$MODEL/seed42${ADAPTER_SUFFIX}/adapter"
    if [ ! -d "$ADAPTER" ] && [ -n "$ADAPTER_SUFFIX" ]; then
        echo "NOTE: $ADAPTER not found, falling back to seed42 for $MODEL"
        ADAPTER="$STAGED_ROOT/$MODEL/seed42/adapter"
    fi
    if [ ! -d "$ADAPTER" ]; then
        echo "ERROR: adapter not found: $ADAPTER"
        echo "       Run run_staged_lora_species7.sh first."
        exit 1
    fi

    OUT="$PROBE_ROOT/$MODEL/adapted"
    if [ -f "$OUT/staged_probe_results.json" ] && [ "${FORCE:-0}" != "1" ]; then
        echo ""
        echo "SKIP $MODEL adapted: $OUT/staged_probe_results.json exists (FORCE=1 to re-run)"
    else
        echo ""
        echo "=== $MODEL | ADAPTED ($ADAPTER) ==="
        mkdir -p "$OUT"
        python scripts/phase3_18_staged_probe.py \
            --model "$MODEL" \
            --adapter-dir "$ADAPTER" \
            --manifest "$MANIFEST" \
            --cache-dir "$CACHE_DIR" \
            --output-dir "$OUT"
    fi
done

echo ""
echo "=== Comparison (C3) ==="
python scripts/phase3_19_staged_probe_analysis.py \
    --probe-root "$PROBE_ROOT" \
    --out-dir "$PROBE_ROOT/summary"

echo ""
echo "========================================"
echo "Phase C complete: $(date)"
echo "Summary: $PROBE_ROOT/summary/STAGED_PROBE_README.md"
echo "========================================"
