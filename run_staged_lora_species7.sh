#!/bin/bash
#SBATCH --job-name=staged_lora_species7
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/staged_lora_species7_%A_%a.out
#SBATCH --error=logs/staged_lora_species7_%A_%a.err

# Phase B1 - LoRA adaptation on the 7-CLASS species task (hyrax excluded)
#
# Produces the ANIMAL-ADAPTED encoders for the staged design. The adapters are
# exported in canonical PEFT format so Phase C can load them onto a fresh base
# encoder, keep it FROZEN, and probe hyrax as an unseen target species.
#
# SCOPE: 2 runs - {hubert_base, xls_r} x 100% data x seed 42. Single seed on
# purpose: this validates the staged path end to end before any sweep.
#
# CONFIG is byte-for-byte the validated setup from run_phase3_lora_sweep.sh:
# r=16 alpha=32 lora_dropout=0.05 on q/k/v/out_proj, LayerDrop 0.0, frozen CNN
# extractor, Dropout(0.3)->Linear head, AdamW adapters 1e-4 / head 1e-3,
# ReduceLROnPlateau on val macro-F1, batch 8, 5s/2.5s windows,
# window_inverse class weights. ONLY the manifest, cache dir, baseline and
# output paths differ.
#
# BASELINE F1 values are the 7-WAY zero-shot numbers from Phase A step A4
# (hubert 0.8736, xls_r 0.8051). The OLD 8-class values (0.8635 / 0.7194) would
# annotate the curves against the wrong label space - chance is 1/7 here, not
# 1/8.
#
# ############################################################################
# # --max-windows-per-file 1 IS MANDATORY, NOT OPTIONAL
# #
# # The window cache key hashes (window params | label key | max_windows_per_file
# # | file list). The 7-class cache was built with max_windows_per_file=1. Drop
# # the flag and it becomes None, the hash changes, and the trainer SILENTLY
# # rebuilds the cache taking EVERY window from every file - far larger than the
# # 2.9 GB that exists, since the anuraset files are long. The preflight below
# # asserts the expected cache files are already on disk so this cannot happen
# # unnoticed mid-run.
# ############################################################################
#
# ISOLATION. Everything lands under outputs/staged_lora/. Nothing touches the
# 8-class checkpoints, lora_sweep_V2, zero_shot/, or outputs/figures_paper/.
# --log-tag keeps the per-run log out of lora_fine_tune_<model>_run.log, which
# the log handler opens in APPEND mode and which already holds an unrelated
# July hyrax run.
#
# WALL CLOCK. 24h matches run_phase3_lora_sweep.sh, the known-good precedent
# for species at 100% (HuBERT ran 20 epochs, XLS-R early-stopped at 13). Each
# epoch is checkpointed, so a task killed at the wall clock resumes from the
# last completed epoch on resubmission:
#   sbatch --array=1 run_staged_lora_species7.sh
# Per-epoch wall time and ms/batch are now logged each epoch - read them off
# the first run to size Phase C.
#
# SUBMIT:
#   sbatch run_staged_lora_species7.sh
#   bash run_staged_lora_species7.sh --list

set -e

MODELS=("hubert_base" "xls_r")
BASELINES=("0.8736" "0.8051")   # 7-WAY zero-shot macro-F1, Phase A step A4
SEED=42
FRACTION=1.0

N_JOBS=${#MODELS[@]}

if [ "${1:-}" = "--list" ]; then
    echo "Job table: $N_JOBS tasks  ->  --array=0-$((N_JOBS - 1))%2"
    printf "%5s  %-14s %-12s %s\n" "index" "model" "baseline_f1" "seed"
    for i in "${!MODELS[@]}"; do
        printf "%5d  %-14s %-12s %s\n" "$i" "${MODELS[$i]}" "${BASELINES[$i]}" "$SEED"
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

MANIFEST="outputs/phase3/manifests_species7/species_id.json"
CACHE_DIR="outputs/phase3/window_cache_species7"
OUT_ROOT="outputs/staged_lora/species7"

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "$N_JOBS" ]; then
    echo "ERROR: array index $IDX is outside the job table (0-$((N_JOBS - 1)))."
    exit 1
fi

MODEL="${MODELS[$IDX]}"
BASELINE_F1="${BASELINES[$IDX]}"
OUT_DIR="$OUT_ROOT/$MODEL/seed${SEED}"
ADAPTER_DIR="$OUT_DIR/adapter"

echo "========================================"
echo "PHASE B1 - STAGED LoRA ADAPTATION (7-class species)"
echo "Array task  : $IDX / $((N_JOBS - 1))"
echo "Job         : ${SLURM_ARRAY_JOB_ID:-local}"
echo "Node        : ${SLURM_NODELIST:-local}"
echo "Model       : $MODEL"
echo "Fraction    : $FRACTION (100%)"
echo "Seed        : $SEED"
echo "Manifest    : $MANIFEST  (7 classes, hyrax EXCLUDED)"
echo "Cache       : $CACHE_DIR"
echo "Baseline F1 : $BASELINE_F1  (7-WAY, from Phase A A4 - NOT the 8-class value)"
echo "Output      : $OUT_DIR"
echo "Adapter     : $ADAPTER_DIR"
echo "HF_HOME     : $HF_HOME"
echo "Started     : $(date)"
echo "========================================"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: 7-class manifest not found: $MANIFEST"
    exit 1
fi

# --- preflight: manifest is really 7-class, and the cache it hashes to exists.
# The cache assertion is what makes the --max-windows-per-file 1 requirement
# enforceable rather than a comment nobody reads.
python - "$MANIFEST" "$CACHE_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path

manifest_path, cache_dir = sys.argv[1], Path(sys.argv[2])
m = json.load(open(manifest_path))

assert m['num_classes'] == 7, f"expected 7 classes, got {m['num_classes']}"
assert m.get('excluded_species') == ['hyrax'], \
    f"expected excluded_species == ['hyrax'], got {m.get('excluded_species')}"
assert 'hyrax' not in m['species_to_idx'], "hyrax still in species_to_idx"
bad = [it['file'] for s in ('train', 'val', 'test') for it in m['splits'][s]
       if str(it['file']).startswith('outputs/phase3/hyrax_data')]
assert not bad, f"{len(bad)} hyrax_data paths present, e.g. {bad[:3]}"

# Same key derivation as WindowedDataset._cache_key, with the values this
# script passes: 5.0s window / 2.5s stride / label key 'species' / mwpf 1.
def key(items):
    h = hashlib.md5()
    h.update(f"5.0|2.5|species|1".encode())
    for it in items:
        h.update(str(it['file']).encode())
    return h.hexdigest()[:12]

missing = []
for split in ('train', 'val', 'test'):
    k = key(m['splits'][split])
    for kind in ('windows', 'labels'):
        f = cache_dir / f"{split}_{k}_{kind}.npy"
        if not f.exists():
            missing.append(f.name)
    print(f"  {split}: expecting cache key {k}")

if missing:
    sys.exit(
        "\nERROR: expected window cache files are missing:\n  "
        + "\n  ".join(missing)
        + "\n\nRun run_phase3_species7_cache.sh first. Proceeding would make the\n"
          "trainer rebuild the cache from scratch - and if --max-windows-per-file 1\n"
          "were ever dropped, it would take EVERY window per file instead of one."
    )

print(f"Preflight OK: {m['num_classes']} classes {m['species']}")
print(f"  splits: {m['split_counts']}")
print("  window cache present for all three splits")
PY

if [ -f "$OUT_DIR/lora_fine_tuning_results.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "SKIP: $OUT_DIR/lora_fine_tuning_results.json already exists."
    echo "      Set FORCE=1 to re-run this cell."
    exit 0
fi

mkdir -p "$OUT_DIR"

python scripts/phase3_10_lora_fine_tuning.py \
    --model "$MODEL" \
    --manifest "$MANIFEST" \
    --output-dir "$OUT_DIR" \
    --cache-dir "$CACHE_DIR" \
    --save-adapter-dir "$ADAPTER_DIR" \
    --log-tag "staged_species7_${MODEL}_seed${SEED}" \
    --data-fraction "$FRACTION" \
    --baseline-f1 "$BASELINE_F1" \
    --max-windows-per-file 1 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --layerdrop 0.0 \
    --head-dropout 0.3 \
    --encoder-lr 1e-4 --head-lr 1e-3 \
    --batch-size 8 \
    --max-epochs 20 \
    --patience 6 --plateau-patience 3 \
    --class-weights window_inverse \
    --seed "$SEED"

echo ""
echo "=== Staged adapter gate (Phase B -> C handoff) ==="
python scripts/phase3_17_verify_staged_adapter.py --adapter-dir "$ADAPTER_DIR"

echo ""
echo "=== Per-epoch cost (for sizing Phase C) ==="
python - "$OUT_DIR/lora_fine_tuning_results.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
h = r['history']
secs = h.get('epoch_seconds') or []
if secs:
    n = len(secs)
    print(f"  epochs run      : {n}")
    print(f"  mean epoch      : {sum(secs)/n/60:.1f} min")
    print(f"  min / max epoch : {min(secs)/60:.1f} / {max(secs)/60:.1f} min")
    print(f"  total train time: {sum(secs)/3600:.2f} h")
else:
    print("  (no epoch timings - run predates the timing instrumentation)")
print(f"  best epoch      : {r['best_epoch']} (val macro-F1 {r['best_val_f1_macro']:.4f})")
t = r.get('test_metrics', {})
if t:
    print(f"  test macro-F1   : {t['f1_macro']:.4f} | acc {t['accuracy']:.4f}")
PY

echo ""
echo "========================================"
echo "Array task $IDX ($MODEL) complete: $(date)"
echo "Adapter: $ADAPTER_DIR"
echo "========================================"
