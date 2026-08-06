#!/bin/bash
#SBATCH --job-name=phase3_species7_cache
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:30:00
#SBATCH --output=logs/phase3_species7_cache_%j.out
#SBATCH --error=logs/phase3_species7_cache_%j.err

# Phase A / Step A3 - window cache for the 7-CLASS species task (hyrax excluded)
#
# WHY A NEW CACHE. The cache key is md5(window params | label key |
# max_windows_per_file | every file path). Dropping the 18 hyrax items changes
# the file list, so all three splits hash to new names:
#
#     split   7-class        8-class
#     train   770e674c6c16   68b595a7064e
#     val     d93b05450547   9785c87b0ea5
#     test    c4c57e75f522   bab3438c5141
#
# Verified distinct by scripts/phase3_16_verify_species7.py. A collision would
# silently feed hyrax-contaminated windows into the 7-class run, which is the
# one thing this whole phase exists to prevent.
#
# ISOLATION. Writes to a SEPARATE directory (window_cache_species7), so the
# 8-class cache is not merely un-collided but physically untouched. The 8-class
# species sweep is NOT being re-run - those results stay as they are.
#
# METHOD-NEUTRAL. This cache is shared by every adaptation method that runs on
# the 7-class task (LoRA and the first-4-layers port both read it), so the two
# are trained on byte-identical windows and the comparison is clean. Nothing
# here is LoRA-specific: phase3_10 is invoked only for its windowing code via
# --build-cache-only, which loads no model. If a method ever needs different
# window parameters, it gets its own cache key automatically - the params are
# part of the hash - so no path here needs changing.
#
# COST. 18162 files to decode (14584 train / 1789 val / 1789 test). Measured
# per-file decode on this manifest: anuraset ~425 ms dominates, wetlands_bird
# mp3 ~53 ms, everything else 1-3 ms => ~12 min single-process locally. 1h30
# is generous headroom for the shared filesystem.
#
# SIZE. max_windows_per_file=1 => one 5s float16 window per file:
#   train 14584 x 80000 x 2B = 2.33 GB | val 0.29 GB | test 0.29 GB = ~2.9 GB
# Check quota before submitting; the 8-class cache already occupies ~2.9 GB.
#
# IDEMPOTENT. phase3_10 reuses any cache file that already exists, so this job
# is safe to resubmit. Pass REBUILD=1 to force a regeneration.
#
# SUBMIT:
#   sbatch run_phase3_species7_cache.sh
# then re-run the gate to complete cache checks 9-11:
#   python scripts/phase3_16_verify_species7.py \
#       --cache-dir outputs/phase3/window_cache_species7

set -e

PROJECT_DIR="/home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning"
cd "$PROJECT_DIR"

export HF_HOME=$WORK/hf_cache/huggingface

module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate
mkdir -p logs

MANIFEST="outputs/phase3/manifests_species7/species_id.json"
CACHE_DIR="outputs/phase3/window_cache_species7"
OLD_CACHE_DIR="outputs/phase3/window_cache"

REBUILD_FLAG=""
if [ "${REBUILD:-0}" = "1" ]; then
    REBUILD_FLAG="--rebuild-cache"
fi

echo "========================================"
echo "PHASE A - 7-CLASS SPECIES WINDOW CACHE"
echo "Job       : ${SLURM_JOB_ID:-local}"
echo "Node      : ${SLURM_NODELIST:-local}"
echo "Manifest  : $MANIFEST"
echo "Cache dir : $CACHE_DIR  (NEW - 8-class cache untouched)"
echo "HF_HOME   : $HF_HOME"
echo "Rebuild   : ${REBUILD:-0}"
echo "Started   : $(date)"
echo "========================================"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: 7-class manifest not found: $MANIFEST"
    echo "Build it first (A2):"
    echo "  python scripts/phase3_02_create_manifests.py --tasks species_only \\"
    echo "      --exclude-species hyrax \\"
    echo "      --output-dir outputs/phase3/manifests_species7 --log-tag species7"
    exit 1
fi

# Refuse to write into the 8-class cache directory under any circumstance.
if [ "$CACHE_DIR" = "$OLD_CACHE_DIR" ]; then
    echo "ERROR: refusing to run - cache dir equals the 8-class cache dir."
    exit 1
fi

# Fail fast if the manifest is not actually the 7-class one.
python - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m['num_classes'] == 7, f"expected 7 classes, got {m['num_classes']}"
assert m.get('excluded_species') == ['hyrax'], \
    f"expected excluded_species == ['hyrax'], got {m.get('excluded_species')}"
assert 'hyrax' not in m['species_to_idx'], "hyrax still in species_to_idx"
bad = [it['file'] for s in ('train', 'val', 'test') for it in m['splits'][s]
       if str(it['file']).startswith('outputs/phase3/hyrax_data')]
assert not bad, f"{len(bad)} hyrax_data paths still present, e.g. {bad[:3]}"
print(f"Manifest OK: {m['num_classes']} classes {m['species']}")
print("  split sizes:", m['split_counts'])
print("  removed by exclusion:", m['excluded_item_counts'])
PY

echo ""
echo "=== Building 7-class species caches (18162 files; anuraset + mp3 dominate) ==="
python scripts/phase3_10_lora_fine_tuning.py \
    --model xls_r \
    --manifest "$MANIFEST" \
    --output-dir outputs/phase3/_cache_build/species7 \
    --cache-dir "$CACHE_DIR" \
    --max-windows-per-file 1 \
    --build-cache-only \
    $REBUILD_FLAG

echo ""
echo "=== New 7-class cache ==="
ls -lh "$CACHE_DIR"

echo ""
echo "=== 8-class cache (must be unchanged) ==="
ls -lh "$OLD_CACHE_DIR"

echo ""
echo "=== Verification gate ==="
python scripts/phase3_16_verify_species7.py --cache-dir "$CACHE_DIR"

echo ""
echo "========================================"
echo "Cache prep complete: $(date)"
echo "========================================"
