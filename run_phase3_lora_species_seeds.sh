#!/bin/bash
#SBATCH --job-name=phase3_lora_species_seeds
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --array=0-45%4
#SBATCH --output=logs/phase3_lora_species_seeds_%A_%a.out
#SBATCH --error=logs/phase3_lora_species_seeds_%A_%a.err

# Phase 3 - LoRA multi-seed + low-fraction runs (species_id task only)
#
# Closes two gaps left by the earlier sweeps:
#
#   1. species_id ran at a single seed (42) while the hyrax task got five
#      (42-46), so the large low-data gaps could not be given error bars:
#      10%  HuBERT 0.8294 vs XLS-R 0.7313
#      25%  HuBERT 0.9739 vs XLS-R 0.7672
#      The 25% XLS-R number is driven by two large-support classes collapsing
#      (bengalese_finch 0.104 with n=292, wetlands_bird 0.321 with n=79) after
#      that same model scored 0.998 on bengalese_finch at 10%. Getting *worse*
#      with more data is either a real multilingual-encoder instability or a
#      one-seed fluke, and n=1 cannot tell the two apart.
#
#   2. species_id saturates at >=50% (both models ~0.977), so the existing grid
#      has no resolution where the models actually differ. 1% / 2% / 5% put
#      points below the ceiling so the curve has visible slope.
#
#   fractions 1% / 2% / 5%    seeds 42-46 (new fractions)  = 3 x 2 x 5 = 30
#   fractions 10% / 25%       seeds 43-46 (42 exists)      = 2 x 2 x 4 = 16
#   total                                                  = 46 array tasks
#
# NOT TOUCHED: every hyrax run, the species 50%/100% runs, and the existing
# species seed-42 runs at 10%/25%. Those stay single-seed by design.
#
# Seed 42 is NOT re-run where it already exists. Re-running would not reproduce
# it anyway - cuDNN kernels are nondeterministic on GPU - so the existing
# results are reused as-is.
#
# RAGGED GRID. 10%/25% need 4 seeds and 1%/2%/5% need 5, so the modular index
# decoding used in run_phase3_lora_sweep.sh would leave dead array indices.
# This script builds an explicit flat job table instead, giving a contiguous
# 0-45 with no gaps. Print it before submitting:
#
#   bash run_phase3_lora_species_seeds.sh --list
#
# OUTPUT PATHS. Everything lands in .../frac<NN>/seed<S>/ - never at the frac
# directory top level, which is where the existing seed-42 10%/25% results
# live. The seed MUST be in the path: checkpoint.pt is written per output
# directory, so seeds sharing a directory would resume each other's runs.
#
# NO OVERWRITES. A task whose lora_fine_tuning_results.json already exists is
# skipped, so the whole array can be resubmitted safely to fill in stragglers.
# Set FORCE=1 to re-run a completed cell anyway.
#
# WALL CLOCK. The largest run here is 25% (3,649 train windows, 457 batches per
# epoch) - a quarter of the 100% run that fit inside the original sweep's 24h.
# 12h is generous, and every task checkpoints per epoch, so a task killed at the
# wall clock resumes from the last completed epoch on resubmission:
#   sbatch --array=17,31 run_phase3_lora_species_seeds.sh
#
# CACHE. The window cache key hashes (window params | label key |
# max_windows_per_file | file list) and includes neither the seed nor the data
# fraction, so all 46 runs reuse the exact float16 cache the seed-42 species
# runs were trained on. Nothing is regenerated. Subsampling happens after the
# cache is memory-mapped, on an index array only.
#
# SUBMIT: use run_phase3_lora_species_submit.sh, which chains the (idempotent)
# cache prep job in front of the array.

set -e

MODELS=("hubert_base" "xls_r")
NEW_FRACTION_SEEDS=(42 43 44 45 46)   # 1% / 2% / 5%: nothing exists yet
TOPUP_SEEDS=(43 44 45 46)             # 10% / 25%: seed 42 already done

# --- build the flat job table: "<model> <fraction> <frac_tag> <seed>" --------
# The tag is carried explicitly rather than derived with bc, so the output path
# can never drift from the fraction actually passed to the trainer.
JOBS=()
add_jobs() {
    local frac="$1" tag="$2"; shift 2
    local model seed
    for model in "${MODELS[@]}"; do
        for seed in "$@"; do
            JOBS+=("$model $frac $tag $seed")
        done
    done
}

add_jobs 0.01 1  "${NEW_FRACTION_SEEDS[@]}"
add_jobs 0.02 2  "${NEW_FRACTION_SEEDS[@]}"
add_jobs 0.05 5  "${NEW_FRACTION_SEEDS[@]}"
add_jobs 0.10 10 "${TOPUP_SEEDS[@]}"
add_jobs 0.25 25 "${TOPUP_SEEDS[@]}"

N_JOBS=${#JOBS[@]}

if [ "${1:-}" = "--list" ]; then
    echo "Job table: $N_JOBS tasks  ->  --array=0-$((N_JOBS - 1))%4"
    echo ""
    printf "%5s  %-12s %-9s %s\n" "index" "model" "fraction" "seed"
    for i in "${!JOBS[@]}"; do
        read -r m f t s <<< "${JOBS[$i]}"
        printf "%5d  %-12s %-9s %s\n" "$i" "$m" "${t}%" "$s"
    done
    exit 0
fi

PROJECT_DIR="/home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning"
cd "$PROJECT_DIR"

module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate
mkdir -p logs

CACHE_DIR="outputs/phase3/window_cache"
SWEEP_ROOT="outputs/phase3/lora_sweep"
MANIFEST="outputs/phase3/manifests/species_id.json"
TASK="species_id"

IDX=$SLURM_ARRAY_TASK_ID

# Guard against an --array range wider than the table (would silently no-op).
if [ "$IDX" -ge "$N_JOBS" ]; then
    echo "ERROR: array index $IDX is outside the job table (0-$((N_JOBS - 1)))."
    exit 1
fi

read -r MODEL FRACTION FRAC_TAG SEED <<< "${JOBS[$IDX]}"

# Zero-shot macro-F1 baseline for the curve annotation (species_id, frozen
# encoder). Same values the seed-42 species runs were annotated with.
if [ "$MODEL" = "xls_r" ]; then BASELINE_F1=0.7194; else BASELINE_F1=0.8635; fi

OUT_DIR="$SWEEP_ROOT/$TASK/$MODEL/frac${FRAC_TAG}/seed${SEED}"

echo "========================================"
echo "PHASE 3 - LoRA SPECIES SEEDS + LOW FRACTIONS"
echo "Array task : $IDX / $((N_JOBS - 1))"
echo "Job        : $SLURM_ARRAY_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Model      : $MODEL"
echo "Task       : $TASK"
echo "Fraction   : $FRACTION (${FRAC_TAG}%)"
echo "Seed       : $SEED"
echo "Manifest   : $MANIFEST"
echo "Baseline F1: $BASELINE_F1"
echo "Output     : $OUT_DIR"
echo "Started    : $(date)"
echo "========================================"

# Never write to the frac-directory top level: that is where the existing
# seed-42 10%/25% results live.
case "$OUT_DIR" in
    */seed[0-9]*) ;;
    *) echo "ERROR: refusing to run, output path has no seed component: $OUT_DIR"; exit 1 ;;
esac

if [ -f "$OUT_DIR/lora_fine_tuning_results.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "SKIP: $OUT_DIR/lora_fine_tuning_results.json already exists."
    echo "      Set FORCE=1 to re-run this cell."
    exit 0
fi

mkdir -p "$OUT_DIR"

# Config is byte-for-byte the one used by run_phase3_lora_sweep.sh for
# species_id. Only --data-fraction, --seed and the output directory differ.
python scripts/phase3_10_lora_fine_tuning.py \
    --model "$MODEL" \
    --manifest "$MANIFEST" \
    --output-dir "$OUT_DIR" \
    --cache-dir "$CACHE_DIR" \
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
echo "========================================"
echo "Array task $IDX complete: $(date)"
echo "Results: $OUT_DIR/lora_fine_tuning_results.json"
echo "========================================"
