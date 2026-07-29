#!/bin/bash
#SBATCH --job-name=phase3_lora_cache
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=logs/phase3_lora_cache_%j.out
#SBATCH --error=logs/phase3_lora_cache_%j.err

# Phase 3 - LoRA sweep, PREP JOB
#
# This job is CPU-only work (audio decoding), but TinyGPU rejects any job that
# does not allocate a GPU, so it requests one v100 like every other job in this
# repo. Headers match run_phase3_part1_zero_shot.sh, which is known to submit.
#
# Materialises the windowed-audio caches once, before the array runs. Without
# this, all 16 array tasks would decode the same audio concurrently - species_id
# alone is ~18k files, many of them mp3.
#
# Submit this first, then make the array depend on it:
#   JID=$(sbatch --parsable run_phase3_lora_cache.sh)
#   sbatch --dependency=afterok:$JID run_phase3_lora_sweep.sh

set -e

PROJECT_DIR="/home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning"
cd "$PROJECT_DIR"

module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate
mkdir -p logs

CACHE_DIR="outputs/phase3/window_cache"
HYRAX_MANIFEST="outputs/phase3/denoiser_screen/manifests/bioda/hyrax_id_session_holdout_ft.json"
SPECIES_MANIFEST="outputs/phase3/manifests/species_id.json"

echo "=== Building hyrax session-holdout caches ==="
python scripts/phase3_10_lora_fine_tuning.py \
    --model xls_r \
    --manifest "$HYRAX_MANIFEST" \
    --output-dir outputs/phase3/lora_sweep/_cache_build/hyrax \
    --cache-dir "$CACHE_DIR" \
    --build-cache-only

echo ""
echo "=== Building species_id caches (~18k files, this is the slow one) ==="
python scripts/phase3_10_lora_fine_tuning.py \
    --model xls_r \
    --manifest "$SPECIES_MANIFEST" \
    --output-dir outputs/phase3/lora_sweep/_cache_build/species \
    --cache-dir "$CACHE_DIR" \
    --max-windows-per-file 1 \
    --build-cache-only

echo ""
echo "Caches:"
ls -lh "$CACHE_DIR"
