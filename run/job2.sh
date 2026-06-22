#!/bin/bash -l

#SBATCH --job-name=p2-zero-shot
#SBATCH --output=logs/out_%j.out
#SBATCH --error=logs/err_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tanver.s.raj@fau.de

# -----------------------------------------------------------
# Environment
# -----------------------------------------------------------

unset SLURM_EXPORT_ENV
set -e
cd "$SLURM_SUBMIT_DIR"

module purge
module load python/3.12-conda
source venv/bin/activate

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# -----------------------------------------------------------
# Stage 2: Zero-Shot Evaluation — All 5 Models × 7 Datasets (35 combos)
# Needs:   outputs/phase2/manifests/  (from job1)
# Outputs: outputs/phase2/zero_shot/per_dataset/
# -----------------------------------------------------------

echo "========================================"
echo "Stage 2: Per-Dataset Zero-Shot Evaluation"
echo "Started: $(date)"
echo "========================================"

bash scripts/phase2_02_run_all_combinations.sh
echo ">>> All 35 combinations done: $(date)"

python3 scripts/phase2_02_aggregate_results.py
echo ">>> Results aggregated: $(date)"

echo "Stage 2 complete: $(date)"
