#!/bin/bash -l
set -e

#SBATCH --job-name=p2-model-select
#SBATCH --output=logs/out_%j.out
#SBATCH --error=logs/err_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tanver.s.raj@fau.de

# -----------------------------------------------------------
# Environment
# -----------------------------------------------------------

unset SLURM_EXPORT_ENV
cd "$SLURM_SUBMIT_DIR"

module purge
module load python/3.12-conda
source venv/bin/activate

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# -----------------------------------------------------------
# Stage 4: Model Comparison & Selection
# Needs:   outputs/phase2/zero_shot/per_dataset/  (from job2)
#          outputs/phase2/zero_shot/pooled/        (from job3)
# Outputs: outputs/phase2/model_selection/
# -----------------------------------------------------------

echo "========================================"
echo "Stage 4: Model Comparison & Selection"
echo "Started: $(date)"
echo "========================================"

python3 scripts/phase2_04_model_selection.py
echo ">>> Best model selected: $(date)"

echo "Stage 4 complete: $(date)"
