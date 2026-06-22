#!/bin/bash -l
set -e

#SBATCH --job-name=p2-finetune
#SBATCH --output=logs/out_%j.out
#SBATCH --error=logs/err_%j.err
#SBATCH --time=24:00:00
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
cd "$SLURM_SUBMIT_DIR"

module purge
module load python/3.12-conda
source venv/bin/activate

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# -----------------------------------------------------------
# Stage 5: Fine-Tuning the Best Model (first 4 layers)
# Needs:   outputs/phase2/model_selection/  (from job4)
# Outputs: outputs/phase2/fine_tuning/
# -----------------------------------------------------------

echo "========================================"
echo "Stage 5: Fine-Tuning Best Model"
echo "Started: $(date)"
echo "========================================"

python3 scripts/phase2_05_fine_tuning.py
echo ">>> Fine-tuning complete: $(date)"

echo "Stage 5 complete: $(date)"
