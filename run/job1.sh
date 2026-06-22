#!/bin/bash -l

#SBATCH --job-name=p2-manifests
#SBATCH --output=logs/out_%j.out
#SBATCH --error=logs/err_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
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
# Stage 1: Create & Validate Manifests
# Outputs: outputs/phase2/manifests/
# -----------------------------------------------------------

echo "========================================"
echo "Stage 1: Create & Validate Manifests"
echo "Started: $(date)"
echo "Working dir: $(pwd)"
echo "========================================"

python3 scripts/phase2_01_create_manifests.py
echo ">>> Manifests created: $(date)"

python3 scripts/phase2_01b_validate_manifests.py
echo ">>> Manifests validated: $(date)"

echo "Stage 1 complete: $(date)"
