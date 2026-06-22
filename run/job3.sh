#!/bin/bash -l
set -e

#SBATCH --job-name=p2-pooled
#SBATCH --output=logs/out_%j.out
#SBATCH --error=logs/err_%j.err
#SBATCH --time=4:00:00
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
# Stage 3: Pooled Zero-Shot Evaluation — All 5 Models on combined dataset
# Needs:   outputs/phase2/manifests/  (from job1)
# Outputs: outputs/phase2/zero_shot/pooled/
# -----------------------------------------------------------

echo "========================================"
echo "Stage 3: Pooled Zero-Shot Evaluation"
echo "Started: $(date)"
echo "========================================"

bash scripts/phase2_03_run_pooled_all_models.sh
echo ">>> All 5 pooled evaluations done: $(date)"

python3 scripts/phase2_03_aggregate_pooled_results.py
echo ">>> Pooled results aggregated: $(date)"

echo "Stage 3 complete: $(date)"
