#!/bin/bash -l

#SBATCH --job-name=p2-report
#SBATCH --output=logs/out_%j.out
#SBATCH --error=logs/err_%j.err
#SBATCH --time=1:00:00
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
# Stage 7: Generate Final Report
# Needs:   All previous stage outputs (jobs 1-6)
# Outputs: outputs/phase2/final_report/
# -----------------------------------------------------------

echo "========================================"
echo "Stage 7: Generate Final Report"
echo "Started: $(date)"
echo "========================================"

python3 scripts/phase2_07_generate_final_report.py
echo ">>> Final report generated: $(date)"

echo "========================================"
echo "Phase 2 Pipeline COMPLETE: $(date)"
echo "========================================"
