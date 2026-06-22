#!/bin/bash -l
# ============================================================
# submit_pipeline.sh — Submit Phase 2 full pipeline to SLURM
# Usage: sbatch run/submit_pipeline.sh  (run from project root)
#
# Submits jobs with afterok dependencies so each stage only
# starts when its upstream stage has finished successfully.
# ============================================================

#SBATCH --job-name=p2-pipeline
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
# Submit all Phase 2 jobs with SLURM dependencies
#
# Execution order:
#   job1 → job2 \
#   job1 → job3  ├─ job4 → job5 → job7
#   job1 → job6 /                ↗
#                                job6 ─┘
# -----------------------------------------------------------

mkdir -p logs

echo "Submitting Phase 2 pipeline at $(date)"

JOB1=$(sbatch run/job1.sh | awk '{print $NF}')
echo "job1 (manifests)             → $JOB1"

JOB2=$(sbatch --dependency=afterok:$JOB1 run/job2.sh | awk '{print $NF}')
echo "job2 (per-dataset zero-shot) → $JOB2  [after $JOB1]"

JOB3=$(sbatch --dependency=afterok:$JOB1 run/job3.sh | awk '{print $NF}')
echo "job3 (pooled zero-shot)      → $JOB3  [after $JOB1]"

JOB6=$(sbatch --dependency=afterok:$JOB1 run/job6.sh | awk '{print $NF}')
echo "job6 (sampling rate)         → $JOB6  [after $JOB1]"

JOB4=$(sbatch --dependency=afterok:$JOB2:$JOB3 run/job4.sh | awk '{print $NF}')
echo "job4 (model selection)       → $JOB4  [after $JOB2 AND $JOB3]"

JOB5=$(sbatch --dependency=afterok:$JOB4 run/job5.sh | awk '{print $NF}')
echo "job5 (fine-tuning)           → $JOB5  [after $JOB4]"

JOB7=$(sbatch --dependency=afterok:$JOB5:$JOB6 run/job7.sh | awk '{print $NF}')
echo "job7 (final report)          → $JOB7  [after $JOB5 AND $JOB6]"

echo ""
echo "Monitor: squeue --job $JOB1,$JOB2,$JOB3,$JOB4,$JOB5,$JOB6,$JOB7"