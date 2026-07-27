#!/bin/bash
#SBATCH --job-name=test_ft_fix
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=15:00:00
#SBATCH --output=logs/test_ft_fix_%j.out
#SBATCH --error=logs/test_ft_fix_%j.err

# TEST: Validate fine-tuning fix on XLS-R species_id 100% before launching all 16 runs

cd /home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning

module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate

echo "========================================"
echo "FINE-TUNING FIX VALIDATION TEST"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================"
echo ""
echo "Testing: XLS-R, species_id, 100% data"
echo "Expected: >89% (zero-shot baseline)"
echo "Old broken: 22% (catastrophic failure)"
echo ""
echo "========================================"

python scripts/phase3_05_fine_tuning.py --model xls_r --task species_id

echo ""
echo "========================================"
echo "TEST COMPLETE"
echo "Finished: $(date)"
echo "========================================"
echo ""
echo "Check: outputs/phase3/fine_tuning/species_id/xls_r/fine_tuning_report.txt"
echo "If 100% fraction > 89% → Fix works! Launch all 16 runs"
echo "If 100% fraction ≈ 22% → Fix failed, switch to LoRA"
