#!/bin/bash
# Run remaining jobs: Fine-tuning + Final Report
# Stage 3 (pooled) completed successfully

PROJECT_DIR="/home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning"

echo "=========================================="
echo "RUNNING REMAINING JOBS"
echo "=========================================="
echo "Jobs to run:"
echo "  1. Fine-tuning (batch_size=8)"
echo "  2. Final report"
echo "=========================================="
echo ""

# Check HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. This may cause rate limiting."
    echo "Set it with: export HF_TOKEN=your_token"
    echo ""
fi

# Step 1: Fine-tuning
echo "[1/2] Submitting fine-tuning (batch_size=8)..."
JOB_FINETUNE=$(sbatch --parsable << 'FINETUNE'
#!/bin/bash
#SBATCH --job-name=07_finetune
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/07_finetune_%j.out
#SBATCH --error=logs/07_finetune_%j.err

cd /home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning
module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate

# Use HF_TOKEN from environment
export HF_TOKEN=${HF_TOKEN}

python scripts/phase2_05_fine_tuning.py
FINETUNE
)
echo "  Fine-tuning: $JOB_FINETUNE"
echo ""

# Step 2: Final report
echo "[2/2] Submitting final report..."
JOB_REPORT=$(sbatch --parsable --dependency=afterok:$JOB_FINETUNE << 'REPORT'
#!/bin/bash
#SBATCH --job-name=08_report
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/08_report_%j.out
#SBATCH --error=logs/08_report_%j.err

cd /home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning
module load python/3.12-conda
source venv/bin/activate

python scripts/phase2_07_generate_final_report.py
REPORT
)
echo "  Final Report: $JOB_REPORT"
echo ""

echo "=========================================="
echo "JOBS SUBMITTED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  • Fine-tuning: 1 job (24h, batch_size=8, max_epochs=16)"
echo "  • Final report: 1 job (10 min)"
echo ""
echo "Total time: ~19 hours (may finish earlier with early stopping)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs directory: logs/"
echo ""
echo "=========================================="
