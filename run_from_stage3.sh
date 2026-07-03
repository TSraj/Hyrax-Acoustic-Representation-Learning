#!/bin/bash
# Resume Pipeline from Stage 3 (Pooled Evaluation)
# Stage 2 results already exist and will be reused

PROJECT_DIR="/home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning"

MODELS=("wav2vec2_base" "wav2vec2_base_960h" "xls_r" "wavlm" "ecapa_tdnn")

echo "=========================================="
echo "RESUMING PIPELINE FROM STAGE 3"
echo "=========================================="
echo "Stage 2 results: REUSING EXISTING"
echo "Stage 3: Submitting 5 pooled jobs"
echo "Stage 4-9: Full pipeline"
echo "=========================================="
echo ""

# Check HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. This may cause rate limiting."
    echo "Set it with: export HF_TOKEN=your_token"
    echo ""
fi

# Step 1: Stage 3 - All 5 pooled models
echo "[1/7] Submitting Stage 3: Pooled zero-shot (5 jobs)..."
STAGE3_JOBS=()

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    JOB=$(sbatch --parsable << STAGE3JOB
#!/bin/bash
#SBATCH --job-name=04_pooled_${model:0:12}
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/04_stage3_${model}_%j.out
#SBATCH --error=logs/04_stage3_${model}_%j.err

cd /home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning
module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate

# Use HF_TOKEN from environment
export HF_TOKEN=\${HF_TOKEN}

python scripts/phase2_03_zero_shot_pooled.py --model ${model}
STAGE3JOB
)
    STAGE3_JOBS+=($JOB)
    echo "  [$((i+1))/5] ${model}: $JOB"
done
echo ""

# Step 2: Aggregate Stage 3
echo "[2/7] Submitting Stage 3 aggregation..."
STAGE3_DEPS=$(IFS=:; echo "${STAGE3_JOBS[*]}")
JOB_AGG3=$(sbatch --parsable --dependency=afterok:$STAGE3_DEPS << 'AGGREGATE3'
#!/bin/bash
#SBATCH --job-name=05_aggregate3
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/05_aggregate3_%j.out
#SBATCH --error=logs/05_aggregate3_%j.err

cd /home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning
module load python/3.12-conda
source venv/bin/activate

python scripts/phase2_03_aggregate_pooled_results.py
AGGREGATE3
)
echo "  Aggregate Stage 3: $JOB_AGG3"
echo ""

# Step 3: Model selection
echo "[3/7] Submitting model selection..."
JOB_SELECT=$(sbatch --parsable --dependency=afterok:$JOB_AGG3 << 'SELECTION'
#!/bin/bash
#SBATCH --job-name=06_selection
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/06_selection_%j.out
#SBATCH --error=logs/06_selection_%j.err

cd /home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning
module load python/3.12-conda
source venv/bin/activate

python scripts/phase2_04_model_selection.py
SELECTION
)
echo "  Model Selection: $JOB_SELECT"
echo ""

# Step 4: Fine-tuning
echo "[4/7] Submitting fine-tuning..."
JOB_FINETUNE=$(sbatch --parsable --dependency=afterok:$JOB_SELECT << 'FINETUNE'
#!/bin/bash
#SBATCH --job-name=07_finetune
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/07_finetune_%j.out
#SBATCH --error=logs/07_finetune_%j.err

cd /home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning
module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate

# Use HF_TOKEN from environment
export HF_TOKEN=\${HF_TOKEN}

python scripts/phase2_05_fine_tuning.py
FINETUNE
)
echo "  Fine-tuning: $JOB_FINETUNE"
echo ""

# Step 5: Final report
echo "[5/7] Submitting final report..."
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
echo "PIPELINE RESUMED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  • Stage 2: REUSING EXISTING (35 jobs already completed)"
echo "  • Stage 3 (pooled): 5 jobs (~1h each with caching)"
echo "  • Model selection: 1 job (10 min)"
echo "  • Fine-tuning: 1 job (12h)"
echo "  • Final report: 1 job (10 min)"
echo ""
echo "Total remaining time: ~13-14 hours"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs directory: logs/"
echo ""
echo "=========================================="
