#!/bin/bash
# Full Phase 2 Pipeline with Staged Submission
# Prevents cluster overload by submitting in batches

PROJECT_DIR="/home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning"

# All models and datasets
MODELS=("wav2vec2_base" "wav2vec2_base_960h" "xls_r" "wavlm" "ecapa_tdnn")
DATASETS=("anuraset" "bengalese_finch" "macaque" "marmoset" "picidae" "wetlands_bird" "zebra_finch")

# Configuration
BATCH_SIZE=7  # Submit 7 jobs at a time
STAGE2_TIME="04:00:00"  # Increased from 02:30:00 to handle downloads

echo "=========================================="
echo "PHASE 2 PIPELINE - STAGED SUBMISSION"
echo "=========================================="
echo "Models: ${MODELS[@]}"
echo "Datasets: ${DATASETS[@]}"
echo "Stage 2: Batched submission (${BATCH_SIZE} jobs at a time)"
echo "Time per Stage 2 job: ${STAGE2_TIME}"
echo "=========================================="
echo ""

# Check HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. This may cause rate limiting."
    echo "Set it with: export HF_TOKEN=your_token"
    echo ""
fi

# Step 1: Create manifests
echo "[1/9] Submitting manifest creation..."
JOB0=$(sbatch --parsable << 'MANIFEST'
#!/bin/bash
#SBATCH --job-name=00_manifest
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/00_manifest_%j.out
#SBATCH --error=logs/00_manifest_%j.err

cd /home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning
module load python/3.12-conda
source venv/bin/activate

python scripts/phase2_01_create_manifests.py
MANIFEST
)
echo "  Manifest Creation: $JOB0"
echo ""

# Step 2: Validate manifests
echo "[2/9] Submitting validation..."
JOB1=$(sbatch --parsable --dependency=afterok:$JOB0 << 'VALIDATE'
#!/bin/bash
#SBATCH --job-name=01_validate
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/01_validate_%j.out
#SBATCH --error=logs/01_validate_%j.err

cd /home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning
module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate

python scripts/phase2_01b_validate_manifests.py
VALIDATE
)
echo "  Validation: $JOB1"
echo ""

# Step 3: Stage 2 - Batched submission
echo "[3/9] Submitting Stage 2: Per-dataset zero-shot (batched)..."
STAGE2_JOBS=()
job_count=0
batch_count=0

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        job_count=$((job_count + 1))

        # Determine dependency (batch after previous batch completes)
        if [ $job_count -le $BATCH_SIZE ]; then
            # First batch depends on validation
            DEPENDENCY="--dependency=afterok:$JOB1"
        else
            # Subsequent batches depend on previous batch
            prev_batch_start=$(( (batch_count - 1) * BATCH_SIZE ))
            prev_batch_end=$(( prev_batch_start + BATCH_SIZE - 1 ))
            if [ $prev_batch_end -ge ${#STAGE2_JOBS[@]} ]; then
                prev_batch_end=$(( ${#STAGE2_JOBS[@]} - 1 ))
            fi
            prev_batch_jobs="${STAGE2_JOBS[@]:$prev_batch_start:$BATCH_SIZE}"
            DEPENDENCY="--dependency=afterany:$(IFS=:; echo "${prev_batch_jobs// /:}")"
        fi

        JOB=$(sbatch --parsable $DEPENDENCY << STAGE2JOB
#!/bin/bash
#SBATCH --job-name=02_${model:0:8}_${dataset:0:8}
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=${STAGE2_TIME}
#SBATCH --output=logs/02_stage2_${model}_${dataset}_%j.out
#SBATCH --error=logs/02_stage2_${model}_${dataset}_%j.err

cd /home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning
module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate

# Use HF_TOKEN from environment
export HF_TOKEN=\${HF_TOKEN}

python scripts/phase2_02_zero_shot_per_dataset.py --model ${model} --dataset ${dataset}
STAGE2JOB
)
        STAGE2_JOBS+=($JOB)
        echo "  [$job_count/35] ${model} × ${dataset}: $JOB"

        # Update batch counter
        if [ $((job_count % BATCH_SIZE)) -eq 0 ]; then
            batch_count=$((batch_count + 1))
        fi
    done
done
echo ""

# Step 4: Aggregate Stage 2
echo "[4/9] Submitting Stage 2 aggregation..."
STAGE2_DEPS=$(IFS=:; echo "${STAGE2_JOBS[*]}")
JOB_AGG2=$(sbatch --parsable --dependency=afterok:$STAGE2_DEPS << 'AGGREGATE2'
#!/bin/bash
#SBATCH --job-name=03_aggregate2
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/03_aggregate2_%j.out
#SBATCH --error=logs/03_aggregate2_%j.err

cd /home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning
module load python/3.12-conda
source venv/bin/activate

python scripts/phase2_02_aggregate_results.py
AGGREGATE2
)
echo "  Aggregate Stage 2: $JOB_AGG2"
echo ""

# Step 5: Stage 3 - All 5 pooled models
echo "[5/9] Submitting Stage 3: Pooled zero-shot (5 jobs)..."
STAGE3_JOBS=()

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    JOB=$(sbatch --parsable --dependency=afterok:$JOB_AGG2 << STAGE3JOB
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

# Step 6: Aggregate Stage 3
echo "[6/9] Submitting Stage 3 aggregation..."
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

# Step 7: Model selection
echo "[7/9] Submitting model selection..."
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

# Step 8: Fine-tuning
echo "[8/9] Submitting fine-tuning..."
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

# Step 9: Final report
echo "[9/9] Submitting final report..."
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
echo "PIPELINE SUBMITTED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Improvements in this version:"
echo "  ✓ HF_TOKEN environment variable used (no rate limits)"
echo "  ✓ Stage 2 time increased: 02:30 → 04:00 hours"
echo "  ✓ Batched submission: $BATCH_SIZE jobs at a time"
echo "  ✓ Each batch waits for previous to complete"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs directory: logs/"
echo ""
echo "=========================================="
