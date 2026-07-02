#!/bin/bash
# Full Phase 2 Pipeline: 5 models × 7 datasets
# Estimated total time: 24-30 hours

PROJECT_DIR="/home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning"

# All models and datasets
MODELS=("wav2vec2_base" "wav2vec2_base_960h" "xls_r" "wavlm" "ecapa_tdnn")
DATASETS=("anuraset" "bengalese_finch" "macaque" "marmoset" "picidae" "wetlands_bird" "zebra_finch")

echo "=========================================="
echo "FULL PHASE 2 PIPELINE SUBMISSION"
echo "=========================================="
echo "Models: ${MODELS[@]}"
echo "Datasets: ${DATASETS[@]}"
echo "Stage 2: $((${#MODELS[@]} * ${#DATASETS[@]})) jobs (5 models × 7 datasets)"
echo "Stage 3: ${#MODELS[@]} jobs"
echo "Total: 46 jobs (1 validation + 35 stage2 + 1 agg2 + 5 stage3 + 1 agg3 + 1 selection + 1 finetune + 1 report)"
echo "Estimated completion: 24-30 hours"
echo "=========================================="
echo ""

# Step 1: Validate manifests
echo "[1/8] Submitting validation..."
JOB1=$(sbatch --parsable << 'VALIDATE'
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

# Step 2: Stage 2 - All 35 model/dataset combinations
echo "[2/8] Submitting Stage 2: Per-dataset zero-shot (35 jobs)..."
STAGE2_JOBS=()
job_count=0

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        job_count=$((job_count + 1))
        JOB=$(sbatch --parsable --dependency=afterok:$JOB1 << STAGE2JOB
#!/bin/bash
#SBATCH --job-name=02_${model:0:8}_${dataset:0:8}
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00
#SBATCH --output=logs/02_stage2_${model}_${dataset}_%j.out
#SBATCH --error=logs/02_stage2_${model}_${dataset}_%j.err

cd /home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning
module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate

python scripts/phase2_02_zero_shot_per_dataset.py --model ${model} --dataset ${dataset}
STAGE2JOB
)
        STAGE2_JOBS+=($JOB)
        echo "  [$job_count/35] ${model} × ${dataset}: $JOB"
    done
done
echo ""

# Step 3: Aggregate Stage 2
echo "[3/8] Submitting Stage 2 aggregation..."
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

# Step 4: Stage 3 - All 5 pooled models
echo "[4/8] Submitting Stage 3: Pooled zero-shot (5 jobs)..."
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

python scripts/phase2_03_zero_shot_pooled.py --model ${model}
STAGE3JOB
)
    STAGE3_JOBS+=($JOB)
    echo "  [$((i+1))/5] ${model}: $JOB"
done
echo ""

# Step 5: Aggregate Stage 3
echo "[5/8] Submitting Stage 3 aggregation..."
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

# Step 6: Model selection
echo "[6/8] Submitting model selection..."
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

# Step 7: Fine-tuning
echo "[7/8] Submitting fine-tuning..."
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

python scripts/phase2_05_fine_tuning.py
FINETUNE
)
echo "  Fine-tuning: $JOB_FINETUNE"
echo ""

# Step 8: Final report
echo "[8/8] Submitting final report..."
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
echo "Summary:"
echo "  • Validation: 1 job (10 min)"
echo "  • Stage 2 (per-dataset): 35 jobs (2.5h each, parallel)"
echo "  • Stage 3 (pooled): 5 jobs (6h each, parallel)"
echo "  • Model selection: 1 job (10 min)"
echo "  • Fine-tuning: 1 job (12h)"
echo "  • Final report: 1 job (10 min)"
echo ""
echo "Total: 46 jobs"
echo "Est. wall-clock time: 24-30 hours"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs directory: logs/"
echo ""
echo "=========================================="
