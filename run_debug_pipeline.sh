#!/bin/bash
# Phase 2 Pipeline Script
# Usage:
#   - Debug mode (default): bash run_debug_pipeline.sh
#   - Full mode: bash run_debug_pipeline.sh --full

PROJECT_DIR="/home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning"

# Check mode
if [[ "$1" == "--full" ]]; then
    MODE="full"
    echo "=========================================="
    echo "FULL PHASE 2 PIPELINE"
    echo "=========================================="
    MODELS=("wav2vec2_base" "wav2vec2_base_960h" "xls_r" "wavlm" "ecapa_tdnn")
    DATASETS=("anuraset" "bengalese_finch" "macaque" "marmoset" "picidae" "wetlands_bird" "zebra_finch")
    STAGE2_TIME="02:30:00"
    STAGE3_TIME="06:00:00"
    FINETUNE_TIME="12:00:00"
    DEBUG_FLAG=""
    echo "Models: ${MODELS[@]}"
    echo "Datasets: ${DATASETS[@]}"
    echo "Total Stage 2 jobs: $((${#MODELS[@]} * ${#DATASETS[@]})) = 35"
    echo "Total Stage 3 jobs: ${#MODELS[@]} = 5"
    echo "Estimated completion: 24-30 hours"
else
    MODE="debug"
    echo "=========================================="
    echo "DEBUG PHASE 2 PIPELINE"
    echo "=========================================="
    MODELS=("wav2vec2_base")
    DATASETS=("anuraset" "picidae")
    STAGE2_TIME="00:45:00"
    STAGE3_TIME="00:45:00"
    FINETUNE_TIME="01:00:00"
    DEBUG_FLAG="--debug"
    echo "Models: ${MODELS[@]}"
    echo "Datasets: ${DATASETS[@]}"
    echo "Total Stage 2 jobs: 2"
    echo "Total Stage 3 jobs: 2 (wav2vec2_base, ecapa_tdnn)"
    echo "Estimated completion: 2-3 hours"
fi
echo "=========================================="
echo ""

echo "Submitting pipeline jobs..."

# Step 1: Validate manifests (needs GPU allocation even though not used)
JOB1=$(sbatch --parsable << VALIDATE
#!/bin/bash
#SBATCH --job-name=01_validate
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/01_validate_%j.out
#SBATCH --error=logs/01_validate_%j.err

cd $PROJECT_DIR
module load python/3.12-conda
source venv/bin/activate

python scripts/phase2_01b_validate_manifests.py
VALIDATE
)
echo "Job 1 (Validate): $JOB1"

# Step 2: Stage 2 - Test 2 datasets (GPU, depends on validation)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 << STAGE2A
#!/bin/bash
#SBATCH --job-name=02a_stage2_anuraset
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=logs/02a_stage2_wav2vec2_anuraset_%j.out
#SBATCH --error=logs/02a_stage2_wav2vec2_anuraset_%j.err

cd $PROJECT_DIR
module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate

python scripts/phase2_02_zero_shot_per_dataset.py \
    --model wav2vec2_base \
    --dataset anuraset \
    --debug
STAGE2A
)
echo "Job 2a (Stage2 wav2vec2+anuraset): $JOB2"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 << STAGE2B
#!/bin/bash
#SBATCH --job-name=02b_stage2_picidae
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=logs/02b_stage2_wav2vec2_picidae_%j.out
#SBATCH --error=logs/02b_stage2_wav2vec2_picidae_%j.err

cd $PROJECT_DIR
module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate

python scripts/phase2_02_zero_shot_per_dataset.py \
    --model wav2vec2_base \
    --dataset picidae \
    --debug
STAGE2B
)
echo "Job 2b (Stage2 wav2vec2+picidae): $JOB3"

# Step 3: Aggregate Stage 2 (needs GPU allocation even though not used)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB2:$JOB3 << AGGREGATE2
#!/bin/bash
#SBATCH --job-name=03_aggregate2
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/03_aggregate2_%j.out
#SBATCH --error=logs/03_aggregate2_%j.err

cd $PROJECT_DIR
module load python/3.12-conda
source venv/bin/activate

python scripts/phase2_02_aggregate_results.py
AGGREGATE2
)
echo "Job 3 (Aggregate Stage2): $JOB4"

# Step 4: Stage 3 - Test 2 pooled models (GPU, depends on aggregate2)
JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 << STAGE3A
#!/bin/bash
#SBATCH --job-name=04a_stage3_wav2vec2
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=logs/04a_stage3_wav2vec2_%j.out
#SBATCH --error=logs/04a_stage3_wav2vec2_%j.err

cd $PROJECT_DIR
module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate

python scripts/phase2_03_zero_shot_pooled.py \
    --model wav2vec2_base \
    --debug
STAGE3A
)
echo "Job 4a (Stage3 wav2vec2): $JOB5"

JOB6=$(sbatch --parsable --dependency=afterok:$JOB4 << STAGE3B
#!/bin/bash
#SBATCH --job-name=04b_stage3_ecapa
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=logs/04b_stage3_ecapa_%j.out
#SBATCH --error=logs/04b_stage3_ecapa_%j.err

cd $PROJECT_DIR
module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate

python scripts/phase2_03_zero_shot_pooled.py \
    --model ecapa_tdnn \
    --debug
STAGE3B
)
echo "Job 4b (Stage3 ecapa): $JOB6"

# Step 5: Aggregate Stage 3 (needs GPU allocation even though not used)
JOB7=$(sbatch --parsable --dependency=afterok:$JOB5:$JOB6 << AGGREGATE3
#!/bin/bash
#SBATCH --job-name=05_aggregate3
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/05_aggregate3_%j.out
#SBATCH --error=logs/05_aggregate3_%j.err

cd $PROJECT_DIR
module load python/3.12-conda
source venv/bin/activate

python scripts/phase2_03_aggregate_pooled_results.py
AGGREGATE3
)
echo "Job 5 (Aggregate Stage3): $JOB7"

# Step 6: Model selection (needs GPU allocation even though not used)
JOB8=$(sbatch --parsable --dependency=afterok:$JOB7 << SELECTION
#!/bin/bash
#SBATCH --job-name=06_selection
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/06_selection_%j.out
#SBATCH --error=logs/06_selection_%j.err

cd $PROJECT_DIR
module load python/3.12-conda
source venv/bin/activate

python scripts/phase2_04_model_selection.py
SELECTION
)
echo "Job 6 (Model Selection): $JOB8"

# Step 7: Fine-tuning (GPU, depends on selection)
JOB9=$(sbatch --parsable --dependency=afterok:$JOB8 << FINETUNE
#!/bin/bash
#SBATCH --job-name=07_finetune
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/07_finetune_%j.out
#SBATCH --error=logs/07_finetune_%j.err

cd $PROJECT_DIR
module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate

python scripts/phase2_05_fine_tuning.py --debug
FINETUNE
)
echo "Job 7 (Fine-tuning): $JOB9"

# Step 8: Final report (needs GPU allocation even though not used)
JOB10=$(sbatch --parsable --dependency=afterok:$JOB9 << REPORT
#!/bin/bash
#SBATCH --job-name=08_report
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/08_report_%j.out
#SBATCH --error=logs/08_report_%j.err

cd $PROJECT_DIR
module load python/3.12-conda
source venv/bin/activate

python scripts/phase2_07_generate_final_report.py
REPORT
)
echo "Job 8 (Final Report): $JOB10"

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo "Monitor with: squeue -u \$USER"
echo "Check logs in: logs/"
echo ""
echo "Job chain:"
echo "1. Validate manifests -> Job $JOB1"
echo "2a. Stage2 wav2vec2+anuraset -> Job $JOB2"
echo "2b. Stage2 wav2vec2+picidae -> Job $JOB3"
echo "3. Aggregate Stage2 -> Job $JOB4"
echo "4a. Stage3 wav2vec2 pooled -> Job $JOB5"
echo "4b. Stage3 ecapa pooled -> Job $JOB6"
echo "5. Aggregate Stage3 -> Job $JOB7"
echo "6. Model selection -> Job $JOB8"
echo "7. Fine-tuning -> Job $JOB9"
echo "8. Final report -> Job $JOB10"
echo ""
echo "Estimated completion: 2-3 hours"
