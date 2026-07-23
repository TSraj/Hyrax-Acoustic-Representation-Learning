#!/bin/bash
#SBATCH --job-name=hubert_phase2
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/hubert_phase2_%j.out
#SBATCH --error=logs/hubert_phase2_%j.err

# Full HuBERT Phase 2 Pipeline (HPC)
# Runs HuBERT through complete zero-shot + model selection pipeline

set -e  # Exit on error

PROJECT_DIR="/home/hpc/iwi5/iwi5452h/project/Hyrax-Acoustic-Representation-Learning"
cd "$PROJECT_DIR"

echo "=========================================="
echo "HuBERT Phase 2 - Full Pipeline (HPC)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Check HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. This may cause rate limiting."
    echo "Set it with: export HF_TOKEN=your_token"
    echo ""
fi

# Load modules
module load cuda/11.8.0
module load python/3.12-conda
source venv/bin/activate

# Configuration
MODEL="hubert_base"
CONFIG_FILE="configs/config.yaml"
DATA_DIR="data/bird_datasets"
OUTPUT_BASE="outputs/phase2"
MANIFEST_DIR="$OUTPUT_BASE/manifests"

# Create logs directory
mkdir -p logs

# Pre-download HuBERT model (avoid repeated downloads)
echo ""
echo "Pre-downloading HuBERT model..."
python3 << EOF
from transformers import HubertModel, Wav2Vec2FeatureExtractor
model_id = "facebook/hubert-base-ls960"
print(f"Downloading {model_id}...")
_ = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
_ = HubertModel.from_pretrained(model_id, use_safetensors=True)
print("✓ Model downloaded and cached")
EOF

# Step 1: Zero-shot evaluation on all 7 datasets
echo ""
echo "=========================================="
echo "Step 1: Zero-shot per-dataset evaluation"
echo "=========================================="

DATASETS=(
    "anuraset"
    "bengalese_finch"
    "macaque"
    "marmoset"
    "picidae"
    "wetlands_bird"
    "zebra_finch"
)

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Processing dataset: $dataset"

    python scripts/phase2_02_zero_shot_per_dataset.py \
        --model "$MODEL" \
        --dataset "$dataset" \
        --batch-size 8

    echo "✓ Completed: $dataset"
done

echo ""
echo "✓ All per-dataset evaluations complete"

# Step 2: Aggregate per-dataset results
echo ""
echo "=========================================="
echo "Step 2: Aggregating per-dataset results"
echo "=========================================="

python scripts/phase2_02_aggregate_results.py

echo "✓ Per-dataset aggregation complete"

# Step 3: Zero-shot pooled evaluation (all datasets combined)
echo ""
echo "=========================================="
echo "Step 3: Zero-shot pooled evaluation"
echo "=========================================="

python scripts/phase2_03_zero_shot_pooled.py \
    --model "$MODEL"

echo "✓ Pooled evaluation complete"

# Step 4: Aggregate pooled results
echo ""
echo "=========================================="
echo "Step 4: Aggregating pooled results"
echo "=========================================="

python scripts/phase2_03_aggregate_pooled_results.py

echo "✓ Pooled aggregation complete"

# Step 5: Model selection (includes HuBERT + existing 5 models)
echo ""
echo "=========================================="
echo "Step 5: Model selection (all 6 models)"
echo "=========================================="

python scripts/phase2_04_model_selection.py

echo "✓ Model selection complete"

# Step 6: Check if HuBERT was selected for fine-tuning
echo ""
echo "=========================================="
echo "Step 6: Fine-tuning (if HuBERT selected)"
echo "=========================================="

SELECTION_FILE="$OUTPUT_BASE/model_selection/selection_results.json"
if [ -f "$SELECTION_FILE" ]; then
    SELECTED_MODEL=$(python3 << EOF
import json
with open("$SELECTION_FILE", 'r') as f:
    data = json.load(f)
    print(data.get('selected_model', ''))
EOF
    )

    echo "Selected model for fine-tuning: $SELECTED_MODEL"

    if [ "$SELECTED_MODEL" = "hubert_base" ]; then
        echo "Running fine-tuning for HuBERT..."

        python scripts/phase2_05_fine_tuning.py \
            --model "$MODEL" \
            --batch-size 8 \
            --max-epochs 16 \
            --lr 5e-5

        echo "✓ Fine-tuning complete"
    else
        echo "HuBERT not selected for fine-tuning (selected: $SELECTED_MODEL)"
        echo "Skipping fine-tuning step"
    fi
else
    echo "WARNING: Selection results not found, skipping fine-tuning"
fi

# Step 7: Generate final report
echo ""
echo "=========================================="
echo "Step 7: Generating final report"
echo "=========================================="

python scripts/phase2_07_generate_final_report.py

echo "✓ Final report generated"

# Summary
echo ""
echo "=========================================="
echo "✓ HuBERT Phase 2 Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results location: $OUTPUT_BASE"
echo "Final report: $OUTPUT_BASE/final_report"
echo ""
echo "Next: Review results and proceed to Phase 3 (Hyrax experiments)"
