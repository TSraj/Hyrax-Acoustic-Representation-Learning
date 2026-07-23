#!/bin/bash
#SBATCH --job-name=phase3_full
#SBATCH --output=logs/phase3_full_%j.out
#SBATCH --error=logs/phase3_full_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Phase 3 - Full Pipeline for ICASSP 2027
# Runs all steps for both tasks: species_id and hyrax_id

set -e  # Exit on error

echo "========================================"
echo "PHASE 3 - FULL PIPELINE"
echo "Started: $(date)"
echo "========================================"

# Activate environment
source venv/bin/activate

# Create log directory
mkdir -p logs

# Models to evaluate
MODELS=("wav2vec2_base" "wav2vec2_base_960h" "hubert_base" "xls_r" "wavlm" "ecapa_tdnn")
TASKS=("species_id" "hyrax_id")

echo ""
echo "========================================"
echo "STEP 1 & 2: Extract Bouts + Create Manifests"
echo "========================================"
python scripts/phase3_01_extract_hyrax_bouts.py
python scripts/phase3_02_create_manifests.py

echo ""
echo "========================================"
echo "STEP 3: Zero-Shot Evaluation (All 6 Models)"
echo "========================================"

for task in "${TASKS[@]}"; do
    echo ""
    echo "--- Task: $task ---"
    for model in "${MODELS[@]}"; do
        echo "Running $model on $task..."
        python scripts/phase3_03_zero_shot_evaluation.py \
            --model "$model" \
            --task "$task"
    done
done

echo ""
echo "========================================"
echo "STEP 4: Model Selection"
echo "========================================"
python scripts/phase3_04_model_selection.py

# Read selected models
MONO_MODEL=$(python -c "import json; print(json.load(open('outputs/phase3/model_selection/selected_models.json'))['monolingual'])")
MULTI_MODEL=$(python -c "import json; print(json.load(open('outputs/phase3/model_selection/selected_models.json'))['multilingual'])")

echo "Selected monolingual: $MONO_MODEL"
echo "Selected multilingual: $MULTI_MODEL"

echo ""
echo "========================================"
echo "STEP 5: Fine-Tuning (Selected Models)"
echo "========================================"

for task in "${TASKS[@]}"; do
    echo ""
    echo "--- Task: $task ---"

    echo "Fine-tuning $MONO_MODEL..."
    python scripts/phase3_05_fine_tuning.py \
        --model "$MONO_MODEL" \
        --task "$task"

    echo "Analyzing $MONO_MODEL..."
    python scripts/phase3_05b_analyze_fine_tuning.py \
        --model "$MONO_MODEL" \
        --task "$task"

    echo "Fine-tuning $MULTI_MODEL..."
    python scripts/phase3_05_fine_tuning.py \
        --model "$MULTI_MODEL" \
        --task "$task"

    echo "Analyzing $MULTI_MODEL..."
    python scripts/phase3_05b_analyze_fine_tuning.py \
        --model "$MULTI_MODEL" \
        --task "$task"

    echo "Comparing models..."
    python scripts/phase3_05c_compare_models.py \
        --task "$task" \
        --mono "$MONO_MODEL" \
        --multi "$MULTI_MODEL"
done

echo ""
echo "========================================"
echo "STEP 6: Final Analysis"
echo "========================================"

for task in "${TASKS[@]}"; do
    echo "Analyzing $task..."
    python scripts/phase3_06_final_analysis.py --task "$task"
done

echo ""
echo "========================================"
echo "STEP 7: Acoustic Characteristics Analysis"
echo "========================================"

for task in "${TASKS[@]}"; do
    echo "Acoustic analysis for $task with $MONO_MODEL..."
    python scripts/phase3_07_acoustic_analysis.py \
        --task "$task" \
        --model "$MONO_MODEL"
done

echo ""
echo "========================================"
echo "STEP 8: Paper Figures (ICASSP 2027)"
echo "========================================"
python scripts/phase3_08_paper_figures.py

echo ""
echo "========================================"
echo "PHASE 3 COMPLETE"
echo "Finished: $(date)"
echo "========================================"
echo ""
echo "All outputs saved to: outputs/phase3/"
echo ""
echo "Summary of generated files:"
echo "  - Zero-shot results: outputs/phase3/zero_shot/{task}/{model}/"
echo "  - Model selection: outputs/phase3/model_selection/"
echo "  - Fine-tuning: outputs/phase3/fine_tuning/{task}/{model}/"
echo "  - Model comparison: outputs/phase3/model_comparison/"
echo "  - Final analysis: outputs/phase3/final_analysis/"
echo "  - Acoustic analysis: outputs/phase3/acoustic_analysis/"
echo "  - Paper figures: outputs/phase3/paper_figures/"
echo ""
echo "KEY PAPER FIGURES (ICASSP 2027):"
echo "  1. monolingual_experiments.png + .csv"
echo "  2. multilingual_experiments.png + .csv"
echo "  3. adaptation_experiments.png + .csv"
echo "  4. statistical_winner.csv + winner_declaration.txt"
