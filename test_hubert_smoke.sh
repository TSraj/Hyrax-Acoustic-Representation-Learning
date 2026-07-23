#!/bin/bash
# Smoke test for HuBERT model integration
# Tests on a small subset before running full pipeline

set -e  # Exit on error

echo "=========================================="
echo "HuBERT Smoke Test - Phase 2 Integration"
echo "=========================================="

# Check HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. This may cause rate limiting."
    echo "Set it with: export HF_TOKEN=your_token"
    echo ""
fi

# Configuration
MODEL="hubert_base"
CONFIG_FILE="configs/config.yaml"
DATA_DIR="data/bird_datasets"
OUTPUT_DIR="outputs/phase2_hubert_test"
MANIFEST_DIR="outputs/phase2/manifests"

# Check if manifests exist
if [ ! -d "$MANIFEST_DIR" ]; then
    echo "ERROR: Manifests directory not found: $MANIFEST_DIR"
    echo "Run phase2_01_create_manifests.py first"
    exit 1
fi

# Get first dataset manifest for quick test
FIRST_MANIFEST=$(find "$MANIFEST_DIR" -name "*_manifest.json" -type f ! -name "pooled_*" | head -1)
if [ -z "$FIRST_MANIFEST" ]; then
    echo "ERROR: No manifests found in $MANIFEST_DIR"
    exit 1
fi

DATASET_NAME=$(basename "$FIRST_MANIFEST" | sed 's/_manifest.json//')
echo "Testing on dataset: $DATASET_NAME"

# Create test output directory
mkdir -p "$OUTPUT_DIR"

# Test 1: Zero-shot per-dataset (debug mode uses small subset)
echo ""
echo "Test 1: Zero-shot evaluation (debug mode)..."
python scripts/phase2_02_zero_shot_per_dataset.py \
    --model "$MODEL" \
    --dataset "$DATASET_NAME" \
    --batch-size 4 \
    --debug

echo ""
echo "✓ Test 1 passed: HuBERT zero-shot works"

# Test 2: Check embedding extraction
echo ""
echo "Test 2: Checking embedding extraction..."
ZERO_SHOT_DIR="outputs/phase2/zero_shot/per_dataset/$DATASET_NAME/$MODEL"
if [ -d "$ZERO_SHOT_DIR/embedding_cache" ]; then
    CACHE_FILES=$(find "$ZERO_SHOT_DIR/embedding_cache" -name "*.npz" | wc -l)
    echo "✓ Found $CACHE_FILES embedding cache files"
else
    echo "WARNING: No embedding cache found"
fi

# Test 3: Check results format
echo ""
echo "Test 3: Checking results format..."
RESULT_FILE="$ZERO_SHOT_DIR/summary.json"
if [ -f "$RESULT_FILE" ]; then
    python3 << EOF
import json
with open("$RESULT_FILE", 'r') as f:
    results = json.load(f)
    print(f"✓ Results file valid")
    print(f"  Model: {results.get('model', 'N/A')}")
    print(f"  Dataset: {results.get('dataset', 'N/A')}")
    print(f"  Best layer: {results.get('best_layer', 'N/A')}")
    print(f"  Best accuracy: {results.get('best_accuracy', 0):.4f}")
    print(f"  Layers tested: {len(results.get('layer_wise_accuracy', []))}")
EOF
else
    echo "ERROR: Results file not found: $RESULT_FILE"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All smoke tests passed!"
echo "Ready to run full HuBERT pipeline"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run full zero-shot: bash scripts/phase2_02_run_all_combinations.sh (add hubert_base)"
echo "2. Run pooled evaluation: python scripts/phase2_03_zero_shot_pooled.py --model hubert_base"
echo "3. Re-run model selection: python scripts/phase2_04_model_selection.py"
echo "4. If HuBERT selected, run fine-tuning"
echo "5. Regenerate final report"
