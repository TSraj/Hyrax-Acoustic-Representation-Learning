# Setup & Run Guide - New Machine

**Quick guide to run the project on a new PC after cloning**

---

## ✅ Prerequisites Checklist

Make sure you've completed these steps:

- [x] Cloned repository from GitHub
- [x] Created virtual environment (`python -m venv venv`)
- [x] Activated virtual environment (`source venv/bin/activate`)
- [x] Installed dependencies (`pip install -r requirements.txt`)
- [x] Copied data to `Data/Macaque/` and `Data/Zebra finch/`

---

## 🔍 Step 1: Verify Your Setup

### 1.1 Check Directory Structure

```bash
cd Hyrax-Acoustic-Representation-Learning

# Verify the structure
ls -la

# You should see:
# - src/
# - scripts/
# - config/
# - Data/
# - venv/
# - requirements.txt
# - PROJECT_STATUS.md
```

### 1.2 Verify Data is in Place

```bash
# Check Macaque data
ls Data/Macaque/
# Should show: AL, BE, IO, MU, QU, SN, TH, TW (8 individual folders)

# Check Zebra Finch data
ls "Data/Zebra finch/"
# Should show: AdultVocalizations, ChickVocalizations

# Count audio files (example for Macaque)
find Data/Macaque -type f -name "*.wav" | wc -l
# Should show around 7,285 files

find "Data/Zebra finch" -type f -name "*.wav" | wc -l
# Should show around 3,433 files
```

### 1.3 Verify Virtual Environment

```bash
# Make sure venv is activated (you should see (venv) in your prompt)
which python
# Should point to: /path/to/your/project/venv/bin/python

# Check Python version
python --version
# Should be Python 3.8 or higher (3.12 recommended)

# Verify key packages are installed
pip list | grep -E "torch|transformers|librosa|scikit-learn|umap"
# Should show all these packages
```

### 1.4 Test Imports

```bash
# Quick test that everything imports correctly
python -c "
import torch
import transformers
import librosa
import sklearn
import umap
import yaml
import numpy as np
print('✅ All imports successful!')
"
```

If all checks pass, you're ready to run! ✅

---

## 🎯 Step 2: Decide What to Run

You have two options:

### Option A: Quick Test Run (Subset Mode - RECOMMENDED FIRST)
- **Time:** ~30-60 minutes total
- **Data:** Uses only 20 samples per individual (~2-6% of data)
- **Purpose:** Verify everything works before full run
- **Config:** Keep `subset.enabled: true` (default)

### Option B: Full Dataset Run
- **Time:** 5-9 hours (CPU) or 2-4 hours (GPU)
- **Data:** All 7,285 Macaque + 3,433 Zebra Finch samples
- **Purpose:** Get real, publication-ready results
- **Config:** Change `subset.enabled: false`

**RECOMMENDATION:** Start with Option A to verify everything works!

---

## 🏃 Step 3: Run the Pipeline (Subset Mode - Test Run)

### Configure for Subset Mode (Default)

```bash
# Check current config
grep "enabled:" config/config.yaml | head -1

# Should show:
#   enabled: true

# If not, edit it:
nano config/config.yaml
# Or use your preferred editor
```

### Run Each Script in Order

**IMPORTANT:** Run scripts in this exact order!

#### Script 01: Analyze Datasets (~1-2 minutes)

```bash
python scripts/01_analyze_datasets.py
```

**What it does:**
- Scans your data directories
- Counts files per individual
- Generates analysis reports

**Expected output:**
```
================================================================================
SCRIPT 01: ANALYZE DATASETS
================================================================================
Analyzing Macaque dataset...
  Found 8 individuals
  Total samples: 7285
  
Analyzing Zebra Finch dataset...
  Found 10 individuals (subset)
  Total samples: 3433

Reports saved to: outputs/reports/
```

**Check results:**
```bash
ls outputs/reports/
# Should see: dataset_analysis_report_macaque.txt
#             dataset_analysis_report_zebra_finch.txt
```

---

#### Script 02: Create Subsets & Preprocess (~5-10 minutes)

```bash
python scripts/02_create_subsets_and_preprocess.py
```

**What it does:**
- Selects 20 samples per individual (subset mode)
- Converts to 16kHz, mono, normalized
- Saves preprocessed audio

**Expected output:**
```
================================================================================
SCRIPT 02: CREATE SUBSETS AND PREPROCESS
================================================================================
Creating subsets...
  Macaque: Selected 160 samples (20 per individual)
  Zebra Finch: Selected 200 samples (20 per individual)

Preprocessing audio...
  Progress: 100% [████████████████████] 360/360

Preprocessed audio saved to: Data/processed/preprocessed_subsets/
```

**Check results:**
```bash
ls Data/processed/subsets/
# Should see: macaque/, zebra_finch/

ls Data/processed/preprocessed_subsets/
# Should see: macaque/, zebra_finch/
```

---

#### Script 03: Extract Embeddings (~15-30 minutes CPU, ~5-10 min GPU)

**This is the slowest script!**

```bash
python scripts/03_extract_embeddings.py
```

**What it does:**
- Loads wav2vec2 models
- Extracts features from all layers
- Pools features (mean, max, first, last)
- Saves embeddings as .npz files

**Expected output:**
```
================================================================================
SCRIPT 03: EXTRACT EMBEDDINGS
================================================================================
Processing macaque with wav2vec2-base...
  Layer 0: 100% [████████████████████] 160/160
  Layer 1: 100% [████████████████████] 160/160
  ...
  Layer 11: 100% [████████████████████] 160/160
  
Processing macaque with wav2vec2-xlsr...
  (similar, but 24 layers)

Processing zebra_finch...
  (similar for both models)

Embeddings saved to: outputs/embeddings/
```

**Check results:**
```bash
ls -lh outputs/embeddings/
# Should see 8 .npz files:
#   macaque_wav2vec2_base_features.npz (~94 MB)
#   macaque_wav2vec2_base_pooled.npz (~22 MB)
#   macaque_wav2vec2_xlsr_features.npz (~240 MB)
#   macaque_wav2vec2_xlsr_pooled.npz (~56 MB)
#   (similar for zebra_finch)
```

**Optional - Use GPU for Speed:**
```bash
# Edit config first:
nano config/config.yaml

# Change this line:
# feature_extraction:
#   device: "cpu"  → device: "mps"  (for Apple Silicon)
#                 → device: "cuda" (for NVIDIA GPU)

# Then run script 03 again (will be 3-4x faster)
```

---

#### Script 04: Visualize Embeddings (~10-20 minutes)

```bash
python scripts/04_visualize_embeddings.py
```

**What it does:**
- Creates PCA, LDA, UMAP visualizations
- Generates layer comparison grids
- Plots for each layer individually

**Expected output:**
```
================================================================================
SCRIPT 04: VISUALIZE EMBEDDINGS
================================================================================
Visualizing macaque wav2vec2_base...
  PCA: 100% [████████████████████] 12/12 layers
  LDA: 100% [████████████████████] 12/12 layers
  UMAP: 100% [████████████████████] 12/12 layers

Visualizing macaque wav2vec2_xlsr...
  (similar, 24 layers)

Visualizing zebra_finch...
  (similar for both models)

Figures saved to: outputs/figures/
```

**Check results:**
```bash
# Count generated figures
find outputs/figures -name "*.png" | wc -l
# Should show ~228 images

# View some examples
ls outputs/figures/macaque/wav2vec2_xlsr/comparison_grids/
# Should see layer comparison plots
```

---

#### Script 05: Evaluate Layers (~5-15 minutes)

```bash
python scripts/05_evaluate_layers.py
```

**What it does:**
- Runs k-NN classifier (multiple k values)
- Runs Linear Probe (multiple C values)
- Runs Logistic Regression (multiple C values)
- Compares all layers
- Generates comprehensive reports

**Expected output:**
```
================================================================================
SCRIPT 05: EVALUATE LAYERS
================================================================================
Evaluating wav2vec2_xlsr on macaque...
  Memory usage before loading embeddings: 150.2 MB
  
  Running k-NN evaluation...
    Layer 0: Best k=5, CV accuracy: 0.8500
    Layer 1: Best k=3, CV accuracy: 0.8625
    ...
    Layer 5: Best k=5, CV accuracy: 0.9167
    
  Running Linear Probe evaluation...
    Layer 0: Best C=1.0, CV accuracy: 0.8750
    Layer 1: Best C=0.1, CV accuracy: 0.8875
    ...
    Layer 5: Best C=1.0, CV accuracy: 0.9250
    
  Running Logistic Regression evaluation...
    (similar output)

  Memory usage after GC: 180.5 MB

================================================================================
OVERALL SUMMARY - ALL CLASSIFIERS
================================================================================

WAV2VEC2_XLSR on MACAQUE:
  k-NN (Layer 5):
    Accuracy: 0.9167
    Balanced Accuracy: 0.9123
    Macro F1: 0.9145

  Linear Probe (Layer 5):
    Accuracy: 0.9250
    Balanced Accuracy: 0.9201
    Macro F1: 0.9220

  Logistic Regression (Layer 5):
    Accuracy: 0.9200
    Balanced Accuracy: 0.9150
    Macro F1: 0.9175
```

**Check results:**
```bash
# View overall summary
cat outputs/reports/evaluation/overall_summary.txt

# View detailed reports
ls outputs/reports/evaluation/macaque/wav2vec2_xlsr/
# Should see:
#   knn_report_layer5.txt
#   linear_probe_report_layer5.txt
#   logreg_report_layer5.txt
#   layer_comparison_report.txt
#   layer_comparison.png
#   confusion_matrices/
```

---

## 🎉 Step 4: Verify Success

### Check All Outputs Were Generated

```bash
# Verify all output directories exist
ls -la outputs/
# Should see: embeddings/, figures/, reports/

# Check sizes
du -sh outputs/embeddings/
du -sh outputs/figures/
du -sh outputs/reports/

# For subset run, expect:
#   embeddings: ~900 MB
#   figures: ~100 MB
#   reports: ~1-5 MB
```

### View Key Results

```bash
# Overall summary
cat outputs/reports/evaluation/overall_summary.txt

# Best layer comparison
open outputs/reports/evaluation/macaque/wav2vec2_xlsr/layer_comparison.png
# (or use your image viewer)

# Layer comparison grid
open outputs/figures/macaque/wav2vec2_xlsr/comparison_grids/layer_comparison_umap.png
```

---

## ✅ Success Criteria

Your test run is successful if:

- [x] All 5 scripts completed without errors
- [x] `outputs/embeddings/` has 8 .npz files (~900 MB total)
- [x] `outputs/figures/` has ~228 images
- [x] `outputs/reports/evaluation/overall_summary.txt` exists
- [x] Summary shows results for all 3 classifiers (k-NN, Linear Probe, LogReg)
- [x] Best accuracy is reasonable (~85-95% for Macaque, ~30-45% for Zebra Finch)

**If all checks pass:** ✅ **Your setup is correct!**

---

## 🚀 Step 5: Run Full Dataset (After Test Success)

Once the subset test works perfectly:

### 5.1 Update Configuration

```bash
# Edit config
nano config/config.yaml

# Change line 30:
# FROM:
  subset:
    enabled: true

# TO:
  subset:
    enabled: false
```

### 5.2 Clean Previous Outputs (Optional)

```bash
# Remove subset outputs to avoid confusion
rm -rf outputs/
rm -rf Data/processed/

# This is optional - outputs will be overwritten anyway
```

### 5.3 Run Full Pipeline

**IMPORTANT:** This will take 5-9 hours on CPU, or 2-4 hours on GPU!

**Recommended: Run overnight or in background**

```bash
# Option A: Run in background with nohup
nohup python scripts/02_create_subsets_and_preprocess.py > run.log 2>&1 && \
nohup python scripts/03_extract_embeddings.py >> run.log 2>&1 && \
nohup python scripts/04_visualize_embeddings.py >> run.log 2>&1 && \
nohup python scripts/05_evaluate_layers.py >> run.log 2>&1 && \
echo "Full experiment complete!" >> run.log &

# Check progress
tail -f run.log

# Option B: Run sequentially (stay at computer)
python scripts/02_create_subsets_and_preprocess.py  # 30-60 min
python scripts/03_extract_embeddings.py             # 3-6 hours (bottleneck!)
python scripts/04_visualize_embeddings.py           # 1-2 hours
python scripts/05_evaluate_layers.py                # 15-30 min
```

### 5.4 Monitor Progress

```bash
# Watch log file
tail -f run.log

# Check memory usage (if monitoring enabled)
grep "Memory usage" run.log

# Check disk space
df -h .

# Estimate remaining time
# Script 03 is the bottleneck - check its progress in the log
```

### 5.5 Expected Results (Full Dataset)

When complete, you'll have:

```
outputs/
├── embeddings/        # ~3-4 GB (8 files)
├── figures/           # ~500 MB (228+ images)
└── reports/           # ~5-10 MB
    └── evaluation/
        └── overall_summary.txt  # Full dataset results
```

**Expected performance (approximate):**
- Macaque: 75-95% accuracy (varies by layer)
- Zebra Finch: 30-50% accuracy
- Linear Probe ≥ k-NN (usually)

---

## ⚠️ Troubleshooting

### Script 01 Fails
```bash
# Check data paths
ls Data/Macaque/
ls "Data/Zebra finch/"

# Verify config
cat config/config.yaml | grep -A5 "datasets:"
```

### Script 02 Fails
```bash
# Check permissions
chmod +x scripts/*.py

# Check disk space
df -h .

# Verify librosa is installed
pip list | grep librosa
```

### Script 03 Fails (Out of Memory)
```bash
# Reduce batch size
nano config/config.yaml

# Change:
feature_extraction:
  batch_size: 8  →  batch_size: 4  # or even 2

# Or use CPU instead of GPU
  device: "mps"  →  device: "cpu"
```

### Script 03 Very Slow
```bash
# Enable GPU if available
nano config/config.yaml

# Change:
  device: "cpu"  →  device: "mps"  # Apple Silicon
                →  device: "cuda" # NVIDIA GPU

# Check if GPU is detected
python -c "import torch; print(f'GPU available: {torch.cuda.is_available() or torch.backends.mps.is_available()}')"
```

### Script 04 or 05 Fails
```bash
# Check if embeddings were generated
ls -lh outputs/embeddings/

# Verify sklearn is installed
pip list | grep scikit-learn

# Check config
cat config/config.yaml | grep -A10 "visualization:"
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Or reinstall specific package
pip install transformers --upgrade
```

---

## 📊 Quick Reference

### Time Estimates (Subset):
```
Script 01: 1-2 minutes
Script 02: 5-10 minutes
Script 03: 15-30 minutes (CPU) / 5-10 min (GPU)
Script 04: 10-20 minutes
Script 05: 5-15 minutes
Total: ~40-75 minutes
```

### Time Estimates (Full Dataset):
```
Script 01: 1-2 minutes (skip if already done)
Script 02: 30-60 minutes
Script 03: 3-6 hours (CPU) / 1-2 hours (GPU) ← BOTTLENECK
Script 04: 1-2 hours
Script 05: 15-30 minutes
Total: 5-9 hours (CPU) / 2-4 hours (GPU)
```

### Disk Space Required:
```
Subset: ~1.5 GB total
Full: ~5-10 GB total
```

### Key Commands:
```bash
# Activate environment
source venv/bin/activate

# Check what's running
ps aux | grep python

# Kill if needed
pkill -f "python scripts"

# View results
cat outputs/reports/evaluation/overall_summary.txt

# Clean and restart
rm -rf outputs/ Data/processed/
```

---

## 🎯 Success!

Once all scripts complete:

1. **Check results:** `cat outputs/reports/evaluation/overall_summary.txt`
2. **View visualizations:** `open outputs/figures/macaque/wav2vec2_xlsr/comparison_grids/`
3. **Review reports:** Browse `outputs/reports/evaluation/`
4. **Update documentation:** Add your results to PROJECT_STATUS.md
5. **Push to git:** `git add PROJECT_STATUS.md && git commit -m "Update with results" && git push`

**You now have complete, publication-ready results!** 🎉

---

**Quick Start Summary:**
```bash
# 1. Verify setup
ls Data/Macaque/ Data/"Zebra finch"/

# 2. Run subset test (recommended first)
python scripts/01_analyze_datasets.py
python scripts/02_create_subsets_and_preprocess.py
python scripts/03_extract_embeddings.py
python scripts/04_visualize_embeddings.py
python scripts/05_evaluate_layers.py

# 3. Check results
cat outputs/reports/evaluation/overall_summary.txt

# 4. If test successful, run full dataset
# (Edit config: subset.enabled: false, then repeat scripts 02-05)
```
