# Hyrax Acoustic Representation Learning - Project Status

**Last Updated:** 2026-04-16  
**Current Phase:** Enhanced Pipeline with 7 Feature Methods - Ready for Full Dataset

---

## 📍 Current Status Summary

### ✅ What's Complete:
1. **7 feature extraction methods implemented** (3 handcrafted + 4 deep learning models)
2. **HuBERT models added** - Alternative to wav2vec2 with better performance
3. **Prosodic features added** - Pitch, energy, duration (26 features)
4. **Comprehensive evaluation system** - 7 classifiers with advanced metrics
5. **All visualization overlaps fixed** - Publication-quality figures
6. **Subset experiments tested** - System validated with ~2-6% of data

### 🎯 Next Step:
**Run full dataset experiments on all 7 feature extraction methods**

---

## 🎓 Project Goal

Investigate whether pretrained audio models (wav2vec2 and HuBERT) trained on human speech can extract meaningful representations from animal vocalizations for individual identification.

### Datasets:
- **Macaque:** 8 individuals, 7,285 vocalizations
- **Zebra Finch:** 10 individuals, 3,433 vocalizations
- **Future:** Hyrax + up to 10 animal datasets

---

## 🔬 Feature Extraction Methods (7 Total)

### Handcrafted Features (3):
1. **MFCC** - 13 coefficients, spectral envelope
2. **Spectral** - ~25 features, frequency characteristics  
3. **Prosodic** ⭐ NEW - 26 features (pitch, energy, duration, spectral-temporal)

### Deep Learning Models (4):
4. **wav2vec2_base** - 13 layers, 768-dim, English speech
5. **wav2vec2_xlsr** - 25 layers, 1024-dim, multilingual
6. **hubert_base** ⭐ NEW - 13 layers, 768-dim, cluster-based learning
7. **hubert_large** ⭐ NEW - 25 layers, 1024-dim, high-capacity

---

## 📁 Pipeline Scripts (Run in Order)

### Script 01: `01_analyze_datasets.py`
- Analyzes dataset statistics
- **Time:** ~5 min
- **Optional:** Not required for main pipeline

### Script 02: `02_create_subsets_and_preprocess.py`
- Preprocesses audio to 16kHz mono
- Creates subsets if enabled
- **Time:** ~30-60 min (full dataset)

### Script 03: `03_extract_embeddings.py`
- Extracts features from all 4 models (wav2vec2_base, wav2vec2_xlsr, hubert_base, hubert_large)
- Extracts all layers for each model
- **Time:** ~3-6 hours CPU, ~1-2 hours GPU
- **Bottleneck:** Slowest step

### Script 03 (pooled): `03_extract_pooled_embeddings.py`
- Pools layer embeddings (mean, max, first, last)
- **Time:** ~10-20 min
- **Note:** May not be needed if using comprehensive evaluation

### Script 03b: `03b_extract_handcrafted_features.py`
- Extracts MFCC, Spectral, and Prosodic features
- **Time:** ~20-40 min

### Script 04: `04_visualize_embeddings.py`
- Creates PCA, LDA, UMAP visualizations
- Layer comparison grids
- **Time:** ~1-2 hours

### Script 05: `05_comprehensive_evaluation.py` ⭐ USE THIS ONE
- Evaluates all 7 feature methods
- 7 classifiers: k-NN, Linear Probe, Logistic Regression, SVM (Linear/RBF), Random Forest, XGBoost
- Finds best layer for each model
- Creates comparison charts
- **Time:** ~15-30 min
- **Note:** Don't use `05_evaluate_layers.py` - this comprehensive version includes everything

---

## 🏗️ Project Structure

```
├── config/
│   └── config.yaml                          # All settings (subset mode here!)
│
├── Data/
│   ├── Macaque/                             # 7,285 samples
│   ├── Zebra finch/                         # 3,433 samples
│   └── processed/                           # Preprocessed audio
│
├── src/
│   ├── models/
│   │   ├── wav2vec_extractor.py             # Wav2Vec2 + HuBERT extraction
│   │   ├── mfcc_extractor.py                # MFCC features
│   │   ├── spectral_extractor.py            # Spectral features
│   │   └── prosodic_extractor.py            # Prosodic features (NEW)
│   │
│   └── evaluation/
│       ├── visualizer.py                    # Visualizations (overlaps fixed)
│       ├── advanced_visualizer.py           # Feature comparison charts (overlaps fixed)
│       ├── knn_classifier.py                # k-NN
│       ├── linear_classifiers.py            # Linear Probe + LogReg
│       ├── svm_classifier.py                # SVM Linear + RBF
│       └── ensemble_classifiers.py          # Random Forest + XGBoost
│
├── scripts/
│   ├── 01_preprocess_data.py
│   ├── 02_extract_embeddings.py
│   ├── 03_extract_pooled_embeddings.py
│   ├── 03b_extract_handcrafted_features.py
│   ├── 04_visualize_embeddings.py
│   ├── 05_comprehensive_evaluation.py       # ⭐ USE THIS
│   └── 05_evaluate_layers.py                # Keep for reference only
│
└── outputs/
    ├── embeddings/                          # Extracted features
    ├── figures/                             # Visualizations
    └── reports/
        └── comprehensive_evaluation/        # Evaluation results
```

---

## 🚀 How to Run

### For Subset (Quick Test):
```bash
# config.yaml line 30
subset:
  enabled: true    # Uses 20 samples per individual
```

### For Full Dataset (Real Results):
```bash
# config.yaml line 30
subset:
  enabled: false   # ⭐ CHANGE THIS FOR FULL DATASET
```

### Run Pipeline:
```bash
cd "/Users/raj/Documents/Hyrax Acoustic Representation Learning"
source venv/bin/activate

# Clean start (optional)
rm -rf outputs/
rm -rf Data/processed/

# Run scripts in order
python scripts/02_create_subsets_and_preprocess.py  # Preprocessing
python scripts/03_extract_embeddings.py             # Slowest step (3-6 hrs)
python scripts/03b_extract_handcrafted_features.py  # Handcrafted features
python scripts/04_visualize_embeddings.py           # Visualizations (1-2 hrs)
python scripts/05_comprehensive_evaluation.py       # ⭐ Main evaluation
```

### Expected Time (Full Dataset):
- **CPU:** 5-9 hours total
- **GPU:** 2-4 hours total
- **Bottleneck:** Script 02 (embedding extraction)

---

## 📊 Evaluation Metrics

### Classifiers (7):
1. k-NN (k=3,5,7,9)
2. Linear Probe (gold standard for representation learning)
3. Logistic Regression
4. SVM Linear
5. SVM RBF
6. Random Forest
7. XGBoost

### Metrics:
- **Accuracy** - Overall correctness
- **Balanced Accuracy** - Handles class imbalance
- **Macro F1** - Standard for multi-class (best for imbalanced data)
- Macro Precision & Recall
- Per-class F1
- Clustering metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)

---

## ✅ Recent Fixes (2026-04-16)

### Visualization Overlaps - FIXED:
1. ✅ Layer comparison grids: Title/legend no longer overlap with plots
2. ✅ Feature comparison charts: Legend moved outside plot area
3. ✅ All plots now publication-quality at 300 DPI PNG

### Key Changes:
- Legends positioned outside plot areas
- Proper spacing with tight_layout
- Title padding adjusted
- All overlap issues resolved

---

## 🔑 Key Configuration

### Full Dataset Mode:
```yaml
# config/config.yaml
subset:
  enabled: false  # Change to false for full dataset

preprocessing:
  min_duration: 0.5  # IMPORTANT: Must be 0.5 (not 0.1) to avoid spectral feature errors
  max_duration: 10.0
```

### GPU Acceleration (Optional):
```yaml
# config/config.yaml
feature_extraction:
  device: "mps"    # For Apple Silicon
  # OR
  device: "cuda"   # For NVIDIA GPU
```

---

## 📈 What You'll Get

### Output Files:
```
outputs/
├── embeddings/
│   ├── {dataset}/
│   │   ├── wav2vec2_base/         # 13 layers
│   │   ├── wav2vec2_xlsr/         # 25 layers
│   │   ├── hubert_base/           # 13 layers (NEW)
│   │   └── hubert_large/          # 25 layers (NEW)
│
├── features/
│   └── {dataset}/
│       ├── mfcc_features_mean.npz
│       ├── spectral_features_mean.npz
│       └── prosodic_features_mean.npz (NEW)
│
├── figures/
│   └── {dataset}/{model}/
│       ├── comparison_grids/
│       │   ├── layer_comparison_pca.png  (fixed)
│       │   ├── layer_comparison_lda.png
│       │   └── layer_comparison_umap.png
│       └── individual_layers/
│
└── reports/
    └── comprehensive_evaluation/
        └── {dataset}/
            ├── evaluation_summary.txt
            ├── feature_comparison_knn.png     (fixed)
            ├── feature_comparison_svm.png     (fixed)
            ├── classifier_comparison_*.png
            └── layer_comparison_combined.png
```

### Key Results:
- **Best feature method** for each dataset
- **Best layer** for each deep learning model
- **Best classifier** for each feature type
- **Handcrafted vs. learned** feature comparison
- **wav2vec2 vs. HuBERT** comparison

---

## 🎯 Research Questions Answered

1. **Do pretrained speech models work for animal vocalizations?**
   - Compare all 4 models vs handcrafted baselines

2. **Which layers capture useful information?**
   - Layer-wise analysis for all models

3. **HuBERT vs wav2vec2 - which is better?**
   - Direct comparison on same data

4. **Are handcrafted features still competitive?**
   - MFCC, Spectral, Prosodic vs deep learning

5. **Which classifier works best?**
   - 7 classifiers tested on each feature type

---

## ⚠️ Important Notes

### For Next Session:
1. **Read this file first** to catch up
2. **Check config.yaml** - Is subset enabled or disabled? Is min_duration set to 0.5?
3. **Script 05 comprehensive** is the main evaluation (not 05_evaluate_layers.py)
4. **All visualization overlaps fixed** - layer_comparison_combined.png legend now outside plot
5. **HuBERT models download on first run** (~1.5GB total)
6. **Proposal focus** - Clustering analysis of wav2vec2_xlsr WITHOUT fine-tuning

### Git Commits:
- **No co-author attribution** - Just commit as yourself

### Pull Latest Changes (Other PC):
```bash
git pull origin main
pip install -r requirements.txt  # Update dependencies
```

---

## 🛠️ Technical Details

### Environment:
- **Python:** 3.12
- **Platform:** macOS (Darwin 25.3.0)
- **Virtual env:** `venv/`

### Key Dependencies:
- PyTorch 2.0+
- Transformers 4.30+ (wav2vec2 + HuBERT)
- scikit-learn 1.3+
- librosa 0.10+ (prosodic features)
- UMAP 0.5+
- XGBoost 2.0+

### Preprocessing:
- Sample rate: 16kHz (required)
- Channels: Mono
- Normalization: Peak normalization
- Duration: 0.1s - 10s

---

## 📝 Quick Commands

### View Results:
```bash
# Summary report
cat outputs/reports/comprehensive_evaluation/macaque/evaluation_summary.txt

# Best visualizations
open outputs/figures/macaque/hubert_base/comparison_grids/layer_comparison_umap.png
open outputs/reports/comprehensive_evaluation/macaque/feature_comparison_knn.png
```

### Check Sizes:
```bash
du -sh outputs/embeddings/      # Embedding storage
du -sh outputs/                 # Total outputs
```

### Monitor Progress:
```bash
# Watch memory during execution
python scripts/02_extract_embeddings.py 2>&1 | grep "Memory"
```

---

## 🎓 For Thesis

### What This Gives You:
- ✅ 7 feature extraction methods compared
- ✅ Multiple baselines (not just one)
- ✅ Robust metrics (handles class imbalance)
- ✅ Layer-wise analysis (which layers work best)
- ✅ Cross-species comparison (Macaque vs Zebra Finch)
- ✅ Publication-quality figures (300 DPI, no overlaps)
- ✅ Comprehensive evaluation (7 classifiers)

### After Full Dataset Run:
- Statistically significant results
- Real performance numbers for proposal
- Credible comparison of methods
- Foundation for scaling to 10 datasets

---

## 🎓 Professor's Requirements for Initial Proposal

### Experiment Focus:
**Use pretrained wav2vec2_xlsr (multilingual) WITHOUT fine-tuning**
- NO training, NO fine-tuning - just pass animal data through as-is
- Extract features from all layers
- Visualize clusters (PCA/UMAP) to see which layers give best separation
- Compute clustering metrics (Silhouette, Calinski-Harabasz)
- Show which layer produces best clusters

### Key Point:
**This is a representation quality check, NOT a classification task**
- Focus: "Which layer captures individual vocal characteristics?"
- NOT: "What's the best classification accuracy?"
- Goal: Show pretrained speech models capture animal individual differences

### Essential Outputs for Proposal:
1. **Layer comparison UMAP grid** (already have: `layer_comparison_umap.png`)
2. **Clustering metrics across layers plot** (MISSING - need to add)
3. **Best layer detailed visualization** (already have: individual layer UMAP)
4. **Feature method comparison** (already have: `feature_comparison_*.png`)

---

## ⚠️ Important Fixes & Known Issues

### Fixed Issues (2026-04-18):

1. **min_duration Configuration** ✅ FIXED
   - **Issue:** `config.yaml` line 42 had `min_duration: 0.1` (too short)
   - **Problem:** Files with 0.1-0.5s failed spectral_contrast extraction (needs 9+ frames, 0.1s gives only 7)
   - **Fix:** Change to `min_duration: 0.5` to ensure enough frames for all feature extractors
   - **Result:** All 7 feature methods process identical file sets (fair comparison)

2. **layer_comparison_combined.png Legend Overlap** ✅ FIXED
   - **Issue:** Legend overlapped with bars when running full dataset
   - **Fix:** Moved legend outside plot area with `bbox_to_anchor=(1.02, 1)`
   - **File:** `scripts/05_comprehensive_evaluation.py` line 402

### Missing Visualization (To Add):

**Clustering Metrics Across Layers Plot**
- X-axis: Layer number (0-24)
- Y-axis: Silhouette Score / Calinski-Harabasz
- Shows which layer achieves best clustering quantitatively
- Complements visual layer comparison grids
- **Status:** Not yet implemented, metrics computed but not plotted

---

## 🚀 Success Criteria

### Pipeline: ✅ COMPLETE
- [x] All 7 feature methods working
- [x] HuBERT models integrated
- [x] Prosodic features added
- [x] Visualization overlaps fixed
- [x] Comprehensive evaluation system
- [x] Code tested on subsets

### Full Dataset: ⏳ NEXT STEP
- [ ] Change `subset.enabled: false`
- [ ] Run scripts 01-05
- [ ] Verify all 7 methods evaluated
- [ ] Check visualization quality
- [ ] Review comprehensive reports

---

**Project Status:** Ready for full dataset experiments with 7 feature methods! 🚀

**Last Major Update:** 2026-04-18 - Fixed min_duration config, legend overlap, added proposal context
