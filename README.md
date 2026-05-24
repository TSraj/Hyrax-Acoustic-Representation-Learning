# Hyrax Acoustic Representation Learning

A comprehensive Python pipeline for evaluating deep learning and handcrafted acoustic features for animal vocalization analysis, with a focus on individual identification across diverse species.

## Project Overview

This project provides a complete evaluation framework comparing:
- **4 Deep Learning Models**: Wav2Vec2, HuBERT, WavLM, Whisper
- **3 Handcrafted Features**: OpenSMILE, Prosodic Features, Librosa Acoustic Features
- **7 Animal Datasets**: Frogs, birds, primates across diverse acoustic characteristics

### Key Research Questions

1. Do pretrained audio foundation models capture useful structure in animal vocalizations?
2. Which model layers provide the most informative representations?
3. How do deep learning features compare to handcrafted acoustic features?
4. Are these representations suitable for individual identification tasks?
5. How do results generalize across different species?

## Datasets

### Active Datasets (7 species)

| Dataset | Species | Individuals | Characteristics |
|---------|---------|-------------|-----------------|
| **AnuraSet** | Frogs | 4 | 60s recordings (auto-chunked) |
| **Bengalese Finch** | Bird | 11 | Song sequences |
| **Macaque** | Primate | 8 | Very short calls (50-100ms) |
| **Marmoset** | Primate | 11 | Diverse call types |
| **Picidae** | Woodpeckers | 13 species | Calls and drumming |
| **Wetlands Bird** | Mediterranean birds | 20 species | MP3 files (auto-converted) |
| **Zebra Finch** | Bird | 2 | Adult and chick vocalizations |

**Total**: ~700-1000 samples per dataset (configurable)

### Dataset Features
- ✅ Automatic MP3→WAV conversion
- ✅ Auto-chunking for long recordings (>60s)
- ✅ Flexible duration handling (50ms - 60min)
- ✅ Multi-format support (WAV, MP3, FLAC)

## Project Structure

```
hyrax-acoustic-representation-learning/
├── config/
│   └── config.yaml                    # All hyperparameters and settings
├── Data/
│   ├── AnuraSet/                      # Frog vocalizations
│   ├── Bengalese finch/              # Bird songs
│   ├── Macaque/                       # Primate calls
│   ├── Marmoset/                      # Primate vocalizations
│   ├── 7 Picidae Species/            # Woodpecker calls
│   ├── Western Mediterranean Wetlands Bird/  # Bird species
│   ├── Zebra finch/                   # Finch vocalizations
│   └── processed/                     # Preprocessed data
├── src/
│   ├── data/
│   │   ├── dataset_analyzer.py        # Dataset inspection
│   │   ├── subset_creator.py          # Subset creation + chunking
│   │   └── audio_preprocessor.py      # Audio preprocessing
│   ├── models/
│   │   ├── wav2vec_extractor.py       # Wav2Vec2/XLSR features
│   │   ├── hubert_extractor.py        # HuBERT features
│   │   ├── wavlm_extractor.py         # WavLM features
│   │   ├── whisper_extractor.py       # Whisper features
│   │   ├── opensmile_extractor.py     # OpenSMILE features
│   │   ├── prosodic_extractor.py      # Prosodic features
│   │   └── librosa_extractor.py       # Librosa features
│   ├── evaluation/
│   │   ├── visualizer.py              # t-SNE, UMAP visualization
│   │   ├── knn_classifier.py          # k-NN evaluation
│   │   ├── svm_classifier.py          # SVM with grid search
│   │   └── ensemble_classifiers.py    # Random Forest, XGBoost
│   └── utils/
│       ├── audio_utils.py             # Audio utilities + chunking
│       └── logging_utils.py           # Logging utilities
├── scripts/
│   ├── 01_analyze_datasets.py         # Dataset analysis
│   ├── 02_create_subsets_and_preprocess.py  # Preprocessing
│   ├── 03_extract_embeddings.py       # Feature extraction (all models)
│   ├── 04_visualize_embeddings.py     # Visualizations
│   └── 05_comprehensive_evaluation.py # Complete evaluation + CSVs
├── outputs/
│   ├── figures/                       # PNG visualizations (300 DPI)
│   ├── embeddings/                    # Saved features (.npz)
│   └── reports/
│       └── comprehensive_evaluation/
│           └── csv_exports/           # CSV result tables
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- 16GB+ RAM (for full dataset)
- GPU optional (speeds up extraction 10x)

### Setup

1. **Clone or navigate to the project directory:**

```bash
cd "Hyrax Acoustic Representation Learning"
```

2. **Create and activate virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Draft Run)

For testing the pipeline with subset of data:

```bash
# Ensure config has: samples_per_individual: 10
python scripts/01_analyze_datasets.py
python scripts/02_create_subsets_and_preprocess.py
python scripts/03_extract_embeddings.py
python scripts/04_visualize_embeddings.py
python scripts/05_comprehensive_evaluation.py
```

**Time: ~60-90 minutes**

### Full Run (Thesis Quality)

For complete results with all data:

```bash
# Update config: samples_per_individual: 100
python scripts/01_analyze_datasets.py
python scripts/02_create_subsets_and_preprocess.py
python scripts/03_extract_embeddings.py
python scripts/04_visualize_embeddings.py
python scripts/05_comprehensive_evaluation.py
```

**Time: ~4-6 hours (GPU recommended)**

---

## Pipeline Scripts

### Step 1: Analyze Datasets

```bash
python scripts/01_analyze_datasets.py
```

**What it does:**
- Scans all active datasets
- Counts files and individuals
- Checks audio formats and sample rates
- Estimates total duration

**Output:**
- `outputs/reports/dataset_analysis_report.txt`
- `outputs/reports/dataset_stats.json`

---

### Step 2: Create Subsets and Preprocess

```bash
python scripts/02_create_subsets_and_preprocess.py
```

**What it does:**
- Samples N files per individual (configurable)
- Converts MP3/FLAC → WAV automatically
- Chunks long files (>60s) into smaller segments
- Resamples to 16kHz mono
- Filters by duration (0.05s - 3600s)
- Normalizes audio

**Features:**
- ✅ Auto-chunking: 30s chunks with 5s overlap
- ✅ MP3 support: Auto-converts to WAV
- ✅ Duration filtering: Handles 50ms to 60min files

**Output:**
- Subsets: `Data/processed/subsets/`
- Preprocessed: `Data/processed/preprocessed_subsets/`

---

### Step 3: Extract Embeddings

```bash
python scripts/03_extract_embeddings.py
```

**What it does:**
- Extracts features from ALL models in parallel
- Saves layer-wise representations
- Applies pooling strategies (mean, max, first, last)

**Models extracted:**
1. **Wav2Vec2 Base** (12 layers, 95M params)
2. **Wav2Vec2 XLSR** (24 layers, 300M params)
3. **HuBERT Base** (12 layers)
4. **HuBERT Large** (24 layers)
5. **WavLM Base** (12 layers)
6. **WavLM Large** (24 layers)
7. **Whisper Base** (encoder features)

**Output:**
- Layer features: `outputs/embeddings/*_features.npz`
- Pooled features: `outputs/embeddings/*_pooled.npz`

**Note:** GPU highly recommended for this step.

---

### Step 4: Visualize Embeddings

```bash
python scripts/04_visualize_embeddings.py
```

**What it does:**
- Applies t-SNE and UMAP dimensionality reduction
- Creates per-layer visualizations
- Generates layer comparison grids
- Saves high-quality figures (300 DPI PNG)

**Output:**
- Individual layer plots
- Comparison grids (all layers)
- Feature comparison charts
- Saved to `outputs/figures/`

---

### Step 5: Comprehensive Evaluation

```bash
python scripts/05_comprehensive_evaluation.py
```

**What it does:**
- Evaluates ALL models and features
- Tests 6 classifiers: k-NN, Linear Probe, Logistic Regression, SVM (Linear & RBF), Random Forest, XGBoost
- Identifies best layers per model
- Generates comparison charts
- **Exports CSV tables** for analysis

**CSV Outputs** (`outputs/reports/comprehensive_evaluation/csv_exports/`):

| File | Description |
|------|-------------|
| `layer_wise_accuracy.csv` | Performance of EVERY layer |
| `best_layer_per_model.csv` | Best layer for each model per dataset |
| `model_comparison.csv` | All models × classifiers × datasets |
| `classifier_comparison.csv` | Classifier performance comparison |
| `cross_species_summary.csv` | **Best method per dataset (winner table)** |

**Figure Outputs:**
- Feature comparison bar charts
- Classifier comparison charts
- Layer comparison heatmaps
- Saved to `outputs/reports/comprehensive_evaluation/`

---

## Configuration

All parameters in `config/config.yaml`:

### Key Settings

**Active Datasets:**
```yaml
datasets:
  active:
    - "anuraset"
    - "bengalese_finch"
    - "macaque"
    - "marmoset"
    - "picidae"
    - "wetlands_bird"
    - "zebra_finch"
```

**Subset Creation:**
```yaml
subset:
  enabled: true
  samples_per_individual: 10    # Draft: 10, Full: 100
  random_seed: 42
```

**Audio Preprocessing:**
```yaml
preprocessing:
  target_sample_rate: 16000
  channels: 1
  normalize: true
  min_duration: 0.05           # 50ms (for short macaque calls)
  max_duration: 3600.0         # 60 minutes
  chunk_long_files: true       # Auto-chunk files >60s
  chunk_threshold: 60.0        # Chunk files longer than this
  chunk_size: 30.0             # Draft: 60s, Full: 30s
  chunk_overlap: 5.0           # Overlap between chunks
```

**Models:**
```yaml
models:
  wav2vec2_base:
    model_id: "facebook/wav2vec2-base-960h"
  hubert_base:
    model_id: "facebook/hubert-base-ls960"
  wavlm_base:
    model_id: "microsoft/wavlm-base"
  whisper_base:
    model_id: "openai/whisper-base"
```

**Evaluation:**
```yaml
knn:
  n_neighbors: [3, 5, 7, 9]
  test_size: 0.2
svm:
  grid_search: true           # Optimizes hyperparameters
  cv_folds: 5
```

---

## Expected Results

After running all scripts:

### 1. Dataset Analysis
- Full statistics for 7 datasets
- Audio format and duration distributions

### 2. Extracted Features
- 7 models × 7 datasets × multiple layers
- ~20-50GB of embeddings (full run)

### 3. Visualizations
- Layer-wise t-SNE/UMAP plots
- Feature comparison charts
- Classifier comparison charts
- Layer comparison grids

### 4. CSV Exports (Key Deliverable!)
- ✅ `layer_wise_accuracy.csv` - Complete layer analysis
- ✅ `best_layer_per_model.csv` - Optimal layers identified
- ✅ `model_comparison.csv` - Full comparison matrix
- ✅ `cross_species_summary.csv` - **Winner table**

### 5. Quantitative Metrics
- Accuracy, Balanced Accuracy, Macro F1
- Per-layer performance
- Cross-dataset comparison

---

## Models Compared

### Deep Learning Models (4)

| Model | Layers | Parameters | Training Data |
|-------|--------|------------|---------------|
| **Wav2Vec2 Base** | 12 | 95M | English speech (960h) |
| **Wav2Vec2 XLSR** | 24 | 300M | 128 languages (436K hours) |
| **HuBERT Base** | 12 | 95M | Masked prediction training |
| **HuBERT Large** | 24 | 300M | Masked prediction training |
| **WavLM Base** | 12 | 95M | Speech + noise robustness |
| **WavLM Large** | 24 | 300M | Speech + noise robustness |
| **Whisper Base** | 6 (enc) | 74M | Multilingual speech (680K hours) |

### Handcrafted Features (3)

| Feature Set | Description | Dimensions |
|-------------|-------------|------------|
| **OpenSMILE** | MFCC + spectral features | ~100 features |
| **Prosodic** | Pitch, formants, jitter, shimmer | ~50 features |
| **Librosa** | Spectral contrast, chroma, zero-crossing | ~80 features |

---

## Computational Requirements

### Minimum (Draft Run)
- **RAM**: 8GB
- **Storage**: 10GB free
- **CPU**: Multi-core (4+)
- **Time**: ~60-90 minutes

### Recommended (Full Run)
- **RAM**: 16-32GB
- **Storage**: 50GB free
- **GPU**: CUDA-compatible (8GB+ VRAM)
- **Time**: 4-6 hours with GPU

**Note:** Works on CPU but 10x slower.

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Errors

**Symptoms:** Script killed, "zsh: killed"

**Solutions:**
- Reduce `samples_per_individual` to 5-10
- Process datasets separately
- Increase `chunk_size` to 60s (reduces file count)

#### 2. Train/Test Split Errors

**Symptoms:** "test_size should be greater or equal to number of classes"

**Solutions:**
- Increase `samples_per_individual` to at least 10
- Ensure `min_duration` allows enough files to pass

#### 3. Cross-Validation Errors

**Symptoms:** "n_splits=5 cannot be greater than number of members"

**Solutions:**
- Increase `samples_per_individual` to allow 5-fold CV
- Or disable grid search: `perform_grid_search: false`

#### 4. Duration Filter Issues

**Symptoms:** Datasets have very few files after preprocessing

**Solutions:**
- Check `min_duration` (set to 0.05s for short calls)
- Check `max_duration` (3600s for long recordings)
- Review preprocessing logs for rejected files

#### 5. MP3 Files Not Found

**Symptoms:** "No audio files found"

**Solutions:**
- Ensure MP3 files are in correct directory structure
- Auto-conversion should work automatically
- Check file permissions

#### 6. Slow Feature Extraction

**Solutions:**
- Enable GPU: Check CUDA availability
- Reduce number of models in config
- Process fewer samples first

---

## Features Implemented

### ✅ Audio Processing
- Multi-format support (WAV, MP3, FLAC)
- Automatic MP3→WAV conversion
- Auto-chunking for long files
- Duration filtering (50ms - 60min)
- Silence trimming (optional)

### ✅ Feature Extraction
- 7 feature extraction methods
- Layer-wise representations
- Multiple pooling strategies
- Parallel processing

### ✅ Evaluation
- 6 classifier types
- Grid search hyperparameter optimization
- Cross-validation
- Stratified train/test split

### ✅ Outputs
- High-quality visualizations (300 DPI PNG)
- CSV exports for analysis
- Text reports
- Layer comparison charts

---

## Future Work

1. Fine-tuning models on animal audio
2. Multi-task learning (individual + call type)
3. Attention-based pooling strategies
4. Real-time inference pipeline
5. Additional species datasets

---

## Project Status

✅ **Complete Feature Set**
- 7 datasets configured and tested
- 7 feature extraction methods
- 6 classifiers with optimization
- Comprehensive CSV exports
- Full visualization pipeline

**Last Updated:** May 2024

---

## Quick Reference

### Draft Run (Test)
```yaml
samples_per_individual: 10
chunk_size: 60.0
```

### Full Run (Thesis)
```yaml
samples_per_individual: 100
chunk_size: 30.0
```

### Critical Settings
- `min_duration: 0.05` - Required for Macaque
- `chunk_long_files: true` - Required for AnuraSet
- `test_size: 0.2` - Standard 80/20 split

---

## Citation

If you use this code for your research, please cite:

```
[Thesis citation to be added]
```

## License

[To be determined]

## Contact

For questions or issues: [your email/info]
