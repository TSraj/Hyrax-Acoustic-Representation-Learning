# Hyrax Acoustic Representation Learning

A modular Python project for investigating foundation models for acoustic representation learning in animal vocalizations, with a focus on individual identification.

## Project Overview

This project evaluates whether pretrained audio foundation models (wav2vec 2.0 and XLSR) can extract meaningful acoustic representations from animal vocalizations useful for individual identification. The initial baseline experiment uses Macaque and Zebra Finch vocalizations.

### Key Research Questions

1. Do pretrained wav2vec representations capture useful structure in animal vocalizations?
2. Which wav2vec layers provide the most informative representations?
3. How do the English-trained wav2vec2-base and multilingual XLSR models compare?
4. Are these representations suitable for individual identification tasks?

## Project Structure

```
hyrax-acoustic-representation-learning/
├── config/
│   └── config.yaml                    # All hyperparameters and settings
├── Data/
│   ├── Macaque/                       # Macaque vocalizations (8 individuals)
│   ├── Zebra finch/                   # Zebra finch vocalizations
│   └── processed/                     # Preprocessed and subset data
├── src/
│   ├── data/
│   │   ├── dataset_analyzer.py        # Dataset inspection and statistics
│   │   ├── subset_creator.py          # Create test subsets
│   │   └── audio_preprocessor.py      # Audio preprocessing pipeline
│   ├── models/
│   │   ├── wav2vec_extractor.py       # Wav2vec feature extraction
│   │   └── feature_pooling.py         # Pooling strategies
│   ├── evaluation/
│   │   ├── visualizer.py              # PCA, LDA, t-SNE, UMAP visualization
│   │   ├── knn_classifier.py          # k-NN evaluation
│   │   └── layer_comparator.py        # Layer-wise comparison
│   └── utils/
│       ├── audio_utils.py             # Audio utility functions
│       └── logging_utils.py           # Logging utilities
├── scripts/
│   ├── 01_analyze_datasets.py         # Dataset analysis
│   ├── 02_create_subsets_and_preprocess.py  # Subset creation and preprocessing
│   ├── 03_extract_embeddings.py       # Feature extraction
│   ├── 04_visualize_embeddings.py     # Generate visualizations
│   └── 05_evaluate_layers.py          # Evaluation and comparison
├── outputs/
│   ├── figures/                       # High-quality visualizations
│   ├── embeddings/                    # Saved features
│   └── reports/                       # Analysis reports
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

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

The project follows a sequential pipeline with 5 main scripts:

### Step 1: Analyze Datasets

Inspect both datasets and generate summary statistics:

```bash
python scripts/01_analyze_datasets.py
```

**Output:**
- Dataset statistics (JSON)
- Analysis report (TXT)
- Saved to `outputs/reports/`

### Step 2: Create Subsets and Preprocess

Create small test subsets and preprocess audio to standard format:

```bash
python scripts/02_create_subsets_and_preprocess.py
```

**What it does:**
- Selects 20 samples per individual (configurable)
- Converts to mono
- Resamples to 16 kHz
- Normalizes audio

**Output:**
- Subsets saved to `Data/processed/subsets/`
- Preprocessed audio saved to `Data/processed/preprocessed_subsets/`

### Step 3: Extract Embeddings

Extract wav2vec features from preprocessed audio using both models:

```bash
python scripts/03_extract_embeddings.py
```

**What it does:**
- Loads pretrained wav2vec2-base and wav2vec2-xls-r models
- Extracts features from all transformer layers
- Applies multiple pooling strategies (mean, max, first, last)
- Saves embeddings

**Output:**
- Raw features: `outputs/embeddings/*_features.npz`
- Pooled features: `outputs/embeddings/*_pooled.npz`

**Note:** This step requires significant computation. GPU recommended but not required.

### Step 4: Visualize Embeddings

Generate visualizations using dimensionality reduction:

```bash
python scripts/04_visualize_embeddings.py
```

**What it does:**
- Applies PCA, LDA, t-SNE, and UMAP
- Creates per-layer visualizations
- Generates layer comparison grids

**Output:**
- Individual layer plots
- Comparison grids showing all layers
- Saved to `outputs/figures/`

### Step 5: Evaluate Layers

Run quantitative evaluation using k-NN classification:

```bash
python scripts/05_evaluate_layers.py
```

**What it does:**
- k-NN classification with cross-validation
- Clustering quality metrics (silhouette score, etc.)
- Layer-wise comparison
- Identifies best-performing layers

**Output:**
- k-NN performance plots
- Confusion matrices
- Layer comparison heatmaps
- Evaluation reports
- Saved to `outputs/reports/evaluation/`

## Configuration

All parameters are configurable in `config/config.yaml`:

### Key Settings

**Subset Creation:**
```yaml
subset:
  enabled: true
  samples_per_individual: 20
```

**Audio Preprocessing:**
```yaml
preprocessing:
  target_sample_rate: 16000
  channels: 1
  normalize: true
```

**Models:**
```yaml
models:
  wav2vec2_base:
    model_id: "facebook/wav2vec2-base-960h"
  wav2vec2_xlsr:
    model_id: "facebook/wav2vec2-xls-r-300m"
```

**Visualization:**
```yaml
visualization:
  methods:
    pca: {enabled: true}
    lda: {enabled: true}
    tsne: {enabled: true}
    umap: {enabled: true}
  dpi: 300
```

**k-NN Evaluation:**
```yaml
knn:
  n_neighbors: [3, 5, 7, 9]
  metric: "cosine"
  cv_folds: 5
```

## Expected Results

After running all scripts, you will have:

1. **Dataset Analysis:** Understanding of data distribution and quality
2. **Embeddings:** Layer-wise representations from both models
3. **Visualizations:** 
   - 2D projections showing clustering by individual
   - Layer comparison grids
4. **Quantitative Metrics:**
   - k-NN classification accuracy per layer
   - Clustering quality scores
   - Best layer identification
5. **Reports:** Text summaries and recommendations

## Models Compared

### wav2vec2-base-960h
- Trained on English speech (LibriSpeech, 960 hours)
- 12 transformer layers
- 95M parameters

### wav2vec2-xls-r-300m
- Trained on 128 languages (436K hours)
- 24 transformer layers
- 300M parameters
- Expected to generalize better to diverse acoustic patterns

## Datasets

### Macaque Vocalizations
- 8 individual macaques
- ~7,285 vocalization clips
- Individual identification task

### Zebra Finch Vocalizations
- ~42 individual birds
- ~3,433 adult vocalizations
- Both individual ID and call type labels

## Computational Requirements

### Minimum Requirements
- RAM: 8GB
- Storage: 5GB free space
- CPU: Multi-core recommended

### Recommended for Speed
- GPU: CUDA-compatible GPU (8GB+ VRAM)
- RAM: 16GB
- Storage: 10GB free space

**Note:** The project works on CPU, but feature extraction will be slower.

## Troubleshooting

### Out of Memory Errors
- Reduce `subset.samples_per_individual` in config
- Process datasets separately
- Use smaller model (wav2vec2-base only)

### Slow Feature Extraction
- Enable GPU in config: `device: "cuda"` or `device: "mps"` (Mac)
- Reduce batch size
- Extract fewer layers

### Visualization Errors
- Ensure at least 2 classes for LDA
- Reduce number of samples if t-SNE/UMAP too slow
- Disable specific methods in config if needed

## Future Work

This baseline experiment sets the foundation for:

1. Fine-tuning wav2vec on animal audio
2. Evaluation on hyrax recordings
3. Advanced pooling strategies (attention-based)
4. Comparison with hand-crafted features (MFCCs)
5. Multi-task learning (individual + call type)

## Citation

If you use this code for your research, please cite:

```
[Thesis citation to be added]
```

## License

[To be determined]

## Contact

For questions or issues, please contact [your email/info].

---

## Quick Start Example

```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run full pipeline
python scripts/01_analyze_datasets.py
python scripts/02_create_subsets_and_preprocess.py
python scripts/03_extract_embeddings.py
python scripts/04_visualize_embeddings.py
python scripts/05_evaluate_layers.py

# 3. Check results
ls outputs/figures/
ls outputs/reports/
```

---

**Project Status:** Baseline Experiment (Initial Phase)

**Last Updated:** April 2026
