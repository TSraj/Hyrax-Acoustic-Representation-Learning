# Second Level Task - Implementation Plan

## Professor's Feedback

> The preliminary results look very good! We should meet and discuss the next steps, but the path you should follow is:
> 
> - **Cross-dataset animal identification with the selected datasets**
>   - You will need to have training, validation, and test splits for each dataset.
> 
> - **Zero-shot vs. Fine-tuning**: You have to evaluate the performance on your test sets with and without fine-tuning.
> 
> - **Hyrax identification (Zero-shot vs Fine-tuning)**.
> 
> - **Regarding your limitation with the sampling rate**, it's a good idea to come up with a strategy to process the audio signals independent of their sampling rate. For now we have to see if this is a real issue. One experiment we can try is training a 1D-ResNet with the training and validation data (as mentioned before) with the original sampling rate and the 16kHz version to find out if there is any considerable change in the performance.

---

## Current System Status

### Existing Pipeline
- **7 Active Datasets**: AnuraSet, Bengalese Finch, Macaque, Marmoset, Picidae, Wetlands Bird, Zebra Finch
- **4 Deep Learning Models**: Wav2Vec2 Base, Wav2Vec2 XLSR, HuBERT Base/Large, WavLM Base/Large, Whisper Base
- **3 Handcrafted Features**: OpenSMILE (MFCC), Prosodic Features, Librosa Acoustic Features
- **Current Approach**: Extract features → Pool → Classify (k-NN, SVM, Random Forest, XGBoost)
- **Current Sampling Rate**: All audio resampled to 16kHz mono

### Existing Scripts
1. `01_analyze_datasets.py` - Dataset analysis
2. `02_create_subsets_and_preprocess.py` - Preprocessing (resampling to 16kHz)
3. `03_extract_embeddings.py` - Feature extraction (layer-wise)
4. `04_visualize_embeddings.py` - t-SNE/UMAP visualizations
5. `05_comprehensive_evaluation.py` - Classifier evaluation

---

## Clarification Questions (PLEASE ANSWER)

### 1. **Cross-dataset Animal Identification**
Cross-dataset can mean multiple things. Which interpretation is correct?

- [ ] **Option A**: Train on Dataset X → Test on Dataset Y (cross-species generalization)
  - Example: Train on Marmoset → Test on Macaque
  
- [ ] **Option B**: Train on multiple datasets combined → Test on all datasets
  - Example: Train on (Marmoset + Macaque + Finches) → Test on each separately
  
- [ ] **Option C**: Train and test on each dataset separately, then compare generalization
  - Example: Train/test on Marmoset, train/test on Macaque, compare which features work best across species

**Your Answer:**

---

### 2. **Train/Validation/Test Splits**
What split ratios do you want for each dataset?

- [ ] **60/20/20** (train/val/test)
- [ ] **70/15/15** (train/val/test)
- [ ] **80/10/10** (train/val/test)
- [ ] **Other**: ___________

Should splits be:
- [ ] **Stratified by individual** (ensure each individual is represented in all splits)
- [ ] **Leave-one-individual-out** (some individuals only in test set)

**Your Answer:**

---

### 3. **Hyrax Dataset**
I don't see a Hyrax dataset in the current `Data/` directory.

**Questions:**
- Where is the Hyrax data located?
- What is the file structure? (individual folders, audio formats, durations)
- How many individuals are in the dataset?
- Should we add it to the existing 7 datasets or treat it separately?

**Your Answer:**

---

### 4. **Zero-shot vs Fine-tuning Scope**

#### Zero-shot Approach:
- [ ] Use pretrained models as-is (current pipeline approach) - **CONFIRM**

#### Fine-tuning Approach:
Which models should we fine-tune?
- [ ] Wav2Vec2 Base
- [ ] Wav2Vec2 XLSR
- [ ] HuBERT Base
- [ ] HuBERT Large
- [ ] WavLM Base
- [ ] WavLM Large
- [ ] Whisper Base
- [ ] All of the above
- [ ] Subset (please specify): ___________

**Fine-tuning strategy:**
- [ ] Fine-tune on each dataset separately (dataset-specific fine-tuning)
- [ ] Fine-tune on combined animal vocalization data (general animal audio fine-tuning)
- [ ] Both approaches

**Fine-tuning objective:**
- [ ] Classification head (freeze encoder, train classifier on top)
- [ ] Full model fine-tuning (update all weights)
- [ ] Other (please specify): ___________

**Your Answer:**

---

### 5. **Sampling Rate Experiment with 1D-ResNet**

The professor suggests training a 1D-ResNet to test whether 16kHz resampling affects performance.

**Questions:**
- [ ] Train 1D-ResNet from scratch on raw waveforms
- [ ] Use pretrained 1D-ResNet architecture (if available)
- [ ] Custom 1D-ResNet architecture

**Experimental design:**
- Train on **original sampling rate** (various: 8kHz, 16kHz, 22kHz, 44.1kHz, 48kHz depending on dataset)
- Train on **16kHz resampled** version
- Compare performance on test set

**Which datasets should we prioritize for this experiment?**
- [ ] All 7 datasets
- [ ] Subset (please specify which ones): ___________
- [ ] Only datasets with non-16kHz original sampling rates

**Confirmation:** This is a **separate experiment** from the pretrained model pipeline, correct?
- [ ] Yes, separate experiment
- [ ] No, integrate into main pipeline

**Your Answer:**

---

### 6. **Priority Order**
In what order should we tackle these tasks?

Please rank 1-5 (1 = highest priority):

- [ ] **Rank ___**: Set up train/val/test splits for all datasets
- [ ] **Rank ___**: Run zero-shot baseline evaluation
- [ ] **Rank ___**: Implement fine-tuning pipeline
- [ ] **Rank ___**: Add and evaluate Hyrax dataset (zero-shot + fine-tuning)
- [ ] **Rank ___**: Sampling rate experiment (1D-ResNet comparison)

**Your Answer:**

---

### 7. **Dataset Scope**
Should we:
- [ ] Keep all 7 current datasets + add Hyrax (8 total)
- [ ] Focus on a subset of datasets for cross-dataset experiments
- [ ] Other (please specify): ___________

**Your Answer:**

---

## Next Steps (After Questions Answered)

Once you provide answers, we will:

1. Create a detailed implementation plan
2. Identify required code changes:
   - New data splitting logic (train/val/test)
   - Fine-tuning training loops
   - 1D-ResNet implementation (if needed)
   - Cross-dataset evaluation framework
   - Hyrax dataset integration
3. Update configuration files
4. Create new scripts or modify existing ones
5. Define evaluation metrics and comparison framework

---

## Notes

- **Current system limitation**: All datasets are currently resampled to 16kHz. We need to preserve original sampling rates for the 1D-ResNet experiment.
- **Fine-tuning consideration**: Fine-tuning 4+ deep learning models on 7-8 datasets will be computationally expensive. GPU access and time estimates needed.
- **Memory constraints**: Previous OOM issues required layer-by-layer extraction. Fine-tuning may require similar memory optimizations.

---

**Status**: Awaiting clarification from professor/discussion
