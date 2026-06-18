# Phase 2 — Professor's Feedback & Confirmed Decisions

**Status:** Finalized after meeting. This document captures every decision and open item from the professor's feedback. It is a planning reference — not yet implementation instructions for Claude Code.

**Note:** Hyrax data and fine-tuning specifics are deferred until the zero-shot stage is complete and the best model is identified.

---

## 0. The Core Conceptual Shift

Phase 1 evaluated each dataset independently using frozen embeddings fed into classical classifiers (SVM, etc.). Phase 2 changes this in three fundamental ways:

1. **New pooled task** — mix all datasets together and test *general* animal identification, not just per-dataset identification. The central scientific question: is the model identifying *the animal itself*, or merely *which dataset the recording came from* (i.e. recording conditions, microphone, environment)?
2. **Classifier methodology changes** — only a fully-connected (FC) head is used now. No SVM, no classical ML. This redefines "zero-shot" in this project as: frozen backbone → trained FC head.
3. **Zero-shot first, everything else later** — all fine-tuning and hyrax decisions are deferred until zero-shot results identify the best model.

---

## 1. Data Splits & Manifests

| Item | Decision |
|------|----------|
| Split ratio | **80 train / 10 validation / 10 test** |
| Format | **Manifest file** per dataset — explicitly lists which sample belongs to train, validation, or test |
| Coverage | **All 7 datasets** get a manifest with train/val/test partitions |
| Stratification | Stratified so each individual is represented across splits |
| Imbalance handling | **Class-weighted loss** (weight prior in the loss function) — prevents overfitting on small datasets like Marmoset, rather than relying on sampling alone |
| Verification | Claude proposes the manifest-creation strategy → **professor verifies partitions before use** |

**Key rule on test splits:** Zero-shot is **always evaluated on the test split**. For methodology comparison, only the test sets are used so different approaches can be compared on equal footing. Once the model/architecture is finalized, the training data is used only for the hyrax zero-shot stage.

**Action for Claude:** propose the best manifest-creation strategy for the individual datasets, for the professor to approve.

---

## 2. Models for Zero-Shot (5 total)

Run zero-shot on **all five**, then carry **only the single best** forward to fine-tuning.

| # | Model | HuggingFace / Source | Notes |
|---|-------|----------------------|-------|
| 1 | Wav2Vec2 Base (pretrained-only) | `facebook/wav2vec2-base` | Self-supervised, never ASR-fine-tuned |
| 2 | Wav2Vec2 Base (ASR-fine-tuned) | `facebook/wav2vec2-base-960h` | Same architecture, fine-tuned on 960h English ASR |
| 3 | XLS-R | `facebook/wav2vec2-xls-r-300m` | Multilingual large (300M, 128 languages) |
| 4 | WavLM | `microsoft/wavlm-base-plus` | — |
| 5 | ECAPA-TDNN | `speechbrain/spkrec-ecapa-voxceleb` | Speaker-verification architecture — see note below |

**Comparison logic for models 1 vs 2:** isolates whether ASR fine-tuning on human speech helps or hurts transfer to animal vocalizations.

**ECAPA-TDNN special handling:** structurally different — it's a TDNN speaker-embedding model (SpeechBrain), not a HuggingFace transformer. It loads via a different API and has **no transformer layers to sweep**. The per-layer analysis applies to models 1–4 only; for ECAPA, extract its single pooled speaker embedding instead.

**⚠️ Open item to confirm with professor:** the original notes listed "Wav2Vec Base" and "Wav2Vec2" as separate entries. Interpreted as `wav2vec2-base` (pretrained-only) vs `wav2vec2-base-960h` (ASR-fine-tuned). *Confirm this interpretation before committing HPC runs.*

---

## 3. Zero-Shot Evaluation Protocol

| Item | Decision |
|------|----------|
| Definition of zero-shot | Frozen backbone → trained FC head, evaluated on the **test split** |
| Per-layer analysis | For models 1–4: test each transformer layer's representation to find which layer works best |
| ECAPA exception | Single pooled embedding (no layer sweep) |
| Classifier head | **Fully-connected layer only** — no SVM or other classical ML |
| Goal | Run zero-shot across all 5 models → identify the single best model → continue with that one only |

---

## 4. Pooled / General Identification Task

This is the new task that did not exist in Phase 1.

- **Mix the datasets** and run zero-shot to get a *general overview* of how the representation looks with everything present at once.
- **Visualize the embedding space** (e.g. t-SNE) of all individuals together.
- **Bird-specific check:** birds span four datasets (Bengalese Finch, Picidae, Wetlands Bird, Zebra Finch — 46 individuals total). Check whether embeddings cluster **by bird** (good — identifying the animal) or **by dataset** (bad — a recording-condition artifact).
- **If clusters form by dataset:** contrastive learning is the proposed fix to remove these "bars" / dataset artifacts — but this is a **later discussion**, only after the professor sees the results.

---

## 5. Fine-Tuning (Deferred — decided after zero-shot)

Specified now, but only executed once the best zero-shot model is chosen.

| Item | Decision |
|------|----------|
| Which model | Only the **single best** model from zero-shot |
| Wav2Vec2 family depth | Fine-tune **first 4 layers only**, freeze everything else |
| WavLM depth | Same structure as Wav2Vec2 (first 4 layers, freeze rest) |
| ECAPA depth | **Claude proposes the best partial-fine-tuning approach** — do NOT fine-tune everything |
| Head | Fully-connected layer |
| Per- vs multi-dataset | **Multi-dataset** — everything together at the same time |
| Sequencing | Focus on zero-shot first, then decide exact fine-tuning details |

**Action for Claude:** propose the best fine-tuning strategy according to the thesis needs (Raj shares this with the professor for his opinion).

---

## 6. Sampling Rate Experiment (the "1D-ResNet experiment")

A **separate, standalone experiment** whose sole goal is to measure whether information is lost by resampling audio down to 16 kHz. The aim is to **demonstrate the influence** of sampling rate on animal identification — **not necessarily to solve the problem**.

### 6.1 Verbatim feedback (so nothing is lost)

The professor's three points, captured directly:

1. **When should this run?** This should be a **separate experiment**. Take the best zero-shot model, fine-tune it on the original-rate data and also on the 16 kHz data, and compare the results. The main goal is to find out **whether we are losing information** by resampling.
2. **Which dataset?** Apply this experiment to **only one dataset** — choose either **Picidae** or **Wetlands Bird**.
3. **Which architecture?** Standard **ResNet-18/34 on mel spectrogram** is the named option, **but** the professor prefers fine-tuning the best wav2vec model instead, because then he doesn't have to change models. So: with one of the best wav2vec models, fine-tune (if possible) on the original sound frequency and test with 16 kHz. He stresses this does not need to fully solve the problem — just show the influence of sound frequency on animal identification.

### 6.2 Resolved decisions

| Item | Decision |
|------|----------|
| Experiment type | **Separate / standalone** — not part of the main zero-shot or fine-tuning sweeps |
| Model used | The **best wav2vec model** identified from zero-shot |
| Comparison | Fine-tune on **original sampling rate** vs **16 kHz**, then compare results |
| Dataset | **One bird dataset only** — Picidae **or** Wetlands Bird |
| Goal | Show whether (and how much) information is lost by 16 kHz resampling — demonstrate influence, not necessarily solve |

### 6.3 Architecture: primary vs fallback

- **Primary (preferred):** fine-tune the **best wav2vec model** at original rate vs 16 kHz. Preferred because it avoids introducing a new model into the pipeline.
- **Fallback:** **standard ResNet-18/34 on a mel spectrogram** — the originally named "1D-ResNet" option, used only if the wav2vec approach doesn't work for this comparison.

### 6.4 Technical notes & flags (not from professor — Claude's flags)

- **16 kHz front-end constraint:** Wav2Vec2's convolutional feature encoder assumes a 16 kHz input. Feeding a different rate changes the effective time resolution, so "original rate" fine-tuning is not a free swap — the front-end must adapt during fine-tuning. Handle this explicitly in implementation.
- **"20 kHz" vs "original rate" inconsistency in the source notes:** point #1 of the feedback mentioned "20 kHz" while point #3 said "the original sound frequency." These are most likely the same thing — "20 kHz" appears to be shorthand for the original rate, since Picidae and Wetlands Bird are recorded above 16 kHz. **Confirm the exact original sampling rate** (from the professor, or by reading it directly from the audio files) before setting the resampling step, since a literal 20 kHz vs the true original rate changes the experiment.

---

## 7. Hyrax Identification (Deferred)

| Item | Detail |
|------|--------|
| Timing | Professor provides the hyrax data **later**, once the foundation-model stage is solid |
| Individuals | **17 hyraxes** |
| Baseline | A previous student achieved **~80% accuracy** |
| Bar to clear | Whatever Raj produces becomes the **new baseline** — must be done properly, and should aim **above 80%** |
| Method | Once architecture is locked on the 7 datasets, hyrax is evaluated zero-shot: FC head trained on hyrax training data, tested on hyrax test split, frozen backbone |
| Priority | This is the make-or-break part of the thesis — high focus required |

---

## 8. Execution Order (implied)

1. Build manifests (80/10/10) for all 7 datasets → professor verifies.
2. Run zero-shot (FC head) across all 5 models; per-layer for models 1–4, pooled embedding for ECAPA.
3. Run the pooled / general identification task + embedding visualization + bird clustering check.
4. Identify the single best model.
5. Propose fine-tuning strategy → professor confirms → fine-tune (first 4 layers, multi-dataset).
6. Run the sampling rate experiment (Section 6) on one bird dataset.
7. Once foundation stage is solid → receive hyrax data → hyrax zero-shot (target >80%).

---

## 9. Open Items to Confirm with Professor

1. **Model identity (Section 2):** confirm `wav2vec2-base` (pretrained) vs `wav2vec2-base-960h` (ASR) interpretation.
2. **Manifest strategy (Section 1):** professor must verify the partitions once Claude proposes them.
3. **Fine-tuning strategy (Section 5):** Raj proposes, professor gives opinion before execution.
4. **Sampling rate dataset (Section 6):** Picidae or Wetlands Bird — pick one.
5. **Original sampling rate (Section 6.4):** confirm the literal rate ("20 kHz" vs true original) before setting the resampling step.

---

## 10. Constraints & Resources

- **HPC access confirmed** — compute is not a bottleneck for any planned work.
- All 5 models can run zero-shot, and fine-tuning the best model is feasible.

---

*This document is a planning reference for the upcoming implementation discussion. The implementation-ready instructions for Claude Code will be produced separately once the plan is finalized.*


✅ Config updated - Only wav2vec2_base will run

---
Step 2: Run Phase 2 Scripts (Step-by-Step)

Run these commands one by one in your terminal:

Command 1: Create Manifests (~1 minute)

python scripts/phase2_01_create_manifests.py
✅ Check: You should see outputs/phase2/manifests/ with 8 JSON files

---
Command 2: Per-Dataset Zero-Shot (~3-5 hours)

bash scripts/phase2_02_run_all_combinations.sh
This will run wav2vec2_base on all 7 datasets (7 evaluations total)

✅ Check: You should see outputs/phase2/zero_shot/per_dataset/[7 datasets]/wav2vec2_base/

---
Command 3: Aggregate Stage 2 Results (~1 minute)

python scripts/phase2_02_aggregate_results.py
✅ Check: You should see outputs/phase2/zero_shot/per_dataset_summary/ with plots

---
Command 4: Pooled Zero-Shot (~2-4 hours)

bash scripts/phase2_03_run_pooled_all_models.sh
This will run wav2vec2_base on pooled dataset (all 7 datasets combined)

✅ Check: You should see outputs/phase2/zero_shot/pooled/wav2vec2_base/

---
Command 5: Aggregate Stage 3 Results (~1 minute)

python scripts/phase2_03_aggregate_pooled_results.py
✅ Check: You should see outputs/phase2/zero_shot/pooled_summary/ with plots

---
Command 6: Model Selection (~1 minute)

python scripts/phase2_04_model_selection.py
✅ Check: You should see outputs/phase2/model_selection/best_model_selection.json
(Will select wav2vec2_base since it's the only model)

---
Command 7: Fine-Tuning (~5-15 hours)

python scripts/phase2_05_fine_tuning.py
✅ Check: You should see outputs/phase2/fine_tuning/wav2vec2_base/checkpoints/best_model.pth

---
Command 8: Sampling Rate Experiment (~4-10 hours)

python scripts/phase2_06_sampling_rate_experiment.py --dataset picidae
✅ Check: You should see outputs/phase2/sampling_rate_experiment/picidae/comparison/results.json

---
Command 9: Generate Final Report (~1 minute)

python scripts/phase2_07_generate_final_report.py
✅ Check: You should see outputs/phase2/final_report/phase2_final_report.txt