# Hyrax Thesis Project Brief

## Project Name
**Hyrax Acoustic Representation Learning**

## Thesis Direction
This project focuses on **foundation models for acoustic representation learning** for animal vocalizations, with a primary application to **individual identification in hyrax vocalizations**.

The main idea is to use a pretrained audio foundation model such as **wav2vec 2.0** to extract meaningful acoustic representations from animal audio and evaluate whether these representations are useful for distinguishing which individual hyrax is vocalizing.

This is **not** a generative-audio thesis at this stage. The scope is centered on:
- acoustic representation learning
- feature extraction from animal vocalizations
- layer-wise analysis of wav2vec representations
- evaluating whether those features help identify individual hyraxes

---

## Core Thesis Question
**Can pretrained foundation audio models learn acoustic representations that are useful for identifying individual hyraxes from vocal recordings?**

### Supporting Questions
1. Do pretrained wav2vec representations already capture useful structure in animal vocalizations without fine-tuning?
2. Which wav2vec layers provide the most informative representations?
3. Are these representations suitable for downstream tasks such as individual hyrax identification?
4. In a later phase, does fine-tuning on animal audio improve the representation quality?

---

## Current Project Scope
The current goal is to:
1. prepare and submit the thesis proposal
2. run an initial baseline experiment
3. build a minimal, working pipeline for audio representation extraction

At this stage, the project should remain simple and focused.

### Do not start with
- complex fine-tuning
- large-scale model training
- generative modeling
- too many datasets at once
- full end-to-end classification pipelines
- advanced interpretability beyond simple layer comparison

---

## Immediate Goal
The immediate goal is to complete the **first experiment** and use it to support the proposal.

### End Goal of the First Experiment
The end goal is:
**Check whether a pretrained wav2vec model already gives useful acoustic representations for the available animal audio datasets.**

In simple terms:
- load audio
- preprocess it properly
- pass it through pretrained wav2vec
- extract embeddings from multiple layers
- visualize the embeddings
- inspect whether meaningful structure or clustering appears

This is a **representation quality experiment**, not a final performance experiment.

---

## Available Data Context
The student has already collected **two animal audio datasets** for the initial experiment.

Later, denoised **hyrax recordings** will be shared by the supervisor/team.

For now, the initial experiment should use the currently collected datasets to validate the basic wav2vec-based feature extraction pipeline.

---

## Initial Experiment Plan

### Objective
Use a **pretrained wav2vec model as a frozen feature extractor** and inspect whether its embeddings show meaningful organization for animal audio.

### Important Constraint
Do **not fine-tune** the model in the first experiment.

### Pipeline Overview
1. Organize the two datasets
2. Inspect audio formats and labels
3. Select a small subset first
4. Preprocess the audio consistently
5. Load pretrained wav2vec
6. Extract hidden-layer embeddings
7. Pool embeddings into one vector per sample
8. Visualize embeddings
9. Compare layers
10. Document observations

---

## Step-by-Step Tasks for the Initial Experiment

### Task 1: Organize the Datasets
For each of the two datasets:
- locate all audio files
- note file format (`.wav`, `.flac`, etc.)
- note sample rate
- note label availability
- note number of files
- note whether there are classes, species labels, or individual IDs

### Task 2: Create a Small Test Subset
Before using full datasets:
- select a small subset of files from dataset 1
- select a small subset of files from dataset 2
- if labels exist, choose samples from different classes or individuals

Purpose:
- verify the pipeline works
- avoid large-scale debugging

### Task 3: Preprocess Audio
Prepare all audio in a common format:
- load audio file
- convert to mono if needed
- resample to 16 kHz if required by the wav2vec model
- trim or chunk long recordings if necessary

Purpose:
- produce consistent input for wav2vec

### Task 4: Load Pretrained wav2vec
Use a publicly available pretrained wav2vec model.

Important:
- use it only as a feature extractor
- do not fine-tune yet

### Task 5: Extract Layer-Wise Embeddings
For each audio sample:
- pass audio through wav2vec
- collect hidden representations from multiple layers
- save outputs for later comparison

Purpose:
- study which layers are most useful

### Task 6: Convert Sequence Features to One Vector per Sample
Since wav2vec outputs frame-level or sequence-level representations, reduce each sample to one vector.

Simple approach:
- mean pooling over time

Output:
- one embedding vector per audio sample per layer

### Task 7: Visualize Embeddings
Use a dimensionality-reduction method such as:
- PCA
- t-SNE
- UMAP

Create plots and color samples by:
- dataset
- class label
- individual identity (if available)

Purpose:
- inspect whether similar sounds group together

### Task 8: Compare Layers
Check whether some layers produce better separation than others.

Questions to inspect:
- do similar samples cluster together?
- do different datasets separate?
- if identity labels exist, do recordings from the same individual appear closer?

### Task 9: Document Results
Write a short summary of observations:
- which layers looked promising
- whether embeddings showed structure
- whether preprocessing worked reliably
- what issues appeared

This summary can later support the proposal and next experiments.

---

## Expected Outputs of the First Experiment
By the end of the first experiment, the agent should help produce:

1. a working pipeline for audio -> wav2vec -> embeddings
2. a small cleaned subset of the two datasets
3. saved embeddings for several wav2vec layers
4. visualization plots
5. a short experimental note describing findings

---

## What Success Looks Like
The first experiment is successful if:
- the pipeline runs end-to-end without major issues
- audio preprocessing works consistently
- embeddings can be extracted from multiple layers
- visualizations can be generated
- there is at least some sign that pretrained representations capture useful structure

The success criterion is **not high classification accuracy**.
The success criterion is a **working baseline representation analysis**.

---

## What Comes After the Initial Experiment
Once the baseline experiment is complete, the next possible steps are:

1. run a simple downstream classifier on the embeddings
2. evaluate which layers perform best
3. prepare the proposal using the clarified methodology
4. later consider fine-tuning wav2vec on animal audio
5. eventually evaluate on hyrax recordings for individual identification

---

## Proposal Alignment
The proposal should be built around the same central idea as the experiment.

### Working Thesis Focus
- foundation models for acoustic representation learning
- animal vocalization analysis
- evaluating usefulness of pretrained wav2vec features
- eventual application to hyrax individual identification

### Suggested Research Question
**Can foundation audio models such as wav2vec produce acoustic representations that are useful for individual hyrax identification?**

### Suggested General Objective
Develop and evaluate a foundation-model-based approach for extracting informative acoustic representations from animal vocalizations, with a focus on hyrax individual identification.

### Suggested Specific Objectives
1. Evaluate pretrained wav2vec representations on animal audio datasets.
2. Analyze which layers provide the most useful acoustic embeddings.
3. Assess whether these representations are suitable for distinguishing individual hyraxes.

---

## Instructions for the Agent
The agent working on this project should:
- keep the scope simple at first
- prioritize a clean, working baseline
- avoid overengineering
- focus on reproducible preprocessing and embedding extraction
- treat the first experiment as a feasibility and representation-analysis study

### Practical Priority Order
1. dataset inspection
2. preprocessing
3. wav2vec embedding extraction
4. visualization
5. observation notes
6. optional simple classification later

### Avoid for Now
- fine-tuning before the baseline is complete
- using all data immediately
- training large models
- adding unnecessary complexity
- turning this into a generative-audio project

---

## One-Sentence Summary
This project investigates whether pretrained foundation audio models such as wav2vec can extract meaningful acoustic representations from animal vocalizations that are useful for individual hyrax identification.
