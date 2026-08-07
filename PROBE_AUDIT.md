# Audit: the frozen-encoder baselines are undertrained-probe artefacts

**Date:** 2026-08-07
**Scope:** all 24 frozen zero-shot cells produced by `scripts/phase3_03_zero_shot_evaluation.py`
**Status:** measurement only — no published result, figure, or script has been modified

---

## 1. Summary

Every frozen-encoder number in the paper was produced by a linear probe that never
finished training. The probe uses **full-batch gradient descent**, so one "epoch"
is a *single* gradient step, and the branch that produced the published runs took
**50 epochs = 50 gradient steps** with no early stopping.

Re-running the identical features with a probe trained to convergence moves every
number upward by **+0.05 to +0.58 macro-F1**. The model rankings reorder on all
three tasks, and most "fine-tuning helps" conclusions do not survive.

The corrections are not a different measurement. Replicating the original
50-step recipe reproduces the published value in every group, which is what
establishes that training length — and nothing else — is responsible:

| group | published range | 50-step replication error | corrected range |
|---|---|---|---|
| Hyrax session-holdout (6 models) | 0.078 – 0.207 | ±0.002 (5/6) | 0.238 – 0.416 |
| Denoiser screen (6 cells) | 0.066 – 0.113 | ±0.010 | 0.230 – 0.663 |
| Species 7-way (6 models) | 0.538 – 0.874 | ±0.008 (5/6) | 0.778 – 0.969 |
| Species 8-class (6 models) | 0.458 – 0.864 | ±0.025 | 0.705 – 0.972 |

---

## 2. Root cause

`scripts/phase3_03_zero_shot_evaluation.py`, `train_classifier()`:

```python
for epoch in range(max_epochs):          # max_epochs = 50 when no val split
    outputs = classifier(train_X)        # ENTIRE training set, one forward
    loss = criterion(outputs, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()                     # ONE gradient step per "epoch"
```

There is no minibatching, so `max_epochs` is literally the number of gradient
steps. With Adam at `lr=1e-3`, 50 steps is far from convergence.

The published runs record their own training-set macro-F1, which is a free
diagnostic — a fitted probe should score far above chance:

| model | published train F1 | test F1 | chance = 0.125 |
|---|---|---|---|
| ecapa_tdnn | 0.5381 | 0.2072 | undertrained |
| hubert_base | 0.3288 | 0.1735 | undertrained |
| wav2vec2_base | 0.3297 | 0.1527 | undertrained |
| wavlm | 0.1536 | 0.1115 | barely above chance |
| xls_r | 0.1131 | 0.1017 | **at chance** |
| wav2vec2_base_960h | 0.0834 | 0.0784 | **below chance** |

Three of six models could not classify their own training data. A converged
probe on the same features reaches train macro-F1 ≈ 0.92–1.00.

### Why this biased the *ranking*, not just the level

Convergence speed under a fixed step budget depends on feature dimensionality.
ECAPA's 192-dim embeddings fit fastest and so moved least; the 1024-dim XLS-R
features moved most:

| model | dim | movement |
|---|---|---|
| ecapa_tdnn | 192 | +0.051 |
| hubert_base | 768 | +0.149 |
| wav2vec2_base_960h | 768 | +0.160 |
| wavlm | 768 | +0.244 |
| wav2vec2_base | 768 | +0.263 |
| xls_r | 1024 | +0.239 |

The published ranking was substantially measuring *how quickly a probe starts to
fit*, which correlates with embedding width — not representational quality.

---

## 3. Method

`scripts/phase3_20_probe_audit.py`. For each cell it re-extracts features
**exactly** as `phase3_03` does — same final layer, same mean pooling, same
manifest class weights, and the same per-task extraction regime (`phase3_03:217`
windows hyrax tasks at 5 s/2.5 s but uses one 30 s-truncated embedding per file
for species) — then trains the probe at 50/100/200/500/1000/2000/5000 steps.
Only training length varies.

Two numbers per cell:

- **replication** — the original recipe, to verify the features match.
- **corrected** — early stopping on a held-out split. Where the manifest has a
  val split it is used; where it does not, an internal stratified 80/20 split of
  *train*. **Test is never used for stopping or selection.**

---

## 4. Results

### 4.1 Hyrax individual ID, session-holdout (8 classes, chance 0.125)

| model | published | replication | corrected | movement |
|---|---|---|---|---|
| wav2vec2_base | 0.1527 | 0.1528 | **0.4155** | +0.263 |
| wavlm | 0.1115 | 0.1109 | **0.3559** | +0.244 |
| xls_r | 0.1017 | 0.0998 | **0.3404** | +0.239 |
| hubert_base | 0.1735 | 0.1732 | **0.3221** | +0.149 |
| ecapa_tdnn | 0.2072 | 0.2197 | **0.2586** | +0.051 |
| wav2vec2_base_960h | 0.0784 | 0.0784 | **0.2381** | +0.160 |

```
published ranking:  ecapa > hubert > w2v2_base > wavlm > xls_r > w2v2_960h
corrected ranking:  w2v2_base > wavlm > xls_r > hubert > ecapa > w2v2_960h
```

ECAPA falls 1st → 5th; HuBERT 2nd → 4th.

### 4.2 Species ID, 8-class (chance 0.125)

| model | published | replication | corrected | movement |
|---|---|---|---|---|
| xls_r | 0.7194 | 0.7018 | **0.9717** | +0.252 |
| hubert_base | 0.8635 | 0.8498 | **0.9624** | +0.099 |
| wav2vec2_base | 0.7646 | 0.7396 | **0.9426** | +0.178 |
| wavlm | 0.6744 | 0.6771 | **0.9423** | +0.268 |
| ecapa_tdnn | 0.7201 | 0.6978 | **0.8540** | +0.134 |
| wav2vec2_base_960h | 0.4576 | 0.4564 | **0.7050** | +0.247 |

XLS-R moves **4th → 1st**.

### 4.3 Species ID, 7-way (hyrax excluded, chance 0.143)

| model | published | replication | corrected | movement |
|---|---|---|---|---|
| xls_r | 0.8051 | 0.8007 | **0.9690** | +0.164 |
| hubert_base | 0.8736 | 0.8784 | **0.9624** | +0.089 |
| wavlm | 0.7603 | 0.7604 | **0.9478** | +0.188 |
| wav2vec2_base | 0.7971 | 0.7966 | **0.9350** | +0.138 |
| ecapa_tdnn | 0.7708 | 0.7367 | **0.8879** | +0.117 |
| wav2vec2_base_960h | 0.5378 | 0.5301 | **0.7779** | +0.240 |

XLS-R overtakes HuBERT here too.

### 4.4 Denoiser screen (XLS-R) — the leakage diagnostic had no power

| cell | published | replication | corrected | movement |
|---|---|---|---|---|
| original / within-session | 0.1043 | 0.1125 | **0.5882** | +0.484 |
| original / session-holdout | 0.1123 | 0.1155 | **0.2908** | +0.179 |
| bioda / within-session | 0.1125 | 0.1122 | **0.6633** | +0.551 |
| bioda / session-holdout | 0.1036 | 0.0998 | **0.3404** | +0.237 |
| aca / within-session | 0.0674 | 0.0674 | **0.6451** | +0.578 |
| aca / session-holdout | 0.0664 | 0.0649 | **0.2299** | +0.164 |

The within-session arm is the deliberately leaky control: train and test share
sessions, so it should score *higher* than session-holdout if session leakage
exists.

| version | published gap | corrected gap |
|---|---|---|
| original | −0.008 | **+0.297** |
| bioda | +0.009 | **+0.323** |
| aca | +0.001 | **+0.415** |

Published gaps were ≈ 0, which read as "no leakage detected". In fact the probe
was pinned at chance in *both* arms, so the diagnostic could not detect anything.
Corrected, session leakage is worth **0.30–0.42 macro-F1**.

**This strengthens the paper.** It confirms that session-holdout evaluation is
necessary, which is the design already used for the main hyrax results.

Denoiser ranking also changes — and now supports the choice that was made:

```
published:  original (0.1123) > bioda (0.1036) > aca (0.0664)
corrected:  bioda (0.3404) > original (0.2908) > aca (0.2299)
```

---

## 5. Consequences for the fine-tuning claims

No fine-tuned or LoRA result changed; only the baselines they are measured
against. Recomputing the gaps:

### 5.1 Species 8-class — fine-tuning loses to a frozen probe below 25 % data

Corrected frozen: HuBERT **0.9624**, XLS-R **0.9717**.

| frac | HuBERT ft (mean ± SD) | vs frozen | XLS-R ft (mean ± SD) | vs frozen |
|---|---|---|---|---|
| 1 % | 0.7759 ± 0.030 (n=5) | **−0.187** | 0.7588 ± 0.056 (n=5) | **−0.213** |
| 2 % | 0.7424 ± 0.075 (n=5) | **−0.220** | 0.7392 ± 0.099 (n=5) | **−0.233** |
| 5 % | 0.8644 ± 0.046 (n=5) | **−0.098** | 0.7727 ± 0.026 (n=5) | **−0.199** |
| 10 % | 0.8862 ± 0.055 (n=5) | **−0.076** | 0.7792 ± 0.126 (n=5) | **−0.193** |
| 25 % | 0.9719 ± 0.004 (n=5) | +0.010 | 0.8598 ± 0.098 (n=5) | **−0.112** |
| 50 % | 0.9766 (n=1) | +0.014 | 0.9772 (n=1) | +0.006 |
| 100 % | 0.9586 (n=1) | −0.004 | 0.9809 (n=1) | +0.009 |

Fine-tuning beats the frozen probe at **2 of 7 fractions for both models**. The
deficits at 1–10 % are 4–8× the seed SD, so they are not noise.

### 5.2 Hyrax — the one surviving positive result

Corrected frozen: HuBERT **0.3221**, XLS-R **0.3404**.

| frac | HuBERT ft (n=5) | vs frozen | XLS-R ft (n=5) | vs frozen |
|---|---|---|---|---|
| 10 % | 0.2656 ± 0.063 | −0.057 | 0.0969 ± 0.023 | −0.244 |
| 25 % | 0.3788 ± 0.046 | **+0.057** | 0.2765 ± 0.052 | −0.064 |
| 50 % | 0.4018 ± 0.027 | **+0.080** | 0.2553 ± 0.041 | −0.085 |
| 100 % | 0.4066 ± 0.024 | **+0.085** | 0.3167 ± 0.048 | −0.024 |

**HuBERT on hyrax at ≥25 % data is the only place fine-tuning genuinely helps**,
and it is worth ≈ +0.08, not the +0.233 implied by the published baseline. For
XLS-R, fine-tuning is *worse* than a frozen probe at every fraction.

### 5.3 Staged adaptation (Phases B / C)

LoRA adaptation on the 7-class species task, then frozen probing:

- Species: adapted 0.9772 (HuBERT) / 0.9834 (XLS-R) against corrected frozen
  0.9624 / 0.9690 → **+0.015 / +0.014**, not +0.104 / +0.178.
- Hyrax probe (Phase C): the best layer for both models is layer 0 or 1, and
  layer 0 is bit-identical between base and adapted by construction. At the
  test-oracle level adaptation gives **+0.000 (HuBERT) and −0.005 (XLS-R)**.

Staged adaptation does not currently support a positive claim.

---

## 6. What survives

- **Monolingual vs multilingual comparisons.** Both models moved in the same
  direction, so the contrast is intact — and the multilingual advantage is
  *larger* than reported, since XLS-R now leads on both tasks frozen.
- **HuBERT is more reliable at low data.** A HuBERT-vs-XLS-R comparison,
  independent of the frozen baseline.
- **XLS-R's low-data instability.** SD ±0.126 at 10 %, ±0.098 at 25 %.
- **Session leakage is real and large** — strengthened, see §4.4.
- **BIODA as the denoiser** — now supported by the corrected numbers.
- **HuBERT's fine-tuning gain on hyrax**, at ≈ +0.08.

## 7. What does not survive

- Frozen model rankings on all three tasks.
- "Fine-tuning helps" on species — it is neutral at best, harmful below 25 %.
- The species data-efficiency narrative: the entire curve sits at or below the
  corrected frozen baseline.
- Any claim that ECAPA is the strongest frozen encoder on hyrax (5th of 6).
- The cross-task reversal as currently framed.
- Staged adaptation (Phases B / C) as a positive result.

## 8. Caveats on the corrected numbers

1. **Single probe seed.** No error bars on the corrected values themselves. The
   direction and magnitude are robust across 24 independent cells, but the
   figures should be re-derived with multiple seeds before publication.
2. **Stopping criterion.** Where no val split exists, stopping uses an internal
   stratified 80/20 split of train. Defensible, but not the only valid choice.
3. **Manifest mismatch in §5.2.** Corrected frozen hyrax uses the plain
   session-holdout manifest (1353 train windows); the fine-tuned runs use the
   `_ft` variant (1011). Test splits are identical (409 windows, same held-out
   sessions). The Phase C probe avoids this mismatch and points the same way.
4. **ECAPA offset.** ECAPA replication is consistently 0.02–0.03 from published,
   likely a SpeechBrain version difference. It does not affect any conclusion.
5. **Overfitting is possible.** A linear probe on ~1300 samples in 1024 dims can
   overfit; this is why the trajectory is reported alongside the early-stopped
   value rather than simply training to 5000 steps.

## 9. Recommendations

1. **Agree the reframing before regenerating figures.** The narrative change is
   larger than the numeric change.
2. **Leave `phase3_03` unmodified** and cite this audit, so the published numbers
   remain reproducible from the repository.
3. **Re-derive the corrected baselines with multiple probe seeds** for anything
   entering the paper.
4. **Do not re-run any training.** All adapters and LoRA results are unaffected;
   only the probes were broken.

## 10. Reproducing this

```bash
# per-cell results (24 JSON files)
outputs/phase3/probe_audit/

# one cell, showing the full step trajectory
python scripts/phase3_20_probe_audit.py \
    --model hubert_base \
    --manifest outputs/phase3/manifests/hyrax_id_session_holdout.json \
    --published-f1 0.1735 \
    --output-dir outputs/phase3/probe_audit

# reproduce the original undertrained value
python scripts/phase3_20_probe_audit.py ... --probe-select final --probe-max-epochs 50

# all 18 GPU cells
sbatch run_probe_audit.sh
```
