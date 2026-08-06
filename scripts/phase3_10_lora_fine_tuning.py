#!/usr/bin/env python3
"""
Phase 3 - Step 10: LoRA Fine-Tuning with Capacity Reduction

Replaces the full-layer-unfreezing approach in phase3_05_fine_tuning.py, which
never actually trained the encoder. Four defects from that script are fixed here:

  #1 The encoder forward pass is NOT wrapped in torch.no_grad() during training,
     so gradients reach the LoRA adapters. (phase3_05 wrapped it, which silently
     reduced every "fine-tuning" run to a linear probe on frozen features.)
  #2 Audio is windowed (5s / 2.5s stride) exactly as in zero-shot evaluation, so
     training sees ~1000 windows instead of 1 concatenated file per class.
  #3 Model input length is bounded by construction: every input is exactly one
     5s window. (phase3_05 fed whole concatenated files, up to 18 minutes.)
  #4 Best-model state is deep-copied before restore. (phase3_05 stored live
     tensor references, so "restore best" restored nothing.)

Anti-memorization strategy:
  - LoRA adapters on attention projections, base encoder frozen (capacity reduction)
  - Dropout on the classifier head
  - ReduceLROnPlateau on validation macro-F1
  - AdamW, adapters 1e-4 / head 1e-3

Run the gradient check FIRST (--grad-check): it performs one optimizer step and
verifies the adapters receive non-zero gradients and their weights actually move,
while the frozen base weights do not. That is the go/no-go for the real run.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
from src.utils.audio_utils import load_audio

WINDOW_SECONDS = 5.0
STRIDE_SECONDS = 2.5
SAMPLE_RATE = 16000


def resolve_path(file_path):
    """Phase 2 manifest paths are relative to Data/; Phase 3 paths are not."""
    p = Path(file_path)
    if not p.exists() and not str(file_path).startswith('outputs/'):
        p = Path("Data") / file_path
    return p


class WindowedDataset:
    """Windows every file in a split into fixed-length audio segments.

    Fix #2 and #3: training operates on 5s windows, never whole concatenated
    files, so the sample count is large and the model input length is bounded
    by construction.

    Windows are materialised once into a float16 memmap on disk and reused by
    every later epoch, run and data fraction. species_id has ~14.6k training
    files, so re-decoding per epoch (many of them mp3) would dominate runtime,
    and holding them as float32 in RAM would cost several GB.
    """

    def __init__(self, items, class_to_idx, label_key, logger, split_name,
                 cache_dir, max_windows_per_file=None, rebuild=False):
        self.logger = logger
        self.split_name = split_name
        self.window_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
        stride_samples = int(STRIDE_SECONDS * SAMPLE_RATE)

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(items, label_key, max_windows_per_file)
        self.data_file = cache_dir / f"{split_name}_{key}_windows.npy"
        self.label_file = cache_dir / f"{split_name}_{key}_labels.npy"

        if rebuild or not (self.data_file.exists() and self.label_file.exists()):
            self._build(items, class_to_idx, label_key, stride_samples,
                        max_windows_per_file)
        else:
            logger.info(f"  {split_name}: reusing window cache {self.data_file.name}")

        self.labels = np.load(self.label_file)
        self.windows = np.load(self.data_file, mmap_mode='r')
        self.indices = np.arange(len(self.labels))

        counts = np.bincount(self.labels, minlength=len(class_to_idx))
        logger.info(f"  {split_name}: {len(self.labels)} windows | per-class {counts.tolist()}")
        self.class_counts = counts

    @staticmethod
    def _cache_key(items, label_key, max_windows_per_file):
        import hashlib
        h = hashlib.md5()
        h.update(f"{WINDOW_SECONDS}|{STRIDE_SECONDS}|{label_key}|{max_windows_per_file}".encode())
        for it in items:
            h.update(str(it['file']).encode())
        return h.hexdigest()[:12]

    def _build(self, items, class_to_idx, label_key, stride_samples, max_windows_per_file):
        self.logger.info(f"  {self.split_name}: building window cache "
                         f"({len(items)} files)...")
        chunks, labels, failed = [], [], 0

        for item in tqdm(items, desc=f"window {self.split_name}", leave=False):
            label = class_to_idx[item[label_key]]
            try:
                audio, _ = load_audio(str(resolve_path(item['file'])),
                                      target_sr=SAMPLE_RATE, mono=True)
            except Exception:
                failed += 1
                continue

            n = 0
            for start in range(0, len(audio) - self.window_samples + 1, stride_samples):
                chunks.append(audio[start:start + self.window_samples].astype(np.float16))
                labels.append(label)
                n += 1
                if max_windows_per_file is not None and n >= max_windows_per_file:
                    break

            if n == 0 and len(audio) > 0:
                # Shorter than one window: right-pad to exactly one window
                padded = np.zeros(self.window_samples, dtype=np.float16)
                padded[:len(audio)] = audio
                chunks.append(padded)
                labels.append(label)

        if failed:
            self.logger.warning(f"  {self.split_name}: {failed} files failed to load")
        if not chunks:
            raise RuntimeError(f"No windows produced for split {self.split_name}")

        arr = np.stack(chunks)
        # Atomic write so concurrent array tasks cannot read a partial cache.
        # Write through a file handle: np.save(path) appends ".npy" to any path
        # that does not already end in it, which would break the rename.
        tmp_data = Path(str(self.data_file) + ".tmp")
        tmp_lab = Path(str(self.label_file) + ".tmp")
        with open(tmp_data, 'wb') as f:
            np.save(f, arr)
        with open(tmp_lab, 'wb') as f:
            np.save(f, np.array(labels, dtype=np.int64))
        tmp_data.replace(self.data_file)
        tmp_lab.replace(self.label_file)
        self.logger.info(f"  {self.split_name}: cached {arr.shape} -> {self.data_file.name}")

    def subsample(self, fraction, seed):
        """Stratified subsample of WINDOWS, keeping every class represented.

        Applied at window level rather than file level because the hyrax task
        has exactly one concatenated file per class per split, which makes a
        file-level fraction meaningless.
        """
        if fraction >= 1.0:
            return
        rng = np.random.default_rng(seed)
        keep = []
        for cls in np.unique(self.labels):
            cls_idx = np.where(self.labels == cls)[0]
            n = max(1, int(round(len(cls_idx) * fraction)))
            keep.append(rng.choice(cls_idx, size=n, replace=False))
        self.indices = np.sort(np.concatenate(keep))
        counts = np.bincount(self.labels[self.indices],
                             minlength=len(self.class_counts))
        self.class_counts = counts
        self.logger.info(f"  {self.split_name}: subsampled to {fraction:.0%} -> "
                         f"{len(self.indices)} windows | per-class {counts.tolist()}")

    def __len__(self):
        return len(self.indices)

    def batches(self, batch_size, shuffle=False, rng=None):
        idx = self.indices.copy()
        if shuffle:
            (rng or np.random).shuffle(idx)
        for i in range(0, len(idx), batch_size):
            sel = np.sort(idx[i:i + batch_size])
            audio = [np.asarray(self.windows[j], dtype=np.float32) for j in sel]
            yield audio, torch.from_numpy(self.labels[sel])


class LoRAFineTuner:
    """LoRA fine-tuning of an SSL encoder with a dropout classifier head."""

    def __init__(self, model_name, manifest_path, output_dir, logger, args):
        self.model_name = model_name
        self.logger = logger
        self.args = args
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.num_classes = self.manifest['num_classes']
        if 'species_to_idx' in self.manifest:      # species_id
            self.class_to_idx = self.manifest['species_to_idx']
            self.label_key = 'species'
        else:                                      # hyrax individual tasks
            self.class_to_idx = self.manifest['class_to_idx']
            self.label_key = 'individual'
        self.class_names = sorted(self.class_to_idx, key=self.class_to_idx.get)
        self.logger_label_key = self.label_key

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {model_name} | classes: {self.num_classes}")
        self.logger.info(f"Manifest: {manifest_path}")

        self._load_model()

    # ------------------------------------------------------------------ model

    def _load_model(self):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, HubertModel
        from peft import LoraConfig, get_peft_model

        if self.model_name == "xls_r":
            model_id, self.embedding_dim = "facebook/wav2vec2-xls-r-300m", 1024
            cls = Wav2Vec2Model
        elif self.model_name == "hubert_base":
            model_id, self.embedding_dim = "facebook/hubert-base-ls960", 768
            cls = HubertModel
        else:
            raise ValueError(f"Unsupported model for LoRA fine-tuning: {self.model_name}")

        # Recorded in adapter_meta.json so Phase C can load the SAME base
        # encoder these adapters were trained against.
        self.base_model_id = model_id

        self.logger.info(f"Loading {model_id}")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        base = cls.from_pretrained(model_id, use_safetensors=True)

        # LayerDrop stochastically skips whole transformer layers in train mode.
        # XLS-R ships with 0.1, which means ~2-3 of 24 layers get no gradient on
        # any given step. Default to 0.0 so every adapter trains on every step.
        self.logger.info(f"LayerDrop: {base.config.layerdrop} -> {self.args.layerdrop}")
        base.config.layerdrop = self.args.layerdrop

        # Freeze the convolutional feature extractor (standard for wav2vec2-style
        # fine-tuning). LoRA freezes everything non-adapter anyway; this is explicit.
        if hasattr(base, "freeze_feature_encoder"):
            base.freeze_feature_encoder()

        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            bias="none",
        )
        self.model = get_peft_model(base, lora_config).to(self.device)

        adapter_names = [n for n, _ in self.model.named_parameters() if "lora_" in n]
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"LoRA r={self.args.lora_r} alpha={self.args.lora_alpha} "
                         f"dropout={self.args.lora_dropout} on q/k/v/out_proj")
        self.logger.info(f"  adapter tensors: {len(adapter_names)}")
        self.logger.info(f"  trainable: {n_trainable:,} / {n_total:,} "
                         f"({100 * n_trainable / n_total:.2f}%)")

        # Classifier head: Dropout -> Linear
        self.classifier = nn.Sequential(
            nn.Dropout(self.args.head_dropout),
            nn.Linear(self.embedding_dim, self.num_classes),
        ).to(self.device)

    def _encode(self, audio_list):
        """Encoder forward + mean-pool.

        Fix #1: deliberately NOT wrapped in torch.no_grad(). The returned
        embedding keeps its autograd graph so loss.backward() reaches the
        LoRA adapters. Callers that want no gradients must use torch.no_grad()
        at the call site (see _evaluate).
        """
        inputs = self.feature_extractor(
            audio_list, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Fix #3: inputs are exactly one window long by construction.
        max_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
        assert inputs['input_values'].shape[-1] <= max_samples, (
            f"input longer than one window: {inputs['input_values'].shape[-1]} > {max_samples}")

        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # [B, D]

    def _forward(self, audio_list):
        return self.classifier(self._encode(audio_list))

    # ------------------------------------------------------------------- data

    def load_splits(self, build_only=False):
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"WINDOWING ({WINDOW_SECONDS}s / {STRIDE_SECONDS}s stride) "
                         f"| label key: {self.label_key}")
        self.logger.info("=" * 80)
        datasets = {}
        for split in ['train', 'val', 'test']:
            items = self.manifest['splits'].get(split, [])
            if not items:
                continue
            datasets[split] = WindowedDataset(
                items, self.class_to_idx, self.label_key, self.logger, split,
                cache_dir=self.args.cache_dir,
                max_windows_per_file=self.args.max_windows_per_file,
                rebuild=self.args.rebuild_cache,
            )

        if not build_only and self.args.data_fraction < 1.0:
            # Only the training split is reduced; val/test stay full.
            datasets['train'].subsample(self.args.data_fraction, self.args.seed)

        return datasets

    def _class_weights(self, train_ds):
        """Inverse-frequency weights over WINDOW counts.

        The manifest's class_weights are file-level and therefore all 1.0 here
        (one file per individual per split), which ignores the large per-class
        window imbalance. Window-level inverse frequency is what actually
        matters for macro-F1.
        """
        counts = train_ds.class_counts.astype(np.float64)
        if self.args.class_weights == "none":
            w = np.ones_like(counts)
        elif self.args.class_weights == "manifest":
            w = np.array([self.manifest['class_weights'].get(c, 1.0)
                          for c in self.class_names], dtype=np.float64)
        else:  # window_inverse
            safe = np.where(counts > 0, counts, 1.0)
            w = counts.sum() / (len(counts) * safe)
            w = np.where(counts > 0, w, 0.0)
        self.logger.info(f"\nClass weights ({self.args.class_weights}): "
                         + ", ".join(f"{n}={v:.3f}" for n, v in zip(self.class_names, w)))
        return torch.FloatTensor(w).to(self.device)

    # -------------------------------------------------------------- grad check

    def _grad_stats(self, adapters):
        """Split adapter tensors by gradient state, separating lora_A from lora_B."""
        stats = {'A_nonzero': [], 'A_zero': [], 'B_nonzero': [], 'B_zero': [], 'none': []}
        norms = []
        for n, p in adapters.items():
            kind = 'A' if 'lora_A' in n else 'B'
            if p.grad is None:
                stats['none'].append(n)
                continue
            gn = p.grad.norm().item()
            norms.append(gn)
            stats[f"{kind}_{'nonzero' if gn > 0 else 'zero'}"].append(n)
        return stats, norms

    def gradient_check(self, train_ds):
        """Two optimizer steps; verify adapters get gradients and actually move.

        Two steps are required, not one. LoRA initialises lora_B to zero, so on
        step 1 the gradient w.r.t. lora_A is exactly zero by construction
        (dL/dA = B^T . dL/dout . x^T, and B = 0). Only lora_B receives gradient
        on the first step. After that step B != 0, so on step 2 lora_A must also
        receive non-zero gradient - that is what proves the whole adapter path
        trains, and it is the real go/no-go for the defect that made every
        earlier fine-tuning run a no-op.
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ENCODER GRADIENT CHECK (two training steps)")
        self.logger.info("=" * 80)

        self.model.train()
        self.classifier.train()

        criterion = nn.CrossEntropyLoss(weight=self._class_weights(train_ds))
        optimizer = optim.AdamW([
            {'params': [p for p in self.model.parameters() if p.requires_grad],
             'lr': self.args.encoder_lr},
            {'params': self.classifier.parameters(), 'lr': self.args.head_lr},
        ])

        # Snapshot adapter weights and a sample of frozen base weights
        adapters = {n: p for n, p in self.model.named_parameters()
                    if p.requires_grad and "lora_" in n}
        frozen = {n: p for n, p in self.model.named_parameters() if not p.requires_grad}
        frozen_sample = dict(list(frozen.items())[:5])

        before_adapters = {n: p.detach().clone() for n, p in adapters.items()}
        before_frozen = {n: p.detach().clone() for n, p in frozen_sample.items()}

        rng = np.random.default_rng(self.args.seed)
        batch_iter = train_ds.batches(self.args.batch_size, shuffle=True, rng=rng)

        step_reports = []
        requires_grad_ok = False
        head_grads = []

        for step in (1, 2):
            audio_list, labels = next(batch_iter)
            labels = labels.to(self.device)

            logits = self._forward(audio_list)
            loss = criterion(logits, labels)

            if step == 1:
                requires_grad_ok = bool(logits.requires_grad)
                self.logger.info(f"\nEmbedding requires_grad: {requires_grad_ok} "
                                 f"(must be True - this is defect #1)")

            optimizer.zero_grad()
            loss.backward()

            stats, norms = self._grad_stats(adapters)
            head_grads = [p.grad.norm().item() for p in self.classifier.parameters()
                          if p.grad is not None]

            self.logger.info(f"\n--- Step {step} | batch {len(audio_list)} windows "
                             f"| loss {loss.item():.4f}")
            self.logger.info(f"  lora_A: {len(stats['A_nonzero'])} non-zero grad, "
                             f"{len(stats['A_zero'])} zero grad")
            self.logger.info(f"  lora_B: {len(stats['B_nonzero'])} non-zero grad, "
                             f"{len(stats['B_zero'])} zero grad")
            self.logger.info(f"  grad is None: {len(stats['none'])} "
                             f"(non-zero here means LayerDrop skipped a layer)")
            if norms:
                self.logger.info(f"  grad norm: min={min(norms):.3e} max={max(norms):.3e} "
                                 f"mean={np.mean(norms):.3e}")
            self.logger.info(f"  head grad norms: {['%.3e' % g for g in head_grads]}")

            optimizer.step()

            step_reports.append({
                'step': step,
                'loss': loss.item(),
                'A_nonzero': len(stats['A_nonzero']),
                'A_zero': len(stats['A_zero']),
                'B_nonzero': len(stats['B_nonzero']),
                'B_zero': len(stats['B_zero']),
                'grad_none': len(stats['none']),
                'grad_norm_mean': float(np.mean(norms)) if norms else 0.0,
            })

        # --- did the weights actually move (across both steps)?
        deltas = {n: (adapters[n].detach() - before_adapters[n]).abs().max().item()
                  for n in adapters}
        moved = [n for n, d in deltas.items() if d > 0]
        moved_A = [n for n in moved if 'lora_A' in n]
        moved_B = [n for n in moved if 'lora_B' in n]
        max_delta = max(deltas.values()) if deltas else 0.0

        frozen_deltas = {n: (frozen_sample[n].detach() - before_frozen[n]).abs().max().item()
                         for n in frozen_sample}
        frozen_moved = [n for n, d in frozen_deltas.items() if d > 0]

        self.logger.info(f"\nAfter 2 optimizer steps:")
        self.logger.info(f"  adapter tensors changed: {len(moved)} / {len(adapters)} "
                         f"(lora_A: {len(moved_A)}, lora_B: {len(moved_B)})")
        self.logger.info(f"  max |delta| on adapters: {max_delta:.3e}")
        self.logger.info(f"  frozen base tensors changed: {len(frozen_moved)} / "
                         f"{len(frozen_sample)} sampled (must be 0)")

        s1, s2 = step_reports
        n_expected = len(adapters) // 2  # per A/B family

        checks = {
            "embedding carries autograd graph (defect #1 fixed)":
                requires_grad_ok,
            "no adapter has grad=None (LayerDrop disabled)":
                s1['grad_none'] == 0 and s2['grad_none'] == 0,
            "step 1: lora_B receives non-zero gradient":
                s1['B_nonzero'] == n_expected,
            "step 1: lora_A gradient is zero (correct, B is zero-init)":
                s1['A_nonzero'] == 0,
            "step 2: lora_A ALSO receives non-zero gradient (full path trains)":
                s2['A_nonzero'] == n_expected,
            "both lora_A and lora_B weights changed":
                len(moved_A) > 0 and len(moved_B) > 0,
            "frozen base weights unchanged":
                len(frozen_moved) == 0,
            "classifier head received gradients":
                len(head_grads) > 0,
        }

        self.logger.info("\n" + "-" * 80)
        for desc, ok in checks.items():
            self.logger.info(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
        self.logger.info("-" * 80)

        passed = all(checks.values())
        self.logger.info(f"\nGRADIENT CHECK: {'PASS' if passed else 'FAIL'}")

        report = {
            'passed': passed,
            'checks': checks,
            'steps': step_reports,
            'n_adapter_tensors': len(adapters),
            'n_adapters_changed': len(moved),
            'n_lora_A_changed': len(moved_A),
            'n_lora_B_changed': len(moved_B),
            'max_adapter_delta': max_delta,
            'frozen_tensors_changed': len(frozen_moved),
            'layerdrop': self.args.layerdrop,
        }
        with open(self.output_dir / "gradient_check.json", 'w') as f:
            json.dump(report, f, indent=2)
        self.logger.info(f"✓ Saved: {self.output_dir / 'gradient_check.json'}")

        return passed

    # --------------------------------------------------------------- training

    def _evaluate(self, dataset, criterion=None):
        self.model.eval()
        self.classifier.eval()

        preds, labels_all, losses = [], [], []
        with torch.no_grad():
            for audio_list, labels in dataset.batches(self.args.batch_size):
                labels = labels.to(self.device)
                logits = self._forward(audio_list)
                if criterion is not None:
                    losses.append(criterion(logits, labels).item())
                preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                labels_all.extend(labels.cpu().numpy().tolist())

        self.model.train()
        self.classifier.train()

        return {
            'accuracy': accuracy_score(labels_all, preds),
            'f1_macro': f1_score(labels_all, preds, average='macro', zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(labels_all, preds),
            'loss': float(np.mean(losses)) if losses else 0.0,
            'preds': preds,
            'labels': labels_all,
        }

    # ----------------------------------------------------------- checkpointing

    def _adapter_state(self):
        """Adapter + head weights only.

        The full PeftModel state_dict is ~318M params (~1.2 GB); the base
        encoder is frozen so only the adapters and head can change. Saving just
        those keeps per-epoch checkpoints at ~12 MB.
        """
        from peft import get_peft_model_state_dict
        return {
            'adapters': copy.deepcopy(get_peft_model_state_dict(self.model)),
            'classifier': copy.deepcopy(self.classifier.state_dict()),
        }

    def _load_adapter_state(self, state):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.model, state['adapters'])
        self.classifier.load_state_dict(state['classifier'])

    def save_adapter(self, adapter_dir, best_f1, best_epoch):
        """Export the adapted encoder in canonical PEFT format for downstream use.

        Written for the STAGED design: Phase C loads these adapters onto a fresh
        base encoder, keeps it frozen, and probes a different task (hyrax).

        Uses save_pretrained() rather than the checkpoint.pt route on purpose:

          - checkpoint.pt is ~50 MB and bundles optimizer + scheduler state; it
            exists for resume, and its layout is an internal detail of this
            script. adapter_model.safetensors is ~5 MB (HuBERT) and is the
            format PeftModel.from_pretrained() reads.
          - PeftModel.from_pretrained() returns a model with ZERO trainable
            parameters, so the frozen-encoder requirement holds by construction
            instead of depending on a caller remembering to freeze it.

        Call AFTER train(), which restores the best-epoch adapters, so what
        lands on disk is the best checkpoint and not the last epoch.
        """
        adapter_dir = Path(adapter_dir)
        adapter_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(adapter_dir))

        # The head is not used by Phase C (which trains its own probe), but it
        # is saved so this run is reproducible end to end.
        torch.save(self.classifier.state_dict(), adapter_dir / "classifier_head.pt")

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        meta = {
            'model': self.model_name,
            'base_model_id': self.base_model_id,
            'embedding_dim': self.embedding_dim,
            'num_transformer_layers': self.model.config.num_hidden_layers,
            # Phase C indexes output_hidden_states, which is num_layers + 1:
            # index 0 is the pre-transformer feature_projection output and
            # indices 1..N are the transformer blocks. Same convention as the
            # base-model layer sweeps, so best layers are directly comparable.
            'num_hidden_states': self.model.config.num_hidden_layers + 1,
            'num_attention_heads': self.model.config.num_attention_heads,
            'head_dim': self.embedding_dim // self.model.config.num_attention_heads,
            'task': self.manifest['task'],
            'manifest': str(self.args.manifest),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'excluded_species': self.manifest.get('excluded_species'),
            'best_epoch': best_epoch,
            'best_val_f1_macro': best_f1,
            'seed': self.args.seed,
            'data_fraction': self.args.data_fraction,
            'trainable_params_at_save': n_trainable,
            'lora': {'r': self.args.lora_r, 'alpha': self.args.lora_alpha,
                     'dropout': self.args.lora_dropout,
                     'target_modules': ["q_proj", "k_proj", "v_proj", "out_proj"],
                     'layerdrop': self.args.layerdrop},
            'config': vars(self.args),
        }
        if self.manifest.get('comparability_note'):
            meta['comparability_note'] = self.manifest['comparability_note']

        with open(adapter_dir / "adapter_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)

        self.logger.info(f"\n✓ Adapter saved for staged transfer: {adapter_dir}")
        self.logger.info(f"    files: {sorted(p.name for p in adapter_dir.iterdir())}")
        self.logger.info(f"    best epoch {best_epoch} (val macro-F1 {best_f1:.4f}) | "
                         f"{self.num_classes} classes | "
                         f"hidden_states index range 0..{self.model.config.num_hidden_layers}")

    def _save_checkpoint(self, path, epoch, optimizer, scheduler, history,
                         best_f1, best_epoch, best_state, patience_counter):
        tmp = Path(str(path) + ".tmp")
        torch.save({
            'epoch': epoch,
            'current_state': self._adapter_state(),
            'best_state': best_state,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'history': history,
            'best_f1': best_f1,
            'best_epoch': best_epoch,
            'patience_counter': patience_counter,
            'config': vars(self.args),
        }, tmp)
        tmp.replace(path)

    def train(self, datasets):
        train_ds, val_ds = datasets['train'], datasets['val']

        criterion = nn.CrossEntropyLoss(weight=self._class_weights(train_ds))
        optimizer = optim.AdamW([
            {'params': [p for p in self.model.parameters() if p.requires_grad],
             'lr': self.args.encoder_lr},
            {'params': self.classifier.parameters(), 'lr': self.args.head_lr},
        ])
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                      patience=self.args.plateau_patience)

        # --- resume if a checkpoint exists
        ckpt_path = self.output_dir / "checkpoint.pt"
        start_epoch = 0
        history = {'train_loss': [], 'train_acc': [], 'val_acc': [],
                   'val_f1_macro': [], 'val_loss': [], 'lr': []}
        best_f1, best_epoch, patience_counter = -1.0, -1, 0
        best_state = None

        if ckpt_path.exists() and not self.args.no_resume:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self._load_adapter_state(ckpt['current_state'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            history = ckpt['history']
            best_f1, best_epoch = ckpt['best_f1'], ckpt['best_epoch']
            best_state = ckpt['best_state']
            patience_counter = ckpt['patience_counter']
            start_epoch = ckpt['epoch']
            self.logger.info(f"\n✓ Resumed from {ckpt_path} at epoch {start_epoch} "
                             f"(best val macro-F1 {best_f1:.4f} @ epoch {best_epoch})")

        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING")
        self.logger.info(f"  adapters lr={self.args.encoder_lr} | head lr={self.args.head_lr}")
        self.logger.info(f"  ReduceLROnPlateau on val macro-F1 "
                         f"(factor 0.5, patience {self.args.plateau_patience})")
        self.logger.info(f"  batch={self.args.batch_size} | max epochs={self.args.max_epochs} "
                         f"| early-stop patience={self.args.patience}")
        self.logger.info("=" * 80)

        rng = np.random.default_rng(self.args.seed + start_epoch)

        self.model.train()
        self.classifier.train()

        for epoch in range(start_epoch, self.args.max_epochs):
            epoch_start = time.time()
            losses, correct, total = [], 0, 0
            n_batches = int(np.ceil(len(train_ds) / self.args.batch_size))

            for audio_list, labels in tqdm(
                train_ds.batches(self.args.batch_size, shuffle=True, rng=rng),
                total=n_batches, desc=f"Epoch {epoch+1}/{self.args.max_epochs}", leave=False
            ):
                labels = labels.to(self.device)
                logits = self._forward(audio_list)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.numel()

            train_seconds = time.time() - epoch_start
            val = self._evaluate(val_ds, criterion)
            epoch_seconds = time.time() - epoch_start
            train_acc = correct / max(total, 1)
            current_lr = optimizer.param_groups[0]['lr']
            history.setdefault('epoch_seconds', []).append(epoch_seconds)
            history.setdefault('train_seconds', []).append(train_seconds)

            history['train_loss'].append(float(np.mean(losses)))
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val['accuracy'])
            history['val_f1_macro'].append(val['f1_macro'])
            history['val_loss'].append(val['loss'])
            history['lr'].append(current_lr)

            scheduler.step(val['f1_macro'])

            self.logger.info(
                f"Epoch {epoch+1:3d} | train loss {history['train_loss'][-1]:.4f} "
                f"| train acc {train_acc:.4f} | val acc {val['accuracy']:.4f} "
                f"| val macro-F1 {val['f1_macro']:.4f} | lr {current_lr:.2e} "
                f"| {epoch_seconds/60:.1f} min "
                f"({train_seconds/max(n_batches,1)*1000:.0f} ms/batch)"
            )

            stop = False
            if val['f1_macro'] > best_f1:
                best_f1, best_epoch, patience_counter = val['f1_macro'], epoch + 1, 0
                # Fix #4: deep-copy, not a live reference
                best_state = self._adapter_state()
            else:
                patience_counter += 1
                if patience_counter >= self.args.patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    stop = True

            # Checkpoint every epoch so a job killed at the wall clock resumes here
            self._save_checkpoint(ckpt_path, epoch + 1, optimizer, scheduler,
                                  history, best_f1, best_epoch, best_state,
                                  patience_counter)
            if stop:
                break

        if best_state is not None:
            self._load_adapter_state(best_state)
            self.logger.info(f"\n✓ Restored best checkpoint "
                             f"(epoch {best_epoch}, val macro-F1 {best_f1:.4f})")

        return history, best_f1, best_epoch

    # ---------------------------------------------------------------- outputs

    def plot_curves(self, history):
        epochs = np.arange(1, len(history['train_acc']) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

        axes[0].plot(epochs, history['train_loss'], color="#0173B2", label="train")
        axes[0].plot(epochs, history['val_loss'], color="#DE8F05", label="val")
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

        axes[1].plot(epochs, history['train_acc'], color="#0173B2", label="train acc")
        axes[1].plot(epochs, history['val_acc'], color="#DE8F05", label="val acc")
        axes[1].axhline(1.0 / self.num_classes, color="grey", ls="--", lw=1.2)
        axes[1].text(0.995, 1.0 / self.num_classes + 0.01, "chance",
                     transform=axes[1].get_yaxis_transform(), ha="right",
                     va="bottom", fontsize=8, color="grey")
        axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylim(0, 1.05); axes[1].legend()

        axes[2].plot(epochs, history['val_f1_macro'], color="#029E73", label="val macro-F1")
        axes[2].axhline(self.args.baseline_f1, color="crimson", ls="--", lw=1.2)
        axes[2].text(0.995, self.args.baseline_f1 + 0.01,
                     f"zero-shot {self.args.baseline_f1:.3f}",
                     transform=axes[2].get_yaxis_transform(), ha="right",
                     va="bottom", fontsize=8, color="crimson")
        axes[2].set_title("Validation macro-F1"); axes[2].set_xlabel("Epoch")
        axes[2].set_ylim(0, 1.05); axes[2].legend()

        for ax in axes:
            ax.grid(alpha=0.3)

        fig.suptitle(f"LoRA fine-tuning: {self.model_name}, {self.num_classes}-class "
                     f"session-holdout", fontsize=13)
        fig.tight_layout()
        out = self.output_dir / "training_curves.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"✓ Curves saved: {out}")

    def full_metrics(self, result):
        y_true, y_pred = result['labels'], result['preds']
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        }


def main():
    parser = argparse.ArgumentParser(description="Phase 3 - LoRA fine-tuning")
    parser.add_argument("--model", default="xls_r", choices=["xls_r", "hubert_base"])
    parser.add_argument("--manifest", required=True, help="Path to the manifest JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--grad-check", action="store_true",
                        help="Run only the one-step encoder gradient check")
    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--layerdrop", type=float, default=0.0,
                        help="Encoder LayerDrop during training (XLS-R default is 0.1; "
                             "0.0 keeps every adapter receiving gradients each step)")
    # Head / optim
    parser.add_argument("--head-dropout", type=float, default=0.3)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--class-weights", default="window_inverse",
                        choices=["window_inverse", "manifest", "none"])
    parser.add_argument("--baseline-f1", type=float, default=0.1017,
                        help="Zero-shot macro-F1 to beat (drawn on the curve)")
    parser.add_argument("--seed", type=int, default=42)
    # Sweep / HPC
    parser.add_argument("--data-fraction", type=float, default=1.0,
                        help="Fraction of TRAINING windows to keep (stratified)")
    parser.add_argument("--cache-dir", default="outputs/phase3/window_cache",
                        help="Where windowed audio caches live (shared across runs)")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--build-cache-only", action="store_true",
                        help="Materialise window caches and exit (run once before an array)")
    parser.add_argument("--max-windows-per-file", type=int, default=None,
                        help="Cap windows per source file (use for species_id, which has "
                             "~14.6k files)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore an existing checkpoint and start fresh")
    # Staged transfer (Phase B/C)
    parser.add_argument("--save-adapter-dir", default=None,
                        help="Export the best-epoch adapters in canonical PEFT format "
                             "(adapter_config.json + adapter_model.safetensors + "
                             "adapter_meta.json) for a later frozen probe. Loadable "
                             "with PeftModel.from_pretrained(), which yields zero "
                             "trainable params.")
    parser.add_argument("--log-tag", default=None,
                        help="Suffix for the log filename. WITHOUT this, every run of "
                             "a given model appends to the same "
                             "lora_fine_tune_<model>_run.log, interleaving unrelated "
                             "runs (the log handler opens in append mode).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log_dir = Path("outputs/phase3/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = "gradcheck" if args.grad_check else "run"
    if args.log_tag:
        tag = f"{tag}_{args.log_tag}"
    logger = setup_logger(f"Phase3_LoRA_{args.model}_{tag}",
                          log_file=str(log_dir / f"lora_fine_tune_{args.model}_{tag}.log"))

    logger.info("=" * 80)
    logger.info("PHASE 3 - LoRA FINE-TUNING")
    logger.info("=" * 80)

    if args.build_cache_only:
        # Cache building needs no model; skip the 300M download/load.
        tuner = LoRAFineTuner.__new__(LoRAFineTuner)
        tuner.args, tuner.logger = args, logger
        tuner.output_dir = Path(args.output_dir)
        tuner.output_dir.mkdir(parents=True, exist_ok=True)
        with open(args.manifest) as f:
            tuner.manifest = json.load(f)
        if 'species_to_idx' in tuner.manifest:
            tuner.class_to_idx, tuner.label_key = tuner.manifest['species_to_idx'], 'species'
        else:
            tuner.class_to_idx, tuner.label_key = tuner.manifest['class_to_idx'], 'individual'
        tuner.load_splits(build_only=True)
        logger.info("\n✓ Window caches built.")
        return 0

    tuner = LoRAFineTuner(args.model, args.manifest, args.output_dir, logger, args)
    datasets = tuner.load_splits()

    if args.grad_check:
        passed = tuner.gradient_check(datasets['train'])
        return 0 if passed else 1

    history, best_f1, best_epoch = tuner.train(datasets)
    tuner.plot_curves(history)

    # train() has restored the best-epoch adapters, so this exports the best
    # checkpoint rather than the final epoch.
    if args.save_adapter_dir:
        tuner.save_adapter(args.save_adapter_dir, best_f1, best_epoch)

    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 80)

    results = {'model': args.model, 'task': tuner.manifest['task'],
               'num_classes': tuner.num_classes, 'class_names': tuner.class_names,
               'config': vars(args), 'history': history,
               'best_val_f1_macro': best_f1, 'best_epoch': best_epoch}

    for split in ['val', 'test']:
        if split not in datasets:
            continue
        res = tuner._evaluate(datasets[split])
        metrics = tuner.full_metrics(res)
        results[f'{split}_metrics'] = metrics
        logger.info(f"\n{split.upper()}: acc={metrics['accuracy']:.4f} "
                    f"macro-F1={metrics['f1_macro']:.4f} "
                    f"balanced-acc={metrics['balanced_accuracy']:.4f}")
        if split == 'test':
            cm = confusion_matrix(res['labels'], res['preds'])
            results['test_confusion_matrix'] = cm.tolist()
            results['test_per_class'] = classification_report(
                res['labels'], res['preds'], labels=list(range(tuner.num_classes)),
                target_names=tuner.class_names, output_dict=True, zero_division=0)

    out_file = Path(args.output_dir) / "lora_fine_tuning_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Results saved: {out_file}")

    baseline = args.baseline_f1
    final_f1 = results.get('test_metrics', {}).get('f1_macro', 0.0)
    logger.info(f"\nZero-shot macro-F1 baseline: {baseline:.4f}")
    logger.info(f"Fine-tuned test macro-F1:    {final_f1:.4f} ({final_f1 - baseline:+.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
