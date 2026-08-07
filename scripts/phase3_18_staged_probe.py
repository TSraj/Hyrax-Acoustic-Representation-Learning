#!/usr/bin/env python3
"""
Phase C / Step C1: frozen linear probe on hyrax, swept over every layer and two
pooling variants, for BOTH the un-adapted base encoder and the species-adapted
encoder.

WHY THE BASE SWEEP EXISTS
-------------------------
The previously reported frozen numbers (HuBERT 0.1735, XLS-R 0.1017) come from
phase3_03_zero_shot_evaluation.py, which uses outputs.last_hidden_state - the
FINAL layer only. They are not best-layer numbers, and no base-frozen layer
sweep existed for this task.

Comparing an adapted best-of-13-or-25 against a base final-layer number would
hand the adapted encoder a max-over-layers selection advantage the base never
had, manufacturing part of the improvement. So this script runs the IDENTICAL
sweep on both encoders: pass --adapter-dir for the adapted one, omit it for the
base. Same code path, same probe, same windows - the only difference is whether
the LoRA deltas are present.

SELECTION IS ON VAL, NEVER TEST
-------------------------------
The sweep produces 26 cells for HuBERT (13 layers x 2 variants) and 50 for
XLS-R. Choosing the best by TEST score across that many candidates would be
badly optimistic on 409 test windows over 8 classes. The best cell is therefore
chosen by VAL macro-F1 and its TEST score is reported. Every cell's test score
is still written out, but the headline number is the val-selected one.

LAYER AXIS - the two variants differ, deliberately
--------------------------------------------------
  mean  : layers 0..N over output_hidden_states. Index 0 is the pre-transformer
          feature_projection output; 1..N are the transformer blocks. N+1 cells.
  head0 : layers 1..N only. Head 0's context vector is a property of a
          transformer block's attention, so it does not exist for index 0.
          N cells.

Both index the SAME block for a given layer number >= 1, so "layer 5" means the
same thing in both variants. head0 simply has no layer 0.

PROBE RECIPE, AND WHY IT DOES NOT MATCH THE PUBLISHED NUMBERS
------------------------------------------------------------
nn.Linear, Adam lr 1e-3, FULL-BATCH gradient descent (one step per epoch, as in
phase3_03), CrossEntropyLoss weighted by the manifest's class_weights (all 1.0
on this task, i.e. uniform), evaluated window-level with no file aggregation.

The one deliberate departure from phase3_03 is how long the probe trains, and
it matters enormously. phase3_03's published run took its NO-VAL branch: 50
epochs, no early stopping, final state. With full-batch GD one epoch is ONE
gradient step, so that probe never came close to fitting.

Measured on frozen HuBERT features, final layer, mean pooled:

    probe training                          train F1   test F1
    50 steps, final state (phase3_03)        0.2350     0.1590   <- published 0.1735
    converged (median 860 steps)             0.9245     0.3280

The 50-step replication lands on the published number (0.1590 vs 0.1735, the
residual explained by the smaller train split below). So the published frozen
baselines are UNDERTRAINED-PROBE ARTEFACTS, not measurements of what a linear
probe can read off frozen features. A converged probe roughly doubles them.

Phase C therefore defaults to a CONVERGED probe (--probe-max-epochs 2000,
--probe-patience 300, selection on val). An undertrained probe would also be
the wrong instrument here: pinned near chance, it has little power to resolve
any base-vs-adapted difference, which is the entire point of the phase.

--probe-select final --probe-max-epochs 50 reproduces the old behaviour and is
kept as a diagnostic.

Consequence worth carrying into the writeup: the frozen-to-fine-tuned gap for
HuBERT shrinks from 0.1735 -> 0.4066 (+0.233) to roughly 0.328 -> 0.4066
(+0.079). Same caveat likely applies to every other phase3_03 zero-shot number,
including the 7-way species baselines, since they share this probe.

Remaining difference from the published setup, unrelated to training length:
train size. The published number used the plain session_holdout manifest (~1353
train windows); this uses the _ft manifest (1011), because only _ft has the val
split that val-based selection requires. The TEST split is identical between
them (409 windows, same held-out sessions).

Windows come from the cache the LoRA runs trained on, so they are byte-identical
(verified: keys 7ef9e0a1822d / a1741a737762 / 9a2edf1d1b2e).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger

WINDOW_SECONDS = 5.0
STRIDE_SECONDS = 2.5
SAMPLE_RATE = 16000

MODEL_IDS = {
    'hubert_base': 'facebook/hubert-base-ls960',
    'xls_r': 'facebook/wav2vec2-xls-r-300m',
}
# Reference points, carried into the output so they are never mis-cited.
PUBLISHED_FINAL_LAYER = {'hubert_base': 0.1735, 'xls_r': 0.1017}
HYRAX_FINETUNED_CEILING = {'hubert_base': 0.4066, 'xls_r': 0.3167}


def cache_key(items, label_key, max_windows_per_file):
    """Reproduce WindowedDataset._cache_key from phase3_10_lora_fine_tuning.py."""
    import hashlib
    h = hashlib.md5()
    h.update(f"{WINDOW_SECONDS}|{STRIDE_SECONDS}|{label_key}|{max_windows_per_file}".encode())
    for it in items:
        h.update(str(it['file']).encode())
    return h.hexdigest()[:12]


def load_windows(manifest, split, cache_dir, label_key, mwpf, logger):
    """Load the cached float16 windows + labels for a split.

    Deliberately cache-only: silently re-decoding audio here would risk
    producing windows that differ from the ones the adapters were trained and
    evaluated on, which would break the comparison this script exists to make.
    """
    key = cache_key(manifest['splits'][split], label_key, mwpf)
    win_f = Path(cache_dir) / f"{split}_{key}_windows.npy"
    lab_f = Path(cache_dir) / f"{split}_{key}_labels.npy"
    if not (win_f.exists() and lab_f.exists()):
        raise FileNotFoundError(
            f"window cache missing for split '{split}': {win_f.name}\n"
            f"Expected key {key} in {cache_dir}. Build it with "
            f"phase3_10_lora_fine_tuning.py --build-cache-only using the same "
            f"manifest, label key '{label_key}' and max_windows_per_file={mwpf}."
        )
    windows = np.load(win_f, mmap_mode='r')
    labels = np.load(lab_f)
    logger.info(f"  {split}: {len(labels)} windows (cache {key}) | "
                f"per-class {np.bincount(labels, minlength=manifest['num_classes']).tolist()}")
    return windows, labels


class Encoder:
    """Frozen encoder exposing per-layer mean-pooled and head-0 embeddings."""

    def __init__(self, model_name, adapter_dir, logger):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, HubertModel

        self.model_name = model_name
        self.logger = logger
        self.adapted = adapter_dir is not None
        model_id = MODEL_IDS[model_name]

        cls = HubertModel if model_name == 'hubert_base' else Wav2Vec2Model
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        base = cls.from_pretrained(model_id, use_safetensors=True)
        # Match training and make eval deterministic layer-for-layer.
        base.config.layerdrop = 0.0

        if self.adapted:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, str(adapter_dir))
            n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            if n_trainable != 0:
                raise RuntimeError(
                    f"adapted encoder has {n_trainable:,} trainable params, expected 0")
            self.encoder_module = self.model.base_model.model.encoder
            with open(Path(adapter_dir) / "adapter_meta.json") as f:
                self.adapter_meta = json.load(f)
        else:
            self.model = base
            for p in self.model.parameters():
                p.requires_grad = False
            self.encoder_module = self.model.encoder
            self.adapter_meta = None

        self.config = self.model.config
        self.n_layers = self.config.num_hidden_layers
        self.dim = self.config.hidden_size
        self.head_dim = self.dim // self.config.num_attention_heads

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        self.model.to(self.device).eval()

        logger.info(f"Encoder: {model_name} | {'ADAPTED' if self.adapted else 'BASE (un-adapted)'}")
        logger.info(f"  layers {self.n_layers} | hidden_states 0..{self.n_layers} "
                    f"| dim {self.dim} | head_dim {self.head_dim} | device {self.device}")

    def assert_layer0_matches_base(self, sample_audio):
        """Layer 0 must be bit-identical to the un-adapted base encoder.

        LoRA sits on the attention projections, which hidden_states[0]
        (feature_projection output) precedes, so nothing can change it. A
        mismatch means the wrong base checkpoint loaded, or something is
        adapting the CNN / feature-projection path - either would invalidate
        every layer comparison this script makes. Kept as a hard assertion
        rather than a warning for exactly that reason.
        """
        if not self.adapted:
            return None
        from transformers import Wav2Vec2Model, HubertModel
        cls = HubertModel if self.model_name == 'hubert_base' else Wav2Vec2Model
        ref = cls.from_pretrained(MODEL_IDS[self.model_name], use_safetensors=True)
        ref.config.layerdrop = 0.0
        ref.to(self.device).eval()

        inputs = self._prep(sample_audio)
        with torch.no_grad():
            a = self.model(**inputs, output_hidden_states=True).hidden_states
            b = ref(**inputs, output_hidden_states=True).hidden_states

        deltas = [(x - y).abs().max().item() for x, y in zip(a, b)]
        del ref
        if deltas[0] != 0.0:
            raise RuntimeError(
                f"LAYER-0 EQUALITY FAILED: max|delta| = {deltas[0]:.3e}, must be 0. "
                f"Wrong base checkpoint, or the feature-projection path is being "
                f"adapted. Every layer comparison would be invalid.")
        zero_drift = [i for i, d in enumerate(deltas) if i > 0 and d == 0.0]
        if zero_drift:
            raise RuntimeError(
                f"layers {zero_drift} identical to base - adapter load was a no-op")
        self.logger.info(f"  layer-0 equality OK (delta 0.0); "
                         f"transformer drift {min(deltas[1:]):.3f}..{max(deltas[1:]):.3f}")
        return deltas

    def _prep(self, audio_list):
        inputs = self.feature_extractor(
            audio_list, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def extract(self, windows, batch_size, logger, desc=""):
        """One forward pass per batch yielding BOTH variants for ALL layers.

        Returns:
            mean  : {layer 0..N -> [n, dim]}
            head0 : {layer 1..N -> [n, head_dim]}

        The head-0 hook slices to head_dim INSIDE the hook, so the full
        [B, T, hidden] activation for every layer is never accumulated - only
        the 64-wide slice survives the forward pass.
        """
        n = len(windows)
        mean_out = {i: np.zeros((n, self.dim), dtype=np.float32)
                    for i in range(self.n_layers + 1)}
        head_out = {i: np.zeros((n, self.head_dim), dtype=np.float32)
                    for i in range(1, self.n_layers + 1)}

        captured = {}

        def make_hook(block_idx):
            def hook(module, args, kwargs, output):
                tensor = args[0] if args else kwargs.get('hidden_states')
                # Heads are concatenated at out_proj's INPUT; head h occupies
                # dims [h*head_dim : (h+1)*head_dim]. After out_proj they are
                # mixed and no slice corresponds to a single head.
                captured[block_idx] = tensor[..., 0:self.head_dim].mean(dim=1).detach()
            return hook

        handles = [
            self.encoder_module.layers[i].attention.out_proj.register_forward_hook(
                make_hook(i + 1), with_kwargs=True)   # +1 to align with hidden_states
            for i in range(self.n_layers)
        ]

        t0 = time.time()
        try:
            with torch.no_grad():
                for start in range(0, n, batch_size):
                    sel = slice(start, min(start + batch_size, n))
                    audio = [np.asarray(w, dtype=np.float32) for w in windows[sel]]
                    captured.clear()
                    out = self.model(**self._prep(audio), output_hidden_states=True)

                    for layer, hs in enumerate(out.hidden_states):
                        mean_out[layer][sel] = hs.mean(dim=1).cpu().numpy()
                    for layer, vec in captured.items():
                        head_out[layer][sel] = vec.cpu().numpy()
        finally:
            for h in handles:
                h.remove()

        if len(captured) != self.n_layers:
            raise RuntimeError(f"head-0 hooks fired on {len(captured)} of "
                               f"{self.n_layers} layers")

        logger.info(f"  {desc}: extracted {n} windows x "
                    f"({self.n_layers + 1} mean + {self.n_layers} head0) layers "
                    f"in {time.time() - t0:.1f}s")
        return mean_out, head_out


def train_probe(train_X, train_y, val_X, val_y, num_classes, class_weights,
                device, lr, max_epochs, patience, select='val'):
    """Linear probe. Architecture, loss and optimizer follow phase3_03.

    Full-batch gradient descent - one optimizer step per epoch on the whole
    training set, exactly as phase3_03 does. Not minibatched.

    select:
      'val'   keep the best-by-val-accuracy state, early stopping with patience.
              This is the mode Phase C uses, with max_epochs large enough for
              the probe to actually converge.
      'final' keep the state after max_epochs, ignoring val entirely. This
              replicates phase3_03's NO-VAL branch, which is what produced the
              published 0.1735 / 0.1017. Kept as a diagnostic: at
              max_epochs=50 it reproduces those numbers and demonstrates that
              they are undertrained-probe artifacts, not converged results.
    """
    torch.manual_seed(42)
    clf = nn.Linear(train_X.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(clf.parameters(), lr=lr)

    tX = torch.FloatTensor(train_X).to(device)
    ty = torch.LongTensor(train_y).to(device)
    vX = torch.FloatTensor(val_X).to(device)
    vy = torch.LongTensor(val_y).to(device)

    best_acc, best_state, bad = -1.0, {k: v.clone() for k, v in clf.state_dict().items()}, 0
    best_epoch = 0
    for epoch in range(max_epochs):
        clf.train()
        loss = criterion(clf(tX), ty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if select == 'final':
            continue  # phase3_03 no-val branch: no selection, take the last state

        clf.eval()
        with torch.no_grad():
            acc = (clf(vX).argmax(dim=1) == vy).float().mean().item()
        if acc > best_acc:
            best_acc, best_epoch, bad = acc, epoch + 1, 0
            best_state = {k: v.clone() for k, v in clf.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if select == 'final':
        return clf, max_epochs

    clf.load_state_dict(best_state)
    return clf, best_epoch


def score(clf, X, y, device):
    clf.eval()
    with torch.no_grad():
        preds = clf(torch.FloatTensor(X).to(device)).argmax(dim=1).cpu().numpy()
    return {
        'f1_macro': float(f1_score(y, preds, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y, preds)),
        'accuracy': float(accuracy_score(y, preds)),
    }, preds


def main():
    p = argparse.ArgumentParser(description="Phase C - staged frozen probe on hyrax")
    p.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    p.add_argument("--adapter-dir", default=None,
                   help="Adapted encoder. OMIT for the un-adapted base sweep.")
    p.add_argument("--manifest",
                   default="outputs/phase3/denoiser_screen/manifests/bioda/"
                           "hyrax_id_session_holdout_ft.json",
                   help="_ft manifest: the only session-holdout variant with a val "
                        "split, which val-based selection requires. Its test split "
                        "is identical to the plain manifest's (409 windows).")
    p.add_argument("--cache-dir", default="outputs/phase3/window_cache")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    # Probe recipe - phase3_03 values.
    p.add_argument("--probe-lr", type=float, default=1e-3)
    p.add_argument("--probe-max-epochs", type=int, default=2000)
    p.add_argument("--probe-patience", type=int, default=300)
    p.add_argument("--probe-select", default="val", choices=["val", "final"],
                   help="'val' = keep best-by-val state (Phase C default). "
                        "'final' = keep the state after --probe-max-epochs, "
                        "replicating phase3_03's no-val branch; use with "
                        "--probe-max-epochs 50 to reproduce the published "
                        "undertrained numbers as a diagnostic.")
    args = p.parse_args()

    condition = "adapted" if args.adapter_dir else "base"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path("outputs/phase3/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        f"Phase3_StagedProbe_{args.model}_{condition}",
        log_file=str(log_dir / f"staged_probe_{args.model}_{condition}.log"))

    logger.info("=" * 80)
    logger.info(f"PHASE C - STAGED FROZEN PROBE | {args.model} | {condition.upper()}")
    logger.info("=" * 80)

    with open(args.manifest) as f:
        manifest = json.load(f)
    num_classes = manifest['num_classes']
    class_names = sorted(manifest['class_to_idx'], key=manifest['class_to_idx'].get)
    label_key = 'individual'

    logger.info(f"Manifest: {args.manifest}")
    logger.info(f"  task {manifest['task']} | {num_classes} classes {class_names}")

    # phase3_03 weights the probe loss by the manifest's class_weights, which are
    # file-level and all 1.0 on this task (one file per individual per split) -
    # i.e. uniform. Held identical across base and adapted.
    weights = torch.FloatTensor([manifest['class_weights'][c] for c in class_names])
    logger.info(f"  probe class weights (phase3_03 style): "
                f"{[round(w, 3) for w in weights.tolist()]}")

    logger.info("\nWindows (cache-only; identical to what the LoRA runs used):")
    data = {s: load_windows(manifest, s, args.cache_dir, label_key, None, logger)
            for s in ('train', 'val', 'test')}

    enc = Encoder(args.model, args.adapter_dir, logger)
    sample = [np.asarray(data['train'][0][0], dtype=np.float32)]
    layer_deltas = enc.assert_layer0_matches_base(sample)

    logger.info("\nExtracting embeddings (one forward pass per split, all layers):")
    feats = {}
    for split in ('train', 'val', 'test'):
        windows, labels = data[split]
        mean_f, head_f = enc.extract(windows, args.batch_size, logger, desc=split)
        feats[split] = {'mean': mean_f, 'head0': head_f, 'y': labels}

    # ------------------------------------------------------------------ sweep
    logger.info("\n" + "=" * 80)
    logger.info("PROBE SWEEP (best cell selected on VAL macro-F1, never test)")
    logger.info("=" * 80)

    rows = []
    for variant in ('mean', 'head0'):
        # mean spans 0..N (index 0 = pre-transformer); head0 spans 1..N, since
        # head 0's context vector is a property of a transformer block.
        layers = sorted(feats['train'][variant].keys())
        for layer in layers:
            clf, ep = train_probe(
                feats['train'][variant][layer], feats['train']['y'],
                feats['val'][variant][layer], feats['val']['y'],
                num_classes, weights, enc.device,
                args.probe_lr, args.probe_max_epochs, args.probe_patience,
                select=args.probe_select)

            rec = {'model': args.model, 'condition': condition,
                   'variant': variant, 'layer': layer,
                   'dim': feats['train'][variant][layer].shape[1],
                   'probe_best_epoch': ep}
            for split in ('train', 'val', 'test'):
                m, preds = score(clf, feats[split][variant][layer],
                                 feats[split]['y'], enc.device)
                for k, v in m.items():
                    rec[f'{split}_{k}'] = v
                if split == 'test':
                    rec['_test_preds'] = preds.tolist()
            rows.append(rec)
            logger.info(f"  {variant:5s} L{layer:2d} (d={rec['dim']:4d}) | "
                        f"val F1 {rec['val_f1_macro']:.4f} | "
                        f"test F1 {rec['test_f1_macro']:.4f} | "
                        f"test bal-acc {rec['test_balanced_accuracy']:.4f} | "
                        f"test acc {rec['test_accuracy']:.4f}")

    # -------------------------------------------------------------- selection
    best = {}
    for variant in ('mean', 'head0'):
        sub = [r for r in rows if r['variant'] == variant]
        pick = max(sub, key=lambda r: r['val_f1_macro'])
        best[variant] = {k: v for k, v in pick.items() if not k.startswith('_')}
    overall = max(best.values(), key=lambda r: r['val_f1_macro'])

    final_layer_mean = next(r for r in rows
                            if r['variant'] == 'mean' and r['layer'] == enc.n_layers)

    logger.info("\n" + "=" * 80)
    logger.info("SELECTED CELLS (by val macro-F1)")
    logger.info("=" * 80)
    for variant, b in best.items():
        logger.info(f"  {variant:5s}: L{b['layer']} -> test macro-F1 "
                    f"{b['test_f1_macro']:.4f} | bal-acc "
                    f"{b['test_balanced_accuracy']:.4f} | acc {b['test_accuracy']:.4f} "
                    f"(val {b['val_f1_macro']:.4f})")
    logger.info(f"  overall: {overall['variant']} L{overall['layer']} -> "
                f"test macro-F1 {overall['test_f1_macro']:.4f}")
    logger.info(f"\n  final-layer + mean cell (comparable to the published "
                f"phase3_03 number): test macro-F1 "
                f"{final_layer_mean['test_f1_macro']:.4f}")
    logger.info(f"    published final-layer for {args.model}: "
                f"{PUBLISHED_FINAL_LAYER[args.model]:.4f} - do NOT expect a match. "
                f"That number came from a 50-step full-batch probe that never fit "
                f"(train macro-F1 ~0.24); a converged probe roughly doubles it. "
                f"See the module docstring.")

    results = {
        'model': args.model,
        'condition': condition,
        'adapter_dir': str(args.adapter_dir) if args.adapter_dir else None,
        'adapter_meta': enc.adapter_meta,
        'manifest': args.manifest,
        'task': manifest['task'],
        'num_classes': num_classes,
        'class_names': class_names,
        'n_transformer_layers': enc.n_layers,
        'embedding_dim': enc.dim,
        'head_dim': enc.head_dim,
        'split_windows': {s: int(len(feats[s]['y'])) for s in feats},
        'probe_recipe': {
            'source': 'replicates phase3_03_zero_shot_evaluation.train_classifier',
            'type': 'nn.Linear', 'optimizer': 'Adam', 'lr': args.probe_lr,
            'batching': 'full-batch (one step per epoch)',
            'max_epochs': args.probe_max_epochs,
            'selection': args.probe_select,
            'early_stopping': (f'val accuracy, patience {args.probe_patience}'
                               if args.probe_select == 'val' else 'none (final state)'),
            'class_weights': 'manifest class_weights (all 1.0 = uniform)',
            'evaluation': 'window-level, no file aggregation',
        },
        'layer_axis': {
            'mean': f'0..{enc.n_layers}; index 0 is the pre-transformer '
                    f'feature_projection output, 1..N are transformer blocks',
            'head0': f'1..{enc.n_layers}; head 0 has no layer-0 counterpart '
                     f'because it is a property of a transformer block',
        },
        'selection': 'best cell chosen by VAL macro-F1; test never used to select',
        'best_by_variant': best,
        'best_overall': {k: v for k, v in overall.items() if not k.startswith('_')},
        'final_layer_mean_cell': {k: v for k, v in final_layer_mean.items()
                                  if not k.startswith('_')},
        'reference_published_final_layer': PUBLISHED_FINAL_LAYER[args.model],
        'reference_hyrax_finetuned_ceiling': HYRAX_FINETUNED_CEILING[args.model],
        'reference_note': (
            'published_final_layer (phase3_03) is an UNDERTRAINED-PROBE ARTEFACT, '
            'not a valid frozen baseline: its 50 full-batch GD steps leave the '
            'probe unfit (train macro-F1 ~0.24). Replicating that recipe here '
            'gives 0.1590 vs the published 0.1735, while a converged probe on the '
            'same features gives 0.3280. Do NOT use published_final_layer as the '
            'base baseline - use this run\'s own base sweep. '
            'hyrax_finetuned_ceiling comes from LoRA fine-tuning ON hyrax '
            '(minibatch, window_inverse class weights, early stop on val '
            'macro-F1) - a loose ceiling, not a like-for-like probe result.'),
        'layer_deltas_vs_base': layer_deltas,
        'cells': [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows],
    }

    with open(out_dir / "staged_probe_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    import csv
    with open(out_dir / "staged_probe_cells.csv", 'w', newline='') as f:
        cols = [k for k in rows[0] if not k.startswith('_')]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    cm = confusion_matrix(feats['test']['y'],
                          next(r for r in rows
                               if r['variant'] == overall['variant']
                               and r['layer'] == overall['layer'])['_test_preds'],
                          labels=list(range(num_classes)))
    with open(out_dir / "best_cell_test_confusion_matrix.json", 'w') as f:
        json.dump({'variant': overall['variant'], 'layer': overall['layer'],
                   'class_names': class_names, 'matrix': cm.tolist()}, f, indent=2)

    logger.info(f"\n✓ Results: {out_dir / 'staged_probe_results.json'}")
    logger.info(f"✓ Per-cell CSV: {out_dir / 'staged_probe_cells.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
