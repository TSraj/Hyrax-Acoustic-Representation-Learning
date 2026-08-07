#!/usr/bin/env python3
"""
Audit: how much of each published phase3_03 zero-shot number is probe
undertraining?

BACKGROUND
----------
phase3_03_zero_shot_evaluation.train_classifier trains its linear probe with
FULL-BATCH gradient descent - one optimizer step per "epoch". Its no-val branch
runs 50 epochs with no early stopping and keeps the final state, i.e. 50
gradient steps total. On the hyrax tasks that leaves the probe unfit: the
published runs report TRAIN macro-F1 of 0.08-0.53 on an 8-class task whose
chance level is 0.125, and three of six models sit at or below chance ON THEIR
OWN TRAINING DATA.

A converged probe on the same frozen HuBERT features reaches train macro-F1
0.92 and test 0.3280, against a published 0.1735 (measured in Phase C).

This script isolates the probe-training variable. It re-extracts features
EXACTLY as phase3_03 does - same 5s/2.5s windowing, same final layer, same mean
pooling, same manifest class weights - and then trains the probe at a range of
step counts, so the trajectory from "published" to "converged" is visible.
Nothing about the features changes; only how long the probe trains.

TWO NUMBERS PER CELL
--------------------
  trajectory   test macro-F1 at fixed step counts, final state, no early
               stopping. Step 50 reproduces phase3_03's no-val branch.
  internal-val a principled corrected number: TRAIN is split 80/20
               (stratified, seeded), the 20% is used only for early stopping,
               and TEST is reported. Needed because these manifests have no val
               split, so there is otherwise no honest place to stop. TEST is
               never used for stopping or selection.

Reporting both matters: a linear probe on ~1300 samples in 1024 dims can overfit,
so "more steps" is not automatically better. The trajectory shows whether the
published number is below the peak, and the internal-val number is what should
replace it.

Read-only with respect to published results: writes only into --output-dir.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
from src.utils.audio_utils import load_audio

WINDOW_SECONDS = 5.0
STRIDE_SECONDS = 2.5
SAMPLE_RATE = 16000

MODEL_IDS = {
    'wav2vec2_base': 'facebook/wav2vec2-base',
    'wav2vec2_base_960h': 'facebook/wav2vec2-base-960h',
    'hubert_base': 'facebook/hubert-base-ls960',
    'xls_r': 'facebook/wav2vec2-xls-r-300m',
    'wavlm': 'microsoft/wavlm-base-plus',
    'ecapa_tdnn': 'speechbrain/spkrec-ecapa-voxceleb',
}
DEFAULT_STEPS = [50, 100, 200, 500, 1000, 2000, 5000]


def resolve(fp):
    p = Path(fp)
    if not p.exists() and not str(fp).startswith('outputs/'):
        p = Path("Data") / fp
    return p


class Extractor:
    """Final-layer mean-pooled embeddings, matching phase3_03 exactly."""

    def __init__(self, model_name, logger):
        self.model_name = model_name
        self.logger = logger
        self.is_ecapa = model_name == 'ecapa_tdnn'

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        model_id = MODEL_IDS[model_name]
        if self.is_ecapa:
            from speechbrain.inference.speaker import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(
                source=model_id, savedir="pretrained_models/ecapa_tdnn")
            self.feature_extractor = None
        else:
            from transformers import (Wav2Vec2FeatureExtractor, Wav2Vec2Model,
                                      HubertModel, WavLMModel)
            cls = {'hubert_base': HubertModel, 'wavlm': WavLMModel}.get(
                model_name, Wav2Vec2Model)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.model = cls.from_pretrained(model_id, use_safetensors=True)
            self.model.config.layerdrop = 0.0
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.to(self.device).eval()

        logger.info(f"Extractor: {model_name} ({model_id}) on {self.device}")

    def embed(self, audio):
        """One window -> one embedding. Mirrors phase3_03.extract_embedding_from_audio."""
        if self.is_ecapa:
            with torch.no_grad():
                e = self.model.encode_batch(torch.FloatTensor(audio).unsqueeze(0))
                e = e.squeeze(0)
                if e.dim() > 1:
                    e = e.mean(dim=0)
            return e.cpu().numpy()

        inputs = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE,
                                        return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
            emb = out.last_hidden_state.mean(dim=1).squeeze(0)
        return emb.cpu().numpy()

    def extract_split(self, items, class_to_idx, label_key, logger, split):
        """Window on the fly at 5s/2.5s, exactly as phase3_03 does for hyrax."""
        window = int(WINDOW_SECONDS * SAMPLE_RATE)
        stride = int(STRIDE_SECONDS * SAMPLE_RATE)
        embs, labels, failed = [], [], 0
        t0 = time.time()

        for item in tqdm(items, desc=f"{split}", leave=False):
            try:
                audio, _ = load_audio(str(resolve(item['file'])),
                                      target_sr=SAMPLE_RATE, mono=True)
            except Exception:
                failed += 1
                continue
            label = class_to_idx[item[label_key]]
            n = 0
            for start in range(0, len(audio) - window + 1, stride):
                embs.append(self.embed(audio[start:start + window]))
                labels.append(label)
                n += 1
            if n == 0 and len(audio) > 0:
                embs.append(self.embed(audio))
                labels.append(label)

        if failed:
            logger.warning(f"  {split}: {failed} files failed to load")
        X = np.array(embs, dtype=np.float32)
        y = np.array(labels)
        logger.info(f"  {split}: {len(y)} windows, dim {X.shape[1]} "
                    f"({time.time() - t0:.0f}s)")
        return X, y


def fit_probe(train_X, train_y, num_classes, weights, device, steps,
              val_X=None, val_y=None, patience=None, lr=1e-3, seed=42):
    """Full-batch GD linear probe, as in phase3_03.

    val_X given  -> early stop on val accuracy, keep best state.
    val_X absent -> run `steps` steps and keep the FINAL state (phase3_03's
                    no-val branch).
    """
    torch.manual_seed(seed)
    clf = nn.Linear(train_X.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = optim.Adam(clf.parameters(), lr=lr)

    tX = torch.FloatTensor(train_X).to(device)
    ty = torch.LongTensor(train_y).to(device)
    use_val = val_X is not None
    if use_val:
        vX = torch.FloatTensor(val_X).to(device)
        vy = torch.LongTensor(val_y).to(device)
        best_acc, best_state, bad, best_step = -1.0, None, 0, 0

    for step in range(steps):
        clf.train()
        loss = criterion(clf(tX), ty)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if use_val:
            clf.eval()
            with torch.no_grad():
                acc = (clf(vX).argmax(dim=1) == vy).float().mean().item()
            if acc > best_acc:
                best_acc, best_step, bad = acc, step + 1, 0
                best_state = {k: v.clone() for k, v in clf.state_dict().items()}
            else:
                bad += 1
                if patience and bad >= patience:
                    break

    if use_val and best_state is not None:
        clf.load_state_dict(best_state)
        return clf, best_step
    return clf, steps


def evaluate(clf, X, y, device):
    clf.eval()
    with torch.no_grad():
        pred = clf(torch.FloatTensor(X).to(device)).argmax(dim=1).cpu().numpy()
    return {
        'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'accuracy': float(accuracy_score(y, pred)),
    }


def stratified_split(y, frac, seed):
    rng = np.random.default_rng(seed)
    keep, held = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_held = max(1, int(round(len(idx) * frac)))
        held.append(idx[:n_held])
        keep.append(idx[n_held:])
    return np.sort(np.concatenate(keep)), np.sort(np.concatenate(held))


def main():
    p = argparse.ArgumentParser(description="Audit phase3_03 probe undertraining")
    p.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    p.add_argument("--manifest", required=True)
    p.add_argument("--label-key", default="individual",
                   choices=["individual", "species"])
    p.add_argument("--published-f1", type=float, default=None,
                   help="The number currently in the paper, for side-by-side")
    p.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    p.add_argument("--internal-val-frac", type=float, default=0.2)
    p.add_argument("--internal-val-patience", type=int, default=300)
    p.add_argument("--internal-val-max-steps", type=int, default=5000)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--emb-cache", default=None,
                   help="Directory to cache extracted embeddings so re-probing "
                        "does not re-extract")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("outputs/phase3/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    logger = setup_logger(f"ProbeAudit_{args.model}{suffix}",
                          log_file=str(log_dir / f"probe_audit_{args.model}{suffix}.log"))

    with open(args.manifest) as f:
        manifest = json.load(f)
    num_classes = manifest['num_classes']
    if args.label_key == 'species':
        class_to_idx = manifest['species_to_idx']
    else:
        class_to_idx = manifest['class_to_idx']
    class_names = sorted(class_to_idx, key=class_to_idx.get)
    weights = torch.FloatTensor([manifest['class_weights'][c] for c in class_names])

    logger.info("=" * 78)
    logger.info(f"PROBE AUDIT | {args.model} | {manifest['task']}")
    logger.info("=" * 78)
    logger.info(f"manifest: {args.manifest}")
    logger.info(f"{num_classes} classes | chance macro-F1 ~ {1/num_classes:.4f}")
    if args.published_f1 is not None:
        logger.info(f"published test macro-F1: {args.published_f1:.4f}")

    # ------------------------------------------------------------- features
    cache_key = None
    if args.emb_cache:
        import hashlib
        h = hashlib.md5()
        h.update(f"{args.model}|{args.manifest}|{WINDOW_SECONDS}|{STRIDE_SECONDS}".encode())
        cache_key = h.hexdigest()[:12]
        cache_f = Path(args.emb_cache) / f"emb_{args.model}_{cache_key}.npz"
    else:
        cache_f = None

    if cache_f is not None and cache_f.exists():
        logger.info(f"\nreusing embedding cache {cache_f.name}")
        z = np.load(cache_f)
        feats = {s: (z[f'{s}_X'], z[f'{s}_y'])
                 for s in ('train', 'val', 'test') if f'{s}_X' in z}
    else:
        ex = Extractor(args.model, logger)
        logger.info("\nextracting (5s/2.5s windows, final layer, mean pooled - "
                    "identical to phase3_03):")
        feats = {}
        for split in ('train', 'val', 'test'):
            if split not in manifest['splits'] or not manifest['splits'][split]:
                continue
            feats[split] = ex.extract_split(manifest['splits'][split],
                                            class_to_idx, args.label_key,
                                            logger, split)
        if cache_f is not None:
            cache_f.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_f, **{f'{s}_{k}': v for s, (X, y) in feats.items()
                            for k, v in (('X', X), ('y', y))})
            logger.info(f"cached embeddings -> {cache_f.name}")

    device = 'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu')
    train_X, train_y = feats['train']
    test_X, test_y = feats['test']

    # --------------------------------------------------------- trajectory
    has_val = 'val' in feats
    val_X, val_y = feats['val'] if has_val else (None, None)

    logger.info("\n" + "=" * 78)
    logger.info("TRAJECTORY - final state after N full-batch steps, no early stopping")
    if has_val:
        logger.info("This manifest HAS a val split, so phase3_03 took its has-val")
        logger.info("branch (100 steps max, early stop patience 10 on val accuracy).")
        logger.info("That replication is reported separately below; the trajectory")
        logger.info("here is still the reference for how much training length matters.")
    else:
        logger.info("No val split, so phase3_03 took its no-val branch:")
        logger.info("step 50 == exactly what it did.")
    logger.info("=" * 78)

    traj = []
    for steps in args.steps:
        clf, _ = fit_probe(train_X, train_y, num_classes, weights, device, steps)
        tr = evaluate(clf, train_X, train_y, device)
        te = evaluate(clf, test_X, test_y, device)
        traj.append({'steps': steps, 'train_f1_macro': tr['f1_macro'],
                     'test_f1_macro': te['f1_macro'],
                     'test_balanced_accuracy': te['balanced_accuracy'],
                     'test_accuracy': te['accuracy']})
        marker = "  <- phase3_03" if steps == 50 else ""
        logger.info(f"  {steps:5d} steps | train F1 {tr['f1_macro']:.4f} | "
                    f"test F1 {te['f1_macro']:.4f} | test acc "
                    f"{te['accuracy']:.4f}{marker}")

    # ------------------------------------------------------- internal val
    phase3_03_repro = None
    if has_val:
        logger.info("\n" + "=" * 78)
        logger.info("phase3_03 HAS-VAL BRANCH REPLICATION (100 steps, patience 10 "
                    "on val acc)")
        logger.info("=" * 78)
        clf_r, step_r = fit_probe(train_X, train_y, num_classes, weights, device,
                                  100, val_X=val_X, val_y=val_y, patience=10)
        tr_r = evaluate(clf_r, train_X, train_y, device)
        te_r = evaluate(clf_r, test_X, test_y, device)
        phase3_03_repro = {'stopped_at_step': step_r,
                           'train_f1_macro': tr_r['f1_macro'],
                           'test_f1_macro': te_r['f1_macro'],
                           'test_accuracy': te_r['accuracy']}
        logger.info(f"  stopped at step {step_r} | train F1 {tr_r['f1_macro']:.4f} "
                    f"| test F1 {te_r['f1_macro']:.4f}")
        if args.published_f1 is not None:
            logger.info(f"  published {args.published_f1:.4f} -> replication "
                        f"{te_r['f1_macro']:.4f}")

    logger.info("\n" + "=" * 78)
    if has_val:
        logger.info("CORRECTED - early stopping on the manifest's REAL val split")
    else:
        logger.info("CORRECTED - early stopping on an internal 80/20 split of TRAIN")
        logger.info("(no val split exists here; TEST is never used to stop)")
    logger.info("=" * 78)

    if has_val:
        fit_train_X, fit_train_y = train_X, train_y
        stop_X, stop_y = val_X, val_y
    else:
        keep, held = stratified_split(train_y, args.internal_val_frac, seed=42)
        fit_train_X, fit_train_y = train_X[keep], train_y[keep]
        stop_X, stop_y = train_X[held], train_y[held]

    clf, best_step = fit_probe(
        fit_train_X, fit_train_y, num_classes, weights, device,
        args.internal_val_max_steps, val_X=stop_X, val_y=stop_y,
        patience=args.internal_val_patience)
    tr = evaluate(clf, fit_train_X, fit_train_y, device)
    te = evaluate(clf, test_X, test_y, device)
    logger.info(f"  stopped at step {best_step} | fit-train F1 {tr['f1_macro']:.4f}")
    logger.info(f"  TEST macro-F1 {te['f1_macro']:.4f} | bal-acc "
                f"{te['balanced_accuracy']:.4f} | acc {te['accuracy']:.4f}")

    peak = max(traj, key=lambda r: r['test_f1_macro'])
    logger.info("\n" + "-" * 78)
    if args.published_f1 is not None:
        logger.info(f"  published        {args.published_f1:.4f}")
    logger.info(f"  50 steps         {traj[0]['test_f1_macro']:.4f}  "
                f"(train F1 {traj[0]['train_f1_macro']:.4f})")
    logger.info(f"  internal-val     {te['f1_macro']:.4f}")
    logger.info(f"  trajectory peak  {peak['test_f1_macro']:.4f} at "
                f"{peak['steps']} steps")
    if args.published_f1:
        logger.info(f"  MOVEMENT vs published: "
                    f"{te['f1_macro'] - args.published_f1:+.4f} (internal-val)")
    logger.info("-" * 78)

    result = {
        'model': args.model,
        'manifest': args.manifest,
        'task': manifest['task'],
        'label_key': args.label_key,
        'num_classes': num_classes,
        'chance_f1_macro': 1.0 / num_classes,
        'published_test_f1_macro': args.published_f1,
        'n_train_windows': int(len(train_y)),
        'n_val_windows': int(len(val_y)) if has_val else None,
        'n_test_windows': int(len(test_y)),
        'embedding_dim': int(train_X.shape[1]),
        'trajectory': traj,
        'has_val_split': has_val,
        'phase3_03_hasval_replication': phase3_03_repro,
        'corrected_internal_val': {
            'stop_split': 'manifest val' if has_val else 'internal 80/20 of train',
            'stopped_at_step': best_step,
            'inner_train_f1_macro': tr['f1_macro'],
            'test_f1_macro': te['f1_macro'],
            'test_balanced_accuracy': te['balanced_accuracy'],
            'test_accuracy': te['accuracy'],
            'val_frac': args.internal_val_frac,
        },
        'trajectory_peak': peak,
        'movement_vs_published': (te['f1_macro'] - args.published_f1
                                  if args.published_f1 is not None else None),
        'note': ('Features are identical to phase3_03 (same windowing, final '
                 'layer, mean pooling, manifest class weights). Only the number '
                 'of probe training steps differs.'),
    }
    name = f"probe_audit_{args.model}{suffix}.json"
    with open(out_dir / name, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"\n✓ {out_dir / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
