#!/usr/bin/env python3
"""
Phase 3 - SQ4 step 0: per-individual transfer success from cached Phase 2 results.

No model is run here. The Phase 2 pooled 69-way individual-ID task cached its
test embeddings for every layer and saved the trained linear head, so
per-individual F1 - which was never written out - can be recovered exactly.

Layer self-check
----------------
The saved head does not record which layer it was trained on. Rather than
guess, this script applies the head to EVERY cached layer and keeps the one
that reproduces the accuracy logged in pooled_results.json. If no layer
reproduces it (within --tol), the model is skipped and reported, rather than
silently falling back to the best-scoring layer.

The correct layer is unmistakable in practice: the head only makes sense on the
representation it was fitted to, so the right layer beats the runner-up by a
wide margin.

ECAPA is excluded: its embedding_cache directory exists but is empty, so its
per-individual F1 cannot be recovered without re-running the model.

Outputs per_individual_f1.csv - one row per individual, with each model's F1,
the mean across models, and the test support.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS = ["hubert_base", "xls_r", "wavlm", "wav2vec2_base", "wav2vec2_base_960h"]
DATASETS = ['anuraset', 'bengalese_finch', 'macaque', 'marmoset',
            'picidae', 'wetlands_bird', 'zebra_finch']


def identify_layer(model_dir, W, b, tol):
    """Return (layer, reproduced_acc, table) for the layer matching the logged accuracy."""
    logged = json.load(open(model_dir / "pooled_results.json"))['test_accuracy']
    with open(model_dir / "embedding_cache/test_all_layers.pkl", 'rb') as f:
        cache = pickle.load(f)

    y = np.asarray(cache['labels'])
    accs = {}
    for k, E in cache['layer_embeddings'].items():
        E = np.asarray(E)
        if E.shape[1] != W.shape[1]:
            continue
        accs[int(k)] = accuracy_score(y, (E @ W.T + b).argmax(1))

    matches = [k for k, v in accs.items() if abs(v - logged) < tol]
    best = max(accs, key=accs.get) if accs else None
    runner = sorted(accs.values(), reverse=True)[1] if len(accs) > 1 else float('nan')
    return matches, logged, accs, best, runner, cache, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled-root", default="outputs/phase2V2/zero_shot/pooled")
    ap.add_argument("--manifest", default="outputs/phase2V2/manifests/pooled_manifest.json")
    ap.add_argument("--out", default="outputs/phase3/sq4/per_individual_f1.csv")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="Tolerance for matching the logged accuracy")
    args = ap.parse_args()

    root = Path(args.pooled_root)
    individuals = json.load(open(args.manifest))['individuals']  # index order == label id
    print(f"classes: {len(individuals)}")

    rows, chosen, skipped = [], {}, []

    for m in MODELS:
        d = root / m
        if not (d / "embedding_cache/test_all_layers.pkl").exists():
            skipped.append((m, "no cached embeddings"))
            continue

        head = torch.load(d / "fc_head_pooled.pth", map_location='cpu', weights_only=False)
        W, b = head['weight'].numpy(), head['bias'].numpy()

        matches, logged, accs, best, runner, cache, y = identify_layer(d, W, b, args.tol)

        if not matches:
            skipped.append((m, f"no layer reproduces logged acc {logged:.6f} "
                               f"(best {accs[best]:.6f} at L{best})"))
            print(f"  {m}: SKIPPED - no layer reproduces logged accuracy")
            continue
        if len(matches) > 1:
            print(f"  {m}: WARNING - {len(matches)} layers match; using lowest ({min(matches)})")

        L = min(matches)
        chosen[m] = {'layer': L, 'acc': accs[L], 'logged': logged,
                     'runner_up': runner, 'n_layers': len(accs)}
        print(f"  {m:20s} layer L{L:<3d} acc={accs[L]:.6f} (logged {logged:.6f}) "
              f"| runner-up {runner:.4f}")

        E = np.asarray(cache['layer_embeddings'][str(L)]
                       if str(L) in cache['layer_embeddings'] else cache['layer_embeddings'][L])
        pred = (E @ W.T + b).argmax(1)
        f1 = f1_score(y, pred, average=None,
                      labels=list(range(len(individuals))), zero_division=0)
        support = np.bincount(y, minlength=len(individuals))
        for i, ind in enumerate(individuals):
            rows.append({'model': m, 'individual': ind,
                         'f1': f1[i], 'support': int(support[i])})
        del cache, E

    if skipped:
        print("\nSkipped models:")
        for m, why in skipped:
            print(f"  {m}: {why}")
    if not rows:
        print("No models usable - aborting.")
        return 1

    df = pd.DataFrame(rows)
    piv = df.pivot(index='individual', columns='model', values='f1')
    sup = df.groupby('individual')['support'].first()
    out = pd.DataFrame({'support': sup,
                        'f1_mean': piv.mean(axis=1),
                        'f1_std': piv.std(axis=1, ddof=1)}).join(piv)
    out['dataset'] = [next(d for d in DATASETS if i.startswith(d + '_')) for i in out.index]
    out = out.sort_values('f1_mean')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out)

    print(f"\nWrote {args.out} ({len(out)} individuals, {len(chosen)} models)")
    print(f"usable (support>=5): {(out['support'] >= 5).sum()}")
    u = out[out['support'] >= 5]
    print(f"f1_mean on usable set: mean={u['f1_mean'].mean():.4f} "
          f"sd={u['f1_mean'].std():.4f} range {u['f1_mean'].min():.4f}-{u['f1_mean'].max():.4f}")
    print("\nper-dataset (usable):")
    print(u.groupby('dataset')['f1_mean'].agg(['count', 'mean', 'std']).round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
