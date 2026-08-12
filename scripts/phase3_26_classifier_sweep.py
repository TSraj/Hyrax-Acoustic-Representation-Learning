#!/usr/bin/env python3
"""
Phase 3 - Step 26: classifier sweep on frozen hyrax embeddings.

THE QUESTION
------------
Every hyrax number in this project comes from ONE readout: a linear probe. The
audit already showed how much that choice matters -- probe training length alone
moved results by +0.05 to +0.58 and reordered every ranking. So a reviewer can
fairly ask: is 0.45 macro-F1 the limit of the REPRESENTATION, or just the limit
of a linear boundary?

This answers it. The encoder and the features are held fixed; only the
classifier on top changes.

  all classifiers score alike  -> the ceiling is the representation
  the non-linear ones win      -> the information was there, the readout was
                                  the bottleneck, and every earlier number
                                  understates the encoder

Either outcome is publishable, and it closes the "your probe was too weak"
objection to the negative adaptation result.

WHAT IT DOES
------------
Reads the cached per-layer embeddings written by phase3_24 (no GPU, no
re-extraction), picks each cell's best layer by an internal split of TRAIN, and
fits:

  linear      logistic regression        the current readout, as a reference
  mlp         1 hidden layer, 256 units  non-linear, still parametric
  svm_rbf     RBF-kernel SVM             non-linear, margin-based
  knn         k-nearest neighbours       non-parametric; asks whether identity
                                         is simply LOCAL in embedding space

Features are standardised inside a Pipeline, so scaling is fit on train only and
never leaks. Several seeds per cell/classifier; mean +- SD reported.

LAYER SELECTION uses an internal 80/20 stratified split of TRAIN, never test --
the same discipline as phase3_24. Pass --layer to pin one instead.

USAGE
-----
    python scripts/phase3_26_classifier_sweep.py

    # a specific probe directory / layer
    python scripts/phase3_26_classifier_sweep.py \
        --probe-dir outputs/phase3/hyrax_probe_adapt_species_id --layer 3
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# same validated pair as phase3_25, plus two further steps for the 4-way
# classifier axis; assigned in fixed order, never cycled
PALETTE = ["#3366CC", "#E8710A", "#137333", "#8430CE"]
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#dcdcdc"

MODEL_LABEL = {"hubert_base": "HuBERT", "xls_r": "XLS-R"}
COND_LABEL = {"base": "frozen", "adapted": "adapted"}


def classifiers(seed, n_train):
    """Fresh, unfitted estimators. Scaling lives inside the pipeline."""
    def pipe(clf):
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])

    return {
        "linear": pipe(LogisticRegression(max_iter=3000, C=1.0,
                                          class_weight="balanced",
                                          random_state=seed)),
        "mlp": pipe(MLPClassifier(hidden_layer_sizes=(256,), max_iter=600,
                                  early_stopping=True, n_iter_no_change=20,
                                  validation_fraction=0.15, alpha=1e-3,
                                  random_state=seed)),
        "svm_rbf": pipe(SVC(kernel="rbf", C=10.0, gamma="scale",
                            class_weight="balanced", random_state=seed)),
        # k grows slowly with n; identity in embedding space is expected to be
        # local, so a small neighbourhood is the honest setting
        "knn": pipe(KNeighborsClassifier(
            n_neighbors=max(3, min(15, int(np.sqrt(n_train) / 3))),
            weights="distance")),
    }


def stratified_split(y, frac, seed):
    rng = np.random.default_rng(seed)
    keep, held = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = max(1, int(round(len(idx) * frac)))
        held.append(idx[:n])
        keep.append(idx[n:])
    return np.sort(np.concatenate(keep)), np.sort(np.concatenate(held))


def pick_layer(train_X, train_y, seed, val_frac):
    """Best layer by a linear fit on an internal split. TEST IS NOT USED."""
    keep, held = stratified_split(train_y, val_frac, seed)
    best, best_f1 = 0, -1.0
    for layer in range(train_X.shape[1]):
        clf = Pipeline([("scale", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=1000,
                                                   class_weight="balanced",
                                                   random_state=seed))])
        clf.fit(train_X[keep, layer, :], train_y[keep])
        f1 = f1_score(train_y[held], clf.predict(train_X[held, layer, :]),
                      average="macro", zero_division=0)
        if f1 > best_f1:
            best, best_f1 = layer, f1
    return best, best_f1


def evaluate_cell(cache, layer, seeds, val_frac):
    z = np.load(cache)
    train_X, train_y = z["train_X"], z["train_y"]
    test_X, test_y = z["test_X"], z["test_y"]

    chosen = layer
    if chosen is None:
        chosen, sel_f1 = pick_layer(train_X, train_y, seeds[0], val_frac)
        print(f"    layer selected on internal split: {chosen} "
              f"(val macro-F1 {sel_f1:.4f})")

    Xtr, Xte = train_X[:, chosen, :], test_X[:, chosen, :]

    out = {}
    for name in classifiers(seeds[0], len(train_y)):
        runs = []
        for seed in seeds:
            clf = classifiers(seed, len(train_y))[name]
            clf.fit(Xtr, train_y)
            pred = clf.predict(Xte)
            runs.append({
                "seed": seed,
                "f1_macro": float(f1_score(test_y, pred, average="macro",
                                           zero_division=0)),
                "accuracy": float(accuracy_score(test_y, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(test_y, pred)),
            })
        f1s = [r["f1_macro"] for r in runs]
        out[name] = {
            "f1_macro_mean": float(np.mean(f1s)),
            "f1_macro_std": float(np.std(f1s)),
            "accuracy_mean": float(np.mean([r["accuracy"] for r in runs])),
            "balanced_accuracy_mean": float(np.mean(
                [r["balanced_accuracy"] for r in runs])),
            "runs": runs,
        }
        print(f"    {name:<9} macro-F1 {out[name]['f1_macro_mean']:.4f} "
              f"+- {out[name]['f1_macro_std']:.4f}")

    return chosen, out, int(len(train_y)), int(len(test_y))


def figure(results, chance, out_png):
    cells = list(results)
    names = list(results[cells[0]]["classifiers"])

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    x = np.arange(len(cells))
    width = 0.8 / len(names)

    for i, name in enumerate(names):
        means = [results[c]["classifiers"][name]["f1_macro_mean"] for c in cells]
        stds = [results[c]["classifiers"][name]["f1_macro_std"] for c in cells]
        pos = x + (i - (len(names) - 1) / 2) * width
        ax.bar(pos, means, width * 0.9, label=name, color=PALETTE[i],
               edgecolor="white", linewidth=0.8, zorder=3)
        ax.errorbar(pos, means, yerr=stds, fmt="none", ecolor=MUTED,
                    elinewidth=0.9, capsize=2, zorder=4)

    # label only the winner per cell -- never a number on every bar
    for xi, cell in zip(x, cells):
        best = max(names, key=lambda n: results[cell]["classifiers"][n]["f1_macro_mean"])
        i = names.index(best)
        v = results[cell]["classifiers"][best]["f1_macro_mean"]
        s = results[cell]["classifiers"][best]["f1_macro_std"]
        ax.annotate(f"{v:.3f}", (xi + (i - (len(names) - 1) / 2) * width, v + s),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    va="bottom", fontsize=9, fontweight="bold",
                    color=PALETTE[i], zorder=6)

    top = max(results[c]["classifiers"][n]["f1_macro_mean"]
              + results[c]["classifiers"][n]["f1_macro_std"]
              for c in cells for n in names)
    ax.set_ylim(0, top * 1.3)
    ax.set_xlim(-0.6, len(cells) - 0.4)

    ax.axhline(chance, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
    ax.annotate(f"chance {chance:.3f}", (-0.55, chance), ha="left", va="bottom",
                fontsize=8, color=MUTED, zorder=6,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{MODEL_LABEL.get(results[c]['model'], results[c]['model'])}\n"
         f"{COND_LABEL.get(results[c]['condition'], results[c]['condition'])}"
         f"  (L{results[c]['layer']})" for c in cells],
        fontsize=9.5, color=INK)
    ax.set_ylabel("Hyrax ID macro-F1", fontsize=10, color=INK)
    ax.set_title("Is 0.45 the representation's ceiling, or the linear probe's?",
                 fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=10)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.xaxis.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.legend(frameon=False, fontsize=9, ncol=len(names), loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Classifier sweep on frozen hyrax embeddings")
    p.add_argument("--probe-dir", default="outputs/phase3/hyrax_probe_adapt_species_id")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--layer", type=int, default=None,
                   help="pin a layer; default selects per cell on an internal split")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.2)
    args = p.parse_args()

    probe_dir = Path(args.probe_dir)
    cache_dir = probe_dir / "emb_cache"
    out_dir = Path(args.out_dir) if args.out_dir else probe_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    caches = sorted(cache_dir.glob("*.npz"))
    if not caches:
        raise SystemExit(f"no cached embeddings in {cache_dir}")

    # chance level comes from the probe JSONs, which carry the class count
    chance, meta = None, {}
    for j in probe_dir.glob("layer_probe_*.json"):
        d = json.load(open(j))
        chance = d["chance"]
        meta[(d["model"], d["condition"])] = d
    if chance is None:
        raise SystemExit(f"no layer_probe_*.json in {probe_dir} to read the class count from")

    seeds = list(range(42, 42 + args.seeds))
    print(f"probe dir : {probe_dir}")
    print(f"cells     : {len(caches)}   seeds: {seeds}   chance: {chance:.3f}\n")

    results, rows = {}, []
    for cache in caches:
        stem = cache.stem                      # e.g. hubert_base_adapted
        condition = "adapted" if stem.endswith("_adapted") else "base"
        model = stem[: -len(condition) - 1]
        cell = f"{model}_{condition}"
        print(f"  {cell}")

        layer, per_clf, n_tr, n_te = evaluate_cell(cache, args.layer, seeds, args.val_frac)
        results[cell] = {"model": model, "condition": condition, "layer": layer,
                         "n_train": n_tr, "n_test": n_te, "classifiers": per_clf}
        for name, r in per_clf.items():
            rows.append({
                "model": model, "condition": condition, "layer": layer,
                "classifier": name,
                "f1_macro_mean": round(r["f1_macro_mean"], 4),
                "f1_macro_std": round(r["f1_macro_std"], 4),
                "accuracy_mean": round(r["accuracy_mean"], 4),
                "balanced_accuracy_mean": round(r["balanced_accuracy_mean"], 4),
                "chance": chance,
            })
        print()

    # stable, readable cell order: model, then frozen before adapted
    order = sorted(results, key=lambda c: (results[c]["model"] != "hubert_base",
                                           results[c]["model"],
                                           results[c]["condition"] != "base"))
    results = {c: results[c] for c in order}

    png = out_dir / "classifier_comparison.png"
    figure(results, chance, png)

    csv_path = out_dir / "classifier_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    json_path = out_dir / "classifier_comparison.json"
    with open(json_path, "w") as f:
        json.dump({"chance": chance, "seeds": seeds,
                   "layer_selection": "pinned" if args.layer is not None
                   else "internal split of train, test never used",
                   "cells": results}, f, indent=2)

    print("=" * 64)
    print(f"wrote {png}")
    print(f"      {csv_path}")
    print(f"      {json_path}\n")

    print("winner per cell, and the gap over the linear probe:")
    for cell, r in results.items():
        lin = r["classifiers"]["linear"]["f1_macro_mean"]
        best = max(r["classifiers"], key=lambda n: r["classifiers"][n]["f1_macro_mean"])
        bv = r["classifiers"][best]["f1_macro_mean"]
        bs = r["classifiers"][best]["f1_macro_std"]
        ls = r["classifiers"]["linear"]["f1_macro_std"]
        verdict = "beyond noise" if (bv - lin) > (bs + ls) else "within noise"
        print(f"  {cell:<22} L{r['layer']:<3} {best:<8} {bv:.4f}   "
              f"linear {lin:.4f}   gap {bv - lin:+.4f}  ({verdict})")

    print("\nReading it: gaps within noise mean the linear probe was already at the\n"
          "representation's ceiling, so every earlier number stands. A gap beyond\n"
          "noise means the information was present but not linearly separable.")


if __name__ == "__main__":
    main()
