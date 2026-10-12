#!/usr/bin/env python3
"""
Phase 3 - Step 41: frozen 5-fold cross-validation at EVERY layer.

phase3_37 scores only the layer its inner folds choose, so it cannot draw a
layer curve. This fits the same linear probe at every layer, on the same
5-fold-by-recording folds, and reports pooled UAR, AUC, macro-F1 and accuracy
per layer. Nothing is re-embedded: the embeddings come from the phase3_37
cache in outputs/phase3/cv_<audio>/emb_cache.

WHAT IS THE SAME AS phase3_37
-----------------------------
Same manifest (cv_by_file_5fold: 5 folds x 5 repeats), same probe (linear,
class-weighted CE, 5 seeds, 5000 steps, patience 500, grouped 20% early-stop
split), same softmax averaging over seeds, same pooled scoring. The only
change is that the layer is fixed per run instead of chosen.

WHAT THIS IS NOT
----------------
The peak of this curve is the best layer ON THE TEST FOLDS -- the right thing
to plot, the wrong thing to quote as a headline score. The headline remains
phase3_37's inner-CV-selected number.

OUTPUT
------
  outputs/phase3/cv_<audio>/layers/cv_layers_<protocol>_<model>.json
      per layer: pooled UAR / AUC / macro-F1 / accuracy, mean and SD over
      the 5 repeats, plus the per-repeat values

RESUMABLE: layers already finished are kept in a .partial.json and skipped
on restart, so a task killed by the walltime loses at most one layer.

USAGE
-----
    python scripts/phase3_41_cv_layer_sweep.py \
        --model hubert_base \
        --manifest outputs/phase3/manifests_cv_original/cv_by_file_5fold.json \
        --cache-dir outputs/phase3/cv_original/emb_cache \
        --output-dir outputs/phase3/cv_original/layers
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from src.utils.logging_utils import setup_logger  # noqa: E402

from phase3_24_hyrax_layer_probe import ALL_MODELS  # noqa: E402
from phase3_37_cv_probe import (  # noqa: E402
    class_weights_for,
    fit_and_prob,
    load_or_extract,
    score,
)

METRICS = ["uar", "auc_macro_ovr", "f1_macro", "accuracy"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=ALL_MODELS)
    p.add_argument("--manifest", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--probe-seeds", type=int, default=5)
    p.add_argument("--probe-steps", type=int, default=5000)
    p.add_argument("--probe-patience", type=int, default=500)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--force", action="store_true")
    # needed by load_or_extract; extraction only happens if the cache is gone
    p.add_argument("--force-extract", action="store_true")
    args = p.parse_args()

    logger = setup_logger("Phase3_CVLayers", "INFO")
    manifest_path = Path(args.manifest)
    with open(manifest_path) as f:
        manifest = json.load(f)

    protocol = manifest["protocol"]
    classes = manifest["individuals"]
    class_to_idx = manifest["class_to_idx"]
    num_classes = len(classes)
    items = manifest["items"]
    n_folds, n_repeats = manifest["n_folds"], manifest["n_repeats"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"cv_layers_{protocol}_{args.model}.json"
    partial_path = result_path.with_suffix(".partial.json")
    if result_path.exists() and not args.force:
        logger.info(f"result exists: {result_path} -- pass --force to redo")
        return

    cache_dir = Path(args.cache_dir)
    if not any(cache_dir.glob(f"{args.model}_*.npz")):
        logger.warning(f"no cached embeddings for {args.model} in {cache_dir}; "
                       f"they will be EXTRACTED (GPU, slower)")
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    X, kept = load_or_extract(items, args, cache_dir, logger)
    kept_items = [items[i] for i in kept]
    y = np.asarray([class_to_idx[it["individual"]] for it in kept_items])
    groups = np.asarray([it["file"] for it in kept_items])
    folds = np.asarray([it["folds"] for it in kept_items])
    n_layers = X.shape[1]
    seeds = list(range(42, 42 + args.probe_seeds))

    logger.info("=" * 72)
    logger.info(f"CV LAYER SWEEP - {args.model} - {protocol}")
    logger.info(f"{n_layers} layers x {n_folds} folds x {n_repeats} repeats x "
                f"{len(seeds)} seeds, {len(y)} bouts, {num_classes} classes")
    logger.info("=" * 72)

    done = {}
    if partial_path.exists() and not args.force:
        with open(partial_path) as f:
            done = {int(k): v for k, v in json.load(f).items()}
        logger.info(f"resuming: layers {sorted(done)} already finished")

    t0 = time.time()
    for layer in range(n_layers):
        if layer in done:
            continue
        per_repeat = []
        for r in range(n_repeats):
            pooled = np.zeros((len(y), num_classes), dtype=np.float64)
            for k in range(n_folds):
                te = np.flatnonzero(folds[:, r] == k)
                tr = np.flatnonzero(folds[:, r] != k)
                if len(te) == 0:
                    continue
                w = class_weights_for(y[tr], num_classes)
                pooled[te] = fit_and_prob(
                    X[tr, layer], y[tr], groups[tr], X[te, layer],
                    num_classes, w, device, seeds, args.probe_steps,
                    args.probe_patience, args.val_frac)
            s = score(y, pooled.argmax(1), classes, [], pooled)
            per_repeat.append({m: s[m] for m in METRICS})

        row = {"layer": layer, "per_repeat": per_repeat}
        for m in METRICS:
            v = [pr[m] for pr in per_repeat]
            row[f"{m}_mean"] = round(float(np.mean(v)), 4)
            row[f"{m}_sd"] = round(float(np.std(v)), 4)
        done[layer] = row
        with open(partial_path, "w") as f:
            json.dump(done, f)
        logger.info(f"  layer {layer:2d}/{n_layers - 1}: "
                    f"UAR {row['uar_mean']:.4f}  AUC {row['auc_macro_ovr_mean']:.4f}"
                    f"  F1 {row['f1_macro_mean']:.4f}  "
                    f"[{(time.time() - t0) / 60:.1f} min]")

    layers = [done[l] for l in range(n_layers)]
    best = max(layers, key=lambda r: (r["uar_mean"], -r["layer"]))
    out = {
        "model": args.model,
        "protocol": protocol,
        "manifest": str(manifest_path),
        "audio_subdir": manifest.get("audio_subdir"),
        "num_classes": num_classes,
        "chance_uar": round(1 / num_classes, 4),
        "n_bouts": int(len(y)),
        "n_layers": n_layers,
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "probe": {"type": "linear, full-batch Adam lr 1e-3, class-weighted CE",
                  "seeds": seeds, "steps": args.probe_steps,
                  "patience": args.probe_patience, "val_frac": args.val_frac,
                  "layer": "fixed per run, every layer scored on the test folds"},
        "split_exceptions": manifest.get("split_exceptions", {}),
        "peak_layer_by_uar": best["layer"],
        "peak_uar": best["uar_mean"],
        "layers": layers,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(result_path, "w") as f:
        json.dump(out, f, indent=2)
    partial_path.unlink(missing_ok=True)
    logger.info(f"peak layer {best['layer']} (UAR {best['uar_mean']:.4f})")
    logger.info(f"wrote {result_path}")


if __name__ == "__main__":
    main()
