#!/usr/bin/env python3
"""
Phase 3 - Step 29: per-layer SPECIES probe (7-class, hyrax excluded).

WHY A SEPARATE SCRIPT FROM phase3_24
------------------------------------
phase3_24 is the hyrax path and cannot be pointed at species_id.json:

  * it reads manifest["individuals"] and item["individual"]; the species
    manifest carries manifest["species"] / species_to_idx.
  * species_id.json has no start/end, so phase3_24 would fall into its 5 s /
    2.5 s WINDOW branch. The published species baselines were NOT measured that
    way -- phase3_03:217 uses one embedding per FILE, truncated to the first
    30 s. PROBE_AUDIT.md records what happens when this is confused: windowing
    species produced 0.6075 against a published 0.8736. That is a ~4x change in
    dataset size and it makes the comparison meaningless.

So the species regime is reproduced here explicitly.

EARLY STOPPING -- matches phase3_20, which is what the reference numbers used
-----------------------------------------------------------------------------
The hyrax manifests have no val split, so phase3_24 stops on an internal 80/20
split of TRAIN. species_id.json DOES have a val split (1789 items), and
phase3_20's corrected species numbers (XLS-R 0.969, HuBERT 0.962) were produced
by fitting on the FULL train set and stopping on that real val split.

Using an internal 80/20 split here instead would train on 80% of what XLS-R saw
and quietly handicap the new model. So the rule is phase3_20's rule:

    val split present  -> fit on full train, stop on the manifest val split
    val split absent   -> fit on 80%, stop on the held-out 20% of train

TEST is never used for stopping or selection under either branch.

AVES AND THE 10.24 s CANVAS
---------------------------
EAT sees a fixed 10.24 s canvas and start-crops anything longer, so it cannot
take a 30 s file the way a wav2vec2 model does. Cropping to the first 10.24 s
and comparing against a 30 s number would hand AVES a strictly harder task and
make any deficit uninterpretable.

Instead the same first 30 s is covered by CONSECUTIVE crops (0-10.24,
10.24-20.48, 20.48-30) and their embeddings are averaged, giving one vector per
file over the same audio and the same information budget. The final crop is
short and is masked accordingly, not zero-weighted into the mean.

Reported per layer: macro F1, macro PRECISION and macro RECALL, mean +- SD over
seeds, plus a per-class precision/recall/F1/support breakdown at the best layer.

USAGE
-----
    python scripts/phase3_29_species_layer_probe.py --model aves2_eat_bio
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from src.utils.audio_utils import load_audio  # noqa: E402
from src.utils.logging_utils import setup_logger  # noqa: E402

from phase3_20_probe_audit import (  # noqa: E402
    MAX_FILE_SECONDS,
    SAMPLE_RATE,
    evaluate,
    fit_probe,
    resolve,
    stratified_split,
)
from phase3_24_hyrax_layer_probe import (  # noqa: E402
    AVEX_MODELS,
    ALL_MODELS,
    build_extractor,
    layer_tag,
    macro_pr,
    predict,
)


def species_chunks(audio, model_name, crop_samples):
    """The species input unit -> list of chunks to average.

    HF models  one chunk, the first 30 s. Exactly phase3_03/phase3_20.
    AVES       consecutive 10.24 s crops spanning the same first 30 s, averaged
               afterwards, because the encoder cannot see more than one canvas
               at a time.
    """
    audio = audio[:int(MAX_FILE_SECONDS * SAMPLE_RATE)]
    if model_name not in AVEX_MODELS:
        return [audio]

    chunks = [audio[s:s + crop_samples]
              for s in range(0, max(1, len(audio)), crop_samples)]
    return [c for c in chunks if len(c) > 0] or [audio]


def extract_split(extractor, items, class_to_idx, label_key, split,
                  model_name, crop_samples, logger):
    """One embedding per FILE -> (X, y), X = (n_files, n_layers, hidden)."""
    embs, labels = [], []
    load_failed, embed_failed = 0, 0
    t0 = time.time()
    batched = hasattr(extractor, "embed_many")

    for item in tqdm(items, desc=split, leave=False):
        try:
            audio, _ = load_audio(str(resolve(item["file"])),
                                  target_sr=SAMPLE_RATE, mono=True)
        except Exception:
            load_failed += 1
            continue

        chunks = species_chunks(audio, model_name, crop_samples)
        try:
            if batched:
                per_crop = extractor.embed_many(chunks)          # (k, L, D)
            else:
                per_crop = np.stack([extractor.embed_all_layers(c) for c in chunks])
        except Exception:
            embed_failed += 1
            continue

        # one vector per file: mean over the crops covering the same 30 s
        embs.append(per_crop.mean(axis=0))
        labels.append(class_to_idx[item[label_key]])

    if load_failed:
        logger.warning(f"  {split}: {load_failed} files failed to load")
    if embed_failed:
        logger.warning(f"  {split}: {embed_failed} files failed to embed")
    if not embs:
        raise RuntimeError(f"no embeddings for split {split}")

    X = np.stack(embs).astype(np.float32)
    y = np.asarray(labels)
    logger.info(f"  {split}: {len(y)} files, {X.shape[1]} layers, dim {X.shape[2]} "
                f"({time.time() - t0:.0f}s)")
    return X, y


def probe_layer(train_X, train_y, test_X, test_y, num_classes, weights, device,
                seeds, steps, patience, val_frac, classes,
                val_X=None, val_y=None):
    """Converged probe, one run per seed. Test never touches selection.

    val_X given -> fit on all of train, stop on the manifest's real val split
                   (the branch phase3_20 used for species).
    otherwise   -> fit on 1-val_frac of train, stop on the rest.
    """
    use_manifest_val = val_X is not None
    runs, last_pred = [], None

    for seed in seeds:
        if use_manifest_val:
            fit_X, fit_y, stop_X, stop_y = train_X, train_y, val_X, val_y
        else:
            keep, held = stratified_split(train_y, val_frac, seed)
            fit_X, fit_y = train_X[keep], train_y[keep]
            stop_X, stop_y = train_X[held], train_y[held]

        clf, best_step = fit_probe(fit_X, fit_y, num_classes, weights, device,
                                   steps=steps, val_X=stop_X, val_y=stop_y,
                                   patience=patience, seed=seed)
        res = evaluate(clf, test_X, test_y, device)
        last_pred = predict(clf, test_X, device)
        res.update(macro_pr(test_y, last_pred))
        res["train_f1_macro"] = evaluate(clf, fit_X, fit_y, device)["f1_macro"]
        res["best_step"] = best_step
        res["seed"] = seed
        runs.append(res)

    def m(k):
        return float(np.mean([r[k] for r in runs]))

    out = {
        "f1_macro_mean": m("f1_macro"),
        "f1_macro_std": float(np.std([r["f1_macro"] for r in runs])),
        "f1_macro_runs": [r["f1_macro"] for r in runs],
        "precision_macro_mean": m("precision_macro"),
        "precision_macro_std": float(np.std([r["precision_macro"] for r in runs])),
        "recall_macro_mean": m("recall_macro"),
        "recall_macro_std": float(np.std([r["recall_macro"] for r in runs])),
        "train_f1_macro_mean": m("train_f1_macro"),
        "accuracy_mean": m("accuracy"),
        "balanced_accuracy_mean": m("balanced_accuracy"),
        "best_step_mean": m("best_step"),
        "runs": runs,
    }

    if last_pred is not None:
        p, r, f, s = precision_recall_fscore_support(
            test_y, last_pred, labels=list(range(num_classes)), zero_division=0)
        out["per_class"] = {
            classes[i]: {"precision": float(p[i]), "recall": float(r[i]),
                         "f1": float(f[i]), "support": int(s[i])}
            for i in range(num_classes)
        }
        out["confusion_matrix"] = confusion_matrix(
            test_y, last_pred, labels=list(range(num_classes))).tolist()
    return out


def main():
    p = argparse.ArgumentParser(description="Per-layer species probe (zero-shot)")
    p.add_argument("--model", required=True, choices=ALL_MODELS)
    p.add_argument("--manifest", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--probe-seeds", type=int, default=5)
    p.add_argument("--probe-steps", type=int, default=5000)
    p.add_argument("--probe-patience", type=int, default=500)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--stopping", default="auto", choices=["auto", "val", "internal"],
                   help="auto = phase3_20's rule: the manifest's val split when "
                        "it exists, else an internal 80/20 of train. This is "
                        "what the 0.969/0.962 reference numbers used.")
    p.add_argument("--layers", default=None)
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--pooling", default="masked_mean",
                   choices=["masked_mean", "unmasked_mean", "cls"])
    p.add_argument("--pad-mode", default="zero", choices=["zero", "tile"])
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    root = SCRIPT_DIR.parent
    with open(root / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)
    out_root = Path(config["paths"]["output_dir"])

    manifest_path = Path(args.manifest) if args.manifest else \
        out_root / "phase3" / "manifests_species7" / "species_id.json"
    output_dir = Path(args.output_dir) if args.output_dir else \
        out_root / "phase3" / "species_probe_zeroshot"
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "emb_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("Phase3_SpeciesLayerProbe", config["experiment"]["log_level"])

    cell = f"{args.model}_base"
    if args.model in AVEX_MODELS:
        variant = [v for v in (args.pooling if args.pooling != "masked_mean" else "",
                               args.pad_mode if args.pad_mode != "zero" else "") if v]
        if variant:
            cell += "_" + "_".join(variant)

    logger.info("=" * 72)
    logger.info(f"PER-LAYER SPECIES PROBE (zero-shot) - {cell}")
    logger.info("=" * 72)

    with open(manifest_path) as f:
        manifest = json.load(f)

    classes = manifest["species"]
    class_to_idx = manifest["species_to_idx"]
    num_classes = manifest["num_classes"]
    weights = torch.FloatTensor([manifest["class_weights"][c] for c in classes])
    splits = manifest["splits"]

    logger.info(f"manifest: {manifest_path.name}")
    logger.info(f"task:     {manifest['task']}  ({num_classes} classes, "
                f"chance {1 / num_classes:.3f})")
    logger.info("splits:   " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    from phase3_28_avex_extractor import CANVAS_SAMPLES
    crop_samples = CANVAS_SAMPLES

    fp = hashlib.sha1(json.dumps(
        {"task": manifest.get("task"),
         "regime": f"file-level, first {MAX_FILE_SECONDS}s",
         "crop_samples": crop_samples if args.model in AVEX_MODELS else None,
         "counts": {k: len(v) for k, v in splits.items()}},
        sort_keys=True, default=str).encode()).hexdigest()[:10]
    cache = cache_dir / f"{cell}_{fp}.npz"

    extractor_provenance = None
    if cache.exists() and not args.force_extract:
        logger.info(f"loading cached embeddings: {cache}")
        z = np.load(cache)
        feats = {s: (z[f"{s}_X"], z[f"{s}_y"]) for s in ("train", "val", "test")
                 if f"{s}_X" in z}
    else:
        extractor = build_extractor(args.model, None, logger,
                                    pooling=args.pooling, pad_mode=args.pad_mode,
                                    batch_size=args.batch_size)
        if args.model in AVEX_MODELS:
            logger.info(f"species regime: consecutive {crop_samples / SAMPLE_RATE:.2f}s "
                        f"crops over the first {MAX_FILE_SECONDS}s, averaged -> "
                        f"one vector per file")
        else:
            logger.info(f"species regime: ONE embedding per file, first "
                        f"{MAX_FILE_SECONDS}s (phase3_03 regime)")

        feats = {}
        for split in ("train", "val", "test"):
            if not splits.get(split):
                continue
            feats[split] = extract_split(extractor, splits[split], class_to_idx,
                                         "species", split, args.model,
                                         crop_samples, logger)
        if hasattr(extractor, "provenance"):
            extractor_provenance = extractor.provenance()
        if hasattr(extractor, "close"):
            extractor.close()

        np.savez_compressed(cache, **{f"{s}_{k}": v for s, (X, y) in feats.items()
                                      for k, v in (("X", X), ("y", y))})
        logger.info(f"cached embeddings -> {cache}")

    train_X, train_y = feats["train"]
    test_X, test_y = feats["test"]
    has_val = "val" in feats
    val_X, val_y = feats["val"] if has_val else (None, None)

    use_val = (args.stopping == "val") or (args.stopping == "auto" and has_val)
    if args.stopping == "val" and not has_val:
        raise SystemExit("--stopping val requested but the manifest has no val split")
    if use_val:
        logger.info("early stopping on the manifest's REAL val split "
                    "(the branch phase3_20 used for species; test never used)")
    else:
        logger.info(f"early stopping on an internal {int((1 - args.val_frac) * 100)}/"
                    f"{int(args.val_frac * 100)} split of TRAIN (test never used)")

    n_layers = train_X.shape[1]
    layer_ids = ([int(x) for x in args.layers.split(",")] if args.layers
                 else list(range(n_layers)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(42, 42 + args.probe_seeds))
    logger.info(f"\nprobing {len(layer_ids)} layers x {len(seeds)} seeds")

    results = {}
    for layer in layer_ids:
        r = probe_layer(
            train_X[:, layer, :], train_y, test_X[:, layer, :], test_y,
            num_classes, weights, device, seeds, args.probe_steps,
            args.probe_patience, args.val_frac, classes,
            val_X=val_X[:, layer, :] if use_val else None,
            val_y=val_y if use_val else None,
        )
        results[str(layer)] = r
        logger.info(f"  layer {layer:>2} ({layer_tag(args.model, layer):<13}) "
                    f"F1 {r['f1_macro_mean']:.4f} +- {r['f1_macro_std']:.4f}  "
                    f"P {r['precision_macro_mean']:.4f}  "
                    f"R {r['recall_macro_mean']:.4f}   "
                    f"train F1 {r['train_f1_macro_mean']:.4f}")

    best = max(results.items(), key=lambda kv: kv[1]["f1_macro_mean"])
    logger.info(f"\nBEST layer {best[0]}: F1 {best[1]['f1_macro_mean']:.4f} "
                f"+- {best[1]['f1_macro_std']:.4f}  "
                f"precision {best[1]['precision_macro_mean']:.4f}  "
                f"recall {best[1]['recall_macro_mean']:.4f}")
    for name, pc in sorted(best[1]["per_class"].items(), key=lambda kv: -kv[1]["recall"]):
        logger.info(f"    {name:<28} P {pc['precision']:.3f}  R {pc['recall']:.3f}  "
                    f"F1 {pc['f1']:.3f}  n={pc['support']}")

    summary = {
        "model": args.model,
        "condition": "base",
        "zero_shot": True,
        "task": manifest["task"],
        "num_classes": num_classes,
        "chance": 1 / num_classes,
        "classes": classes,
        "n_layers": n_layers,
        "regime": {
            "unit": "file",
            "max_file_seconds": MAX_FILE_SECONDS,
            "crop_seconds": (crop_samples / SAMPLE_RATE
                             if args.model in AVEX_MODELS else None),
            "crops_averaged": args.model in AVEX_MODELS,
        },
        "probe": {
            "seeds": seeds,
            "max_steps": args.probe_steps,
            "patience": args.probe_patience,
            "stopping": ("manifest val split" if use_val
                         else f"internal {args.val_frac} split of train"),
            "selection": "test never used for stopping or selection",
        },
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        "best_layer": int(best[0]),
        "best_f1_macro": best[1]["f1_macro_mean"],
        "best_precision_macro": best[1]["precision_macro_mean"],
        "best_recall_macro": best[1]["recall_macro_mean"],
        "layers": results,
    }
    if extractor_provenance is not None:
        summary["extractor"] = extractor_provenance
        summary["layer0_is_cnn_frontend"] = args.model not in AVEX_MODELS

    out_path = output_dir / f"layer_probe_{cell}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
