#!/usr/bin/env python3
"""
Phase 3 - Step 37: cross-validated layer probe, all 18 individuals.

WHAT IS DIFFERENT FROM phase3_24
--------------------------------
phase3_24 probes ONE fixed train/test partition and reports the layer with the
best TEST macro-F1. Two problems the supervisor asked us to fix:

  1. the reported layer is chosen on the test set, so the headline number is
     the maximum over 13-49 correlated options and is optimistically biased,
     and biased MORE for a deep model than a shallow one
  2. one partition gives no error bar, and eight individuals were never
     predicted at all

Here every recording is tested exactly once, all 18 individuals are included,
and THE LAYER IS CHOSEN INSIDE THE TRAINING DATA of each outer fold. The test
bouts of a fold are touched exactly once: to score the probe that was fitted
and layer-selected without them.

phase3_24 is NOT modified. The numbers already shown to the supervisor stay
reproducible from it.

THE LOOP
--------
    extract every bout once, all layers            <- the only GPU cost
    for each repeat r, each outer fold k:
        test    = bouts with folds[r] == k
        train   = the rest
        inner   = 3 grouped folds carved from TRAIN only
        pick the layer with the best mean inner macro-F1   (1 seed)
        refit on the whole of TRAIN at that layer          (5 seeds)
        average the 5 softmaxes -> one probability per test bout
    pool the out-of-fold probabilities and score them

Extraction is fold-independent, so it happens once per (model, audio) and both
protocols reuse the same cache. That is why this is 30 jobs rather than 60:
the split axis moved inside the job.

THE INNER SPLIT IS GROUPED, NOT RANDOM
--------------------------------------
Layer selection holds out whole RECORDINGS from the training set, never random
bouts. A random inner split would put bouts from one recording on both sides,
so the inner score would reward the layer that best encodes recording
condition -- exactly the confound the outer split exists to avoid, reintroduced
at the point where the layer is chosen.

WHAT IS REPORTED
----------------
  pooled          every bout has one out-of-fold prediction; macro-F1,
                  accuracy and a full per-individual breakdown are computed
                  on that pool. This is the headline.
  per_fold        the same metrics fold by fold -> mean and SD, the error bar
  aggregated      out-of-fold probabilities averaged within a recording, one
                  decision per recording
  cohort_100      macro-F1 restricted to the 11 individuals with >= 100 bouts.
                  The mean of those classes' F1 from the SAME 18-way
                  predictions -- not a separate 11-way model. Reported because
                  macro over 18 is dominated by classes with 16-38 bouts.

Every per-individual row carries `leaky_split`, `n_sessions` and
`n_recordings`, so a score standing on one recording is never read as if it
stood on fifteen.

USAGE
-----
    python scripts/phase3_37_cv_probe.py \
        --model hubert_base \
        --manifest outputs/phase3/manifests_cv/cv_by_file_5fold.json \
        --output-dir outputs/phase3/cv_bioda

RESUMABLE: the embedding cache is written once per (model, manifest) and the
result JSON is skipped if it already exists, so a killed job re-runs only what
is missing.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from src.utils.logging_utils import setup_logger  # noqa: E402

from phase3_20_probe_audit import fit_probe, resolve  # noqa: E402
from phase3_24_hyrax_layer_probe import (  # noqa: E402
    ALL_MODELS,
    STRIDE_SECONDS,
    WINDOW_SECONDS,
    SAMPLE_RATE,
    _load_cached,
    build_extractor,
    chunks_for_item,
    macro_pr,
)

# selection is a comparison between layers, not a published number, so one
# seed is enough; the seeds are spent where they affect the reported score
SELECTION_SEED = 42
COHORT_MIN_BOUTS = 100


# --------------------------------------------------------------- extraction

def extract_all(extractor, items, logger, batch_size=16):
    """Embed every bout once -> (X, kept_indices).

    Returns the indices of the items that produced a row, because a dropped
    bout would otherwise shift every fold id by one and silently mis-score the
    whole run. Nothing downstream indexes `items` directly; everything goes
    through `kept`.
    """
    window = int(WINDOW_SECONDS * SAMPLE_RATE)
    stride = int(STRIDE_SECONDS * SAMPLE_RATE)
    batched = hasattr(extractor, "embed_many")

    pending, kept, rows = [], [], []
    load_failed = empty = 0
    t0 = time.time()

    def flush():
        if not pending:
            return
        rows.append(extractor.embed_many(pending) if batched
                    else np.stack([extractor.embed_all_layers(c)
                                   for c in pending]))
        pending.clear()

    for i, item in enumerate(tqdm(items, desc="embedding", leave=False)):
        try:
            audio = _load_cached(str(resolve(item["file"])))
        except Exception:
            load_failed += 1
            continue
        chunks, is_bout = chunks_for_item(item, audio, window, stride)
        if not chunks:
            empty += 1
            continue
        if not is_bout:
            raise RuntimeError(
                f"item {i} did not slice as a bout. This driver assumes the "
                f"bout manifests from phase3_36; a windowed manifest would "
                f"emit several rows per item and break the fold mapping."
            )
        pending.append(chunks[0])
        kept.append(i)
        if len(pending) >= batch_size:
            flush()
    flush()

    if load_failed or empty:
        logger.warning(f"  dropped {load_failed} unreadable and {empty} empty "
                       f"bouts; fold ids follow the KEPT rows only")
    X = np.concatenate(rows, axis=0).astype(np.float32)
    if len(X) != len(kept):
        raise RuntimeError(f"{len(X)} embedding rows for {len(kept)} kept "
                           f"items -- refusing to continue")
    logger.info(f"  embedded {len(X)} bouts, {X.shape[1]} layers, "
                f"dim {X.shape[2]} ({time.time() - t0:.0f}s)")
    return X, np.asarray(kept)


def bout_key(item):
    """What identifies a bout independently of which manifest lists it."""
    return (item["file"], round(float(item["start"]), 4),
            round(float(item["end"]), 4))


def load_or_extract(items, args, cache_dir, logger):
    """Embeddings for `items`, in `items` order, extracting only if needed.

    THE CACHE IS KEYED ON THE BOUT SET, NOT THE MANIFEST ORDER.

    Both protocols hold exactly the same 4141 bouts but list them in different
    orders -- session_loso groups by session, by_file_5fold by recording, and
    they diverge from item 38 on. Hashing the ordered list therefore produced
    two different keys for one set of embeddings, and the pilot re-extracted
    everything for the second protocol. Harmless for wav2vec2 base at 105 s;
    for xls_r_1b it is tens of wasted GPU-minutes per task.

    So the fingerprint, and the stored row order, are the bouts sorted
    canonically. On load the rows are mapped back to whatever order the caller
    asked for, by key, and any bout the extractor dropped is simply absent
    from the map.
    """
    keys = [bout_key(it) for it in items]
    unique = len(set(keys)) == len(keys)

    if not unique:
        # two bouts with identical (file, start, end) cannot be told apart by
        # key, so fall back to the order-sensitive cache rather than risk
        # mapping a row onto the wrong bout
        logger.warning("  manifest contains duplicate (file, start, end) "
                       "bouts; using an order-sensitive embedding cache")
        order = list(range(len(items)))
    else:
        order = sorted(range(len(items)), key=lambda i: keys[i])

    fp_src = [keys[i] for i in order] if unique else keys
    fp = hashlib.sha1(json.dumps(fp_src).encode()).hexdigest()[:10]
    cache = cache_dir / f"{args.model}_{fp}.npz"

    ordered_items = [items[i] for i in order]

    if cache.exists() and not args.force_extract:
        z = np.load(cache, allow_pickle=False)
        Xc = z["X"]
        cached_keys = [(f, round(float(s), 4), round(float(e), 4))
                       for f, s, e in zip(z["files"], z["starts"], z["ends"])]
        logger.info(f"reusing cached embeddings: {cache.name} {Xc.shape}")
    else:
        extractor = build_extractor(args.model, None, logger,
                                    batch_size=args.batch_size)
        Xc, kept_c = extract_all(extractor, ordered_items, logger,
                                 args.batch_size)
        cached_keys = [bout_key(ordered_items[i]) for i in kept_c]
        tmp = cache.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp, X=Xc,
            files=np.array([k[0] for k in cached_keys]),
            starts=np.array([k[1] for k in cached_keys], dtype=np.float64),
            ends=np.array([k[2] for k in cached_keys], dtype=np.float64))
        tmp.replace(cache)
        logger.info(f"cached embeddings -> {cache.name}")

    if len(cached_keys) != len(Xc):
        raise RuntimeError(f"cache holds {len(Xc)} rows for "
                           f"{len(cached_keys)} bout keys -- refusing to use it")

    row_of = {k: i for i, k in enumerate(cached_keys)}
    if len(row_of) != len(cached_keys):
        raise RuntimeError("cached bout keys are not unique -- refusing to "
                           "map embeddings back onto the manifest")

    rows, kept = [], []
    for i, k in enumerate(keys):
        r = row_of.get(k)
        if r is not None:
            rows.append(r)
            kept.append(i)

    if len(kept) < len(items):
        logger.warning(f"  {len(items) - len(kept)} bouts have no embedding "
                       f"and are excluded from every fold")
    if not kept:
        raise RuntimeError("no manifest bout matched a cached embedding")

    return Xc[np.asarray(rows)], np.asarray(kept)


# ------------------------------------------------------------ fold machinery

def grouped_holdout(groups, y, frac, seed):
    """Hold out whole GROUPS covering ~`frac` of rows, keeping every class.

    Used for the inner layer-selection split. Groups are taken in a shuffled
    order until the target fraction is reached, but a group is skipped if
    removing it would leave its class with no training rows -- an unlearnable
    class in the inner fit would make the layer comparison meaningless for
    exactly the thin individuals this whole exercise is about.
    """
    import random
    rng = random.Random(seed)

    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    order = list(by_group)
    rng.shuffle(order)

    target = frac * len(y)
    remaining = defaultdict(int)
    for lab in y:
        remaining[int(lab)] += 1

    held, n = [], 0
    for g in order:
        rows = by_group[g]
        labs = defaultdict(int)
        for i in rows:
            labs[int(y[i])] += 1
        if any(remaining[c] - k <= 0 for c, k in labs.items()):
            continue
        for c, k in labs.items():
            remaining[c] -= k
        held.extend(rows)
        n += len(rows)
        if n >= target:
            break

    held = np.asarray(sorted(held), dtype=int)
    mask = np.ones(len(y), dtype=bool)
    mask[held] = False
    return np.flatnonzero(mask), held


def class_weights_for(y, num_classes):
    """Inverse-frequency weights from THIS fold's training rows.

    Recomputed per fold rather than taken from the manifest: a fold that holds
    out most of a thin individual's bouts has a different balance from the
    dataset as a whole, and weighting by the global counts would quietly
    under-weight it.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    w = np.where(counts > 0, len(y) / (num_classes * np.maximum(counts, 1)), 0.0)
    return torch.FloatTensor(w)


def fit_and_prob(Xtr, ytr, groups_tr, Xte, num_classes, weights, device,
                 seeds, steps, patience, val_frac):
    """Fit at one layer over `seeds` and return the averaged softmax.

    The early-stopping validation set is GROUPED by recording, like the inner
    selection split and for the same reason: a random one would let the probe
    stop at the point that best fits recording condition.

    Probabilities are averaged, not votes or logits: averaging bounded
    per-class scores keeps a confident seed above an unsure one without
    letting an arbitrary logit scale dominate.
    """
    probs = None
    for seed in seeds:
        if val_frac:
            keep, held = grouped_holdout(groups_tr, ytr, val_frac, seed)
        else:
            keep, held = np.arange(len(ytr)), np.array([], dtype=int)
        clf, _ = fit_probe(
            Xtr[keep], ytr[keep], num_classes, weights, device, steps,
            val_X=Xtr[held] if len(held) else None,
            val_y=ytr[held] if len(held) else None,
            patience=patience, seed=seed)
        clf.eval()
        with torch.no_grad():
            p = torch.softmax(clf(torch.FloatTensor(Xte).to(device)),
                              dim=1).cpu().numpy()
        probs = p if probs is None else probs + p
    return probs / len(seeds)


def grouped_kfold(groups, k, seed):
    """Partition rows into k folds by whole GROUP -> [(train_idx, val_idx)].

    Recordings are dealt largest-first into the emptiest fold, so the folds
    carry a similar number of bouts without ever splitting a recording. Same
    rule as the outer by-file partition, applied one level down.
    """
    import random
    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)

    order = list(by_group)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: len(by_group[g]), reverse=True)

    load = [0] * k
    assign = {}
    for g in order:
        f = min(range(k), key=lambda j: (load[j], j))
        assign[g] = f
        load[f] += len(by_group[g])

    fold_of = np.empty(len(groups), dtype=int)
    for g, rows in by_group.items():
        for i in rows:
            fold_of[i] = assign[g]

    return [(np.flatnonzero(fold_of != j), np.flatnonzero(fold_of == j))
            for j in range(k) if (fold_of == j).any()]


def inner_score(y_true, y_pred, trainable):
    """Macro-F1 over the classes the inner fit could actually learn.

    A class whose every bout sits in the inner validation fold has no training
    rows there, so its F1 is 0 for EVERY layer. Including it adds a constant
    to all layers and shrinks the differences that decide the choice, so it is
    excluded from the inner comparison only -- never from the reported score.
    """
    from sklearn.metrics import f1_score
    labels = sorted(trainable)
    if not labels:
        return 0.0
    return float(f1_score(y_true, y_pred, labels=labels, average="macro",
                          zero_division=0))


def select_layer(X, y, groups, num_classes, layers, device, args, logger):
    """Best layer by macro-F1 averaged over INNER FOLDS of the training data.

    k-fold rather than a single hold-out: with 13-49 candidate layers whose
    inner scores sit within a few points of each other, one split picks the
    winner largely by which recordings happened to land in it. Averaging over
    k folds is what the supervisor asked for, and it costs only probe fits --
    the embeddings are already in memory.

    The test bouts of the outer fold are not present here in any form.
    """
    splits = grouped_kfold(groups, args.inner_folds, SELECTION_SEED)
    degraded = False

    if len(splits) < 2:
        # every recording landed in one fold: nothing to cross-validate over
        from phase3_20_probe_audit import stratified_split
        tr, va = stratified_split(y, args.inner_frac, SELECTION_SEED)
        splits = [(tr, va)]
        degraded = True
        logger.warning("    could not build grouped inner folds; layer "
                       "selected on a STRATIFIED row split instead (bouts "
                       "from one recording may span it)")

    totals = {layer: 0.0 for layer in layers}
    n_used = 0
    for tr, va in splits:
        if len(va) == 0 or len(tr) == 0:
            continue
        trainable = set(np.unique(y[tr]).tolist()) & set(np.unique(y[va]).tolist())
        w = class_weights_for(y[tr], num_classes)
        for layer in layers:
            p = fit_and_prob(X[tr, layer], y[tr], groups[tr], X[va, layer],
                             num_classes, w, device, [SELECTION_SEED],
                             args.selection_steps, args.probe_patience,
                             val_frac=0.0)
            totals[layer] += inner_score(y[va], p.argmax(1), trainable)
        n_used += 1

    if n_used == 0:
        raise RuntimeError("no usable inner fold -- this outer fold cannot "
                           "select a layer honestly.")

    scores = {l: round(totals[l] / n_used, 4) for l in layers}
    best = max(layers, key=lambda l: (scores[l], -l))
    return best, {"scores": scores,
                  "inner_folds_used": n_used,
                  "degraded_inner_split": degraded}


# ------------------------------------------------------------------ metrics

def per_individual(y_true, y_pred, classes, inventory):
    """F1 / precision / recall / support per individual, with its provenance.

    The provenance columns are not decoration. A 0.9 F1 from an animal whose
    train and test share one recording is not the same evidence as 0.9 from an
    animal with fifteen sessions, and the table must make that visible.
    """
    from sklearn.metrics import precision_recall_fscore_support
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    out = []
    for i, name in enumerate(classes):
        inv = inventory.get(name, {})
        out.append({
            "individual": name,
            "f1": round(float(f[i]), 4),
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "support_bouts": int(s[i]),
            "n_sessions": inv.get("sessions"),
            "n_recordings": inv.get("recordings"),
            "leaky_split": inv.get("leaky_split", False),
        })
    return out


def score(y_true, y_pred, classes, cohort_idx):
    m = macro_pr(y_true, y_pred)
    m["accuracy"] = float((y_true == y_pred).mean())
    from sklearn.metrics import f1_score
    per = f1_score(y_true, y_pred, labels=list(range(len(classes))),
                   average=None, zero_division=0)
    m["f1_macro_cohort100"] = float(np.mean([per[i] for i in cohort_idx])) \
        if cohort_idx else None
    return {k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in m.items()}


def aggregate(probs, y_true, groups):
    """One decision per source recording: mean softmax, single argmax."""
    order = defaultdict(list)
    for i, g in enumerate(groups):
        order[g].append(i)
    gt, gp = [], []
    for g, rows in order.items():
        labs = y_true[rows]
        if not np.all(labs == labs[0]):
            raise RuntimeError(f"recording {g!r} carries several individuals")
        gt.append(labs[0])
        gp.append(int(probs[rows].mean(axis=0).argmax()))
    return np.asarray(gt), np.asarray(gp)


# --------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=ALL_MODELS)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--probe-seeds", type=int, default=5)
    p.add_argument("--probe-steps", type=int, default=5000)
    p.add_argument("--selection-steps", type=int, default=2000,
                   help="steps for the layer-selection fits; shorter than the "
                        "final fit because it only has to RANK layers")
    p.add_argument("--probe-patience", type=int, default=500)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--inner-folds", type=int, default=3,
                   help="inner grouped folds used to choose the layer")
    p.add_argument("--inner-frac", type=float, default=0.2,
                   help="only used by the degraded stratified fallback")
    p.add_argument("--force", action="store_true")
    p.add_argument("--force-extract", action="store_true")
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "emb_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("Phase3_CVProbe", "INFO")
    with open(manifest_path) as f:
        manifest = json.load(f)

    protocol = manifest["protocol"]
    classes = manifest["individuals"]
    class_to_idx = manifest["class_to_idx"]
    num_classes = len(classes)
    items = manifest["items"]
    n_folds, n_repeats = manifest["n_folds"], manifest["n_repeats"]

    result_path = out_dir / f"cv_{protocol}_{args.model}.json"
    if result_path.exists() and not args.force:
        logger.info(f"result exists: {result_path} -- pass --force to redo")
        return

    logger.info("=" * 72)
    logger.info(f"CV PROBE - {args.model} - {protocol}")
    logger.info("=" * 72)
    logger.info(f"manifest : {manifest_path}")
    logger.info(f"task     : {num_classes} individuals, chance "
                f"{1 / num_classes:.3f}, {len(items)} bouts")
    logger.info(f"folds    : {n_folds} x {n_repeats} repeat(s)")

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    # ---- embeddings: once per (model, audio), shared by BOTH protocols ----
    X, kept = load_or_extract(items, args, cache_dir, logger)
    kept_items = [items[i] for i in kept]
    y = np.asarray([class_to_idx[it["individual"]] for it in kept_items])
    groups = np.asarray([it["file"] for it in kept_items])
    folds = np.asarray([it["folds"] for it in kept_items])
    n_layers = X.shape[1]
    layers = list(range(n_layers))

    inventory = manifest.get("inventory", {})
    cohort_idx = [i for i, c in enumerate(classes)
                  if inventory.get(c, {}).get("bouts", 0) >= COHORT_MIN_BOUTS]
    logger.info(f"cohort100: {len(cohort_idx)} individuals with "
                f">= {COHORT_MIN_BOUTS} bouts")

    seeds = list(range(42, 42 + args.probe_seeds))
    repeats_out, t0 = [], time.time()

    for r in range(n_repeats):
        pooled = np.zeros((len(y), num_classes), dtype=np.float64)
        fold_rows, chosen = [], []

        for k in range(n_folds):
            te = np.flatnonzero(folds[:, r] == k)
            tr = np.flatnonzero(folds[:, r] != k)
            if len(te) == 0:
                continue

            layer, sel = select_layer(X[tr], y[tr], groups[tr], num_classes,
                                      layers, device, args, logger)
            chosen.append(layer)

            w = class_weights_for(y[tr], num_classes)
            probs = fit_and_prob(X[tr, layer], y[tr], groups[tr],
                                 X[te, layer], num_classes, w, device, seeds,
                                 args.probe_steps, args.probe_patience,
                                 args.val_frac)
            pooled[te] = probs

            s = score(y[te], probs.argmax(1), classes, cohort_idx)
            s.update(fold=k, layer=layer, n_test=len(te),
                     n_individuals_tested=int(len(np.unique(y[te]))),
                     degraded_inner_split=sel["degraded_inner_split"],
                     inner_folds_used=sel["inner_folds_used"])
            fold_rows.append(s)
            logger.info(f"  r{r} fold {k:2d}: layer {layer:2d}  "
                        f"n={len(te):4d}  F1 {s['f1_macro']:.4f}  "
                        f"acc {s['accuracy']:.4f}")

        pooled_pred = pooled.argmax(1)
        gt, gp = aggregate(pooled, y, groups)
        f1s = [f["f1_macro"] for f in fold_rows]

        repeats_out.append({
            "repeat": r,
            "pooled": score(y, pooled_pred, classes, cohort_idx),
            "pooled_aggregated": score(gt, gp, classes, cohort_idx),
            "n_recordings": int(len(gt)),
            "per_fold": fold_rows,
            "fold_f1_mean": round(float(np.mean(f1s)), 4),
            "fold_f1_std": round(float(np.std(f1s)), 4),
            "layers_chosen": chosen,
            "per_individual": per_individual(y, pooled_pred, classes,
                                             inventory),
        })
        logger.info(f"  r{r} POOLED: F1 {repeats_out[-1]['pooled']['f1_macro']:.4f}"
                    f"  acc {repeats_out[-1]['pooled']['accuracy']:.4f}"
                    f"  aggregated F1 "
                    f"{repeats_out[-1]['pooled_aggregated']['f1_macro']:.4f}")

    pooled_f1 = [r["pooled"]["f1_macro"] for r in repeats_out]
    agg_f1 = [r["pooled_aggregated"]["f1_macro"] for r in repeats_out]

    out = {
        "model": args.model,
        "protocol": protocol,
        "manifest": str(manifest_path),
        "audio_subdir": manifest.get("audio_subdir"),
        "num_classes": num_classes,
        "chance": round(1 / num_classes, 4),
        "classes": classes,
        "n_bouts": int(len(y)),
        "n_bouts_dropped": int(len(items) - len(kept)),
        "n_layers": n_layers,
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "cohort100_individuals": [classes[i] for i in cohort_idx],
        "probe": {
            "type": "linear, full-batch Adam lr 1e-3, class-weighted CE",
            "seeds": seeds,
            "steps": args.probe_steps,
            "selection_steps": args.selection_steps,
            "patience": args.probe_patience,
            "val_frac": args.val_frac,
            "inner_folds": args.inner_folds,
            "inner_frac": args.inner_frac,
            "layer_selection": f"{args.inner_folds}-fold grouped CV inside the "
                               f"training rows of each outer fold, macro-F1 "
                               f"averaged over inner folds; test never used",
            "seed_combination": "softmax averaged over seeds before argmax",
        },
        "split_exceptions": manifest.get("split_exceptions", {}),
        "leakage_note": manifest.get("leakage_note"),
        "repeats": repeats_out,
        "summary": {
            "pooled_f1_macro_mean": round(float(np.mean(pooled_f1)), 4),
            "pooled_f1_macro_std": round(float(np.std(pooled_f1)), 4),
            "pooled_accuracy_mean": round(
                float(np.mean([r["pooled"]["accuracy"] for r in repeats_out])), 4),
            "pooled_f1_cohort100_mean": round(float(np.mean(
                [r["pooled"]["f1_macro_cohort100"] for r in repeats_out])), 4),
            "aggregated_f1_macro_mean": round(float(np.mean(agg_f1)), 4),
            "aggregated_f1_macro_std": round(float(np.std(agg_f1)), 4),
            "fold_f1_std_mean": round(float(np.mean(
                [r["fold_f1_std"] for r in repeats_out])), 4),
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(result_path, "w") as f:
        json.dump(out, f, indent=2)

    s = out["summary"]
    logger.info("")
    logger.info(f"POOLED over {n_repeats} repeat(s): "
                f"F1 {s['pooled_f1_macro_mean']:.4f} "
                f"+/- {s['pooled_f1_macro_std']:.4f}  "
                f"acc {s['pooled_accuracy_mean']:.4f}  "
                f"cohort100 F1 {s['pooled_f1_cohort100_mean']:.4f}  "
                f"aggregated F1 {s['aggregated_f1_macro_mean']:.4f}")
    logger.info(f"wrote {result_path}")


if __name__ == "__main__":
    main()
