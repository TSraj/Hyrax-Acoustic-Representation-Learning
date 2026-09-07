#!/usr/bin/env python3
"""
Phase 3 - Step 24: per-layer hyrax probe, base encoder vs species-adapted.

THE QUESTION
------------
Does adapting a speech encoder on 7 animal species (hyrax excluded) improve its
representation of hyrax individual identity -- a species it never saw?

Every previous attempt answered ~0.000, but that was structural. Hyrax peaks at
hidden_states[0], which is the CNN front-end output, and both the LoRA path and
phase2_05 kept that stack frozen. The best-scoring layer could not change. Step
23 unfroze it, so a non-zero delta is now possible for the first time.

This script measures the delta layer by layer.

WHAT IT DOES
------------
For one (model, condition) cell:

  1. loads the encoder -- pretrained weights, or the step-23 adapted checkpoint
  2. extracts MEAN-POOLED embeddings from EVERY layer, in a single forward pass
     per window (hidden_states[0] is the CNN front-end; 1..N are the transformer
     blocks)
  3. trains a converged linear probe per layer, several seeds, early stopping on
     an internal split of TRAIN
  4. writes per-layer test macro-F1, mean and SD across seeds

The probe is imported from phase3_20_probe_audit, not reimplemented, so these
numbers sit on exactly the same measurement as the corrected baselines in
outputs/phase3/results_corrected/.

INPUT UNIT is chosen by the manifest, per item:

  bout manifests (phase3_27) carry `start`/`end`, and the exact ground-truth
  segment is sliced at whatever length it is -- no window, no stride, long bouts
  kept whole, one embedding per bout. This is the honest unit: it is what the
  animal actually produced.

  legacy manifests have no timings, so 5 s windows at 2.5 s stride are used,
  matching phase3_03. Those windows span 3-4 CONCATENATED bouts with artificial
  splices, which is why the bout manifests exist -- but the regime is preserved
  so the published baselines stay reproducible.

Do not compare the two regimes as if they measured the same task.

TEST IS NEVER USED for stopping or model selection. Where the manifest has no
val split (session-holdout does not), TRAIN is split 80/20 stratified.

MEMORY: hidden states are mean-pooled immediately after each forward pass and
the full-resolution activations are dropped, so peak memory is one window's
worth regardless of how many layers the model has.

USAGE
-----
    # frozen baseline
    python scripts/phase3_24_hyrax_layer_probe.py --model xls_r --condition base

    # species-adapted
    python scripts/phase3_24_hyrax_layer_probe.py --model xls_r --condition adapted \
        --checkpoint outputs/phase3/species7_finetune/xls_r/checkpoints/best_model.pth

Embeddings are cached, so re-probing with different seeds costs no GPU time.
"""

import argparse
import hashlib
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score,
                             recall_score)
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from src.utils.audio_utils import load_audio  # noqa: E402
from src.utils.logging_utils import setup_logger  # noqa: E402

# the probe itself comes from the audit, so this is the SAME measurement that
# produced the corrected frozen baselines -- not a lookalike
from phase3_20_probe_audit import (  # noqa: E402
    MODEL_IDS,
    SAMPLE_RATE,
    SINGLE_EMBEDDING_MODELS,
    evaluate,
    fit_probe,
    model_class,
    resolve,
    stratified_split,
)

WINDOW_SECONDS = 5.0
STRIDE_SECONDS = 2.5
# phase3_03.extract_embedding truncates non-windowed audio to the first 30 s
MAX_FILE_SECONDS = 30

# Models that do NOT load through HuggingFace and need the phase3_28 wrapper.
# Kept separate from MODEL_IDS on purpose: MODEL_IDS maps to HF ids and is
# consumed by phase3_20's own Extractor, which would try
# Wav2Vec2Model.from_pretrained on anything it finds there.
AVEX_MODELS = {"aves2_eat_bio", "aves2_eat_all"}
ALL_MODELS = sorted(set(MODEL_IDS) | AVEX_MODELS)


def build_extractor(model_name, checkpoint, logger, pooling="masked_mean",
                    pad_mode="zero", batch_size=16):
    """Pick the loader for this model family. HF models are unchanged."""
    if model_name in AVEX_MODELS:
        from phase3_28_avex_extractor import AvesLayerExtractor
        return AvesLayerExtractor(model_name, checkpoint, logger,
                                  pooling=pooling, pad_mode=pad_mode,
                                  batch_size=batch_size)
    if model_name in SINGLE_EMBEDDING_MODELS:
        return EcapaExtractor(model_name, checkpoint, logger)
    return LayerExtractor(model_name, checkpoint, logger)


def layer_tag(model_name, layer):
    """Index 0 means different things across architectures -- say which."""
    if model_name in SINGLE_EMBEDDING_MODELS:
        return "utterance emb"
    if layer != 0:
        return f"block {layer - 1}"
    return "patch embed" if model_name in AVEX_MODELS else "CNN front-end"


def chunks_for_item(item, audio, window, stride, per_file=False):
    """The INPUT UNIT for one manifest entry -> (chunks, is_bout).

    Single source of truth, shared by every extractor. The AVES wrapper batches
    its forward passes and so cannot reuse LayerExtractor.extract_split
    directly; routing both through this function is what guarantees the two
    paths slice audio identically, rather than "identically as far as anyone
    checked".

      per_file=True      SPECIES manifests: ONE embedding per file, truncated
                         to the first 30 s. This is phase3_03:217's
                         non-windowed branch and must not be windowed -- the
                         published species numbers (XLS-R 0.969, HuBERT 0.962)
                         were measured this way, and windowing changes the
                         dataset size about fourfold.
      start/end present  bout manifest (phase3_27): the exact ground-truth
                         segment, whatever length it is. No window, no stride.
      start/end absent   legacy manifests: 5 s windows at 2.5 s stride, the
                         phase3_03 regime the published baselines used.
    """
    if per_file:
        return ([] if len(audio) == 0
                else [audio[:int(MAX_FILE_SECONDS * SAMPLE_RATE)]]), False

    if "start" in item and "end" in item:
        a = max(0, int(float(item["start"]) * SAMPLE_RATE))
        b = min(len(audio), int(float(item["end"]) * SAMPLE_RATE))
        return ([] if b <= a else [audio[a:b]]), True

    chunks = [audio[s:s + window]
              for s in range(0, len(audio) - window + 1, stride)]
    if not chunks and len(audio) > 0:
        chunks = [audio]
    return chunks, False


@lru_cache(maxsize=8)
def _load_cached(path):
    """Bout manifests reference the SAME recording dozens of times.

    Without this, extracting ~3000 bouts means ~3000 full-file decodes of a
    handful of long wavs. Manifests are emitted grouped by individual and
    ordered by (file, start), so a small cache gets nearly every hit.
    """
    audio, _ = load_audio(path, target_sr=SAMPLE_RATE, mono=True)
    return audio


class LayerExtractor:
    """Mean-pooled embedding from every layer, one forward pass per window."""

    def __init__(self, model_name, checkpoint, logger):
        from transformers import Wav2Vec2FeatureExtractor

        self.logger = logger
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")

        model_id = MODEL_IDS[model_name]
        # shared with phase3_20 so the two scripts cannot drift onto different
        # classes for the same checkpoint
        cls = model_class(model_name)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.model = cls.from_pretrained(model_id, use_safetensors=True)
        self.model.config.layerdrop = 0.0

        if checkpoint is not None:
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            missing, unexpected = self.model.load_state_dict(
                ck["backbone_state_dict"], strict=False
            )
            if missing:
                raise RuntimeError(f"checkpoint is missing {len(missing)} keys: {missing[:5]}")
            logger.info(f"loaded adapted weights from epoch {ck['epoch']} "
                        f"(val macro-F1 {ck['val_f1_macro']:.4f})")
            if unexpected:
                logger.info(f"  ignored {len(unexpected)} unexpected keys")
            cfg = ck.get("config", {})
            logger.info(f"  adaptation config: {cfg}")
            if cfg.get("freeze_conv", True):
                logger.warning("  checkpoint was trained with a FROZEN conv stack -- "
                               "layer 0 will be identical to base by construction")

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device).eval()

        self.num_layers = self.model.config.num_hidden_layers + 1  # +1 for the CNN output
        logger.info(f"{model_name} on {self.device}: {self.num_layers} layers "
                    f"(0 = CNN front-end, 1..{self.num_layers - 1} = transformer blocks)")

    def embed_all_layers(self, audio):
        """-> (num_layers, hidden) mean-pooled over time."""
        inputs = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE,
                                        return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            # pool immediately; the full activations are dropped here
            pooled = torch.stack([h.mean(dim=1).squeeze(0) for h in out.hidden_states])
        return pooled.cpu().numpy()

    def extract_split(self, items, class_to_idx, split, label_key="individual"):
        """One embedding per input unit -> (X, y, g).

        `g` is the SOURCE RECORDING each row came from. It exists so score-level
        aggregation can average a probe's per-bout scores over the bouts of one
        recording. Without it the row-to-recording link is gone by the time the
        probe runs, and grouping would have to be inferred from row order --
        which is only correct while nothing is ever dropped.

        THREE regimes, chosen automatically:

          start/end in the item   bout-level hyrax manifest (phase3_27). The
                                  exact GT segment, whatever its length.
          label_key == 'species'  ONE embedding per FILE, truncated to the first
                                  30 s -- phase3_03:217's non-windowed branch.
                                  This must not be windowed: the published
                                  species numbers (XLS-R 0.969, HuBERT 0.962)
                                  were measured this way, and windowing changes
                                  the dataset size about fourfold.
          otherwise               legacy concatenated hyrax manifests: 5 s
                                  windows at 2.5 s stride.
        """
        window = int(WINDOW_SECONDS * SAMPLE_RATE)
        stride = int(STRIDE_SECONDS * SAMPLE_RATE)
        max_file = int(MAX_FILE_SECONDS * SAMPLE_RATE)
        per_file = label_key == "species"

        embs, labels, groups = [], [], []
        load_failed, embed_failed = 0, 0
        t0 = time.time()
        n_bout_items = 0

        for item in tqdm(items, desc=split, leave=False):
            try:
                audio = _load_cached(str(resolve(item["file"])))
            except Exception:
                load_failed += 1
                continue

            label = class_to_idx[item[label_key]]

            chunks, is_bout = chunks_for_item(item, audio, window, stride, per_file)
            if not chunks:
                embed_failed += 1
                continue
            n_bout_items += int(is_bout)

            for chunk in chunks:
                try:
                    embs.append(self.embed_all_layers(chunk))
                except Exception:
                    embed_failed += 1
                    continue
                labels.append(label)
                # appended in lockstep with the embedding, INSIDE the try's
                # success path, so a dropped chunk can never shift the mapping
                groups.append(item["file"])

        if load_failed:
            self.logger.warning(f"  {split}: {load_failed} files failed to load")
        if embed_failed:
            self.logger.warning(f"  {split}: {embed_failed} windows failed to embed")
        if not embs:
            raise RuntimeError(f"no embeddings for split {split}")

        X = np.stack(embs).astype(np.float32)  # (n_samples, n_layers, hidden)
        y = np.asarray(labels)
        g = np.asarray(groups)
        unit = "files" if per_file else ("bouts" if n_bout_items else "windows")
        self.logger.info(f"  {split}: {len(y)} {unit} from {len(set(groups))} "
                         f"recordings, {X.shape[1]} layers, "
                         f"dim {X.shape[2]} ({time.time() - t0:.0f}s)")
        return X, y, g


class EcapaExtractor(LayerExtractor):
    """ECAPA-TDNN: ONE utterance embedding, not a layer profile.

    ECAPA is the most relevant baseline in the set -- it is the only encoder
    here purpose-built for speaker identity -- but it is a SpeechBrain
    EncoderClassifier with no `hidden_states` interface. Its channels are
    aggregated by attentive statistics pooling into a single 192-d vector, and
    there is no stack of per-block outputs comparable to a transformer's.

    So it is modelled as a ONE-LAYER encoder: `num_layers == 1` and
    `embed_all_layers` returns shape (1, 192). Everything downstream -- the
    per-layer loop, best-layer selection, the per-individual breakdown -- then
    works untouched and simply finds one layer. `single_embedding` is recorded
    in the summary so a reader cannot mistake "best layer 0" here for the
    "layer 0 = CNN front-end" finding that applies to the wav2vec2 family.

    extract_split is INHERITED, not reimplemented, so ECAPA slices bouts
    through exactly the same code path as every other model.
    """

    def __init__(self, model_name, checkpoint, logger):
        if checkpoint is not None:
            raise ValueError(
                "ECAPA is evaluated zero-shot only: there is no species-adapted "
                "ECAPA checkpoint in this experiment."
            )
        from speechbrain.inference.speaker import EncoderClassifier

        self.logger = logger
        # SpeechBrain's MPS support is patchy; CUDA or CPU only.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_id = MODEL_IDS[model_name]
        self.model = EncoderClassifier.from_hparams(
            source=model_id, savedir="pretrained_models/ecapa_tdnn",
            run_opts={"device": self.device},
        )
        self.feature_extractor = None
        self.num_layers = 1
        logger.info(f"{model_name} on {self.device}: single 192-d utterance "
                    f"embedding (attentive stat pooling) -- NO layer profile")

    def embed_all_layers(self, audio):
        """-> (1, hidden). Mirrors phase3_20.Extractor.embed's ECAPA branch."""
        with torch.no_grad():
            e = self.model.encode_batch(
                torch.FloatTensor(audio).unsqueeze(0).to(self.device))
            e = e.squeeze(0)
            if e.dim() > 1:
                e = e.mean(dim=0)
        return e.cpu().numpy()[None, :]


def predict(clf, X, device):
    clf.eval()
    with torch.no_grad():
        return clf(torch.FloatTensor(X).to(device)).argmax(dim=1).cpu().numpy()


def macro_pr(y_true, y_pred):
    """Precision and recall alongside F1.

    Recall answers 'how many of this individual's calls did we miss', precision
    answers 'when we said it was this individual, how often were we right'. F1
    alone hides which of the two is failing.
    """
    return {
        "precision_macro": float(precision_score(y_true, y_pred, average="macro",
                                                 zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def predict_proba(clf, X, device):
    """Per-row class probabilities. Softmax, so scores are comparable across
    rows before they are averaged -- raw logits are not."""
    clf.eval()
    with torch.no_grad():
        logits = clf(torch.FloatTensor(X).to(device))
        return torch.softmax(logits, dim=1).cpu().numpy()


def aggregate_by_group(probs, y_true, groups):
    """Score-level aggregation -> one decision per group.

    THE SECOND EVALUATION MODE. Per-bout asks "can you name the animal from one
    1.4 s call"; this asks "can you name it from all the calls in a recording",
    which is what a deployed system would actually have.

    Mean of the probability vectors over the group, then a single argmax. Mean
    of probabilities (not of logits, and not a majority vote) because it keeps
    a confident bout weighted above an uncertain one while staying bounded.

    A group's label is taken from its first row and asserted to be unanimous:
    a recording holds one individual by construction, so a mixed group means
    the group ids and labels have desynchronised, and that must fail loudly
    rather than quietly score against a fabricated label.

    -> (group_true, group_pred, group_keys)
    """
    order = {}
    for i, gid in enumerate(groups):
        order.setdefault(gid, []).append(i)

    keys = list(order)
    g_true = np.empty(len(keys), dtype=y_true.dtype)
    g_pred = np.empty(len(keys), dtype=np.int64)
    for j, gid in enumerate(keys):
        rows = order[gid]
        labels = y_true[rows]
        if not np.all(labels == labels[0]):
            raise RuntimeError(
                f"group {gid!r} spans {len(set(labels.tolist()))} individuals. "
                f"A recording is one animal by construction, so the group ids "
                f"and labels are misaligned -- refusing to score this."
            )
        g_true[j] = labels[0]
        g_pred[j] = int(probs[rows].mean(axis=0).argmax())
    return g_true, g_pred, keys


def group_metrics(probs, y_true, groups, num_classes):
    """Aggregated accuracy / macro-F1 / precision / recall, plus group counts."""
    g_true, g_pred, keys = aggregate_by_group(probs, y_true, groups)
    out = {
        "accuracy": float((g_true == g_pred).mean()),
        "n_groups": len(keys),
    }
    out.update(macro_pr(g_true, g_pred))
    # how many groups each individual actually got: a macro average over
    # classes with one or two groups is volatile, and the reader must see that
    counts = {int(c): int((g_true == c).sum()) for c in np.unique(g_true)}
    out["groups_per_class"] = counts
    out["min_groups_per_class"] = min(counts.values()) if counts else 0
    return out


def probe_layer(train_X, train_y, test_X, test_y, num_classes, weights,
                device, seeds, steps, patience, val_frac, classes=None,
                test_groups=None):
    """Converged probe, one run per seed. Test never touches selection."""
    runs, last_pred = [], None
    for seed in seeds:
        keep, held = stratified_split(train_y, val_frac, seed)
        clf, best_step = fit_probe(
            train_X[keep], train_y[keep], num_classes, weights, device,
            steps=steps, val_X=train_X[held], val_y=train_y[held],
            patience=patience, seed=seed,
        )
        res = evaluate(clf, test_X, test_y, device)
        last_pred = predict(clf, test_X, device)
        res.update(macro_pr(test_y, last_pred))
        res["train_f1_macro"] = evaluate(clf, train_X[keep], train_y[keep], device)["f1_macro"]
        res["best_step"] = best_step
        res["seed"] = seed

        # ---- second evaluation mode: one decision per source recording ----
        if test_groups is not None:
            probs = predict_proba(clf, test_X, device)
            # SELF-CHECK: with every row in its own group, aggregation must
            # reproduce the per-bout result exactly. If this ever fails the
            # grouping or averaging is wrong, and every aggregated number in
            # this run is untrustworthy -- so it fails here, not in a figure.
            solo = aggregate_by_group(probs, test_y, np.arange(len(test_y)))[1]
            if not np.array_equal(solo, last_pred):
                raise RuntimeError(
                    "aggregation self-check failed: groups of one did not "
                    "reproduce the per-bout predictions."
                )
            res["grouped"] = group_metrics(probs, test_y, test_groups, num_classes)
        runs.append(res)

    def m(k):
        return float(np.mean([r[k] for r in runs]))

    out = {
        "f1_macro_mean": m("f1_macro"),
        "f1_macro_std": float(np.std([r["f1_macro"] for r in runs])),
        "f1_macro_runs": [r["f1_macro"] for r in runs],
        "precision_macro_mean": m("precision_macro"),
        "recall_macro_mean": m("recall_macro"),
        "train_f1_macro_mean": m("train_f1_macro"),
        "accuracy_mean": m("accuracy"),
        "balanced_accuracy_mean": m("balanced_accuracy"),
        "best_step_mean": m("best_step"),
        "runs": runs,
    }

    # aggregated mode, averaged over the same seeds. Reported ALONGSIDE the
    # per-bout numbers above, never replacing them.
    if test_groups is not None and "grouped" in runs[0]:
        gm = [r["grouped"] for r in runs]
        out["grouped"] = {
            "f1_macro_mean": float(np.mean([r["f1_macro"] for r in gm])),
            "f1_macro_std": float(np.std([r["f1_macro"] for r in gm])),
            "precision_macro_mean": float(np.mean([r["precision_macro"] for r in gm])),
            "recall_macro_mean": float(np.mean([r["recall_macro"] for r in gm])),
            "accuracy_mean": float(np.mean([r["accuracy"] for r in gm])),
            "n_groups": gm[0]["n_groups"],
            "groups_per_class": gm[0]["groups_per_class"],
            "min_groups_per_class": gm[0]["min_groups_per_class"],
        }

    # per-individual breakdown from the last seed: which animals are being
    # missed, and how they are confused
    if classes is not None and last_pred is not None:
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
    p = argparse.ArgumentParser(description="Per-layer hyrax probe, base vs adapted")
    p.add_argument("--model", required=True, choices=ALL_MODELS)
    p.add_argument("--condition", required=True, choices=["base", "adapted"])
    p.add_argument("--checkpoint", default=None,
                   help="required when --condition adapted")
    p.add_argument("--manifest", default=None,
                   help="default: outputs/phase3/manifests/hyrax_id_session_holdout.json")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--cache-dir", default=None)

    p.add_argument("--probe-seeds", type=int, default=5,
                   help="probe seeds per layer; the audit's single seed was a "
                        "stated caveat, so this reports mean +- SD")
    p.add_argument("--probe-steps", type=int, default=5000)
    p.add_argument("--probe-patience", type=int, default=500)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--layers", default=None,
                   help="comma-separated subset, e.g. 0,1,2 (default: all)")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="re-probe even if the result JSON already exists")

    # avex-only knobs. Ignored by the HF models, whose pooling is unchanged.
    p.add_argument("--pooling", default="masked_mean",
                   choices=["masked_mean", "unmasked_mean", "cls"],
                   help="AVES only. masked_mean is the primary setting; the "
                        "others exist for sensitivity checks and are tagged "
                        "into the output filename so they cannot overwrite it.")
    p.add_argument("--pad-mode", default="zero", choices=["zero", "tile"],
                   help="AVES only. zero = real bout + padding (primary). "
                        "tile = repeat the bout to fill the 10.24s canvas, "
                        "which is a DIFFERENT stimulus and is a sensitivity "
                        "check only.")
    p.add_argument("--batch-size", type=int, default=16,
                   help="AVES only; the canvas is fixed so batching is free.")
    args = p.parse_args()

    if args.condition == "adapted" and not args.checkpoint:
        raise SystemExit("--condition adapted requires --checkpoint")
    if args.model in AVEX_MODELS and args.condition == "adapted":
        raise SystemExit(f"{args.model} is evaluated ZERO-SHOT only; there is "
                         f"no adapted AVES cell in this experiment.")
    if args.model in SINGLE_EMBEDDING_MODELS and args.condition == "adapted":
        raise SystemExit(f"{args.model} is evaluated ZERO-SHOT only; there is "
                         f"no adapted ECAPA cell in this experiment.")

    root = SCRIPT_DIR.parent
    with open(root / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)
    out_root = Path(config["paths"]["output_dir"])

    manifest_path = Path(args.manifest) if args.manifest else \
        out_root / "phase3" / "manifests" / "hyrax_id_session_holdout.json"
    # one probe directory PER adaptation experiment -- the frozen "base" cells are
    # identical across experiments and can simply be copied over to skip the GPU
    output_dir = Path(args.output_dir) if args.output_dir else \
        out_root / "phase3" / "hyrax_probe_adapt_species_id"
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "emb_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("Phase3_HyraxLayerProbe", config["experiment"]["log_level"])
    cell = f"{args.model}_{args.condition}"
    # A sensitivity run must never land on the primary result's filename or its
    # embedding cache, so any non-default pooling/padding is tagged into the cell.
    if args.model in AVEX_MODELS:
        variant = []
        if args.pooling != "masked_mean":
            variant.append(args.pooling)
        if args.pad_mode != "zero":
            variant.append(args.pad_mode)
        if variant:
            cell += "_" + "_".join(variant)

    logger.info("=" * 72)
    logger.info(f"PER-LAYER PROBE - {cell}")
    logger.info("=" * 72)

    result_path = output_dir / f"layer_probe_{cell}.json"
    if result_path.exists() and not args.force_extract and not args.force:
        logger.info(f"result already exists: {result_path}")
        logger.info("nothing to do -- pass --force to redo it")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    # species manifests key on 'species', hyrax manifests on 'individual'
    if "species" in manifest and "species_to_idx" in manifest:
        label_key = "species"
        classes = manifest["species"]
    else:
        label_key = "individual"
        classes = manifest["individuals"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    weights = torch.FloatTensor([manifest["class_weights"][c] for c in classes])
    splits = manifest["splits"]

    logger.info(f"manifest: {manifest_path.name}")
    logger.info(f"task:     {manifest['task']}  ({num_classes} classes, "
                f"chance {1 / num_classes:.3f})")
    logger.info(f"splits:   " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    # ---------------------------------------------------------------- extract
    # The cache key includes a fingerprint of the MANIFEST, not just the cell.
    # Without it, running a bout manifest into a directory that already holds
    # window embeddings would silently reuse the wrong features and report a
    # confident, wrong number.
    fp = hashlib.sha1(json.dumps(
        {"task": manifest.get("task"),
         "unit": manifest.get("unit", "file" if label_key == "species" else "window"),
        "label_key": label_key,
         "counts": {k: len(v) for k, v in splits.items()},
         "first": splits["train"][0] if splits.get("train") else None},
        sort_keys=True, default=str).encode()).hexdigest()[:10]
    # PER-SPLIT caches, not one combined file. A cell that dies extracting the
    # test split used to lose the train split too, because nothing was written
    # until both finished. Species extraction is hours long, so that mattered:
    # the next job in the chain now reuses whatever completed.
    cache_train = cache_dir / f"{cell}_{fp}_train.npz"
    cache_test = cache_dir / f"{cell}_{fp}_test.npz"
    legacy_cache = cache_dir / f"{cell}_{fp}.npz"

    extractor_provenance = None
    extractor = None

    def _extractor():
        """Built once, and only if something actually needs extracting."""
        nonlocal extractor
        if extractor is None:
            extractor = build_extractor(
                args.model, args.checkpoint if args.condition == "adapted" else None,
                logger, pooling=args.pooling, pad_mode=args.pad_mode,
                batch_size=args.batch_size,
            )
        return extractor

    def recover_groups(split_name, y):
        """Group ids for a cache written before `g` was stored.

        Bout and file regimes emit exactly ONE row per manifest item, so row i
        is item i -- but ONLY if nothing was dropped. That is checked, not
        assumed: if the counts disagree the mapping is unknowable and we return
        None, which disables aggregation for this cell rather than silently
        grouping the wrong bouts together. Windowed regimes emit several rows
        per item and are never recoverable this way.
        """
        items = splits[split_name]
        if len(y) != len(items):
            logger.warning(
                f"  {split_name}: cache has {len(y)} rows for {len(items)} "
                f"manifest items -- group ids cannot be recovered, so "
                f"score-level aggregation is DISABLED for this cell. "
                f"Re-run with --force-extract to record them properly."
            )
            return None
        logger.info(f"  {split_name}: recovered group ids from the manifest "
                    f"({len(y)} rows == {len(items)} items)")
        return np.asarray([it["file"] for it in items])

    def load_or_extract(split_name, cache_path):
        if cache_path.exists() and not args.force_extract:
            logger.info(f"reusing cached {split_name}: {cache_path.name}")
            z = np.load(cache_path)
            X, y = z["X"], z["y"]
            g = z["g"] if "g" in z.files else recover_groups(split_name, y)
            return X, y, g
        X, y, g = _extractor().extract_split(splits[split_name], class_to_idx,
                                             split_name, label_key)
        tmp = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, X=X, y=y, g=g)
        tmp.replace(cache_path)      # atomic: a kill mid-write leaves no half file
        logger.info(f"cached {split_name} -> {cache_path.name}")
        return X, y, g

    if legacy_cache.exists() and not args.force_extract:
        # a cache written before the split was separated
        logger.info(f"loading combined cache: {legacy_cache.name}")
        z = np.load(legacy_cache)
        train_X, train_y, test_X, test_y = z["train_X"], z["train_y"], z["test_X"], z["test_y"]
        test_g = recover_groups("test", test_y)
    else:
        train_X, train_y, _ = load_or_extract("train", cache_train)
        test_X, test_y, test_g = load_or_extract("test", cache_test)

    if extractor is not None:
        if hasattr(extractor, "provenance"):
            extractor_provenance = extractor.provenance()
        if hasattr(extractor, "close"):
            extractor.close()

    logger.info(f"  train {train_X.shape}  test {test_X.shape}")

    n_layers = train_X.shape[1]
    layer_ids = ([int(x) for x in args.layers.split(",")] if args.layers
                 else list(range(n_layers)))

    # ---------------------------------------------------------------- probe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(42, 42 + args.probe_seeds))

    logger.info(f"\nprobing {len(layer_ids)} layers x {len(seeds)} seeds "
                f"(early stopping on {int(args.val_frac * 100)}% of train, "
                f"test never used for selection)")

    results = {}
    for layer in layer_ids:
        r = probe_layer(
            train_X[:, layer, :], train_y, test_X[:, layer, :], test_y,
            num_classes, weights, device, seeds,
            args.probe_steps, args.probe_patience, args.val_frac,
            classes=classes, test_groups=test_g,
        )
        results[str(layer)] = r
        tag = layer_tag(args.model, layer)
        agg = (f"   | grouped F1 {r['grouped']['f1_macro_mean']:.4f}"
               if "grouped" in r else "")
        logger.info(f"  layer {layer:>2} ({tag:<13}) "
                    f"F1 {r['f1_macro_mean']:.4f} +- {r['f1_macro_std']:.4f}  "
                    f"P {r['precision_macro_mean']:.4f}  "
                    f"R {r['recall_macro_mean']:.4f}   "
                    f"train F1 {r['train_f1_macro_mean']:.4f}{agg}")

    best = max(results.items(), key=lambda kv: kv[1]["f1_macro_mean"])
    logger.info(f"\nBEST layer {best[0]}: F1 {best[1]['f1_macro_mean']:.4f} "
                f"+- {best[1]['f1_macro_std']:.4f}  "
                f"precision {best[1]['precision_macro_mean']:.4f}  "
                f"recall {best[1]['recall_macro_mean']:.4f}")
    if "per_class" in best[1]:
        logger.info("  per individual (last seed):")
        for name, pc in sorted(best[1]["per_class"].items(),
                               key=lambda kv: -kv[1]["recall"]):
            logger.info(f"    {name:<10} P {pc['precision']:.3f}  R {pc['recall']:.3f}  "
                        f"F1 {pc['f1']:.3f}  n={pc['support']}")

    # Which AUDIO VERSION this cell used, read off the manifest's own paths
    # rather than inferred from a directory name. The original-vs-denoised
    # comparison lives or dies on not mixing these two up, and a result file
    # that cannot say which audio produced it is not comparable to anything.
    first_file = str(splits["train"][0]["file"]) if splits.get("train") else ""
    if "/BIODA/" in first_file:
        audio_version = "bioda_denoised"
    elif "/Audio/" in first_file:
        audio_version = "original"
    else:
        audio_version = "unknown"

    summary = {
        "model": args.model,
        "condition": args.condition,
        "checkpoint": args.checkpoint,
        "task": manifest["task"],
        "manifest": str(manifest_path),
        "audio_version": audio_version,
        "num_classes": num_classes,
        "chance": 1 / num_classes,
        "classes": classes,
        "n_layers": n_layers,
        "windowing": {"window_s": WINDOW_SECONDS, "stride_s": STRIDE_SECONDS},
        "probe": {
            "seeds": seeds,
            "max_steps": args.probe_steps,
            "patience": args.probe_patience,
            "val_frac": args.val_frac,
            "selection": "early stopping on internal split of train; test never used",
        },
        "unit": manifest.get("unit", "file" if label_key == "species" else "window"),
        "label_key": label_key,
        "split_by": manifest.get("split_by", "session"),
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        # ECAPA has no layer stack: one embedding, so n_layers == 1 and
        # "best_layer 0" means the utterance embedding, NOT a CNN front-end.
        "single_embedding": args.model in SINGLE_EMBEDDING_MODELS,
        "best_layer": int(best[0]),
        "best_f1_macro": best[1]["f1_macro_mean"],
        "best_precision_macro": best[1]["precision_macro_mean"],
        "best_recall_macro": best[1]["recall_macro_mean"],
        "layers": results,
    }

    # Aggregation is a SECOND evaluation mode, so it gets its own best layer.
    # The layer that is best per bout need not be the layer that is best once
    # scores are pooled, and quietly reusing the per-bout layer would hide that.
    if "grouped" in best[1]:
        gbest = max(results.items(),
                    key=lambda kv: kv[1]["grouped"]["f1_macro_mean"])
        summary["aggregation"] = {
            "mode": "score-level: mean of per-bout softmax within a group, "
                    "then one argmax per group",
            "group_by": "source recording",
            "n_groups": best[1]["grouped"]["n_groups"],
            "groups_per_class": best[1]["grouped"]["groups_per_class"],
            "min_groups_per_class": best[1]["grouped"]["min_groups_per_class"],
            "best_layer": int(gbest[0]),
            "best_f1_macro": gbest[1]["grouped"]["f1_macro_mean"],
            "best_accuracy": gbest[1]["grouped"]["accuracy_mean"],
            "best_precision_macro": gbest[1]["grouped"]["precision_macro_mean"],
            "best_recall_macro": gbest[1]["grouped"]["recall_macro_mean"],
            # the aggregated score AT the per-bout best layer, so the two modes
            # can also be compared at a single fixed layer
            "f1_macro_at_bout_best_layer": best[1]["grouped"]["f1_macro_mean"],
        }
        logger.info(
            f"\nAGGREGATED (one decision per recording, {best[1]['grouped']['n_groups']} groups)"
            f"\n  best layer {gbest[0]}: F1 {gbest[1]['grouped']['f1_macro_mean']:.4f}  "
            f"acc {gbest[1]['grouped']['accuracy_mean']:.4f}  "
            f"(per-bout best was layer {best[0]} at {best[1]['f1_macro_mean']:.4f})")
        if best[1]["grouped"]["min_groups_per_class"] < 3:
            logger.warning(
                f"  an individual has only {best[1]['grouped']['min_groups_per_class']} "
                f"test recording(s): the aggregated macro average is volatile here")
    if extractor_provenance is not None:
        summary["extractor"] = extractor_provenance
        summary["layer0_is_cnn_frontend"] = args.model not in AVEX_MODELS

    out_path = output_dir / f"layer_probe_{cell}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
