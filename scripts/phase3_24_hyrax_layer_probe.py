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

WINDOWING matches phase3_03 for hyrax tasks: 5 s windows, 2.5 s stride, one
embedding per window. Do not change it -- the corrected frozen baselines were
measured this way and the comparison is void otherwise.

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
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
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
    evaluate,
    fit_probe,
    resolve,
    stratified_split,
)

WINDOW_SECONDS = 5.0
STRIDE_SECONDS = 2.5


class LayerExtractor:
    """Mean-pooled embedding from every layer, one forward pass per window."""

    def __init__(self, model_name, checkpoint, logger):
        from transformers import (HubertModel, Wav2Vec2FeatureExtractor,
                                  Wav2Vec2Model, WavLMModel)

        self.logger = logger
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")

        model_id = MODEL_IDS[model_name]
        cls = {"hubert_base": HubertModel, "wavlm": WavLMModel}.get(model_name, Wav2Vec2Model)

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

    def extract_split(self, items, class_to_idx, split):
        """Windowed at 5 s / 2.5 s, matching phase3_03 for hyrax tasks."""
        window = int(WINDOW_SECONDS * SAMPLE_RATE)
        stride = int(STRIDE_SECONDS * SAMPLE_RATE)

        embs, labels = [], []
        load_failed, embed_failed = 0, 0
        t0 = time.time()

        for item in tqdm(items, desc=split, leave=False):
            try:
                audio, _ = load_audio(str(resolve(item["file"])),
                                      target_sr=SAMPLE_RATE, mono=True)
            except Exception:
                load_failed += 1
                continue

            label = class_to_idx[item["individual"]]
            chunks = [audio[s:s + window]
                      for s in range(0, len(audio) - window + 1, stride)]
            if not chunks and len(audio) > 0:
                chunks = [audio]

            for chunk in chunks:
                try:
                    embs.append(self.embed_all_layers(chunk))
                except Exception:
                    embed_failed += 1
                    continue
                labels.append(label)

        if load_failed:
            self.logger.warning(f"  {split}: {load_failed} files failed to load")
        if embed_failed:
            self.logger.warning(f"  {split}: {embed_failed} windows failed to embed")
        if not embs:
            raise RuntimeError(f"no embeddings for split {split}")

        X = np.stack(embs).astype(np.float32)  # (n_windows, n_layers, hidden)
        y = np.asarray(labels)
        self.logger.info(f"  {split}: {len(y)} windows, {X.shape[1]} layers, "
                         f"dim {X.shape[2]} ({time.time() - t0:.0f}s)")
        return X, y


def probe_layer(train_X, train_y, test_X, test_y, num_classes, weights,
                device, seeds, steps, patience, val_frac):
    """Converged probe, one run per seed. Test never touches selection."""
    runs = []
    for seed in seeds:
        keep, held = stratified_split(train_y, val_frac, seed)
        clf, best_step = fit_probe(
            train_X[keep], train_y[keep], num_classes, weights, device,
            steps=steps, val_X=train_X[held], val_y=train_y[held],
            patience=patience, seed=seed,
        )
        res = evaluate(clf, test_X, test_y, device)
        res["train_f1_macro"] = evaluate(clf, train_X[keep], train_y[keep], device)["f1_macro"]
        res["best_step"] = best_step
        res["seed"] = seed
        runs.append(res)

    f1s = [r["f1_macro"] for r in runs]
    return {
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
        "f1_macro_runs": f1s,
        "train_f1_macro_mean": float(np.mean([r["train_f1_macro"] for r in runs])),
        "accuracy_mean": float(np.mean([r["accuracy"] for r in runs])),
        "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in runs])),
        "best_step_mean": float(np.mean([r["best_step"] for r in runs])),
        "runs": runs,
    }


def main():
    p = argparse.ArgumentParser(description="Per-layer hyrax probe, base vs adapted")
    p.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
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
    args = p.parse_args()

    if args.condition == "adapted" and not args.checkpoint:
        raise SystemExit("--condition adapted requires --checkpoint")

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

    logger.info("=" * 72)
    logger.info(f"PER-LAYER HYRAX PROBE - {cell}")
    logger.info("=" * 72)

    with open(manifest_path) as f:
        manifest = json.load(f)

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
    cache = cache_dir / f"{cell}.npz"
    if cache.exists() and not args.force_extract:
        logger.info(f"loading cached embeddings: {cache}")
        z = np.load(cache)
        train_X, train_y, test_X, test_y = z["train_X"], z["train_y"], z["test_X"], z["test_y"]
        logger.info(f"  train {train_X.shape}  test {test_X.shape}")
    else:
        extractor = LayerExtractor(
            args.model, args.checkpoint if args.condition == "adapted" else None, logger
        )
        train_X, train_y = extractor.extract_split(splits["train"], class_to_idx, "train")
        test_X, test_y = extractor.extract_split(splits["test"], class_to_idx, "test")
        np.savez_compressed(cache, train_X=train_X, train_y=train_y,
                            test_X=test_X, test_y=test_y)
        logger.info(f"cached embeddings -> {cache}")

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
        )
        results[str(layer)] = r
        tag = "CNN front-end" if layer == 0 else f"block {layer - 1}"
        logger.info(f"  layer {layer:>2} ({tag:<13}) "
                    f"test F1 {r['f1_macro_mean']:.4f} +- {r['f1_macro_std']:.4f}   "
                    f"train F1 {r['train_f1_macro_mean']:.4f}")

    best = max(results.items(), key=lambda kv: kv[1]["f1_macro_mean"])
    logger.info(f"\nBEST layer {best[0]}: {best[1]['f1_macro_mean']:.4f} "
                f"+- {best[1]['f1_macro_std']:.4f}")

    summary = {
        "model": args.model,
        "condition": args.condition,
        "checkpoint": args.checkpoint,
        "task": manifest["task"],
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
        "n_train_windows": int(len(train_y)),
        "n_test_windows": int(len(test_y)),
        "best_layer": int(best[0]),
        "best_f1_macro": best[1]["f1_macro_mean"],
        "layers": results,
    }

    out_path = output_dir / f"layer_probe_{cell}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
