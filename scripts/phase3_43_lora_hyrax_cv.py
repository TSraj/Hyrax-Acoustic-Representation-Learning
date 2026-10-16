#!/usr/bin/env python3
"""
Phase 3 - Step 43: LoRA fine-tuning on hyrax individual ID, cross-validated.

The two encoders the frozen cross-validation picked (phase3_40):
WavLM Large (best speech encoder) and AVES 2 EAT (all) (better AVES).

WHAT IS TRAINED
---------------
  task        hyrax individual ID, 18 classes, the bout manifest the frozen
              runs used (Original audio)
  folds       5-fold by recording, REPETITION 0 ONLY -- the same five folds the
              frozen layer sweep (phase3_41) scored as per_repeat[0], so the
              frozen-vs-fine-tuned comparison is like for like
  LoRA        rank 16 on the attention projections of EVERY transformer block,
              applied to the weights themselves (LoRAWeight) -- see there for
              why peft cannot be used on WavLM
                WavLM Large   q/k/v/out_proj, 24 blocks
                AVES 2 EAT    attn.qkv / attn.proj, 12 blocks
  front end   trained fully at a low LR, for both encoders alike
                WavLM Large   CNN feature extractor + feature projection
                AVES 2 EAT    local_encoder (the mel patch embedding)
              LoRA cannot reach these, and the frozen runs found hyrax
              identity peaking at or near them; a frozen front end is what
              made the original LoRA run do nothing
  head        linear, on the masked-mean of the last block
  loss        class-weighted cross-entropy, weights from the fold's training
              rows (phase3_37.class_weights_for -- the frozen runs' function)

THE TEST FOLD IS NEVER TOUCHED BEFORE SCORING
---------------------------------------------
Early stopping and best-epoch selection use a VALIDATION SLICE: ~20% of the
training folds' rows, held out by whole RECORDING (phase3_37.grouped_holdout),
never a random bout split. The per-layer probe after training is fitted on
training-fold rows only, with its own grouped early-stop split, exactly as the
frozen probe was. No layer is chosen at all: every layer is reported.

WHAT IS SAVED PER FOLD  (outputs/phase3/lora_hyrax_cv/<model>/fold_<k>/)
------------------------------------------------------------------------
  checkpoint.pth   the trainable weights at the best validation epoch:
                   LoRA adapters + front end + head (the frozen rest is the
                   public checkpoint, so this plus the model id is the model)
  result.json      training history, best epoch, validation UAR, test UAR and
                   AUC end-to-end, and test UAR / AUC / F1 / accuracy at every
                   layer from the frozen-style linear probe
  probs.npz        predicted probabilities for every TEST bout: end-to-end head
                   (bouts x 18) and per-layer probe (layers x bouts x 18), with
                   file / start / end / label / flags per bout

Once all five folds exist, summary.json and layer_comparison.csv pool the
out-of-fold predictions -- as the frozen runs did -- and put the fine-tuned
per-layer UAR next to the frozen per_repeat[0] UAR for the same layer.

FLAGGED INDIVIDUALS
-------------------
  X0  one recording: its bouts are split WITHIN that recording, so train and
      test share it (manifest `leaky_split`). Optimistic.
  O1  one session, three recordings: train and test never share a recording
      but do share the session. Optimistic in the session sense.
Both are carried as per-bout flags and in every per-individual row.

TIMING TEST FIRST
-----------------
    python scripts/phase3_43_lora_hyrax_cv.py --model wavlm_large --timing

trains ONE epoch of fold 0, times a validation pass, an embedding pass and a
5-seed probe on one layer, and writes timing.json with a full-run estimate.

USAGE
-----
    python scripts/phase3_43_lora_hyrax_cv.py --model wavlm_large --fold 0
    python scripts/phase3_43_lora_hyrax_cv.py --model wavlm_large --summarize
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from src.utils.logging_utils import setup_logger  # noqa: E402

from phase3_20_probe_audit import SAMPLE_RATE, resolve  # noqa: E402
from phase3_24_hyrax_layer_probe import (  # noqa: E402
    AVEX_MODELS,
    STRIDE_SECONDS,
    WINDOW_SECONDS,
    _load_cached,
    build_extractor,
    chunks_for_item,
)
from phase3_37_cv_probe import (  # noqa: E402
    class_weights_for,
    extract_all,
    fit_and_prob,
    grouped_holdout,
    score,
)

# wavlm (Base+) is NOT a candidate: it is the same WavLMModel class at a size
# that fits on a laptop, kept only so the code path can be smoke-tested locally
MODELS = ["wavlm_large", "aves2_eat_all", "wavlm"]

# Regexes, not bare names: EAT's patch embedding is ALSO called `proj`
# (local_encoder.proj, a Conv2d), and a bare "proj" target would put a LoRA
# adapter on the front end instead of training it.
LORA_TARGETS = {
    "hf":   r".*encoder\.layers\.\d+\.attention\.(q_proj|k_proj|v_proj|out_proj)",
    "avex": r".*blocks\.\d+\.attn\.(qkv|proj)",
}
FRONT_END = {
    "hf":   ["feature_extractor", "feature_projection"],
    "avex": ["backbone.model.local_encoder"],
}
SINGLE_SESSION = {"O1"}     # flagged beside the manifest's leaky_split (X0)
FROZEN_LAYERS = "outputs/phase3/cv_original/layers/cv_layers_by_file_5fold_{}.json"


# ------------------------------------------------------------------ data

class BoutDataset(Dataset):
    """One manifest bout -> (waveform, label, row). Sliced by chunks_for_item,
    the function every frozen extractor uses, so the unit cannot drift."""

    def __init__(self, items, rows, y):
        self.items, self.rows, self.y = items, rows, y
        self.window = int(WINDOW_SECONDS * SAMPLE_RATE)
        self.stride = int(STRIDE_SECONDS * SAMPLE_RATE)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        audio = _load_cached(str(resolve(self.items[r]["file"])))
        chunks, is_bout = chunks_for_item(self.items[r], audio,
                                          self.window, self.stride)
        if not is_bout or len(chunks) != 1:
            raise RuntimeError(f"row {r} did not slice as exactly one bout")
        return np.asarray(chunks[0], dtype=np.float32), int(self.y[r]), r


def collate(batch):
    audios, labels, rows = zip(*batch)
    return list(audios), torch.LongTensor(labels), list(rows)


# ----------------------------------------------------------------- model

class LoRAWeight(nn.Module):
    """W -> W + (B @ A) * alpha / r, as a parametrization of the weight itself.

    WHY NOT peft. WavLM's attention never calls q_proj/k_proj/v_proj/out_proj:
    it reads their `.weight` tensors straight into
    F.multi_head_attention_forward. peft's LoRA wraps the module's forward, so
    on WavLM the adapters sit outside the computation and receive exactly zero
    gradient -- measured here, |grad| = 0.0 -- while training appears to run.
    (peft's target_parameters path fails the same way.) A parametrization
    changes what `.weight` RETURNS, so every reader, including that one, sees
    the adapted matrix. It is used for both encoders so they are adapted by
    the same mechanism.

    Standard LoRA init: A Kaiming-uniform, B zero, so training starts exactly
    at the pretrained model. Weight-space LoRA has no input dropout.
    """

    def __init__(self, out_features, in_features, r, alpha):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.scale = alpha / r

    def forward(self, W):
        return W + (self.lora_B @ self.lora_A).to(W.dtype) * self.scale


def apply_lora(model, pattern, r, alpha):
    """Parametrize the weight of every nn.Linear whose name matches `pattern`."""
    import re

    from torch.nn.utils import parametrize
    rx = re.compile(pattern)
    hits = [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and rx.fullmatch(n)]
    for _, m in hits:
        out_f, in_f = m.weight.shape
        parametrize.register_parametrization(
            m, "weight", LoRAWeight(out_f, in_f, r, alpha).to(m.weight.device))
    return [n for n, _ in hits]


class Encoder:
    """The frozen extractor, with LoRA injected and the front end unfrozen.

    Built FROM the frozen extractor on purpose: the per-layer embedding after
    training then runs through the exact code that produced the frozen
    numbers (phase3_37.extract_all -> extractor.embed_*), so any difference
    between frozen and fine-tuned is the weights, not the measurement.
    """

    def __init__(self, model_name, args, logger):
        self.kind = "avex" if model_name in AVEX_MODELS else "hf"
        self.ex = build_extractor(model_name, None, logger,
                                  batch_size=args.embed_batch_size)
        self.model = self.ex.model
        self.device = self.ex.device
        self.logger = logger
        # training input is capped at the AVES canvas for BOTH encoders, so
        # neither sees more context than the other; 14 of 4141 bouts are longer
        self.max_samples = int(10.24 * SAMPLE_RATE)

        adapted = apply_lora(self.model, LORA_TARGETS[self.kind],
                             args.lora_r, args.lora_alpha)

        for p in self.model.parameters():
            p.requires_grad = False
        self.lora_params = [p for n, p in self.model.named_parameters()
                            if "lora_" in n]
        for p in self.lora_params:
            p.requires_grad = True

        modules = dict(self.model.named_modules())
        self.front_params = []
        for name in FRONT_END[self.kind]:
            if name not in modules:
                raise RuntimeError(f"front-end module '{name}' not found")
            for p in modules[name].parameters():
                p.requires_grad = True
                self.front_params.append(p)

        n_adapted = len(adapted)
        self.hidden = (self.ex.hidden_size if self.kind == "avex"
                       else self.model.config.hidden_size)
        self.num_layers = self.ex.num_layers
        self.model.to(self.device)
        logger.info(f"LoRA (weight-space) r={args.lora_r} alpha={args.lora_alpha} on "
                    f"{n_adapted} attention projections "
                    f"({sum(p.numel() for p in self.lora_params):,} params)")
        logger.info(f"front end {FRONT_END[self.kind]} trainable "
                    f"({sum(p.numel() for p in self.front_params):,} params)")
        if n_adapted == 0:
            raise RuntimeError("LoRA matched no modules")

    def pooled_last(self, audios):
        """list of waveforms -> (B, hidden), masked mean of the last block."""
        audios = [a[:self.max_samples] for a in audios]
        if self.kind == "avex":
            from phase3_28_avex_extractor import (LAYER_MODULES, fit_to_canvas,
                                                  valid_time_columns)
            canv, cols = [], []
            for a in audios:
                wav, dur, _ = fit_to_canvas(a, self.ex.pad_mode)
                canv.append(wav)
                cols.append(valid_time_columns(dur))
            x = torch.from_numpy(np.stack(canv)).to(self.device)
            valid = torch.tensor(cols, dtype=torch.float32, device=self.device)
            self.ex._captured.clear()
            self.model(x)
            out = self.ex._pool(self.ex._captured[LAYER_MODULES[-1]], valid)
            self.ex._captured.clear()
            return out

        fe = self.ex.feature_extractor
        lengths = torch.LongTensor([len(a) for a in audios])
        # normalised at the TRUE length, then padded -- as the frozen
        # extractor normalises each bout on its own
        normed = [fe(a, sampling_rate=SAMPLE_RATE, return_tensors="np")
                  ["input_values"][0] for a in audios]
        x = np.zeros((len(normed), int(lengths.max())), dtype=np.float32)
        for i, a in enumerate(normed):
            x[i, :len(a)] = a
        x = torch.from_numpy(x).to(self.device)
        inputs = {"input_values": x}
        if fe.return_attention_mask:
            pos = torch.arange(x.size(1), device=self.device)
            inputs["attention_mask"] = (pos[None] < lengths.to(self.device)[:, None]).long()
        h = self.model(**inputs).last_hidden_state
        n = torch.as_tensor(self.model._get_feat_extract_output_lengths(lengths),
                            device=self.device).long().clamp(max=h.size(1))
        mask = (torch.arange(h.size(1), device=self.device)[None] < n[:, None])
        mask = mask.unsqueeze(-1).to(h.dtype)
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


def trainable_state(enc, head):
    return {"backbone": {n: p.detach().cpu().clone()
                         for n, p in enc.model.named_parameters() if p.requires_grad},
            "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}


def load_trainable_state(enc, head, state):
    params = dict(enc.model.named_parameters())
    missing = set(state["backbone"]) ^ {n for n, p in params.items() if p.requires_grad}
    if missing:
        raise RuntimeError(f"trainable-state mismatch: {sorted(missing)[:5]}")
    with torch.no_grad():
        for n, v in state["backbone"].items():
            params[n].copy_(v.to(params[n].device))
    head.load_state_dict(state["head"])


# --------------------------------------------------------------- helpers

def predict(enc, head, ds, args):
    """-> (probs, rows) for every bout in `ds`, no grad."""
    enc.eval()
    head.eval()
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)
    probs, rows = [], []
    with torch.no_grad():
        for audios, _, r in loader:
            probs.append(torch.softmax(head(enc.pooled_last(audios)), 1).cpu().numpy())
            rows.extend(r)
    return np.concatenate(probs).astype(np.float64), np.asarray(rows)


def grad_gate(enc, head, loader, criterion, logger):
    """One backward pass before any training: every adapted projection, the
    front end and the head must receive gradient, and nothing frozen may.

    This is the check that would have caught the zero-gradient LoRA on WavLM
    (see LoRAWeight). It costs one batch and refuses to run otherwise.
    """
    enc.train()
    head.train()
    audios, labels, _ = next(iter(loader))
    criterion(head(enc.pooled_last(audios)), labels.to(enc.device)).backward()
    named = list(enc.model.named_parameters())
    adapters = {n.split(".parametrizations")[0] for n, p in named
                if "lora_B" in n}
    live = {n.split(".parametrizations")[0] for n, p in named
            if "lora_B" in n and p.grad is not None and float(p.grad.abs().sum()) > 0}
    front = sum(float(p.grad.abs().sum()) for p in enc.front_params
                if p.grad is not None)
    leaked = [n for n, p in named if not p.requires_grad and p.grad is not None]
    head_ok = all(p.grad is not None for p in head.parameters())
    logger.info(f"gradient gate: adapters {len(live)}/{len(adapters)} live, "
                f"front end {'live' if front > 0 else 'DEAD'}, "
                f"frozen leaks {len(leaked)}, head {'live' if head_ok else 'DEAD'}")
    enc.model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    if live != adapters or front == 0 or leaked or not head_ok:
        raise RuntimeError("gradient gate FAILED -- refusing to train")


def run_epoch(enc, head, loader, optimizer, criterion, params, args, limit=None):
    enc.train()
    head.train()
    total, n = 0.0, 0
    for b, (audios, labels, _) in enumerate(loader):
        if limit is not None and b >= limit:
            break
        labels = labels.to(enc.device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(head(enc.pooled_last(audios)), labels)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        total += float(loss.detach()) * len(labels)
        n += len(labels)
    return total / max(n, 1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------------ fold

def setup(args, logger):
    manifest = json.load(open(args.manifest))
    if manifest["protocol"] != "by_file_5fold":
        raise SystemExit("this script runs the 5-fold-by-recording manifest only")
    if "Audio" not in manifest["items"][0]["file"].split("/"):
        raise SystemExit("manifest does not point at Original audio")
    items = manifest["items"]
    if args.limit_items:
        rng = random.Random(0)
        items = rng.sample(items, args.limit_items)
    classes = manifest["individuals"]
    y = np.asarray([manifest["class_to_idx"][it["individual"]] for it in items])
    groups = np.asarray([it["file"] for it in items])
    folds = np.asarray([it["folds"][args.repeat] for it in items])
    return manifest, items, classes, y, groups, folds


def train_fold(args, logger, k, timing=False):
    manifest, items, classes, y, groups, folds = setup(args, logger)
    C = len(classes)
    out = Path(args.output_dir) / args.model / f"fold_{k}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists() and not args.force and not timing:
        logger.info(f"fold {k} done ({out / 'result.json'}) -- skipping")
        return

    seed_everything(args.seed + k)
    te = np.flatnonzero(folds == k)
    tr_all = np.flatnonzero(folds != k)
    keep, held = grouped_holdout(groups[tr_all], y[tr_all], args.val_frac,
                                 args.seed + k)
    tr, va = tr_all[keep], tr_all[held]
    if set(tr) & set(te) or set(va) & set(te):
        raise RuntimeError("test rows leaked into training or validation")
    logger.info(f"fold {k}: train {len(tr)}  val {len(va)} "
                f"({len(set(groups[va]))} recordings)  test {len(te)}  "
                f"[repeat {args.repeat}]")

    enc = Encoder(args.model, args, logger)
    head = nn.Linear(enc.hidden, C).to(enc.device)
    w = class_weights_for(y[tr], C)
    criterion = nn.CrossEntropyLoss(weight=w.to(enc.device))
    groups_opt = [{"params": enc.lora_params, "lr": args.lr_lora},
                  {"params": enc.front_params, "lr": args.lr_front},
                  {"params": list(head.parameters()), "lr": args.lr_head}]
    params = [p for g in groups_opt for p in g["params"]]
    optimizer = optim.AdamW(groups_opt)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience)

    train_loader = DataLoader(BoutDataset(items, tr, y), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate, drop_last=False)
    val_ds = BoutDataset(items, va, y)
    grad_gate(enc, head, train_loader, criterion, logger)

    if timing:
        return time_fold(args, logger, enc, head, train_loader, val_ds, optimizer,
                         criterion, params, items, y, groups, tr_all, te, C, out)

    # ---- resume
    resume = out / "resume.pth"
    start, best_uar, best_epoch, stale, history, best_state = 0, -1.0, 0, 0, [], None
    if resume.exists() and not args.force:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        load_trainable_state(enc, head, ck["current"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start, best_uar, best_epoch = ck["epoch"], ck["best_uar"], ck["best_epoch"]
        stale, history, best_state = ck["stale"], ck["history"], ck["best"]
        logger.info(f"resumed after epoch {start} (best val UAR {best_uar:.4f} "
                    f"at epoch {best_epoch})")

    for epoch in range(start + 1, args.max_epochs + 1):
        t0 = time.time()
        loss = run_epoch(enc, head, train_loader, optimizer, criterion, params, args)
        t_train = time.time() - t0
        vp, vrows = predict(enc, head, val_ds, args)
        v = score(y[vrows], vp.argmax(1), classes, [], None)
        scheduler.step(v["uar"])
        history.append({"epoch": epoch, "train_loss": round(loss, 5),
                        "val_uar": v["uar"], "val_f1_macro": v["f1_macro"],
                        "val_accuracy": v["accuracy"],
                        "lr_lora": optimizer.param_groups[0]["lr"],
                        "train_seconds": round(t_train, 1),
                        "seconds": round(time.time() - t0, 1)})
        logger.info(f"fold {k} epoch {epoch:2d}  loss {loss:.4f}  val UAR "
                    f"{v['uar']:.4f}  F1 {v['f1_macro']:.4f}  "
                    f"({time.time() - t0:.0f}s)")
        if v["uar"] > best_uar:
            best_uar, best_epoch, stale = v["uar"], epoch, 0
            best_state = trainable_state(enc, head)
        else:
            stale += 1
        tmp = resume.with_suffix(".tmp")
        torch.save({"epoch": epoch, "current": trainable_state(enc, head),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "best_uar": best_uar,
                    "best_epoch": best_epoch, "stale": stale,
                    "history": history, "best": best_state}, tmp)
        tmp.replace(resume)
        if stale >= args.patience:
            logger.info(f"early stop after epoch {epoch}")
            break

    # ---- the best validation epoch is the model; the test fold is used now
    load_trainable_state(enc, head, best_state)
    torch.save({"model": args.model, "fold": k, "repeat": args.repeat,
                "best_epoch": best_epoch, "val_uar": best_uar,
                "trainable_state": best_state, "classes": classes,
                "config": config_of(args)}, out / "checkpoint.pth")

    test_ds = BoutDataset(items, te, y)
    tp, trows = predict(enc, head, test_ds, args)
    if not np.array_equal(trows, te):
        raise RuntimeError("test prediction order does not match the test rows")
    e2e = score(y[te], tp.argmax(1), classes, [], tp)
    logger.info(f"fold {k} TEST end-to-end: UAR {e2e['uar']:.4f}  "
                f"AUC {e2e.get('auc_macro_ovr')}  F1 {e2e['f1_macro']:.4f}")

    layer_rows, layer_probs = probe_layers(args, logger, enc, items, y, groups,
                                           tr_all, te, C, classes)

    np.savez_compressed(
        out / "probs.npz",
        rows=te, y=y[te].astype(np.int16), classes=np.array(classes),
        e2e_probs=tp.astype(np.float32),
        layer_probs=layer_probs.astype(np.float32),     # (layers, bouts, C)
        files=groups[te],
        starts=np.array([float(items[i]["start"]) for i in te]),
        ends=np.array([float(items[i]["end"]) for i in te]),
        individual=np.array([items[i]["individual"] for i in te]),
        leaky_split=np.array([bool(items[i].get("leaky_split")) for i in te]),
        single_session=np.array([items[i]["individual"] in SINGLE_SESSION
                                 for i in te]))
    result = {
        "model": args.model, "fold": k, "repeat": args.repeat,
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "best_epoch": best_epoch, "best_val_uar": best_uar, "history": history,
        "test_end_to_end": e2e, "test_per_layer": layer_rows,
        "config": config_of(args),
    }
    json.dump(result, open(out / "result.json", "w"), indent=2)
    resume.unlink(missing_ok=True)
    logger.info(f"wrote {out}")


def probe_layers(args, logger, enc, items, y, groups, tr_all, te, C, classes):
    """Frozen-style probe at every layer of the FINE-TUNED encoder.

    Embeds every bout through phase3_37.extract_all -- the frozen code path --
    then fits phase3_37.fit_and_prob on the training-fold rows only.
    """
    enc.eval()
    t0 = time.time()
    X, kept = extract_all(enc.ex, items, logger, args.embed_batch_size)
    if len(kept) != len(items):
        raise RuntimeError(f"{len(items) - len(kept)} bouts failed to embed")
    logger.info(f"embedded {len(kept)} bouts x {X.shape[1]} layers "
                f"({time.time() - t0:.0f}s)")
    seeds = list(range(42, 42 + args.probe_seeds))
    w = class_weights_for(y[tr_all], C)
    rows, all_probs = [], []
    for layer in range(X.shape[1]):
        p = fit_and_prob(X[tr_all, layer], y[tr_all], groups[tr_all],
                         X[te, layer], C, w, enc.device, seeds,
                         args.probe_steps, args.probe_patience, args.val_frac)
        s = score(y[te], p.argmax(1), classes, [], p)
        rows.append({"layer": layer, **{m: s.get(m) for m in
                     ("uar", "auc_macro_ovr", "f1_macro", "accuracy")}})
        all_probs.append(p)
        logger.info(f"  layer {layer:2d}: UAR {s['uar']:.4f}  "
                    f"AUC {s.get('auc_macro_ovr')}")
    return rows, np.stack(all_probs)


def time_fold(args, logger, enc, head, train_loader, val_ds, optimizer,
              criterion, params, items, y, groups, tr_all, te, C, out):
    """One real training epoch, one validation pass, a partial embedding pass
    and a one-layer probe -- scaled up to a full-run estimate."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    loss = run_epoch(enc, head, train_loader, optimizer, criterion, params, args,
                     limit=args.timing_batches)
    n_batches = min(len(train_loader), args.timing_batches or len(train_loader))
    t_epoch = (time.time() - t0) * len(train_loader) / n_batches
    t0 = time.time()
    predict(enc, head, val_ds, args)
    t_val = time.time() - t0

    sub = random.Random(0).sample(range(len(items)), min(200, len(items)))
    t0 = time.time()
    X, _ = extract_all(enc.ex, [items[i] for i in sub], logger,
                       args.embed_batch_size)
    t_embed = (time.time() - t0) * len(items) / len(sub)

    # one layer, full probe settings, on stand-in features of the real size
    rng = np.random.default_rng(0)
    Xf = rng.normal(size=(len(items), X.shape[2])).astype(np.float32)
    seeds = list(range(42, 42 + args.probe_seeds))
    t0 = time.time()
    fit_and_prob(Xf[tr_all], y[tr_all], groups[tr_all], Xf[te], C,
                 class_weights_for(y[tr_all], C), enc.device, seeds,
                 args.probe_steps, args.probe_patience, args.val_frac)
    t_probe_layer = time.time() - t0
    n_layers = X.shape[1]

    per_epoch = t_epoch + t_val
    per_fold_probe = t_embed + n_layers * t_probe_layer
    est = {
        "model": args.model,
        "train_bouts": len(train_loader.dataset), "batch_size": args.batch_size,
        "epoch_seconds_train": round(t_epoch, 1),
        "epoch_seconds_val": round(t_val, 1),
        "embed_all_bouts_seconds": round(t_embed, 1),
        "probe_one_layer_seconds": round(t_probe_layer, 1),
        "n_layers": n_layers,
        "train_loss_first_epoch": round(loss, 4),
        "peak_gpu_memory_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 2)
                               if torch.cuda.is_available() else None),
        "max_epochs": args.max_epochs, "patience": args.patience,
        "per_fold_hours_worst": round((args.max_epochs * per_epoch
                                       + per_fold_probe) / 3600, 2),
        "all_5_folds_hours_worst": round(5 * (args.max_epochs * per_epoch
                                              + per_fold_probe) / 3600, 2),
        "all_5_folds_hours_if_stop_at_half": round(
            5 * (args.max_epochs / 2 * per_epoch + per_fold_probe) / 3600, 2),
    }
    if args.timing_batches:
        est["note"] = f"epoch time extrapolated from {n_batches} batches"
    json.dump(est, open(out.parent / "timing.json", "w"), indent=2)
    logger.info("TIMING " + json.dumps(est, indent=2))


# --------------------------------------------------------------- summary

def summarize(args, logger):
    root = Path(args.output_dir) / args.model
    folds = sorted(root.glob("fold_*/result.json"))
    if len(folds) != 5:
        raise SystemExit(f"{len(folds)}/5 folds finished in {root}")
    manifest, items, classes, _, _, _ = setup(args, logger)
    P = [np.load(f.parent / "probs.npz") for f in folds]
    rows = np.concatenate([p["rows"] for p in P])
    if len(set(rows)) != len(rows) or len(rows) != len(items):
        raise SystemExit("folds do not cover every bout exactly once")
    y = np.concatenate([p["y"] for p in P]).astype(int)
    e2e = np.concatenate([p["e2e_probs"] for p in P])
    lay = np.concatenate([p["layer_probs"] for p in P], axis=1)

    pooled_e2e = score(y, e2e.argmax(1), classes, [], e2e)
    frozen = json.load(open(FROZEN_LAYERS.format(args.model)))
    comp = []
    for l in range(lay.shape[0]):
        s = score(y, lay[l].argmax(1), classes, [], lay[l])
        f = frozen["layers"][l]["per_repeat"][args.repeat]
        comp.append({"layer": l,
                     "depth_pct": round(100 * l / (lay.shape[0] - 1), 1),
                     "finetuned_uar": s["uar"], "frozen_uar": f["uar"],
                     "delta_uar": round(s["uar"] - f["uar"], 4),
                     "finetuned_auc": s["auc_macro_ovr"],
                     "frozen_auc": f["auc_macro_ovr"],
                     # None only when the pooled set lacks a class (subset runs)
                     "delta_auc": (round(s["auc_macro_ovr"] - f["auc_macro_ovr"], 4)
                                   if s["auc_macro_ovr"] is not None else None)})

    from sklearn.metrics import recall_score
    rec = recall_score(y, e2e.argmax(1), labels=list(range(len(classes))),
                       average=None, zero_division=0)
    inv = manifest["inventory"]
    per_ind = [{"individual": c, "recall_end_to_end": round(float(rec[i]), 4),
                "bouts": inv[c]["bouts"], "sessions": inv[c]["sessions"],
                "recordings": inv[c]["recordings"],
                "leaky_split": inv[c]["leaky_split"],
                "single_session": c in SINGLE_SESSION}
               for i, c in enumerate(classes)]

    best_ft = max(comp, key=lambda r: r["finetuned_uar"])
    best_fr = max(comp, key=lambda r: r["frozen_uar"])
    summary = {
        "model": args.model, "repeat": args.repeat,
        "pooled_end_to_end": pooled_e2e,
        "best_layer_finetuned": best_ft, "best_layer_frozen": best_fr,
        "per_layer": comp, "per_individual": per_ind,
        "folds": [json.load(open(f))["best_epoch"] for f in folds],
        "flags": {"X0": "single recording, split within it (leaky_split)",
                  "O1": "single session, split by recording"},
    }
    json.dump(summary, open(root / "summary.json", "w"), indent=2)
    import csv
    with open(root / "layer_comparison.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(comp[0]))
        wr.writeheader()
        wr.writerows(comp)
    logger.info(f"pooled end-to-end: UAR {pooled_e2e['uar']:.4f}  "
                f"AUC {pooled_e2e['auc_macro_ovr']}")
    logger.info(f"best layer fine-tuned {best_ft['layer']} UAR "
                f"{best_ft['finetuned_uar']:.4f} vs frozen best {best_fr['layer']} "
                f"UAR {best_fr['frozen_uar']:.4f}")
    logger.info(f"wrote {root / 'summary.json'}, layer_comparison.csv")


def config_of(args):
    return {k: getattr(args, k) for k in (
        "lora_r", "lora_alpha", "lr_lora", "lr_front", "lr_head",
        "batch_size", "max_epochs", "patience", "lr_patience", "val_frac",
        "grad_clip", "probe_seeds", "probe_steps", "probe_patience", "seed")} | {
        "lora_targets": "attention projections, every block",
        "lora_mechanism": "weight parametrization (W + BA*alpha/r), no dropout",
        "front_end_trained": True,
        "train_crop_seconds": 10.24,
        "selection": "best validation UAR (grouped 20% of training folds)"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--manifest",
                   default="outputs/phase3/manifests_cv_original/cv_by_file_5fold.json")
    p.add_argument("--output-dir", default="outputs/phase3/lora_hyrax_cv")
    p.add_argument("--fold", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--timing", action="store_true",
                   help="one epoch of fold 0 plus probe timing -> timing.json")
    p.add_argument("--timing-batches", type=int, default=None,
                   help="time only this many batches and extrapolate (smoke tests)")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--limit-items", type=int, default=None,
                   help="smoke tests only: random subset of bouts")
    p.add_argument("--force", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--lr-front", type=float, default=1e-5)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--embed-batch-size", type=int, default=16)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--lr-patience", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--probe-seeds", type=int, default=5)
    p.add_argument("--probe-steps", type=int, default=5000)
    p.add_argument("--probe-patience", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logger = setup_logger("Phase3_LoRAHyraxCV", "INFO")
    if args.summarize:
        return summarize(args, logger)
    if args.timing:
        return train_fold(args, logger, 0, timing=True)
    for k in args.fold:
        train_fold(args, logger, k)


if __name__ == "__main__":
    main()
