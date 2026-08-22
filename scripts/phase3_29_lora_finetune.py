#!/usr/bin/env python3
"""
Phase 3 - Step 29: staged 7-species adaptation by LoRA.

WHY THIS IS A COPY OF phase3_23 RATHER THAN A FIX TO phase3_10
--------------------------------------------------------------
The question is "does the adaptation METHOD matter", so everything except the
method has to be identical. phase3_10 differs from phase3_23 in ways that would
confound that: it trains on 5 s windows at 2.5 s stride out of a window cache,
where phase3_23 uses one 30 s-truncated sample per FILE, and its defaults for
layerdrop, pooling and model selection differ too.

This file is phase3_23's training regime unchanged, with LoRA substituted for
direct weight updates. Same manifest, same dataset unit, same masked pooling,
same per-sample normalisation, same optimiser families, same selection metric,
same seed. Only the parameterisation differs.

phase3_10's two known defects are also fixed here:

  1. It trained on the 8-class species manifest, which INCLUDES hyrax. That
     teaches the encoder to collapse every hyrax into one class -- discarding
     exactly the within-hyrax variation individual ID depends on. This uses the
     7-class manifest; hyrax is never seen.
  2. It froze the CNN front-end. The hyrax signal peaks at hidden_states[0],
     which the conv stack produces, and LoRA only touches attention projections,
     so nothing could move it. Here the conv stack is unfrozen at a low LR
     alongside the adapters.

SCOPE MATCHES phase3_23 DELIBERATELY
------------------------------------
LoRA is applied to blocks 0-3 ONLY, because phase3_23 unfroze blocks 0-3 only.
Adapting all 12/24 blocks would make the comparison "4 blocks vs every block"
rather than "direct updates vs low-rank updates". Use --lora-layers to override.

RESUME AND JOB CHAINING
-----------------------
Full training state -- weights, optimiser, scheduler, epoch, best score,
history -- is written every epoch to checkpoints/resume.pth. On start, if that
file exists, training continues from it. When training finishes, a DONE marker
is written and any later job in the chain exits immediately.

This exists because the previous run lost XLS-R to the cluster's 24 h limit at
epoch 14 of 16. Chain three jobs with --dependency=afterany and a 24 h cap
becomes 72 h of effective compute.

CHECKPOINT FORMAT
-----------------
best_model.pth carries a MERGED full backbone_state_dict, not adapters, so the
bout probe (phase3_24) loads it exactly like the partial-fine-tuning checkpoints
with no LoRA-specific code path.

USAGE
-----
    # gradient gate only
    python scripts/phase3_29_lora_finetune.py --model xls_r --check-grads

    # full run (resumes automatically if interrupted)
    python scripts/phase3_29_lora_finetune.py --model xls_r
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from src.utils.logging_utils import setup_logger  # noqa: E402

# the data path is imported, not reimplemented, so the training regime is
# provably the same one the partial-fine-tuning result was measured under
from phase3_23_species7_finetune import (  # noqa: E402
    MODEL_IDS,
    SR,
    SpeciesDataset,
    make_collate,
)

CONV_MODULES = ("feature_extractor", "feature_projection")
ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


class LoraFineTuner:

    def __init__(self, config, args, manifest_path, output_dir, logger):
        self.config = config
        self.args = args
        self.logger = logger
        self.output_dir = Path(output_dir)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        if "species" in self.manifest and "species_to_idx" in self.manifest:
            self.label_key = "species"
            self.classes = self.manifest["species"]
        else:
            self.label_key = "individual"
            self.classes = self.manifest["individuals"]

        self.label_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self.class_weights = torch.FloatTensor(
            [self.manifest["class_weights"][c] for c in self.classes])

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.logger.info(f"Device:      {self.device}")
        self.logger.info(f"Model:       {args.model}")
        self.logger.info(f"Task:        {self.manifest.get('task')} ({self.label_key})")
        self.logger.info(f"Classes:     {self.num_classes} -> {self.classes}")
        excluded = self.manifest.get("excluded_species")
        if excluded:
            self.logger.info(f"EXCLUDED:    {excluded}  (never seen during adaptation)")

        self._build()

    # ------------------------------------------------------------------ model

    def _build(self):
        from peft import LoraConfig, get_peft_model
        from transformers import (HubertModel, WavLMModel,
                                  Wav2Vec2FeatureExtractor, Wav2Vec2Model)

        model_id = MODEL_IDS[self.args.model]
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        cls = {"hubert_base": HubertModel, "wavlm": WavLMModel}.get(
            self.args.model, Wav2Vec2Model)

        self.logger.info(f"Loading {model_id} ...")
        base = cls.from_pretrained(model_id, use_safetensors=True)

        old_ld = getattr(base.config, "layerdrop", None)
        base.config.layerdrop = self.args.layerdrop
        self.logger.info(f"layerdrop:   {old_ld} -> {self.args.layerdrop}")

        n_layers = len(base.encoder.layers)
        lora_layers = list(range(min(self.args.lora_layers, n_layers)))

        lora_cfg = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            target_modules=ATTN_TARGETS,
            # restrict to the same blocks phase3_23 unfroze, so the comparison
            # isolates the method rather than the scope
            layers_to_transform=lora_layers,
            layers_pattern="layers",
        )

        self.backbone = get_peft_model(base, lora_cfg)
        self.base = self.backbone.get_base_model()
        self.backbone.to(self.device)

        n_lora = sum(p.numel() for n, p in self.backbone.named_parameters()
                     if p.requires_grad)
        self.logger.info(f"LoRA:        r={self.args.lora_r} alpha={self.args.lora_alpha} "
                         f"dropout={self.args.lora_dropout} on {ATTN_TARGETS}")
        self.logger.info(f"             blocks {lora_layers} of {n_layers}  "
                         f"({n_lora:,} adapter params) at lr={self.args.lr_lora}")

        # get_peft_model freezes everything but the adapters, so the conv stack
        # has to be re-enabled AFTER wrapping
        self.conv_params = []
        if not self.args.freeze_conv:
            for mod_name in CONV_MODULES:
                mod = getattr(self.base, mod_name, None)
                if mod is None:
                    self.logger.warning(f"  no submodule '{mod_name}' -- skipped")
                    continue
                for p in mod.parameters():
                    p.requires_grad = True
                    self.conv_params.append(p)
            n_conv = sum(p.numel() for p in self.conv_params)
            self.logger.info(f"CNN:         UNFROZEN, {n_conv:,} params at "
                             f"lr={self.args.lr_conv}  (LoRA cannot reach convolutions, "
                             f"and hidden_states[0] is where hyrax identity peaks)")
        else:
            self.logger.info("CNN:         FROZEN -- hidden_states[0] cannot change")

        hidden = self.base.config.hidden_size
        self.classifier = nn.Linear(hidden, self.num_classes).to(self.device)
        self.logger.info(f"Head:        {hidden} -> {self.num_classes} at lr={self.args.lr_head}")

        if self.args.grad_checkpoint:
            self.base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            self.logger.info("gradient checkpointing enabled (use_reentrant=False)")

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        self.logger.info(f"Trainable:   {trainable:,} / {total:,} "
                         f"({trainable / total * 100:.2f}%)")

    # --------------------------------------------------------------- forward

    def _masked_mean(self, hidden_states, lengths):
        feat_lens = self.base._get_feat_extract_output_lengths(lengths)
        feat_lens = torch.as_tensor(feat_lens, device=hidden_states.device).long()
        feat_lens = feat_lens.clamp(max=hidden_states.size(1))
        pos = torch.arange(hidden_states.size(1), device=hidden_states.device)
        mask = (pos[None, :] < feat_lens[:, None]).unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def forward(self, audio, lengths):
        audio = audio.to(self.device)
        inputs = {"input_values": audio}
        if self.feature_extractor.return_attention_mask:
            pos = torch.arange(audio.size(1), device=self.device)
            inputs["attention_mask"] = (
                pos[None, :] < lengths.to(self.device)[:, None]).long()
        hidden = self.backbone(**inputs).last_hidden_state
        return self.classifier(self._masked_mean(hidden, lengths))

    # ------------------------------------------------------------------ gate

    def check_grads(self, loader):
        self.logger.info("\n" + "=" * 72)
        self.logger.info("GRADIENT CHECK")
        self.logger.info("=" * 72)

        self.backbone.train()
        self.classifier.train()

        audio, lengths, labels = next(iter(loader))
        criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        criterion(self.forward(audio, lengths), labels.to(self.device)).backward()

        def norm_of(named):
            got = [p for _, p in named if p.requires_grad and p.grad is not None]
            return (sum(float(p.grad.norm()) ** 2 for p in got) ** 0.5 if got else 0.0,
                    len(got))

        results = {}
        for mod_name in CONV_MODULES:
            mod = getattr(self.base, mod_name, None)
            if mod is None:
                continue
            n, c = norm_of(list(mod.named_parameters()))
            results[mod_name] = n
            self.logger.info(f"  {mod_name:<28} tensors={c:<4} |grad|={n:.6e}  "
                             f"{'OK ' if n > 0 else 'ZERO'}")

        lora_named = [(n, p) for n, p in self.backbone.named_parameters() if "lora_" in n]
        n, c = norm_of(lora_named)
        results["lora"] = n
        self.logger.info(f"  {'lora adapters':<28} tensors={c:<4} |grad|={n:.6e}  "
                         f"{'OK ' if n > 0 else 'ZERO'}")

        # a block OUTSIDE the LoRA scope must receive nothing
        last = len(self.base.encoder.layers) - 1
        n_last, _ = norm_of(list(self.base.encoder.layers[last].named_parameters()))
        results["deep_frozen"] = n_last
        self.logger.info(f"  {f'encoder.layers[{last}] (frozen)':<28} "
                         f"|grad|={n_last:.6e}  {'LEAK' if n_last > 0 else 'OK '}")

        n_head, _ = norm_of(list(self.classifier.named_parameters()))
        self.logger.info(f"  {'classifier':<28} |grad|={n_head:.6e}")

        conv_ok = self.args.freeze_conv or any(
            (results.get(m) or 0) > 0 for m in CONV_MODULES)
        passed = conv_ok and results["lora"] > 0 and results["deep_frozen"] == 0 and n_head > 0

        self.logger.info("-" * 72)
        self.logger.info(f"  conv stack receives gradient : {conv_ok}")
        self.logger.info(f"  LoRA adapters receive gradient: {results['lora'] > 0}")
        self.logger.info(f"  layers outside scope frozen   : {results['deep_frozen'] == 0}")
        self.logger.info(f"\n  GATE: {'PASS' if passed else 'FAIL'}")
        self.logger.info("=" * 72)

        self.backbone.zero_grad(set_to_none=True)
        self.classifier.zero_grad(set_to_none=True)
        return passed

    # --------------------------------------------------------------- loaders

    def loaders(self):
        data_dir = Path(self.config["paths"]["data_dir"])
        splits = self.manifest["splits"] if "splits" in self.manifest else self.manifest
        collate = make_collate(self.feature_extractor)

        out = {}
        for name in ("train", "val", "test"):
            items = list(splits[name])
            if self.args.debug:
                items = items[: {"train": 64, "val": 32, "test": 32}[name]]
            elif self.args.limit_train and name == "train":
                items = items[: self.args.limit_train]

            ds = SpeciesDataset(items, self.label_to_idx, data_dir,
                                self.label_key, self.args.max_duration)
            out[name] = DataLoader(ds, batch_size=self.args.batch_size,
                                   shuffle=(name == "train"),
                                   num_workers=self.args.num_workers,
                                   collate_fn=collate,
                                   pin_memory=(self.device == "cuda"))
            self.logger.info(f"  {name:<6} {len(ds):>6} files, {len(out[name]):>5} batches")
        return out["train"], out["val"], out["test"]

    def _eval(self, loader, desc):
        self.backbone.eval()
        self.classifier.eval()
        preds, golds, total = [], [], 0.0
        criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        with torch.no_grad():
            for audio, lengths, labels in tqdm(loader, desc=desc, leave=False):
                logits = self.forward(audio, lengths)
                total += float(criterion(logits, labels.to(self.device)))
                preds.extend(logits.argmax(1).cpu().numpy())
                golds.extend(labels.numpy())
        return {"loss": total / max(len(loader), 1),
                "acc": accuracy_score(golds, preds),
                "f1_macro": f1_score(golds, preds, average="macro", zero_division=0)}

    # ------------------------------------------------------- resume plumbing

    def _trainable_state(self):
        """Adapters + conv + head only. A few MB, not the whole 1.2 GB backbone."""
        from peft import get_peft_model_state_dict
        conv = {}
        for mod_name in CONV_MODULES:
            mod = getattr(self.base, mod_name, None)
            if mod is not None:
                for k, v in mod.state_dict().items():
                    conv[f"{mod_name}.{k}"] = v.detach().cpu().clone()
        return {
            "adapters": {k: v.detach().cpu().clone()
                         for k, v in get_peft_model_state_dict(self.backbone).items()},
            "conv": conv,
            "classifier": {k: v.detach().cpu().clone()
                           for k, v in self.classifier.state_dict().items()},
        }

    def _load_trainable_state(self, state):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.backbone, state["adapters"])
        for mod_name in CONV_MODULES:
            mod = getattr(self.base, mod_name, None)
            if mod is None:
                continue
            sub = {k[len(mod_name) + 1:]: v for k, v in state["conv"].items()
                   if k.startswith(mod_name + ".")}
            if sub:
                mod.load_state_dict(sub)
        self.classifier.load_state_dict(state["classifier"])

    def _save_resume(self, epoch, optimizer, scheduler, best_f1, best_epoch, stale,
                     history, best_state):
        path = self.output_dir / "checkpoints" / "resume.pth"
        tmp = path.with_suffix(".tmp")
        torch.save({
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "stale": stale,
            "history": history,
            "current_state": self._trainable_state(),
            "best_state": best_state,
            "rng": {"torch": torch.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate()},
        }, tmp)
        # atomic: a job killed mid-write leaves the previous resume intact
        tmp.replace(path)

    def _try_resume(self, optimizer, scheduler):
        path = self.output_dir / "checkpoints" / "resume.pth"
        if not path.exists() or self.args.no_resume:
            return 0, -1.0, 0, 0, [], None
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self._load_trainable_state(ck["current_state"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        try:
            torch.set_rng_state(ck["rng"]["torch"])
            np.random.set_state(ck["rng"]["numpy"])
            random.setstate(ck["rng"]["python"])
        except Exception:
            self.logger.warning("could not restore RNG state; continuing")
        self.logger.info(f"RESUMED from epoch {ck['epoch']} "
                         f"(best val macro-F1 {ck['best_f1']:.4f} at epoch {ck['best_epoch']})")
        return (ck["epoch"], ck["best_f1"], ck["best_epoch"], ck["stale"],
                ck["history"], ck["best_state"])

    def _export_merged(self, best_state, best_f1, best_epoch):
        """Merge adapters into the base weights and save a full backbone_state_dict.

        The probe then loads this exactly like a partial-fine-tuning checkpoint --
        no LoRA-aware code path anywhere downstream.
        """
        import copy
        from peft import get_peft_model_state_dict  # noqa: F401  (symmetry)

        self._load_trainable_state(best_state)
        merged = copy.deepcopy(self.backbone).merge_and_unload()
        sd = {k: v.detach().cpu() for k, v in merged.state_dict().items()}

        path = self.output_dir / "checkpoints" / "best_model.pth"
        torch.save({
            "epoch": best_epoch,
            "model_name": self.args.model,
            "model_id": MODEL_IDS[self.args.model],
            "backbone_state_dict": sd,
            "classifier_state_dict": best_state["classifier"],
            "classes": self.classes,
            "label_key": self.label_key,
            "val_f1_macro": best_f1,
            "config": {
                "method": "lora",
                "lora_r": self.args.lora_r,
                "lora_alpha": self.args.lora_alpha,
                "lora_dropout": self.args.lora_dropout,
                "lora_target_modules": ATTN_TARGETS,
                "lora_layers": self.args.lora_layers,
                "freeze_conv": self.args.freeze_conv,
                "layerdrop": self.args.layerdrop,
                "lr_lora": self.args.lr_lora,
                "lr_conv": self.args.lr_conv,
                "lr_head": self.args.lr_head,
                "masked_pooling": True,
                "merged": True,
            },
        }, path)
        del merged
        self.logger.info(f"merged checkpoint -> {path}")
        return path

    # ----------------------------------------------------------------- train

    def train(self, train_loader, val_loader):
        self.logger.info("\n" + "=" * 72)
        self.logger.info("LoRA ADAPTATION")
        self.logger.info("=" * 72)

        criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))

        lora_params = [p for n, p in self.backbone.named_parameters()
                       if p.requires_grad and "lora_" in n]
        groups = [{"params": lora_params, "lr": self.args.lr_lora}]
        if self.conv_params:
            groups.append({"params": self.conv_params, "lr": self.args.lr_conv})
        groups.append({"params": list(self.classifier.parameters()),
                       "lr": self.args.lr_head})

        optimizer = optim.AdamW(groups)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=self.args.lr_patience)

        start_epoch, best_f1, best_epoch, stale, history, best_state = \
            self._try_resume(optimizer, scheduler)

        for epoch in range(start_epoch + 1, self.args.max_epochs + 1):
            self.backbone.train()
            self.classifier.train()

            run_loss, preds, golds = 0.0, [], []
            t0 = time.time()

            for audio, lengths, labels in tqdm(train_loader, desc=f"epoch {epoch}",
                                               leave=False):
                labels = labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.forward(audio, lengths)
                loss = criterion(logits, labels)
                loss.backward()
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in groups for p in g["params"]], self.args.grad_clip)
                optimizer.step()

                run_loss += float(loss.detach())
                preds.extend(logits.argmax(1).detach().cpu().numpy())
                golds.extend(labels.cpu().numpy())

            tr_loss = run_loss / max(len(train_loader), 1)
            tr_f1 = f1_score(golds, preds, average="macro", zero_division=0)
            va = self._eval(val_loader, "val")
            scheduler.step(va["f1_macro"])

            history.append({
                "epoch": epoch, "train_loss": tr_loss, "train_f1_macro": tr_f1,
                "train_acc": accuracy_score(golds, preds),
                "val_loss": va["loss"], "val_f1_macro": va["f1_macro"],
                "val_acc": va["acc"], "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - t0,
            })
            self.logger.info(
                f"epoch {epoch:>3}  train loss {tr_loss:.4f} F1 {tr_f1:.4f}  |  "
                f"val loss {va['loss']:.4f} F1 {va['f1_macro']:.4f} acc {va['acc']:.4f}  "
                f"({time.time() - t0:.0f}s)")

            if va["f1_macro"] > best_f1:
                best_f1, best_epoch, stale = va["f1_macro"], epoch, 0
                best_state = self._trainable_state()
                self.logger.info(f"           new best (val F1 {va['f1_macro']:.4f})")
            else:
                stale += 1

            # written every epoch, so a timeout costs at most one epoch
            self._save_resume(epoch, optimizer, scheduler, best_f1, best_epoch,
                              stale, history, best_state)

            if stale >= self.args.patience:
                self.logger.info(f"early stop at epoch {epoch}")
                break

        self.logger.info(f"\nbest val macro-F1 {best_f1:.4f} at epoch {best_epoch}")
        return history, best_f1, best_epoch, best_state

    def run(self):
        done_marker = self.output_dir / "DONE"
        if done_marker.exists() and not self.args.force:
            self.logger.info(f"DONE marker present ({done_marker}) -- already complete, "
                             f"exiting so the next job in the chain does no work")
            return {"already_done": True}

        train_loader, val_loader, test_loader = self.loaders()

        if self.args.check_grads:
            return {"gate_passed": self.check_grads(train_loader)}

        history, best_f1, best_epoch, best_state = self.train(train_loader, val_loader)
        if best_state is None:
            raise RuntimeError("no epoch completed; nothing to export")

        ckpt_path = self._export_merged(best_state, best_f1, best_epoch)
        test = self._eval(test_loader, "test")
        self.logger.info(f"\nTEST macro-F1 {test['f1_macro']:.4f}  acc {test['acc']:.4f}")

        summary = {
            "model": self.args.model,
            "method": "lora",
            "task": self.manifest.get("task"),
            "num_classes": self.num_classes,
            "classes": self.classes,
            "excluded_species": self.manifest.get("excluded_species"),
            "lora": {"r": self.args.lora_r, "alpha": self.args.lora_alpha,
                     "dropout": self.args.lora_dropout,
                     "target_modules": ATTN_TARGETS,
                     "layers": self.args.lora_layers},
            "conv_unfrozen": not self.args.freeze_conv,
            "layerdrop": self.args.layerdrop,
            "masked_pooling": True,
            "best_epoch": best_epoch,
            "best_val_f1_macro": best_f1,
            "test_f1_macro": test["f1_macro"],
            "test_accuracy": test["acc"],
            "checkpoint": str(ckpt_path),
            "history": history,
        }
        with open(self.output_dir / "adaptation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        done_marker.write_text(
            f"model={self.args.model} best_epoch={best_epoch} "
            f"val_f1={best_f1:.4f} test_f1={test['f1_macro']:.4f}\n")
        self.logger.info(f"wrote {self.output_dir / 'adaptation_summary.json'}")
        self.logger.info(f"wrote DONE marker -- later jobs in the chain will exit")

        if test["f1_macro"] < self.args.min_species_f1:
            self.logger.warning(f"\nGATE WARNING: test macro-F1 {test['f1_macro']:.4f} "
                                f"< {self.args.min_species_f1}")
        return summary


def main():
    p = argparse.ArgumentParser(description="Staged 7-species adaptation by LoRA")
    p.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    p.add_argument("--manifest", default=None)
    p.add_argument("--output-dir", default=None)

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-layers", type=int, default=4,
                   help="apply LoRA to the first N blocks; 4 matches phase3_23's scope")
    p.add_argument("--freeze-conv", action="store_true",
                   help="reproduce phase3_10's frozen CNN, which could not move layer 0")
    p.add_argument("--layerdrop", type=float, default=0.0)

    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--lr-conv", type=float, default=1e-5)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lr-patience", type=int, default=3)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=16)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-duration", type=float, default=30.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--check-grads", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--no-resume", action="store_true",
                   help="ignore an existing resume.pth and start over")
    p.add_argument("--force", action="store_true",
                   help="run even if a DONE marker exists")
    p.add_argument("--min-species-f1", type=float, default=0.90)
    p.add_argument("--no-cudnn", action="store_true")

    args = p.parse_args()

    if args.no_cudnn:
        torch.backends.cudnn.enabled = False
    if args.debug:
        args.max_epochs = min(args.max_epochs, 3)
        args.patience = 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    root = SCRIPT_DIR.parent
    with open(root / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)
    out_root = Path(config["paths"]["output_dir"])

    manifest = Path(args.manifest) if args.manifest else \
        out_root / "phase3" / "manifests_species7" / "species_id.json"
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")

    tag = args.model + ("_convfrozen" if args.freeze_conv else "")
    output_dir = Path(args.output_dir) if args.output_dir else \
        out_root / "phase3" / "ft_lora" / tag

    logger = setup_logger("Phase3_LoRA", config["experiment"]["log_level"])
    logger.info("=" * 72)
    logger.info("PHASE 3 - STEP 29: STAGED 7-SPECIES ADAPTATION BY LoRA")
    logger.info("=" * 72)
    logger.info(f"manifest:  {manifest}")
    logger.info(f"output:    {output_dir}")

    trainer = LoraFineTuner(config, args, manifest, output_dir, logger)
    result = trainer.run()

    if args.check_grads:
        raise SystemExit(0 if result.get("gate_passed") else 1)


if __name__ == "__main__":
    main()
