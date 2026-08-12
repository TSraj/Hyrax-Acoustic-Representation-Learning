#!/usr/bin/env python3
"""
Phase 3 - Step 23: Staged adaptation by REAL fine-tuning (no LoRA).

Ports the phase2_05 recipe -- which was verified to be genuine fine-tuning --
into the staged design:

    adapt on 7-class species (hyrax EXCLUDED)  ->  freeze  ->  probe hyrax

Differences from phase2_05, all deliberate:

  1. CNN front-end is UNFROZEN by default.
     The hyrax signal peaks at hidden_states[0], which is the conv feature
     extractor + feature_projection output -- BEFORE any transformer block.
     phase2_05 froze it, so adaptation provably could not move the best layer
     (verified: perturbing blocks 0-3 leaves hidden_states[0] delta == 0.000).
     It gets its own, lower LR because conv fine-tuning is unstable.
     Use --freeze-conv to reproduce the old behaviour for an A/B.

  2. layerdrop forced to 0.0. phase2_05 left it at the checkpoint default
     (0.1 for xls-r / hubert-base / w2v2-960h), adding gradient noise.

  3. Mean pooling is MASKED. phase2_05 averaged over zero padding, and for
     group-norm models no attention mask reaches the encoder at all. Here the
     valid frame count is derived from true audio lengths via
     _get_feat_extract_output_lengths, so padding never enters the mean.

  4. Per-sample normalisation. The feature extractor is applied to each
     unpadded clip, then clips are padded -- not the reverse.

  5. Model selection on val macro-F1 (not accuracy), matching how every
     downstream number in this project is reported.

The checkpoint written here is the input to the per-layer hyrax probe.

Usage
-----
Smoke test (verifies gradients actually reach the conv stack, then stops):

    python scripts/phase3_23_species7_finetune.py --model xls_r --check-grads

Short end-to-end run on a subset:

    python scripts/phase3_23_species7_finetune.py --model xls_r --debug

Full run:

    python scripts/phase3_23_species7_finetune.py --model xls_r \
        --batch-size 8 --max-epochs 16
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.audio_utils import load_audio  # noqa: E402
from src.utils.logging_utils import setup_logger  # noqa: E402


MODEL_IDS = {
    "wav2vec2_base": "facebook/wav2vec2-base",
    "wav2vec2_base_960h": "facebook/wav2vec2-base-960h",
    "hubert_base": "facebook/hubert-base-ls960",
    "wavlm": "microsoft/wavlm-base-plus",
    "xls_r": "facebook/wav2vec2-xls-r-300m",
}

SR = 16000


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class SpeciesDataset(Dataset):
    """One sample per FILE, truncated to max_duration. Matches phase2_05."""

    def __init__(self, items, label_to_idx, data_dir, label_key,
                 max_duration=30.0, min_duration=0.5):
        self.items = items
        self.label_to_idx = label_to_idx
        self.data_dir = Path(data_dir)
        self.label_key = label_key
        self.max_samples = int(max_duration * SR)
        self.min_samples = int(min_duration * SR)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio, _ = load_audio(str(self.data_dir / item["file"]), target_sr=SR, mono=True)

        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]
        if len(audio) < self.min_samples:
            audio = np.pad(audio, (0, self.min_samples - len(audio)), mode="constant")

        return np.asarray(audio, dtype=np.float32), self.label_to_idx[item[self.label_key]]


def make_collate(feature_extractor):
    """Normalise each clip at its TRUE length, then pad. Returns lengths too."""

    def collate(batch):
        audios, labels = zip(*batch)
        lengths = torch.LongTensor([len(a) for a in audios])

        # per-sample normalisation -- padding is never included in the statistics
        normed = [
            feature_extractor(a, sampling_rate=SR, return_tensors="np")["input_values"][0]
            for a in audios
        ]

        max_len = int(lengths.max())
        padded = np.zeros((len(normed), max_len), dtype=np.float32)
        for i, a in enumerate(normed):
            padded[i, : len(a)] = a

        return torch.from_numpy(padded), lengths, torch.LongTensor(labels)

    return collate


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class StagedFineTuner:

    def __init__(self, config, args, manifest_path, output_dir, logger):
        self.config = config
        self.args = args
        self.logger = logger
        self.output_dir = Path(output_dir)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        # species_id manifests key on 'species'; individual manifests on 'individual'
        if "species" in self.manifest and "species_to_idx" in self.manifest:
            self.label_key = "species"
            self.classes = self.manifest["species"]
        else:
            self.label_key = "individual"
            self.classes = self.manifest["individuals"]

        self.label_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        weights = [self.manifest["class_weights"][c] for c in self.classes]
        self.class_weights = torch.FloatTensor(weights)

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.logger.info(f"Device:      {self.device}")
        self.logger.info(f"Model:       {args.model}")
        self.logger.info(f"Task:        {self.manifest.get('task', '?')} ({self.label_key})")
        self.logger.info(f"Classes:     {self.num_classes} -> {self.classes}")

        excluded = self.manifest.get("excluded_species")
        if excluded:
            self.logger.info(f"EXCLUDED:    {excluded}  (never seen during adaptation)")

        self._build()

    def _build(self):
        from transformers import HubertModel, WavLMModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model

        model_id = MODEL_IDS[self.args.model]
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

        if self.args.model == "hubert_base":
            cls = HubertModel
        elif self.args.model == "wavlm":
            cls = WavLMModel
        else:
            cls = Wav2Vec2Model

        self.logger.info(f"Loading {model_id} ...")
        self.backbone = cls.from_pretrained(model_id, use_safetensors=True)

        # --- fix 2: layerdrop ------------------------------------------------
        old_ld = getattr(self.backbone.config, "layerdrop", None)
        self.backbone.config.layerdrop = self.args.layerdrop
        self.logger.info(f"layerdrop:   {old_ld} -> {self.args.layerdrop}")

        self.backbone.to(self.device)

        # --- freeze everything, then selectively unfreeze --------------------
        for p in self.backbone.parameters():
            p.requires_grad = False

        n_layers = len(self.backbone.encoder.layers)
        n_ft = min(self.args.num_layers, n_layers)
        for i in range(n_ft):
            for p in self.backbone.encoder.layers[i].parameters():
                p.requires_grad = True

        # --- fix 1: the CNN front-end ----------------------------------------
        # hidden_states[0] is produced here. If this stays frozen, the layer
        # that scores best on hyrax cannot change, and the whole experiment
        # measures zero by construction.
        self.conv_param_names = []
        if not self.args.freeze_conv:
            for mod_name in ("feature_extractor", "feature_projection"):
                mod = getattr(self.backbone, mod_name, None)
                if mod is None:
                    self.logger.warning(f"  no submodule '{mod_name}' -- skipped")
                    continue
                for pname, p in mod.named_parameters():
                    p.requires_grad = True
                    self.conv_param_names.append(f"{mod_name}.{pname}")
            self.logger.info(
                f"CNN front-end UNFROZEN ({len(self.conv_param_names)} tensors) "
                f"at lr={self.args.lr_conv}"
            )
        else:
            self.logger.info("CNN front-end FROZEN (--freeze-conv): "
                             "hidden_states[0] cannot change")

        self.logger.info(f"Transformer: layers 0..{n_ft - 1} of {n_layers} unfrozen "
                         f"at lr={self.args.lr_backbone}")

        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden, self.num_classes).to(self.device)
        self.logger.info(f"Head:        {hidden} -> {self.num_classes} at lr={self.args.lr_head}")

        if self.args.grad_checkpoint:
            # non-reentrant: works correctly when the segment inputs do not
            # themselves require grad, which is the case for the frozen deep
            # layers we still have to backprop THROUGH to reach blocks 0-3.
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.logger.info("gradient checkpointing enabled (use_reentrant=False)")

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        self.logger.info(f"Trainable:   {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

    # ----------------------------------------------------------------------

    def _masked_mean(self, hidden_states, lengths):
        """fix 3: average only over real frames."""
        feat_lens = self.backbone._get_feat_extract_output_lengths(lengths)
        feat_lens = torch.as_tensor(feat_lens, device=hidden_states.device).long()
        feat_lens = feat_lens.clamp(max=hidden_states.size(1))

        pos = torch.arange(hidden_states.size(1), device=hidden_states.device)
        mask = (pos[None, :] < feat_lens[:, None]).unsqueeze(-1).to(hidden_states.dtype)

        return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def forward(self, audio, lengths):
        audio = audio.to(self.device)
        inputs = {"input_values": audio}

        # group-norm models (wav2vec2-base, hubert-base) take no attention mask
        if self.feature_extractor.return_attention_mask:
            pos = torch.arange(audio.size(1), device=self.device)
            inputs["attention_mask"] = (
                pos[None, :] < lengths.to(self.device)[:, None]
            ).long()

        hidden = self.backbone(**inputs).last_hidden_state
        return self.classifier(self._masked_mean(hidden, lengths))

    # ----------------------------------------------------------------------

    def check_grads(self, loader):
        """Step-1 gate: prove gradients reach the conv stack and blocks 0..n."""
        self.logger.info("\n" + "=" * 72)
        self.logger.info("GRADIENT CHECK")
        self.logger.info("=" * 72)

        self.backbone.train()
        self.classifier.train()

        audio, lengths, labels = next(iter(loader))
        criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        loss = criterion(self.forward(audio, lengths), labels.to(self.device))
        loss.backward()

        def report(tag, named):
            named = list(named)
            if not named:
                self.logger.info(f"  {tag:<28} (no parameters)")
                return None
            got = [(n, p) for n, p in named if p.requires_grad and p.grad is not None]
            norm = sum(float(p.grad.norm()) ** 2 for _, p in got) ** 0.5 if got else 0.0
            n_train = sum(1 for _, p in named if p.requires_grad)
            status = "OK " if norm > 0 else "ZERO"
            self.logger.info(
                f"  {tag:<28} trainable={n_train:<4} with_grad={len(got):<4} "
                f"|grad|={norm:.6e}  {status}"
            )
            return norm

        results = {}
        for mod_name in ("feature_extractor", "feature_projection"):
            mod = getattr(self.backbone, mod_name, None)
            results[mod_name] = report(mod_name, mod.named_parameters()) if mod else None

        n_ft = min(self.args.num_layers, len(self.backbone.encoder.layers))
        for i in range(n_ft):
            results[f"encoder.layers[{i}]"] = report(
                f"encoder.layers[{i}]", self.backbone.encoder.layers[i].named_parameters()
            )
        last = len(self.backbone.encoder.layers) - 1
        results[f"encoder.layers[{last}]"] = report(
            f"encoder.layers[{last}] (frozen)", self.backbone.encoder.layers[last].named_parameters()
        )
        results["classifier"] = report("classifier", self.classifier.named_parameters())

        self.logger.info("-" * 72)
        conv_ok = (self.args.freeze_conv
                   or (results.get("feature_extractor") or 0) > 0
                   or (results.get("feature_projection") or 0) > 0)
        blocks_ok = all((results.get(f"encoder.layers[{i}]") or 0) > 0 for i in range(n_ft))
        frozen_ok = not results.get(f"encoder.layers[{last}]")

        self.logger.info(f"  conv stack receives gradient : {conv_ok}")
        self.logger.info(f"  blocks 0..{n_ft - 1} receive gradient : {blocks_ok}")
        self.logger.info(f"  deep layers stay frozen      : {frozen_ok}")

        passed = conv_ok and blocks_ok and frozen_ok
        self.logger.info(f"\n  GATE: {'PASS' if passed else 'FAIL'}")
        self.logger.info("=" * 72)

        self.backbone.zero_grad(set_to_none=True)
        self.classifier.zero_grad(set_to_none=True)
        return passed

    # ----------------------------------------------------------------------

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
            out[name] = DataLoader(
                ds,
                batch_size=self.args.batch_size,
                shuffle=(name == "train"),
                num_workers=self.args.num_workers,
                collate_fn=collate,
                pin_memory=(self.device == "cuda"),
            )
            self.logger.info(f"  {name:<6} {len(ds):>6} files, {len(out[name]):>5} batches")

        return out["train"], out["val"], out["test"]

    def _eval(self, loader, desc):
        self.backbone.eval()
        self.classifier.eval()

        preds, golds, total_loss = [], [], 0.0
        criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))

        with torch.no_grad():
            for audio, lengths, labels in tqdm(loader, desc=desc, leave=False):
                logits = self.forward(audio, lengths)
                total_loss += float(criterion(logits, labels.to(self.device)))
                preds.extend(logits.argmax(1).cpu().numpy())
                golds.extend(labels.numpy())

        return {
            "loss": total_loss / max(len(loader), 1),
            "acc": accuracy_score(golds, preds),
            "f1_macro": f1_score(golds, preds, average="macro", zero_division=0),
            "preds": preds,
            "labels": golds,
        }

    def train(self, train_loader, val_loader):
        self.logger.info("\n" + "=" * 72)
        self.logger.info("ADAPTATION")
        self.logger.info("=" * 72)

        criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))

        conv_ids = set()
        groups = []
        if not self.args.freeze_conv:
            conv_params = []
            for mod_name in ("feature_extractor", "feature_projection"):
                mod = getattr(self.backbone, mod_name, None)
                if mod is None:
                    continue
                for p in mod.parameters():
                    if p.requires_grad:
                        conv_params.append(p)
                        conv_ids.add(id(p))
            if conv_params:
                groups.append({"params": conv_params, "lr": self.args.lr_conv})

        block_params = [p for p in self.backbone.parameters()
                        if p.requires_grad and id(p) not in conv_ids]
        groups.append({"params": block_params, "lr": self.args.lr_backbone})
        groups.append({"params": list(self.classifier.parameters()), "lr": self.args.lr_head})

        optimizer = optim.AdamW(groups)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=self.args.lr_patience
        )

        history = []
        best_f1, best_epoch, stale = -1.0, 0, 0
        ckpt_path = self.output_dir / "checkpoints" / "best_model.pth"

        for epoch in range(1, self.args.max_epochs + 1):
            self.backbone.train()
            self.classifier.train()

            run_loss, preds, golds = 0.0, [], []
            t0 = time.time()

            for audio, lengths, labels in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
                labels = labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)

                logits = self.forward(audio, lengths)
                loss = criterion(logits, labels)
                loss.backward()

                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in groups for p in g["params"]], self.args.grad_clip
                    )
                optimizer.step()

                run_loss += float(loss.detach())
                preds.extend(logits.argmax(1).detach().cpu().numpy())
                golds.extend(labels.cpu().numpy())

            tr_loss = run_loss / max(len(train_loader), 1)
            tr_f1 = f1_score(golds, preds, average="macro", zero_division=0)
            va = self._eval(val_loader, "val")
            scheduler.step(va["f1_macro"])

            history.append({
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_f1_macro": tr_f1,
                "train_acc": accuracy_score(golds, preds),
                "val_loss": va["loss"],
                "val_f1_macro": va["f1_macro"],
                "val_acc": va["acc"],
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - t0,
            })

            self.logger.info(
                f"epoch {epoch:>3}  train loss {tr_loss:.4f} F1 {tr_f1:.4f}  |  "
                f"val loss {va['loss']:.4f} F1 {va['f1_macro']:.4f} acc {va['acc']:.4f}  "
                f"({time.time() - t0:.0f}s)"
            )

            if va["f1_macro"] > best_f1:
                best_f1, best_epoch, stale = va["f1_macro"], epoch, 0
                torch.save({
                    "epoch": epoch,
                    "model_name": self.args.model,
                    "model_id": MODEL_IDS[self.args.model],
                    "backbone_state_dict": self.backbone.state_dict(),
                    "classifier_state_dict": self.classifier.state_dict(),
                    "classes": self.classes,
                    "label_key": self.label_key,
                    "val_f1_macro": va["f1_macro"],
                    "val_acc": va["acc"],
                    "config": {
                        "num_layers": self.args.num_layers,
                        "freeze_conv": self.args.freeze_conv,
                        "layerdrop": self.args.layerdrop,
                        "lr_conv": self.args.lr_conv,
                        "lr_backbone": self.args.lr_backbone,
                        "lr_head": self.args.lr_head,
                        "masked_pooling": True,
                    },
                }, ckpt_path)
                self.logger.info(f"           saved (val F1 {va['f1_macro']:.4f})")
            else:
                stale += 1
                if stale >= self.args.patience:
                    self.logger.info(f"early stop at epoch {epoch}")
                    break

        self.logger.info(f"\nbest val macro-F1 {best_f1:.4f} at epoch {best_epoch}")
        return history, best_f1, best_epoch, ckpt_path

    def run(self):
        train_loader, val_loader, test_loader = self.loaders()

        if self.args.check_grads:
            return {"gate_passed": self.check_grads(train_loader)}

        history, best_f1, best_epoch, ckpt_path = self.train(train_loader, val_loader)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.backbone.load_state_dict(ckpt["backbone_state_dict"])
        self.classifier.load_state_dict(ckpt["classifier_state_dict"])
        self.logger.info(f"loaded best checkpoint (epoch {ckpt['epoch']})")

        test = self._eval(test_loader, "test")
        self.logger.info(f"\nTEST macro-F1 {test['f1_macro']:.4f}  acc {test['acc']:.4f}")

        summary = {
            "model": self.args.model,
            "task": self.manifest.get("task"),
            "num_classes": self.num_classes,
            "classes": self.classes,
            "excluded_species": self.manifest.get("excluded_species"),
            "num_layers_finetuned": self.args.num_layers,
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

        self.logger.info(f"\nwrote {self.output_dir / 'adaptation_summary.json'}")
        self.logger.info(f"checkpoint for probing: {ckpt_path}")

        # step-2 gate
        if test["f1_macro"] < self.args.min_species_f1:
            self.logger.warning(
                f"\nGATE WARNING: test macro-F1 {test['f1_macro']:.4f} < "
                f"{self.args.min_species_f1}. Adaptation may have damaged the "
                f"encoder -- inspect before probing hyrax."
            )
        return summary


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Staged 7-species adaptation by real fine-tuning")
    p.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    p.add_argument("--manifest", default=None,
                   help="default: outputs/phase3/manifests_species7/species_id.json")
    p.add_argument("--output-dir", default=None)

    p.add_argument("--num-layers", type=int, default=4,
                   help="transformer blocks to unfreeze, from the input side (default 4)")
    p.add_argument("--freeze-conv", action="store_true",
                   help="keep the CNN front-end frozen (reproduces phase2_05 / LoRA behaviour)")
    p.add_argument("--layerdrop", type=float, default=0.0)

    p.add_argument("--lr-conv", type=float, default=1e-5)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
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

    p.add_argument("--check-grads", action="store_true",
                   help="one batch, verify gradient flow, exit (step-1 gate)")
    p.add_argument("--debug", action="store_true", help="tiny subset, 3 epochs")
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--min-species-f1", type=float, default=0.90,
                   help="step-2 gate: warn if test macro-F1 falls below this")
    p.add_argument("--no-cudnn", action="store_true",
                   help="disable cuDNN (V100 workaround inherited from phase2_05)")

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

    root = Path(__file__).parent.parent
    with open(root / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)

    out_root = Path(config["paths"]["output_dir"])
    manifest = Path(args.manifest) if args.manifest else \
        out_root / "phase3" / "manifests_species7" / "species_id.json"

    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")

    tag = args.model + ("_convfrozen" if args.freeze_conv else "")
    # adapt_species_id = adaptation supervised by SPECIES labels. The individual-ID
    # variant writes to adapt_individual_id; keep the two apart, their checkpoints
    # are not interchangeable.
    output_dir = Path(args.output_dir) if args.output_dir else \
        out_root / "phase3" / "adapt_species_id" / tag

    logger = setup_logger("Phase3_Species7_FT", config["experiment"]["log_level"])
    logger.info("=" * 72)
    logger.info("PHASE 3 - STEP 23: STAGED 7-SPECIES ADAPTATION (real fine-tuning)")
    logger.info("=" * 72)
    logger.info(f"manifest:  {manifest}")
    logger.info(f"output:    {output_dir}")

    trainer = StagedFineTuner(config, args, manifest, output_dir, logger)
    result = trainer.run()

    if args.check_grads:
        raise SystemExit(0 if result.get("gate_passed") else 1)


if __name__ == "__main__":
    main()
