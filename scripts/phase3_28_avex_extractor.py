#!/usr/bin/env python3
"""
Phase 3 - Step 28: AVES 2 (EAT) layer extractor.

WHY THIS FILE EXISTS
--------------------
The six models already evaluated are HuggingFace `Wav2Vec2Model`-likes: they
load with `from_pretrained`, accept a raw waveform of any length, and hand back
`hidden_states` when asked. `esp-aves2-eat-bio` is none of those things, so
phase3_24's `LayerExtractor` cannot load it. This module provides a wrapper with
the same three things phase3_24 needs -- a per-layer list, a hidden size, and a
layer count -- and nothing about how the existing models load is touched.

WHAT WAS MEASURED (not read off the docs), avex 1.3.0, 2026-08-20
-----------------------------------------------------------------
  loading      `avex.load_model("esp_aves2_eat_bio", return_features_only=True)`.
               The avex id uses UNDERSCORES; the HF repo uses hyphens.
               Backbone is worstchan/EAT-base_epoch30_pretrain.

  canvas       forward() returns (B, 513, 768) for EVERY input length. Inside:
               1024 mel frames x 128 mels at hop 160 = 10.24 s exactly.
               513 tokens = 1 CLS + 512 patches, laid out TIME-MAJOR as
               64 time x 8 freq. Proved directly: local_encoder.proj emits
               (B, 768, 64, 8).

  long input   deterministic start-crop of the first 10.24 s. NOT the random
               crop that AudioConfig.window_selection="random" implies -- the
               EAT path uses its own EATAudioProcessor with target_length=1024
               frames. Verified by sweeping a 1 s burst through a 30 s file:
               the peak column tracks the offset linearly up to ~10 s and is
               exactly zero beyond it.

  onset lag    a burst at t peaks ~2-3 columns LATE, so the mask is widened by
               MASK_MARGIN_COLS columns rather than clipping real signal.

MASKED POOLING IS MANDATORY, NOT A PREFERENCE
---------------------------------------------
Bouts here have a median duration near 1.0 s, so a typical bout occupies ~6 of
the 64 time columns and the other ~58 are zero padding. Measured:

    bout    cos(unmasked_pool, silence_pool)    cos(masked_pool, silence_pool)
    0.4 s               0.9992                            0.8658
    1.0 s               0.9986                            0.9339
    1.4 s               0.9981                            0.9453

Unmasked pooling of a real bout is ~99.8% identical to pooling pure silence --
it measures the padding, not the animal. Two different 1.4 s tones come out at
cos 0.9997 unmasked versus 0.9935 masked, i.e. ~20x more separable in (1-cos).

So neither of avex's own pooling paths can be used: `aggregation="mean"` is an
unmasked `torch.mean(dim=1)`, and `forward(padding_mask=...)` is documented as
"kept for interface compatibility" and is genuinely ignored by the model.

LAYER MAPPING (13 indices, documented because it does NOT match the others)
---------------------------------------------------------------------------
    0        backbone.model.local_encoder      512 tokens, no CLS
             the pre-transformer patch embedding
    1..12    backbone.model.blocks.{0..11}     513 tokens (CLS + 512)

Index 0 is the analogue of "pre-transformer output" required by the probe
convention, but it is a MEL-PATCH embedding, not a waveform CNN front-end. The
existing "hyrax peaks at layer 0 = CNN front-end" reading does not transfer to
this model and must not be reported as if it did.

Two further mismatches with avex's own tooling, both deliberate:
  * `target_layers=["all"]` discovers `attn.proj` SUB-modules, which are not
    residual-stream outputs. We hook the block roots instead, so index i means
    the same kind of quantity as HF `hidden_states[i]`.
  * forward()'s final output != block 11 (a final norm is applied). Index 12 is
    block 11, matching HF semantics where hidden_states[-1] is the last block.

CLS is excluded from pooling. The convention is "mean-pool over time"; CLS is
not a time position, and including it would make index 0 (which has no CLS)
incomparable to the rest.

ZERO-SHOT ONLY. This module refuses a checkpoint on purpose -- there is no
fine-tuned AVES cell in this experiment.
"""

import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

AVEX_MODEL_ID = "esp_aves2_eat_bio"
EAT_REMOTE_REPO = "worstchan/EAT-base_epoch30_pretrain"

SAMPLE_RATE = 16000
N_FRAMES = 1024          # mel frames the EAT processor targets
HOP_LENGTH = 160
CANVAS_SAMPLES = N_FRAMES * HOP_LENGTH        # 163840 = 10.24 s
CANVAS_SECONDS = CANVAS_SAMPLES / SAMPLE_RATE
TIME_COLS = 64           # patch grid: 64 time x 8 freq
FREQ_ROWS = 8
N_PATCHES = TIME_COLS * FREQ_ROWS             # 512
HIDDEN_SIZE = 768
N_BLOCKS = 12
NUM_LAYERS = N_BLOCKS + 1                     # 0 = patch embedding, 1..12 = blocks

# a burst at time t peaks ~2-3 columns late, so widen rather than clip
MASK_MARGIN_COLS = 3

LAYER_MODULES = (["backbone.model.local_encoder"]
                 + [f"backbone.model.blocks.{i}" for i in range(N_BLOCKS)])


def shim_eat_for_transformers5(logger=None):
    """Make the remote EAT checkpoint loadable under transformers >= 5.

    The remote `EATModel` calls `super().__init__(config)` but never
    `post_init()`, so `all_tied_weights_keys` -- which transformers 5.x sets
    there and then dereferences in `_move_missing_keys_from_meta_to_device` --
    never exists, and `from_pretrained` dies with

        AttributeError: 'EATModel' object has no attribute 'all_tied_weights_keys'

    avex's documented workaround is to pin transformers<5.0.0. That is not an
    option here: the six existing models were all measured on the installed
    5.x, and downgrading would put every published number in this repo at risk
    of moving for reasons unrelated to the experiment.

    So patch the one class instead. The dict is set PER INSTANCE (not as a
    shared class attribute, which transformers would later mutate in place via
    `.update()`), and only when genuinely absent, so if upstream ever fixes
    this the shim quietly does nothing. EATModel is an audio encoder with no
    tied weights, so {} is the correct value rather than a placeholder.
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module("modeling_eat.EATModel", EAT_REMOTE_REPO)
    if getattr(cls, "_avex_tied_weights_shim", False):
        return cls

    original_init = cls.__init__

    def patched_init(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        if "all_tied_weights_keys" not in self.__dict__:
            self.all_tied_weights_keys = {}

    cls.__init__ = patched_init
    cls._avex_tied_weights_shim = True
    if logger:
        logger.info("applied transformers>=5 shim to remote EATModel "
                    "(missing all_tied_weights_keys)")
    return cls


def valid_time_columns(duration_seconds):
    """How many of the 64 time columns carry real audio, not padding.

    Widened by MASK_MARGIN_COLS because the patch/attention receptive field
    makes a burst peak 2-3 columns after its true onset; clipping those would
    throw away the tail of every bout.
    """
    frac = min(1.0, max(0.0, duration_seconds) / CANVAS_SECONDS)
    cols = int(np.ceil(frac * TIME_COLS)) + MASK_MARGIN_COLS
    return int(np.clip(cols, 1, TIME_COLS))


def fit_to_canvas(audio, pad_mode="zero"):
    """Return exactly CANVAS_SAMPLES of waveform, plus the real duration.

    Doing the crop/pad here rather than leaving it to avex keeps the behaviour
    explicit and independent of any future change to EATAudioProcessor.

    pad_mode
        "zero"  what avex does: trailing zeros. The real duration is the
                original one, so the mask covers only the real part.
        "tile"  repeat the bout until it fills the canvas. Every column is then
                real audio, so the mask covers all 64. This is the sensitivity
                check ONLY -- a tiled 1.4 s bout is a different stimulus (a
                repeated call), not the unit the bout manifests define.
    """
    n = len(audio)
    if n >= CANVAS_SAMPLES:
        return audio[:CANVAS_SAMPLES], CANVAS_SECONDS, n > CANVAS_SAMPLES

    if pad_mode == "tile":
        reps = int(np.ceil(CANVAS_SAMPLES / max(1, n)))
        return np.tile(audio, reps)[:CANVAS_SAMPLES], CANVAS_SECONDS, False

    out = np.zeros(CANVAS_SAMPLES, dtype=np.float32)
    out[:n] = audio
    return out, n / SAMPLE_RATE, False


class AvesLayerExtractor:
    """Mean-pooled embedding from all 13 layers, one forward pass per batch.

    Interface-compatible with phase3_24.LayerExtractor: exposes `num_layers`
    and `embed_all_layers(audio) -> (num_layers, hidden)`. Adds `embed_batch`,
    which the runners use because the fixed canvas makes batching free.
    """

    def __init__(self, model_name, checkpoint, logger, pooling="masked_mean",
                 pad_mode="zero", batch_size=16, device=None):
        if checkpoint is not None:
            raise ValueError(
                "phase3_28 is the ZERO-SHOT AVES path: there is no fine-tuned "
                "AVES cell in this experiment, so --checkpoint is refused here."
            )
        if pooling not in {"masked_mean", "unmasked_mean", "cls"}:
            raise ValueError(f"unknown pooling: {pooling}")
        if pad_mode not in {"zero", "tile"}:
            raise ValueError(f"unknown pad_mode: {pad_mode}")

        self.logger = logger
        self.pooling = pooling
        self.pad_mode = pad_mode
        self.batch_size = batch_size
        self.num_layers = NUM_LAYERS
        self.hidden_size = HIDDEN_SIZE
        self.n_cropped = 0          # inputs longer than the 10.24 s canvas

        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available()
                                 else "cpu")

        shim_eat_for_transformers5(logger)
        from avex import load_model

        # avex resolves the weights from an hf:// URI and calls a REMOTE
        # exists() on it before ever consulting its local cache
        # (avex/models/utils/load.py:537). On a compute node with no outbound
        # internet that either hangs until the connect timeout or, under
        # HF_HUB_OFFLINE, fails outright with "Checkpoint not found" -- even
        # though the file is sitting in ESP_CACHE_HOME.
        #
        # So when the cached file is already present, hand avex the local path
        # and skip the network entirely. The cache name is a sha256 of the
        # source URI, hence deterministic across machines.
        local_ckpt = self._cached_checkpoint()
        if local_ckpt is not None:
            logger.info(f"using cached weights, no network: {local_ckpt}")

        self.model = load_model(AVEX_MODEL_ID, device=self.device,
                                checkpoint_path=local_ckpt,
                                return_features_only=True)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        modules = dict(self.model.named_modules())
        missing = [n for n in LAYER_MODULES if n not in modules]
        if missing:
            raise RuntimeError(
                f"expected EAT modules not found: {missing[:3]} -- the avex or "
                f"upstream EAT layout has changed and the layer mapping in this "
                f"file is no longer valid."
            )

        self._captured = {}
        self._handles = [
            modules[name].register_forward_hook(self._make_hook(name))
            for name in LAYER_MODULES
        ]

        logger.info(f"{model_name} (avex {AVEX_MODEL_ID}) on {self.device}: "
                    f"{self.num_layers} layers, dim {self.hidden_size}")
        logger.info(f"  0 = patch embedding (pre-transformer, mel patches -- "
                    f"NOT a waveform CNN front-end), 1..{N_BLOCKS} = blocks")
        logger.info(f"  canvas {CANVAS_SECONDS:.2f}s, grid {TIME_COLS}x{FREQ_ROWS}, "
                    f"pooling={pooling}, pad_mode={pad_mode}, batch={batch_size}")

    @staticmethod
    def _cached_checkpoint():
        """Local path to the already-downloaded AVES weights, or None.

        Mirrors avex's own cache naming: ESP_CACHE_HOME (default ~/.cache/esp)
        holding a file named from the first 16 hex of sha256(source_uri).
        """
        import hashlib
        import os

        root = Path(os.environ.get("ESP_CACHE_HOME", Path.home() / ".cache" / "esp"))
        uri = f"hf://EarthSpeciesProject/{AVEX_MODEL_ID.replace('_', '-')}/" \
              f"{AVEX_MODEL_ID.replace('_', '-')}.safetensors"
        digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
        path = root / f"{AVEX_MODEL_ID.replace('_', '-')}-{digest}.safetensors"
        if path.exists():
            return str(path)
        # fall back to any matching file, in case the naming scheme shifts
        hits = sorted(root.glob(f"{AVEX_MODEL_ID.replace('_', '-')}-*.safetensors"))
        return str(hits[0]) if hits else None

    def _make_hook(self, name):
        def hook(_module, _inputs, output):
            self._captured[name] = output[0] if isinstance(output, tuple) else output
        return hook

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _pool(self, tokens, valid_cols):
        """(B, T, D) tokens -> (B, D), masking padded time columns.

        Index 0 arrives with 512 tokens and no CLS; the blocks arrive with 513.
        CLS is dropped so every layer is pooled over the same time grid.
        """
        B, T, D = tokens.shape
        if T == N_PATCHES + 1:
            cls_tok, patches = tokens[:, 0], tokens[:, 1:]
        elif T == N_PATCHES:
            cls_tok, patches = None, tokens
        else:
            raise RuntimeError(f"unexpected token count {T}; expected "
                               f"{N_PATCHES} or {N_PATCHES + 1}")

        if self.pooling == "cls":
            if cls_tok is None:                 # layer 0 has no CLS
                return patches.mean(dim=1)
            return cls_tok

        grid = patches.reshape(B, TIME_COLS, FREQ_ROWS, D)
        if self.pooling == "unmasked_mean":
            return grid.reshape(B, -1, D).mean(dim=1)

        # masked_mean: average only the columns that carry real audio
        idx = torch.arange(TIME_COLS, device=grid.device).view(1, TIME_COLS)
        keep = (idx < valid_cols.view(B, 1)).float()                # (B, cols)
        w = keep.view(B, TIME_COLS, 1, 1)
        return (grid * w).sum(dim=(1, 2)) / (keep.sum(dim=1).view(B, 1) * FREQ_ROWS)

    def embed_batch(self, audios):
        """list of 1-D float arrays -> (B, num_layers, hidden)."""
        canvases, cols = [], []
        for a in audios:
            wav, dur, cropped = fit_to_canvas(np.asarray(a, dtype=np.float32),
                                              self.pad_mode)
            canvases.append(wav)
            cols.append(valid_time_columns(dur))
            self.n_cropped += int(cropped)

        x = torch.from_numpy(np.stack(canvases)).to(self.device)
        valid = torch.tensor(cols, dtype=torch.float32, device=self.device)

        self._captured.clear()
        with torch.no_grad():
            self.model(x)
            pooled = torch.stack(
                [self._pool(self._captured[n], valid) for n in LAYER_MODULES],
                dim=1,
            )
        self._captured.clear()
        return pooled.float().cpu().numpy()

    def embed_all_layers(self, audio):
        """One input -> (num_layers, hidden). phase3_24-compatible."""
        return self.embed_batch([audio])[0]

    def embed_many(self, audios):
        """Batched over the whole list -> (N, num_layers, hidden)."""
        out = []
        for i in range(0, len(audios), self.batch_size):
            out.append(self.embed_batch(audios[i:i + self.batch_size]))
        return np.concatenate(out, axis=0) if out else np.empty(
            (0, self.num_layers, self.hidden_size), dtype=np.float32)

    def extract_split(self, items, class_to_idx, split, label_key="individual"):
        """One embedding per input unit -> (X, y), X = (n, num_layers, hidden).

        The chunking rules come from phase3_24.chunks_for_item, NOT from a copy
        living here, so the bout/window semantics cannot drift away from the
        path the other six models take. The only difference is that chunks are
        collected and forwarded in batches, which the fixed canvas makes free.
        """
        import time

        from tqdm import tqdm

        from phase3_20_probe_audit import resolve
        from phase3_24_hyrax_layer_probe import (STRIDE_SECONDS, WINDOW_SECONDS,
                                                 _load_cached, chunks_for_item)

        window = int(WINDOW_SECONDS * SAMPLE_RATE)
        stride = int(STRIDE_SECONDS * SAMPLE_RATE)

        pending, labels = [], []
        load_failed, skipped, n_bout_items = 0, 0, 0
        t0 = time.time()

        for item in tqdm(items, desc=split, leave=False):
            try:
                audio = _load_cached(str(resolve(item["file"])))
            except Exception:
                load_failed += 1
                continue

            chunks, is_bout = chunks_for_item(item, audio, window, stride)
            if not chunks:
                skipped += 1
                continue
            n_bout_items += int(is_bout)

            label = class_to_idx[item[label_key]]
            for chunk in chunks:
                pending.append(chunk)
                labels.append(label)

        if load_failed:
            self.logger.warning(f"  {split}: {load_failed} files failed to load")
        if skipped:
            self.logger.warning(f"  {split}: {skipped} items had an empty segment")
        if not pending:
            raise RuntimeError(f"no embeddings for split {split}")

        X = self.embed_many(pending).astype(np.float32)
        y = np.asarray(labels)
        unit = "bouts" if n_bout_items else "windows"
        self.logger.info(f"  {split}: {len(y)} {unit}, {X.shape[1]} layers, "
                         f"dim {X.shape[2]} ({time.time() - t0:.0f}s)")
        if self.n_cropped:
            self.logger.warning(
                f"  {self.n_cropped} input(s) exceeded the {CANVAS_SECONDS:.2f}s "
                f"canvas and were start-cropped")
        return X, y

    def provenance(self):
        """Recorded into every result JSON so the run is self-describing."""
        return {
            "loader": "avex",
            "avex_model_id": AVEX_MODEL_ID,
            "backbone_repo": EAT_REMOTE_REPO,
            "sample_rate": SAMPLE_RATE,
            "canvas_seconds": CANVAS_SECONDS,
            "patch_grid": {"time": TIME_COLS, "freq": FREQ_ROWS},
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "layer_mapping": {
                "0": "backbone.model.local_encoder (pre-transformer mel-patch "
                     "embedding; NOT a waveform CNN front-end)",
                "1..12": "backbone.model.blocks.{0..11} residual-stream outputs",
            },
            "pooling": self.pooling,
            "pad_mode": self.pad_mode,
            "mask_margin_cols": MASK_MARGIN_COLS,
            "cls_excluded_from_pooling": self.pooling != "cls",
            "long_input_policy": f"deterministic start-crop to {CANVAS_SECONDS:.2f}s",
            "n_inputs_cropped": self.n_cropped,
        }
