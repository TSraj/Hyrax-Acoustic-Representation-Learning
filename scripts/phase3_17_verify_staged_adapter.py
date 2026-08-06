#!/usr/bin/env python3
"""
Phase B1 / Step B1-3: verification gate for a saved staged-adaptation adapter.

This is the Phase B -> Phase C handoff contract. It proves that the artefact on
disk can be loaded as a FROZEN encoder exposing every layer, which is exactly
what the frozen probe needs, before any probing work is built on top of it.

Checks:

  LOAD
    1. adapter_config.json + adapter_model.safetensors + adapter_meta.json exist.
    2. PeftModel.from_pretrained() loads onto a fresh base encoder.
    3. trainable params == 0. The frozen requirement holds BY CONSTRUCTION
       here, not by a caller remembering to freeze anything.
    4. adapter_meta.json declares the task we expect: 7 classes, hyrax excluded.

  LAYER ACCESS (what Phase C sweeps)
    5. output_hidden_states passes through the PEFT wrapper and returns
       num_hidden_layers + 1 states (13 for HuBERT, 25 for XLS-R).
    6. Indexing matches the base-model sweeps: index 0 is the pre-transformer
       feature_projection output, indices 1..N are transformer blocks, and
       last_hidden_state == hidden_states[-1].
    7. LAYER-0 EQUALITY (permanent assertion). hidden_states[0] must be
       bit-identical between the base and adapted encoders: LoRA sits on the
       attention projections, which layer 0 precedes, so nothing can change it.
       A mismatch means the wrong base checkpoint was loaded, or that something
       is adapting the CNN/feature-projection path - either would silently
       corrupt every layer comparison in Phase C.
    8. Every transformer layer 1..N DOES differ from base, so the adaptation is
       actually present and not a no-op load.

  POOLING VARIANTS (what Phase C extracts)
    9. Mean pooling over time yields [D] per layer, D == embedding_dim.
   10. Head-0 context vectors are recoverable via a forward hook on
       attention.out_proj's INPUT (heads are concatenated at that point;
       head h occupies dims [h*head_dim : (h+1)*head_dim]). Verifies one hook
       fires per transformer layer and the slice has width head_dim.

Exit 0 = the adapter is usable by Phase C.

Usage:
    python scripts/phase3_17_verify_staged_adapter.py \
        --adapter-dir outputs/staged_lora/species7/hubert_base/seed42/adapter
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

EXPECTED_LAYERS = {'hubert_base': 12, 'xls_r': 24}
EXPECTED_DIM = {'hubert_base': 768, 'xls_r': 1024}
SAMPLE_RATE = 16000
WINDOW_SECONDS = 5.0


class Gate:
    def __init__(self):
        self.results = []

    def check(self, name, ok, detail="", always_show_detail=False):
        self.results.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"\n         {detail}" if detail and (not ok or always_show_detail)
                 else ""))
        return bool(ok)

    def section(self, title):
        print(f"\n{title}\n" + "-" * 78)

    @property
    def passed(self):
        return all(ok for _, ok, _ in self.results)

    def summary(self, adapter_dir):
        n_fail = sum(1 for _, ok, _ in self.results if not ok)
        print("\n" + "=" * 78)
        if self.passed:
            print(f"STAGED ADAPTER GATE: PASS ({len(self.results)} checks)")
            print(f"{adapter_dir} is usable as a frozen encoder for Phase C.")
        else:
            print(f"STAGED ADAPTER GATE: FAIL "
                  f"({n_fail}/{len(self.results)} checks failed)")
            print("DO NOT probe with this adapter.")
            for name, ok, detail in self.results:
                if not ok:
                    print(f"  - {name}: {detail}")
        print("=" * 78)


def load_base(model_name, model_id):
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, HubertModel
    cls = HubertModel if model_name == 'hubert_base' else Wav2Vec2Model
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    base = cls.from_pretrained(model_id, use_safetensors=True)
    # Match training: LayerDrop off, so eval is deterministic layer-for-layer.
    base.config.layerdrop = 0.0
    return fe, base


def head0_by_hook(model, inputs, head_dim):
    """Capture head 0's context vector per layer.

    Hooks attention.out_proj and grabs its INPUT rather than its output: at that
    point the per-head context vectors are still concatenated along the feature
    axis, so head h is dims [h*head_dim : (h+1)*head_dim]. After out_proj the
    heads are mixed and no slice corresponds to a single head.

    Works regardless of the attention implementation (eager/sdpa), because
    out_proj's input shape is [B, T, hidden] either way.
    """
    encoder = model.base_model.model.encoder
    captured = {}

    def make_hook(idx):
        def hook(module, args, kwargs, output):
            tensor = args[0] if args else kwargs.get('hidden_states')
            captured[idx] = tensor.detach()
        return hook

    handles = [encoder.layers[i].attention.out_proj.register_forward_hook(
        make_hook(i), with_kwargs=True) for i in range(len(encoder.layers))]
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    return {i: t[..., 0:head_dim] for i, t in captured.items()}


def main():
    p = argparse.ArgumentParser(description="Phase B1 staged-adapter gate")
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--expect-classes", type=int, default=7)
    p.add_argument("--expect-excluded", default="hyrax")
    args = p.parse_args()

    from peft import PeftModel

    adapter_dir = Path(args.adapter_dir)
    print("=" * 78)
    print("PHASE B1 - STAGED ADAPTER GATE")
    print(f"adapter: {adapter_dir}")
    print("=" * 78)

    gate = Gate()
    gate.section("LOAD")

    required = ['adapter_config.json', 'adapter_model.safetensors', 'adapter_meta.json']
    missing = [f for f in required if not (adapter_dir / f).exists()]
    if not gate.check("required files present", not missing, f"missing: {missing}"):
        gate.summary(adapter_dir)
        return 1

    with open(adapter_dir / "adapter_meta.json") as f:
        meta = json.load(f)
    model_name = meta['model']
    print(f"         model={model_name} base={meta['base_model_id']} "
          f"best_epoch={meta['best_epoch']} val_f1={meta['best_val_f1_macro']:.4f}")

    gate.check("adapter_meta declares the expected class count",
               meta['num_classes'] == args.expect_classes,
               f"expected {args.expect_classes}, got {meta['num_classes']}")
    gate.check(f"'{args.expect_excluded}' excluded and absent from class_names",
               args.expect_excluded in (meta.get('excluded_species') or [])
               and args.expect_excluded not in meta['class_names'],
               f"excluded_species={meta.get('excluded_species')}, "
               f"class_names={meta['class_names']}")

    fe, base = load_base(model_name, meta['base_model_id'])
    _, base_ref = load_base(model_name, meta['base_model_id'])
    adapted = PeftModel.from_pretrained(base, str(adapter_dir))
    gate.check("PeftModel.from_pretrained() loaded the adapter", True)

    n_trainable = sum(q.numel() for q in adapted.parameters() if q.requires_grad)
    gate.check("trainable params == 0 (frozen by construction)",
               n_trainable == 0, f"got {n_trainable:,}")

    n_lora = sum(1 for n, _ in adapted.named_parameters() if 'lora_' in n)
    expected_lora = EXPECTED_LAYERS[model_name] * 4 * 2
    gate.check(f"adapter tensor count == layers x 4 modules x 2 ({expected_lora})",
               n_lora == expected_lora, f"got {n_lora}")

    # ------------------------------------------------------------------ layers
    gate.section("LAYER ACCESS")

    adapted.eval()
    base_ref.eval()
    rng = np.random.default_rng(0)
    audio = [rng.standard_normal(int(WINDOW_SECONDS * SAMPLE_RATE)).astype(np.float32)]
    inputs = fe(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)

    with torch.no_grad():
        out_a = adapted(**inputs, output_hidden_states=True)
        out_b = base_ref(**inputs, output_hidden_states=True)

    n_layers = EXPECTED_LAYERS[model_name]
    gate.check(f"output_hidden_states passes through PEFT and returns "
               f"{n_layers + 1} states",
               len(out_a.hidden_states) == n_layers + 1,
               f"got {len(out_a.hidden_states)}")
    gate.check("state count matches base encoder (indexing unchanged by PEFT)",
               len(out_a.hidden_states) == len(out_b.hidden_states),
               f"adapted {len(out_a.hidden_states)} vs base {len(out_b.hidden_states)}")
    gate.check("last_hidden_state == hidden_states[-1]",
               torch.allclose(out_a.last_hidden_state, out_a.hidden_states[-1]))

    deltas = [(a - b).abs().max().item()
              for a, b in zip(out_a.hidden_states, out_b.hidden_states)]

    # PERMANENT ASSERTION: layer 0 is pre-transformer, so adapters cannot reach it.
    gate.check("LAYER-0 EQUALITY: hidden_states[0] bit-identical to base",
               deltas[0] == 0.0,
               f"max|delta| = {deltas[0]:.3e}, must be exactly 0. A non-zero "
               f"value means the wrong base checkpoint was loaded, or something "
               f"is adapting the CNN / feature-projection path - either would "
               f"invalidate every layer comparison in Phase C.")
    gate.check(f"all {n_layers} transformer layers differ from base "
               f"(adaptation is present, load was not a no-op)",
               all(d > 0 for d in deltas[1:]),
               f"zero-drift layers: {[i for i, d in enumerate(deltas) if i > 0 and d == 0]}")

    print("         per-layer max|delta| vs base encoder:")
    for start in range(0, len(deltas), 13):
        chunk = deltas[start:start + 13]
        print(f"           L{start:2d}-{start + len(chunk) - 1:2d}: "
              + " ".join(f"{d:.3f}" for d in chunk))

    # ---------------------------------------------------------------- pooling
    gate.section("POOLING VARIANTS")

    dim = EXPECTED_DIM[model_name]
    mean_pooled = out_a.hidden_states[-1].mean(dim=1)
    gate.check(f"mean pooling over time -> [{dim}]",
               mean_pooled.shape[-1] == dim and mean_pooled.ndim == 2,
               f"got {tuple(mean_pooled.shape)}")

    head_dim = dim // meta['num_attention_heads']
    gate.check(f"head_dim == {meta['head_dim']} (from meta)",
               head_dim == meta['head_dim'], f"computed {head_dim}")

    heads = head0_by_hook(adapted, inputs, head_dim)
    gate.check(f"head-0 hook fired on all {n_layers} transformer layers",
               len(heads) == n_layers, f"captured {len(heads)}")
    if heads:
        h0 = heads[0]
        pooled = h0.mean(dim=1)
        gate.check(f"head-0 slice has width head_dim ({head_dim}) and pools to "
                   f"[{head_dim}]",
                   h0.shape[-1] == head_dim and pooled.shape[-1] == head_dim,
                   f"slice {tuple(h0.shape)} -> pooled {tuple(pooled.shape)}")
        print(f"         head-0 per-layer shape {tuple(h0.shape)} "
              f"-> mean-pooled {tuple(pooled.shape)}")

    gate.summary(adapter_dir)

    if meta.get('comparability_note'):
        print("\nCOMPARABILITY NOTE (carried from the manifest):")
        print(f"  {meta['comparability_note']}")

    return 0 if gate.passed else 1


if __name__ == "__main__":
    sys.exit(main())
