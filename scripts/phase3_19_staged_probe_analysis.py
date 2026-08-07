#!/usr/bin/env python3
"""
Phase C / Step C3: compare base-frozen against species-adapted frozen probes.

Consumes the four sweeps produced by phase3_18_staged_probe.py (2 models x
{base, adapted}) and emits the comparison the phase exists to make.

WHAT THIS GUARDS AGAINST
------------------------
The headline claim is "adapting on animals excluding hyrax improves frozen
hyrax representations". Three ways that claim can be inflated, all handled here:

  1. Selection asymmetry. Base and adapted are both best-of-N chosen on VAL, so
     neither gets a max-over-layers advantage the other lacks. The published
     0.1735 / 0.1017 are FINAL-LAYER numbers and are reported in a separate
     column, never as the base baseline.
  2. Selection on test. phase3_18 selects on val. This script additionally
     reports each condition's best-possible TEST cell (the oracle) so the size
     of the selection gap is visible rather than hidden.
  3. Layer-axis confusion. mean spans 0..N, head0 spans 1..N. Stated in every
     output.

Outputs (all under the --out-dir):
  staged_probe_comparison.csv    one row per (model, variant): base vs adapted
  staged_probe_layer_curves.csv  every cell, long format, for plotting
  staged_probe_layer_curves.png  test macro-F1 vs layer, base vs adapted
  STAGED_PROBE_README.md         the numbers in prose, with provenance caveats
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ['hubert_base', 'xls_r']
MODEL_LABELS = {'hubert_base': 'HuBERT (monolingual)', 'xls_r': 'XLS-R (multilingual)'}
CONDITIONS = ['base', 'adapted']
VARIANTS = ['mean', 'head0']
VARIANT_LABELS = {'mean': 'mean pooling', 'head0': 'head 0 (out_proj input)'}
COLORS = {'base': '#0173B2', 'adapted': '#DE8F05'}


def load(root, model, condition):
    p = Path(root) / model / condition / "staged_probe_results.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Phase C - staged probe comparison")
    ap.add_argument("--probe-root", default="outputs/phase3/staged_lora/probe")
    ap.add_argument("--out-dir", default="outputs/phase3/staged_lora/probe/summary")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded, missing = {}, []
    for model in MODELS:
        for cond in CONDITIONS:
            r = load(args.probe_root, model, cond)
            if r is None:
                missing.append(f"{model}/{cond}")
            else:
                loaded[(model, cond)] = r

    if missing:
        print(f"WARNING: missing sweeps: {missing}")
        print("The base-vs-adapted comparison needs all four. Continuing with "
              "what is present.")
    if not loaded:
        print("ERROR: no sweeps found under " + str(args.probe_root))
        return 2

    # ------------------------------------------------------- comparison table
    comp_rows = []
    for model in MODELS:
        base, adapted = loaded.get((model, 'base')), loaded.get((model, 'adapted'))
        if not (base and adapted):
            continue
        for variant in VARIANTS:
            b = base['best_by_variant'][variant]
            a = adapted['best_by_variant'][variant]

            # Oracle = best TEST cell. Reported only to expose the size of the
            # val-selection gap; never used as a headline number.
            def oracle(res):
                sub = [c for c in res['cells'] if c['variant'] == variant]
                return max(sub, key=lambda c: c['test_f1_macro'])
            ob, oa = oracle(base), oracle(adapted)

            comp_rows.append({
                'model': model,
                'variant': variant,
                'base_best_layer': b['layer'],
                'base_val_f1': round(b['val_f1_macro'], 4),
                'base_test_f1': round(b['test_f1_macro'], 4),
                'base_test_bal_acc': round(b['test_balanced_accuracy'], 4),
                'base_test_acc': round(b['test_accuracy'], 4),
                'adapted_best_layer': a['layer'],
                'adapted_val_f1': round(a['val_f1_macro'], 4),
                'adapted_test_f1': round(a['test_f1_macro'], 4),
                'adapted_test_bal_acc': round(a['test_balanced_accuracy'], 4),
                'adapted_test_acc': round(a['test_accuracy'], 4),
                'delta_test_f1': round(a['test_f1_macro'] - b['test_f1_macro'], 4),
                'delta_test_bal_acc': round(
                    a['test_balanced_accuracy'] - b['test_balanced_accuracy'], 4),
                'base_final_layer_mean_test_f1': round(
                    base['final_layer_mean_cell']['test_f1_macro'], 4),
                'published_final_layer': base['reference_published_final_layer'],
                'hyrax_finetuned_ceiling': base['reference_hyrax_finetuned_ceiling'],
                'base_oracle_test_f1': round(ob['test_f1_macro'], 4),
                'base_oracle_layer': ob['layer'],
                'adapted_oracle_test_f1': round(oa['test_f1_macro'], 4),
                'adapted_oracle_layer': oa['layer'],
            })

    if comp_rows:
        with open(out_dir / "staged_probe_comparison.csv", 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(comp_rows[0]))
            w.writeheader()
            w.writerows(comp_rows)
        print(f"wrote {out_dir / 'staged_probe_comparison.csv'}")

    # ------------------------------------------------------------ long format
    long_rows = []
    for (model, cond), res in loaded.items():
        for c in res['cells']:
            long_rows.append({
                'model': model, 'condition': cond, 'variant': c['variant'],
                'layer': c['layer'], 'dim': c['dim'],
                'val_f1_macro': round(c['val_f1_macro'], 4),
                'test_f1_macro': round(c['test_f1_macro'], 4),
                'test_balanced_accuracy': round(c['test_balanced_accuracy'], 4),
                'test_accuracy': round(c['test_accuracy'], 4),
            })
    long_rows.sort(key=lambda r: (r['model'], r['variant'], r['condition'], r['layer']))
    with open(out_dir / "staged_probe_layer_curves.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0]))
        w.writeheader()
        w.writerows(long_rows)
    print(f"wrote {out_dir / 'staged_probe_layer_curves.csv'}")

    # ----------------------------------------------------------------- figure
    fig, axes = plt.subplots(len(MODELS), len(VARIANTS),
                             figsize=(11, 4.2 * len(MODELS)), squeeze=False)
    for i, model in enumerate(MODELS):
        for j, variant in enumerate(VARIANTS):
            ax = axes[i][j]
            plotted = False
            for cond in CONDITIONS:
                res = loaded.get((model, cond))
                if not res:
                    continue
                cells = sorted((c for c in res['cells'] if c['variant'] == variant),
                               key=lambda c: c['layer'])
                if not cells:
                    continue
                xs = [c['layer'] for c in cells]
                ys = [c['test_f1_macro'] for c in cells]
                ax.plot(xs, ys, marker='o', ms=3.5, lw=1.6,
                        color=COLORS[cond], label=f"{cond}")
                sel = res['best_by_variant'][variant]
                ax.scatter([sel['layer']], [sel['test_f1_macro']], s=110,
                           facecolors='none', edgecolors=COLORS[cond], lw=1.8,
                           zorder=5)
                plotted = True

            res_any = loaded.get((model, 'base')) or loaded.get((model, 'adapted'))
            if res_any:
                ax.axhline(res_any['reference_published_final_layer'],
                           color='grey', ls=':', lw=1.2)
                ax.axhline(res_any['reference_hyrax_finetuned_ceiling'],
                           color='crimson', ls='--', lw=1.2)
                ax.axhline(1.0 / res_any['num_classes'], color='black',
                           ls='-.', lw=0.9, alpha=0.5)

            if plotted:
                ax.legend(fontsize=8, loc='best')
            ax.set_title(f"{MODEL_LABELS[model]} - {VARIANT_LABELS[variant]}",
                         fontsize=10)
            ax.set_xlabel("layer (hidden_states index)")
            ax.set_ylabel("test macro-F1")
            ax.grid(alpha=0.3)

    fig.suptitle("Frozen hyrax probe: base vs species-adapted encoder\n"
                 "circled = cell selected on val; dotted = published final-layer; "
                 "dashed = hyrax-fine-tuned ceiling; dash-dot = chance",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "staged_probe_layer_curves.png", dpi=300,
                bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_dir / 'staged_probe_layer_curves.png'}")

    # ---------------------------------------------------------------- readme
    lines = [
        "# Phase C: frozen hyrax probe, base vs species-adapted encoder", "",
        "Every number here is a **window-level** linear probe on the hyrax",
        "session-holdout task, trained on frozen features. Base and adapted use",
        "the identical probe recipe, identical windows and identical splits; the",
        "only difference is whether the LoRA deltas are present.", "",
        "## How to read the columns", "",
        "- `*_best_layer` / `*_test_f1`: cell chosen by **val** macro-F1, its",
        "  **test** score reported. Both conditions get best-of-N, so neither",
        "  has a selection advantage.",
        "- `base_final_layer_mean_test_f1`: the final-layer + mean cell, i.e. the",
        "  cell corresponding to what phase3_03 measured.",
        "- `published_final_layer`: the previously reported frozen number",
        "  (0.1735 / 0.1017). **Not a valid baseline - do not compare against it.**",
        "  Two problems. It is final-layer only, not best-layer. And it comes from",
        "  phase3_03's 50-epoch no-val branch, where one epoch is a single",
        "  full-batch gradient step, so the probe never fit (train macro-F1 ~0.24",
        "  on HuBERT). Replicating that recipe reproduces it (0.1590 vs 0.1735),",
        "  while a converged probe on the same frozen features reaches 0.3280.",
        "  It is an undertrained-probe artefact. Use the `base_*` columns of this",
        "  table as the frozen baseline instead.",
        "- `hyrax_finetuned_ceiling`: LoRA fine-tuned **on hyrax** (0.4066 /",
        "  0.3167), with minibatch training and window_inverse class weights. A",
        "  loose ceiling, **not** a like-for-like probe result.",
        "- `*_oracle_test_f1`: the best-possible TEST cell. Shown only to expose",
        "  how large the val-selection gap is. Never a headline number.", "",
        "## Layer axis differs by variant", "",
        "- `mean`: layers 0..N. Index 0 is the pre-transformer",
        "  feature_projection output; 1..N are transformer blocks.",
        "- `head0`: layers 1..N. Head 0's context vector is a property of a",
        "  transformer block, so it has no layer-0 counterpart.",
        "",
        "A given layer number >= 1 refers to the same block in both variants.",
        "",
        "## Results", "",
    ]

    if comp_rows:
        lines += [
            "| Model | Variant | Base best L | Base test F1 | Adapted best L | "
            "Adapted test F1 | Delta | Base final-layer | Published final-layer | "
            "FT ceiling |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in comp_rows:
            lines.append(
                f"| {MODEL_LABELS[r['model']]} | {r['variant']} | "
                f"{r['base_best_layer']} | {r['base_test_f1']:.4f} | "
                f"{r['adapted_best_layer']} | {r['adapted_test_f1']:.4f} | "
                f"{r['delta_test_f1']:+.4f} | "
                f"{r['base_final_layer_mean_test_f1']:.4f} | "
                f"{r['published_final_layer']:.4f} | "
                f"{r['hyrax_finetuned_ceiling']:.4f} |")
        lines += ["", "### Selection gap (val-selected vs test-oracle)", "",
                  "| Model | Variant | Base val-sel / oracle | Adapted val-sel / oracle |",
                  "|---|---|---|---|"]
        for r in comp_rows:
            lines.append(
                f"| {MODEL_LABELS[r['model']]} | {r['variant']} | "
                f"{r['base_test_f1']:.4f} / {r['base_oracle_test_f1']:.4f} "
                f"(L{r['base_oracle_layer']}) | "
                f"{r['adapted_test_f1']:.4f} / {r['adapted_oracle_test_f1']:.4f} "
                f"(L{r['adapted_oracle_layer']}) |")
    else:
        lines.append("_No complete model pair available yet._")

    if missing:
        lines += ["", f"**Incomplete:** missing sweeps for {missing}. The "
                      "base-vs-adapted comparison requires all four."]

    with open(out_dir / "STAGED_PROBE_README.md", 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_dir / 'STAGED_PROBE_README.md'}")

    if comp_rows:
        print("\n--- summary ---")
        for r in comp_rows:
            print(f"{r['model']:12s} {r['variant']:5s} | base L{r['base_best_layer']:<2d} "
                  f"{r['base_test_f1']:.4f} -> adapted L{r['adapted_best_layer']:<2d} "
                  f"{r['adapted_test_f1']:.4f} | delta {r['delta_test_f1']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
