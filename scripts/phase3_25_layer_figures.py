#!/usr/bin/env python3
"""
Phase 3 - Step 25: figures for the per-layer hyrax probe.

Reads the four step-24 cells (2 models x {base, adapted}) and produces the two
things the professor asked for, each with the CSV behind it:

  hyrax_layer_probe.png / .csv
      macro-F1 per layer, base vs species-adapted, one panel per model.
      Answers "which layer is best, and does adaptation improve it".

  hyrax_best_layer.png / .csv
      best layer per model and condition, head to head, against chance.
      Answers "how well are hyraxes identified by the two selected models".

  adaptation_delta.csv
      per-layer adapted-minus-base, and the layer-0 check. Layer 0 is the CNN
      front-end: under the old LoRA design its delta was exactly 0.0000 because
      the stack was frozen. A non-zero value here is the mechanism working.

CPU only. Run it wherever the JSONs are.

    python scripts/phase3_25_layer_figures.py
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# two-series categorical pair, validated: CVD deltaE 28.5 (protan),
# normal-vision 36.0, contrast >= 3:1 against a light surface
C_BASE = "#3366CC"
C_ADAPTED = "#E8710A"

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dcdcdc"

MODEL_LABEL = {
    "hubert_base": "HuBERT (monolingual)",
    "xls_r": "XLS-R (multilingual)",
    "wav2vec2_base": "wav2vec2",
    "wavlm": "WavLM",
    "wav2vec2_base_960h": "wav2vec2-960h",
}


def load_cells(in_dir):
    cells = {}
    for path in sorted(Path(in_dir).glob("layer_probe_*.json")):
        with open(path) as f:
            d = json.load(f)
        cells[(d["model"], d["condition"])] = d
    return cells


def series(cell):
    layers = sorted(int(k) for k in cell["layers"])
    mean = np.array([cell["layers"][str(l)]["f1_macro_mean"] for l in layers])
    std = np.array([cell["layers"][str(l)]["f1_macro_std"] for l in layers])
    return np.array(layers), mean, std


def style(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.xaxis.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def fig_per_layer(cells, models, out_png, out_csv):
    fig, axes = plt.subplots(len(models), 1, figsize=(11, 3.4 * len(models)))
    if len(models) == 1:
        axes = [axes]

    rows = []
    chance = None

    for ax, model in zip(axes, models):
        base, adapted = cells.get((model, "base")), cells.get((model, "adapted"))
        present = [c for c in (base, adapted) if c]
        if not present:
            continue
        chance = present[0]["chance"]

        width = 0.42
        peak = 0.0
        for cell, colour, label, offset in (
            (base, C_BASE, "frozen (base)", -width / 2),
            (adapted, C_ADAPTED, "species-adapted", width / 2),
        ):
            if cell is None:
                continue
            layers, mean, std = series(cell)
            peak = max(peak, float(np.max(mean + std)))
            ax.bar(layers + offset, mean, width * 0.94, label=label,
                   color=colour, edgecolor="white", linewidth=0.8, zorder=3)
            ax.errorbar(layers + offset, mean, yerr=std, fmt="none",
                        ecolor=MUTED, elinewidth=0.9, capsize=2, zorder=4)

            # label only this series' peak, nudged in points so the two series'
            # labels cannot collide when their best layers are adjacent
            best = int(np.argmax(mean))
            ax.annotate(f"{mean[best]:.3f}",
                        (layers[best] + offset, mean[best] + std[best]),
                        textcoords="offset points",
                        xytext=(0, 5 if colour == C_BASE else 16),
                        ha="center", va="bottom", fontsize=8.5,
                        color=colour, fontweight="bold", zorder=6)

            for l, m, s in zip(layers, mean, std):
                rows.append({
                    "model": model, "condition": cell["condition"], "layer": int(l),
                    "layer_role": "cnn_front_end" if l == 0 else f"transformer_block_{l - 1}",
                    "f1_macro_mean": round(float(m), 4),
                    "f1_macro_std": round(float(s), 4),
                    "is_best_layer": bool(l == layers[best]),
                })

        n_layers = int(max(series(present[0])[0]))
        ax.set_xlim(-0.7, n_layers + 0.7)
        # headroom so the peak labels clear the legend row
        ax.set_ylim(0, peak * 1.34)

        ax.axhline(chance, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
        ax.annotate(f"chance {chance:.3f}", (-0.6, chance), ha="left", va="bottom",
                    fontsize=8, color=MUTED, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))

        ax.set_title(MODEL_LABEL.get(model, model), fontsize=12,
                     fontweight="bold", color=INK, loc="left", pad=8)
        ax.set_ylabel("Hyrax ID macro-F1", fontsize=10, color=INK)
        ax.set_xticks(range(0, n_layers + 1))
        style(ax)
        ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)

    axes[-1].set_xlabel("Layer   (0 = CNN front-end, 1+ = transformer blocks)",
                        fontsize=10, color=INK)
    fig.suptitle("Where hyrax identity lives, and whether species adaptation moves it",
                 fontsize=13, fontweight="bold", color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    write_csv(out_csv, rows)
    return rows


def fig_best_layer(cells, models, out_png, out_csv):
    labels, values, errs, colours, rows = [], [], [], [], []
    chance = None

    for model in models:
        for cond, colour in (("base", C_BASE), ("adapted", C_ADAPTED)):
            cell = cells.get((model, cond))
            if cell is None:
                continue
            chance = cell["chance"]
            layers, mean, std = series(cell)
            b = int(np.argmax(mean))
            labels.append(f"{MODEL_LABEL.get(model, model)}\n{cond}  (L{layers[b]})")
            values.append(float(mean[b]))
            errs.append(float(std[b]))
            colours.append(colour)
            rows.append({
                "model": model, "condition": cond, "best_layer": int(layers[b]),
                "f1_macro_mean": round(float(mean[b]), 4),
                "f1_macro_std": round(float(std[b]), 4),
                "chance": chance,
            })

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(values))
    ax.bar(x, values, 0.6, color=colours, edgecolor="white", linewidth=1.2, zorder=3)
    ax.errorbar(x, values, yerr=errs, fmt="none", ecolor=MUTED,
                elinewidth=1.0, capsize=3, zorder=4)

    for xi, v, e in zip(x, values, errs):
        ax.annotate(f"{v:.3f}", (xi, v + e), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=INK, zorder=5)

    if chance is not None:
        ax.axhline(chance, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
        ax.annotate(f"chance {chance:.3f}", (len(values) - 0.4, chance),
                    ha="right", va="bottom", fontsize=8.5, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_ylabel("Hyrax ID macro-F1 (best layer)", fontsize=10, color=INK)
    ax.set_title("Hyrax individual identification, best layer per condition",
                 fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=10)
    style(ax)

    handles = [plt.Rectangle((0, 0), 1, 1, color=C_BASE),
               plt.Rectangle((0, 0), 1, 1, color=C_ADAPTED)]
    ax.legend(handles, ["frozen (base)", "species-adapted"],
              frameon=False, fontsize=9, loc="upper left", ncol=2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    write_csv(out_csv, rows)
    return rows


def deltas(cells, models, out_csv):
    rows = []
    for model in models:
        base, adapted = cells.get((model, "base")), cells.get((model, "adapted"))
        if not (base and adapted):
            continue
        layers, bm, bs = series(base)
        _, am, asd = series(adapted)
        for l, b, a, sb, sa in zip(layers, bm, am, bs, asd):
            rows.append({
                "model": model, "layer": int(l),
                "layer_role": "cnn_front_end" if l == 0 else f"transformer_block_{l - 1}",
                "base_f1": round(float(b), 4), "adapted_f1": round(float(a), 4),
                "delta": round(float(a - b), 4),
                "base_std": round(float(sb), 4), "adapted_std": round(float(sa), 4),
                "exceeds_combined_sd": bool(abs(a - b) > (sb + sa)),
            })
    write_csv(out_csv, rows)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="Figures for the per-layer hyrax probe")
    p.add_argument("--in-dir", default="outputs/phase3/hyrax_probe_adapt_species_id")
    p.add_argument("--out-dir",
                   default="outputs/phase3/hyrax_probe_adapt_species_id/figures")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_cells(args.in_dir)
    if not cells:
        raise SystemExit(f"no layer_probe_*.json found in {args.in_dir}")

    models = []
    for m, _ in cells:
        if m not in models:
            models.append(m)
    models.sort(key=lambda m: (m != "hubert_base", m))

    print(f"cells found: {sorted(f'{m}/{c}' for m, c in cells)}")

    fig_per_layer(cells, models, out_dir / "hyrax_layer_probe.png",
                  out_dir / "hyrax_layer_probe.csv")
    best = fig_best_layer(cells, models, out_dir / "hyrax_best_layer.png",
                          out_dir / "hyrax_best_layer.csv")
    dl = deltas(cells, models, out_dir / "adaptation_delta.csv")

    print(f"\nwrote figures + CSVs -> {out_dir}")
    print("\nbest layer per cell:")
    for r in best:
        print(f"  {r['model']:<12} {r['condition']:<8} L{r['best_layer']:<3} "
              f"{r['f1_macro_mean']:.4f} +- {r['f1_macro_std']:.4f}")

    l0 = [r for r in dl if r["layer"] == 0]
    if l0:
        print("\nlayer-0 check (CNN front-end; was exactly 0.0000 under LoRA):")
        for r in l0:
            flag = "MOVED" if abs(r["delta"]) > 1e-6 else "IDENTICAL - still frozen?"
            print(f"  {r['model']:<12} delta {r['delta']:+.4f}   {flag}")

    best_deltas = {}
    for r in dl:
        m = r["model"]
        if m not in best_deltas or r["adapted_f1"] > best_deltas[m]["adapted_f1"]:
            best_deltas[m] = r
    print("\nadaptation effect at each model's best adapted layer:")
    for m, r in best_deltas.items():
        print(f"  {m:<12} L{r['layer']:<3} {r['base_f1']:.4f} -> {r['adapted_f1']:.4f}  "
              f"delta {r['delta']:+.4f}"
              f"{'  (> combined SD)' if r['exceeds_combined_sd'] else '  (within noise)'}")


if __name__ == "__main__":
    main()
