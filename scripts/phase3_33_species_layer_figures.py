#!/usr/bin/env python3
"""
Phase 3 - Step 33: species layer figures, and the species-vs-hyrax comparison.

WHAT THIS SETTLES
-----------------
The published species numbers (XLS-R 0.969, HuBERT 0.962) were final-layer
readings, because they came from phase3_20_probe_audit -- a script written to
test probe undertraining, not to sweep layers. This is the sweep.

It reproduces those numbers at the final layer (XLS-R 0.9710 vs 0.9690, HuBERT
0.9614 vs 0.9624), which is what licenses trusting the rest of the curve.

AND IT REFUTES A PREDICTION, WHICH IS WORTH RECORDING
-----------------------------------------------------
The expectation going in was that species would peak LATE and hyrax individual
identity EARLY, so that "the two tasks occupy different parts of the network"
could be shown rather than argued.

That is not what happened. Species peaks at layers 2-4; hyrax peaks at 1-3. Both
live in the early stack. The depth-separation story does not hold, and the figure
here is built to show that honestly rather than to hide it -- the two tasks are
plotted on a shared layer axis so the overlap is visible.

The behavioural evidence for the central claim is unaffected: species adaptation
still degrades hyrax, and AVES2 is still best at species and worst at
individuals. Only the mechanistic layer-position explanation is withdrawn.

WHAT IT WRITES  (FINAL/09_species_layer_analysis/)
--------------------------------------------------
  species_layers_<model>.png/.csv     F1 and accuracy at every layer, frozen and
                                      LoRA where both exist
  species_layers_all.csv              every model, every layer, one table
  species_vs_hyrax_best_layer.png/.csv  where each task peaks, side by side
  final_vs_best_layer.csv             what reading the final layer costs

    python scripts/phase3_33_species_layer_figures.py
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#dcdcdc"
C_FROZEN, C_LORA = "#0072B2", "#009E73"
C_SPECIES, C_HYRAX = "#785EF0", "#DC267F"

LABEL = {
    "hubert_base": "HuBERT (monolingual)",
    "xls_r": "XLS-R (multilingual)",
    "wavlm": "WavLM (monolingual)",
    "wav2vec2_base": "wav2vec2 (monolingual)",
    "aves2_eat_bio": "AVES2 EAT (bioacoustic)",
}


def series(cell):
    layers = sorted(int(k) for k in cell["layers"])
    f1 = np.array([cell["layers"][str(l)]["f1_macro_mean"] for l in layers])
    sd = np.array([cell["layers"][str(l)]["f1_macro_std"] for l in layers])
    acc = np.array([cell["layers"][str(l)]["accuracy_mean"] for l in layers])
    return np.array(layers), f1, sd, acc


def style(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.xaxis.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def load_species(root):
    cells = {}
    for f in sorted(glob.glob(str(root / "species_layer_probe_lora" / "layer_probe_*.json"))):
        j = json.load(open(f))
        cells[(j["model"], j["condition"])] = j
    p = root / "aves2_zeroshot" / "species7" / "layer_probe_aves2_eat_bio_base.json"
    if p.exists():
        j = json.load(open(p))
        cells[(j["model"], "base")] = j
    return cells


def load_hyrax_best(root):
    """Best hyrax layer per model, frozen, session-holdout."""
    out = {}
    for d in (root / "hyrax_probe_bout_session_holdout",
              root / "aves2_zeroshot" / "hyrax_bout_session_holdout"):
        for f in sorted(d.glob("layer_probe_*_base.json")):
            j = json.load(open(f))
            e = j["layers"][str(j["best_layer"])]
            out[j["model"]] = {"layer": j["best_layer"], "n_layers": j["n_layers"] - 1,
                               "f1": e["f1_macro_mean"], "acc": e["accuracy_mean"]}
    return out


def fig_per_model(cells, model, out_dir):
    conds = [(c, lab, col) for c, lab, col in
             (("base", "frozen", C_FROZEN), ("adapted", "LoRA-adapted", C_LORA))
             if (model, c) in cells]
    if not conds:
        return []

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    rows, lo, hi = [], 1.0, 0.0

    for i, (cond, lab, col) in enumerate(conds):
        cell = cells[(model, cond)]
        layers, f1, sd, acc = series(cell)
        lo, hi = min(lo, float((f1 - sd).min())), max(hi, float((f1 + sd).max()))

        ax.plot(layers, f1, "-o", color=col, label=f"{lab} — macro-F1",
                linewidth=2, markersize=4.5, zorder=3)
        ax.fill_between(layers, f1 - sd, f1 + sd, color=col, alpha=0.15,
                        linewidth=0, zorder=2)

        b = int(np.argmax(f1))
        ax.annotate(f"L{layers[b]}  {f1[b]:.3f}", (layers[b], f1[b] + sd[b]),
                    textcoords="offset points", xytext=(0, 7 + i * 14),
                    ha="center", fontsize=9, fontweight="bold", color=col, zorder=6)
        # the final layer is where the published number was read
        ax.scatter([layers[-1]], [f1[-1]], s=90, facecolor="white",
                   edgecolor=col, linewidth=2, zorder=5)

        for l, a, b_, s_ in zip(layers, acc, f1, sd):
            rows.append({
                "model": model, "condition": cond, "layer": int(l),
                "layer_role": "cnn_front_end" if l == 0 else f"transformer_block_{l - 1}",
                "f1_macro_mean": round(float(b_), 4),
                "f1_macro_std": round(float(s_), 4),
                "accuracy": round(float(a), 4),
                "is_best_layer": bool(l == layers[int(np.argmax(f1))]),
                "is_final_layer": bool(l == layers[-1]),
            })

    span = max(hi - lo, 0.01)
    ax.set_ylim(lo - span * 0.20, hi + span * 0.42)
    ax.set_xlim(-0.6, layers[-1] + 0.6)
    ax.set_xticks(range(0, layers[-1] + 1))
    ax.set_xlabel("Layer   (0 = CNN front-end, 1+ = transformer blocks)",
                  fontsize=10, color=INK)
    ax.set_ylabel("Species ID macro-F1 (7-way)", fontsize=10, color=INK)
    ax.set_title(f"{LABEL.get(model, model)} — species identity by layer\n"
                 f"hollow marker = final layer, where the published number was read",
                 fontsize=12, fontweight="bold", color=INK, loc="left", pad=34)
    style(ax)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower left",
              bbox_to_anchor=(0, 1.01))

    fig.tight_layout()
    fig.savefig(out_dir / f"species_layers_{model}.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    write_csv(out_dir / f"species_layers_{model}.csv", rows)
    return rows


def fig_species_vs_hyrax(cells, hyrax, out_dir):
    """Both tasks on a shared layer axis. They overlap; the figure says so."""
    models = [m for m in ("hubert_base", "xls_r", "wavlm", "wav2vec2_base",
                          "aves2_eat_bio") if (m, "base") in cells and m in hyrax]
    rows = []
    for m in models:
        layers, f1, _, _ = series(cells[(m, "base")])
        b = int(np.argmax(f1))
        rows.append({
            "model": m, "label": LABEL.get(m, m),
            "n_transformer_layers": int(layers[-1]),
            "species_best_layer": int(layers[b]),
            "species_best_f1": round(float(f1[b]), 4),
            "hyrax_best_layer": hyrax[m]["layer"],
            "hyrax_best_f1": round(float(hyrax[m]["f1"]), 4),
            "layer_gap": int(layers[b]) - int(hyrax[m]["layer"]),
        })

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    y = np.arange(len(rows))

    for yi, r in zip(y, rows):
        ax.plot([0, r["n_transformer_layers"]], [yi, yi], color=GRID,
                linewidth=6, solid_capstyle="round", zorder=1)
        ax.scatter(r["species_best_layer"], yi, s=170, color=C_SPECIES,
                   edgecolor="white", linewidth=1.6, zorder=3)
        ax.scatter(r["hyrax_best_layer"], yi, s=170, color=C_HYRAX,
                   edgecolor="white", linewidth=1.6, zorder=3, marker="D")

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Layer   (grey bar = full depth of that model)", fontsize=10, color=INK)
    ax.set_title("Both tasks peak in the SAME early layers\n"
                 "the depth-separation hypothesis is not supported",
                 fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=34)
    style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)

    handles = [plt.Line2D([], [], marker="o", linestyle="", color=C_SPECIES,
                          markersize=11, label="best layer for SPECIES ID"),
               plt.Line2D([], [], marker="D", linestyle="", color=C_HYRAX,
                          markersize=10, label="best layer for HYRAX individual ID")]
    ax.legend(handles=handles, frameon=False, fontsize=9, ncol=2,
              loc="lower left", bbox_to_anchor=(0, 1.01))

    fig.tight_layout()
    fig.savefig(out_dir / "species_vs_hyrax_best_layer.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    write_csv(out_dir / "species_vs_hyrax_best_layer.csv", rows)
    return rows


def final_vs_best(cells, out_dir):
    """What reading the final layer costs -- the published numbers did exactly that."""
    rows = []
    for (model, cond), cell in sorted(cells.items()):
        layers, f1, _, acc = series(cell)
        b = int(np.argmax(f1))
        rows.append({
            "model": model, "condition": cond,
            "best_layer": int(layers[b]),
            "best_f1": round(float(f1[b]), 4),
            "best_accuracy": round(float(acc[b]), 4),
            "final_layer": int(layers[-1]),
            "final_f1": round(float(f1[-1]), 4),
            "final_accuracy": round(float(acc[-1]), 4),
            "cost_of_reading_final": round(float(f1[-1] - f1[b]), 4),
        })
    write_csv(out_dir / "final_vs_best_layer.csv", rows)
    return rows


def main():
    p = argparse.ArgumentParser(description="Species layer figures")
    p.add_argument("--root", default="outputs/phase3")
    p.add_argument("--out", default="outputs/phase3/FINAL/09_species_layer_analysis")
    args = p.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cells = load_species(root)
    if not cells:
        raise SystemExit("no species layer results found")
    hyrax = load_hyrax_best(root)

    all_rows = []
    for model in sorted({m for m, _ in cells}):
        all_rows.extend(fig_per_model(cells, model, out))
    write_csv(out / "species_layers_all.csv", all_rows)

    cmp_rows = fig_species_vs_hyrax(cells, hyrax, out)
    fb = final_vs_best(cells, out)

    print(f"wrote -> {out}")
    print("\nspecies best layer vs final layer:")
    for r in fb:
        print(f"  {r['model']:<14} {r['condition']:<8} best L{r['best_layer']:<3} "
              f"F1 {r['best_f1']:.4f} acc {r['best_accuracy']:.4f}   "
              f"final L{r['final_layer']:<3} F1 {r['final_f1']:.4f}  "
              f"({r['cost_of_reading_final']:+.4f})")

    print("\nspecies vs hyrax best layer (frozen):")
    for r in cmp_rows:
        print(f"  {r['label']:<26} species L{r['species_best_layer']:<3} "
              f"hyrax L{r['hyrax_best_layer']:<3} gap {r['layer_gap']:+d}")


if __name__ == "__main__":
    main()
