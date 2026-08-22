#!/usr/bin/env python3
"""
Phase 3 - Step 30: AVES 2 vs XLS-R vs HuBERT, zero-shot.

Companion to phase3_25, which plots the LAYER PROFILE of a single probe run.
This one answers the other question: how do the three encoders compare, overall
and class by class.

FIGURES (PNG, 300 DPI, no PDF)
------------------------------
  model_comparison_macro          macro-F1 per task per model, with chance
  hyrax_per_individual_<split>    per-animal F1, three models side by side
  species_model_ranking           all frozen models on species, AVES inserted
  species_per_species_aves        AVES per-species F1

A NOTE ON WHAT CANNOT BE PLOTTED
--------------------------------
There is no per-species breakdown for XLS-R or HuBERT anywhere in the repo --
results_corrected/frozen_transfer_species.csv stores macro-F1 only, and the
phase3_20 species runs did not persist per_class. So `species_per_species_aves`
is AVES alone, and says so on the figure rather than implying the comparison
was made and AVES won it. The hyrax figures DO compare all three, because the
phase3_24 layer-probe JSONs carry per_class for every model.

Baselines are read from their existing result files and are never rewritten.

USAGE
-----
    python scripts/phase3_30_aves_comparison_figures.py
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Three-series categorical set. The first two are phase3_25's validated pair,
# kept identical so a reader moving between figures does not have to relearn
# which colour is which model. The third was chosen by running the palette
# validator over candidates with --pairs all (a grouped bar chart makes EVERY
# pair adjacent, not just neighbours in the list): teal passes all pairs,
# where the obvious green collides with orange under protanopia (dE 1.7) and
# purple collides with blue for normal vision (dE 14.6).
C_XLSR = "#3366CC"
C_HUBERT = "#E8710A"
C_AVES = "#00897B"

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dcdcdc"

MODEL_ORDER = ["xls_r", "hubert_base", "aves2_eat_bio"]
MODEL_COLOUR = {"xls_r": C_XLSR, "hubert_base": C_HUBERT, "aves2_eat_bio": C_AVES}
MODEL_LABEL = {
    "xls_r": "XLS-R (speech, 300M)",
    "hubert_base": "HuBERT (speech, base)",
    "aves2_eat_bio": "AVES 2 EAT (bio, base)",
}


def style(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.xaxis.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def write_csv(path, rows, fields):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def chance_line(ax, chance):
    """Chance is stated in the title, not annotated on the line.

    An inline label has nowhere safe to sit: bars occupy the full width at
    this height, so it lands on top of one of them whichever end it is put.
    """
    ax.axhline(chance, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1, zorder=4)


# ---------------------------------------------------------------- gather
def collect(root):
    """-> {task: {model: {'f1','p','r','layer','per_class','chance'}}}"""
    aves_root = root / "outputs" / "phase3" / "aves2_zeroshot"
    tasks = {
        "hyrax_session_holdout": {
            "title": "Hyrax individual ID - bouts, session-holdout (8 animals)",
            "aves": aves_root / "hyrax_bout_session_holdout" / "layer_probe_aves2_eat_bio_base.json",
            "baseline_dir": root / "outputs" / "phase3" / "hyrax_probe_bout_session_holdout",
        },
        "hyrax_by_file": {
            "title": "Hyrax individual ID - bouts, by-file (10 animals)",
            "aves": aves_root / "hyrax_bout_by_file" / "layer_probe_aves2_eat_bio_base.json",
            "baseline_dir": root / "outputs" / "phase3" / "hyrax_probe_bout_by_file",
        },
        "species7": {
            "title": "Species ID - 7-class (hyrax excluded)",
            "aves": aves_root / "species7" / "layer_probe_aves2_eat_bio_base.json",
            "baseline_dir": None,
        },
    }

    out = {}
    for task, spec in tasks.items():
        cells = {}

        d = load_json(spec["aves"])
        if d:
            best = d["layers"][str(d["best_layer"])]
            cells["aves2_eat_bio"] = {
                "f1": d["best_f1_macro"], "p": d["best_precision_macro"],
                "r": d["best_recall_macro"], "layer": d["best_layer"],
                "per_class": best.get("per_class"), "chance": d["chance"],
            }

        if spec["baseline_dir"]:
            for m in ("xls_r", "hubert_base"):
                b = load_json(spec["baseline_dir"] / f"layer_probe_{m}_base.json")
                if not b:
                    continue
                bl = b["layers"][str(b["best_layer"])]
                cells[m] = {
                    "f1": b["best_f1_macro"], "p": b["best_precision_macro"],
                    "r": b["best_recall_macro"], "layer": b["best_layer"],
                    "per_class": bl.get("per_class"), "chance": b["chance"],
                }
        else:
            # species macro-F1 for the baselines lives only in the corrected CSV
            csv_path = root / "outputs" / "phase3" / "results_corrected" / "frozen_transfer_species.csv"
            if csv_path.exists():
                with open(csv_path) as f:
                    for row in csv.DictReader(f):
                        if row["model"] in ("xls_r", "hubert_base"):
                            cells[row["model"]] = {
                                "f1": float(row["test_f1_macro_corrected"]),
                                "p": None, "r": None, "layer": None,
                                "per_class": None, "chance": float(row["chance"]),
                            }
        out[task] = {"title": spec["title"], "cells": cells}
    return out


# ---------------------------------------------------------------- figures
def fig_macro(data, out_png, out_csv):
    tasks = [t for t in ("hyrax_session_holdout", "hyrax_by_file", "species7") if data.get(t)]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    width, rows = 0.26, []

    for i, model in enumerate(MODEL_ORDER):
        xs, ys = [], []
        for j, task in enumerate(tasks):
            c = data[task]["cells"].get(model)
            if not c:
                continue
            xs.append(j + (i - 1) * width)
            ys.append(c["f1"])
            rows.append({"task": task, "model": model, "f1_macro": round(c["f1"], 4),
                         "precision_macro": c["p"], "recall_macro": c["r"],
                         "best_layer": c["layer"], "chance": c["chance"]})
        if not xs:
            continue
        bars = ax.bar(xs, ys, width * 0.94, label=MODEL_LABEL[model],
                      color=MODEL_COLOUR[model], edgecolor="white", linewidth=0.8, zorder=3)
        for b, v in zip(bars, ys):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        va="bottom", fontsize=8.5, color=INK)

    for j, task in enumerate(tasks):
        ch = next((c["chance"] for c in data[task]["cells"].values() if c.get("chance")), None)
        if ch:
            ax.plot([j - 1.6 * width, j + 1.6 * width], [ch, ch], color=MUTED,
                    linestyle=(0, (4, 3)), linewidth=1.1, zorder=4)

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(["Hyrax ID\nsession-holdout", "Hyrax ID\nby-file", "Species ID\n7-class"][:len(tasks)],
                       fontsize=9.5, color=INK)
    ax.set_ylabel("Test macro-F1", fontsize=10, color=INK)
    ax.set_ylim(0, 1.08)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Frozen encoders, zero-shot: bio-pretrained vs speech-pretrained\n"
                 "(dashed = chance)", fontsize=12, fontweight="bold", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    write_csv(out_csv, rows, ["task", "model", "f1_macro", "precision_macro",
                              "recall_macro", "best_layer", "chance"])
    return rows


def fig_per_individual(data, task, out_png, out_csv):
    cells = data[task]["cells"]
    present = [m for m in MODEL_ORDER if cells.get(m) and cells[m].get("per_class")]
    if not present:
        return []

    names = list(cells[present[0]]["per_class"].keys())
    # order by mean F1 across models: best-recognised animals first
    names.sort(key=lambda n: -np.mean([cells[m]["per_class"][n]["f1"] for m in present]))

    fig, ax = plt.subplots(figsize=(max(9, 1.05 * len(names) + 3), 4.6))
    width, rows = 0.26, []

    for i, model in enumerate(present):
        pc = cells[model]["per_class"]
        xs = [j + (i - 1) * width for j in range(len(names))]
        ys = [pc[n]["f1"] for n in names]
        ax.bar(xs, ys, width * 0.94, label=MODEL_LABEL[model], color=MODEL_COLOUR[model],
               edgecolor="white", linewidth=0.8, zorder=3)
        for n in names:
            rows.append({"task": task, "model": model, "individual": n,
                         "precision": round(pc[n]["precision"], 4),
                         "recall": round(pc[n]["recall"], 4),
                         "f1": round(pc[n]["f1"], 4), "support": pc[n]["support"]})

    ch = cells[present[0]]["chance"]
    chance_line(ax, ch)

    sup = cells[present[0]]["per_class"]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\nn={sup[n]['support']}" for n in names], fontsize=9, color=INK)
    ax.set_ylabel("Test F1", fontsize=10, color=INK)
    ax.set_ylim(0, 1.05)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=1)
    ax.set_title(f"{data[task]['title']}\nper-animal F1 at each model's best layer "
                 f"(dashed = chance {ch:.3f})",
                 fontsize=12, fontweight="bold", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    write_csv(out_csv, rows, ["task", "model", "individual", "precision", "recall",
                              "f1", "support"])
    return rows


def fig_species_ranking(root, data, out_png, out_csv):
    csv_path = root / "outputs" / "phase3" / "results_corrected" / "frozen_transfer_species.csv"
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        base = [{"model": r["model"], "label": r["model_label"],
                 "f1": float(r["test_f1_macro_corrected"]),
                 "chance": float(r["chance"])} for r in csv.DictReader(f)]

    aves = data.get("species7", {}).get("cells", {}).get("aves2_eat_bio")
    if aves:
        base.append({"model": "aves2_eat_bio", "label": "AVES 2 EAT",
                     "f1": aves["f1"], "chance": aves["chance"]})
    base.sort(key=lambda r: -r["f1"])

    fig, ax = plt.subplots(figsize=(9, 4.4))
    colours = [MODEL_COLOUR.get(r["model"], "#9aa0a6") for r in base]
    bars = ax.bar(range(len(base)), [r["f1"] for r in base], 0.66, color=colours,
                  edgecolor="white", linewidth=0.8, zorder=3)
    for b, r in zip(bars, base):
        ax.annotate(f"{r['f1']:.3f}", (b.get_x() + b.get_width() / 2, r["f1"]),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    va="bottom", fontsize=8.5, color=INK)
    chance_line(ax, base[0]["chance"])

    ax.set_xticks(range(len(base)))
    ax.set_xticklabels([r["label"] for r in base], fontsize=9, color=INK,
                       rotation=20, ha="right")
    ax.set_ylabel("Test macro-F1", fontsize=10, color=INK)
    ax.set_ylim(0, 1.08)
    style(ax)
    ax.set_title(f"Frozen species ID (7-class), model ranking\n"
                 f"AVES 2 in teal; speech encoders from the corrected baselines "
                 f"(dashed = chance {base[0]['chance']:.3f})",
                 fontsize=12, fontweight="bold", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    write_csv(out_csv, [{"model": r["model"], "label": r["label"],
                         "f1_macro": round(r["f1"], 4), "chance": r["chance"]}
                        for r in base], ["model", "label", "f1_macro", "chance"])
    return base


def fig_species_per_class(data, out_png, out_csv):
    c = data.get("species7", {}).get("cells", {}).get("aves2_eat_bio")
    if not c or not c.get("per_class"):
        return []
    pc = c["per_class"]
    names = sorted(pc, key=lambda n: -pc[n]["f1"])

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    bars = ax.bar(range(len(names)), [pc[n]["f1"] for n in names], 0.62,
                  color=C_AVES, edgecolor="white", linewidth=0.8, zorder=3)
    for b, n in zip(bars, names):
        ax.annotate(f"{pc[n]['f1']:.3f}", (b.get_x() + b.get_width() / 2, pc[n]["f1"]),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    va="bottom", fontsize=8.5, color=INK)
    chance_line(ax, c["chance"])

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\nn={pc[n]['support']}" for n in names], fontsize=8.5,
                       color=INK, rotation=18, ha="right")
    ax.set_ylabel("Test F1", fontsize=10, color=INK)
    ax.set_ylim(0, 1.05)
    style(ax)
    # single series: no legend box, the title names it
    ax.set_title(f"Frozen species ID (7-class), per-species F1 - AVES 2 EAT\n"
                 f"no per-species breakdown exists for XLS-R/HuBERT, so this panel "
                 f"is AVES alone (dashed = chance {c['chance']:.3f})",
                 fontsize=12, fontweight="bold", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    write_csv(out_csv, [{"species": n, "precision": round(pc[n]["precision"], 4),
                         "recall": round(pc[n]["recall"], 4),
                         "f1": round(pc[n]["f1"], 4), "support": pc[n]["support"]}
                        for n in names], ["species", "precision", "recall", "f1", "support"])
    return names


def main():
    p = argparse.ArgumentParser(description="AVES vs XLS-R vs HuBERT figures")
    p.add_argument("--out-dir", default="outputs/phase3/aves2_zeroshot/figures_comparison")
    args = p.parse_args()

    root = Path(__file__).parent.parent
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect(root)
    found = {t: sorted(v["cells"]) for t, v in data.items() if v["cells"]}
    print("cells found:")
    for t, ms in found.items():
        print(f"  {t:<24} {ms}")

    fig_macro(data, out_dir / "model_comparison_macro.png",
              out_dir / "model_comparison_macro.csv")
    for task, tag in (("hyrax_session_holdout", "session_holdout"),
                      ("hyrax_by_file", "by_file")):
        if data.get(task, {}).get("cells"):
            fig_per_individual(data, task,
                               out_dir / f"hyrax_per_individual_{tag}.png",
                               out_dir / f"hyrax_per_individual_{tag}.csv")
    fig_species_ranking(root, data, out_dir / "species_model_ranking.png",
                        out_dir / "species_model_ranking.csv")
    fig_species_per_class(data, out_dir / "species_per_species_aves.png",
                          out_dir / "species_per_species_aves.csv")

    print(f"\nwrote figures + CSVs -> {out_dir}")
    print("\nmacro-F1 summary:")
    for task, v in data.items():
        for m in MODEL_ORDER:
            c = v["cells"].get(m)
            if c:
                layer = f"L{c['layer']}" if c["layer"] is not None else "  -"
                print(f"  {task:<24} {MODEL_LABEL[m]:<24} {layer:<4} {c['f1']:.4f}")


if __name__ == "__main__":
    main()
