#!/usr/bin/env python3
"""
Phase 3 - Step 32: the zero-shot table, F1 AND accuracy, both tasks.

WHY IT PULLS FROM TWO PLACES
----------------------------
Species and hyrax were measured by different scripts, and the numbers live in
different shapes:

  species, 6 speech encoders   outputs/phase3/probe_audit/*species7*.json
                               -> corrected_internal_val, the converged-probe
                                  result that replaced the undertrained ones
  species, AVES2               aves2_zeroshot/species7/layer_probe_*.json
  hyrax, all encoders          hyrax_probe_bout_{session_holdout,by_file}/
                               plus aves2_zeroshot/hyrax_bout_*/

Accuracy was recorded by every one of those runs from the start. It was simply
never written into a CSV, which is why it looked missing. Nothing is re-run here.

UNITS DIFFER BETWEEN THE TWO TASKS AND THAT IS NOT A DEFECT
-----------------------------------------------------------
Species is one embedding per FILE, truncated to 30 s. Hyrax is one embedding per
ground-truth BOUT, ~1.4 s. They are different tasks with different natural units,
and the column is labelled so the two are never averaged together. What matters
is that within each task, every encoder is measured identically.

ACCURACY vs BALANCED ACCURACY
-----------------------------
Both are reported. Plain accuracy flatters the frequent individuals on an
imbalanced 8/10-class task; balanced accuracy is the mean per-class recall and
is the honest twin. On these manifests balanced accuracy equals macro recall.

    python scripts/phase3_32_zeroshot_table.py
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
C_F1, C_ACC = "#0072B2", "#E69F00"

LABEL = {
    "aves2_eat_bio": "AVES2 EAT (bioacoustic)",
    "xls_r": "XLS-R (multilingual)",
    "hubert_base": "HuBERT (monolingual)",
    "wavlm": "WavLM (monolingual)",
    "wav2vec2_base": "wav2vec2 (monolingual)",
    "wav2vec2_base_960h": "wav2vec2-960h (monolingual)",
    "ecapa_tdnn": "ECAPA-TDNN (speaker-ID)",
}
FAMILY = {
    "aves2_eat_bio": "bioacoustic",
    "xls_r": "multilingual",
    "hubert_base": "monolingual",
    "wavlm": "monolingual",
    "wav2vec2_base": "monolingual",
    "wav2vec2_base_960h": "monolingual",
    "ecapa_tdnn": "speaker-ID",
}


def species_rows(root):
    rows = []
    for path in sorted(glob.glob(str(root / "probe_audit" / "*species7*.json"))):
        j = json.load(open(path))
        c = j.get("corrected_internal_val")
        if not c:
            continue
        rows.append({
            "task": "species_7way",
            "model": j["model"],
            "label": LABEL.get(j["model"], j["model"]),
            "family": FAMILY.get(j["model"], "?"),
            "best_layer": "final",   # the audit probes the final layer only
            "f1_macro": round(float(c["test_f1_macro"]), 4),
            "accuracy": round(float(c["test_accuracy"]), 4),
            "balanced_accuracy": round(float(c["test_balanced_accuracy"]), 4),
            "chance": round(1 / j["num_classes"], 4),
            "unit": "file (30s)",
            "split": "n/a",
            "source": Path(path).name,
        })

    p = root / "aves2_zeroshot" / "species7" / "layer_probe_aves2_eat_bio_base.json"
    if p.exists():
        j = json.load(open(p))
        d = j["layers"][str(j["best_layer"])]
        rows.append({
            "task": "species_7way",
            "model": j["model"],
            "label": LABEL.get(j["model"], j["model"]),
            "family": FAMILY.get(j["model"], "?"),
            "best_layer": j["best_layer"],
            "f1_macro": round(float(d["f1_macro_mean"]), 4),
            "accuracy": round(float(d["accuracy_mean"]), 4),
            "balanced_accuracy": round(float(d["balanced_accuracy_mean"]), 4),
            "chance": round(float(j["chance"]), 4),
            "unit": "file (30s)",
            "split": "n/a",
            "source": p.name,
        })
    return sorted(rows, key=lambda r: -r["f1_macro"])


def hyrax_rows(root):
    rows = []
    sources = [
        ("session_holdout", root / "hyrax_probe_bout_session_holdout"),
        ("session_holdout", root / "aves2_zeroshot" / "hyrax_bout_session_holdout"),
        ("by_file", root / "hyrax_probe_bout_by_file"),
        ("by_file", root / "aves2_zeroshot" / "hyrax_bout_by_file"),
    ]
    for split, d in sources:
        for path in sorted(d.glob("layer_probe_*_base.json")):
            j = json.load(open(path))
            k = str(j["best_layer"])
            e = j["layers"][k]
            rows.append({
                "task": f"hyrax_individual_{split}",
                "model": j["model"],
                "label": LABEL.get(j["model"], j["model"]),
                "family": FAMILY.get(j["model"], "?"),
                "best_layer": j["best_layer"],
                "f1_macro": round(float(e["f1_macro_mean"]), 4),
                "f1_macro_std": round(float(e["f1_macro_std"]), 4),
                "precision_macro": round(float(e["precision_macro_mean"]), 4),
                "recall_macro": round(float(e["recall_macro_mean"]), 4),
                "accuracy": round(float(e["accuracy_mean"]), 4),
                "balanced_accuracy": round(float(e["balanced_accuracy_mean"]), 4),
                "chance": round(float(j["chance"]), 4),
                "unit": "bout",
                "split": split,
                "source": f"{d.name}/{path.name}",
            })
    return rows


def fig_task(rows, title, chance, out_png, ylab):
    rows = sorted(rows, key=lambda r: -r["f1_macro"])
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    x = np.arange(len(rows))
    w = 0.38

    ax.bar(x - w / 2, [r["f1_macro"] for r in rows], w, label="macro-F1",
           color=C_F1, edgecolor="white", linewidth=1.0, zorder=3)
    ax.bar(x + w / 2, [r["accuracy"] for r in rows], w, label="accuracy",
           color=C_ACC, edgecolor="white", linewidth=1.0, zorder=3)

    for xi, r in zip(x, rows):
        ax.annotate(f"{r['f1_macro']:.3f}", (xi - w / 2, r["f1_macro"]),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=8.5, fontweight="bold", color=C_F1, zorder=5)
        ax.annotate(f"{r['accuracy']:.3f}", (xi + w / 2, r["accuracy"]),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=8.5, fontweight="bold", color=C_ACC, zorder=5)

    top = max(max(r["f1_macro"], r["accuracy"]) for r in rows)
    ax.set_ylim(0, top * 1.22)
    ax.axhline(chance, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
    ax.annotate(f"chance {chance:.3f}", (-0.5, chance), ha="left", va="top",
                textcoords="offset points", xytext=(0, -3),
                fontsize=8.5, color=MUTED, zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels([r["label"].replace(" (", "\n(") for r in rows],
                       fontsize=8.5, color=INK)
    ax.set_ylabel(ylab, fontsize=10, color=INK)
    ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK,
                 loc="left", pad=10)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r}, key=lambda k: (
        list(rows[0]).index(k) if k in rows[0] else 99))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="Zero-shot table with F1 and accuracy")
    p.add_argument("--root", default="outputs/phase3")
    p.add_argument("--out", default="outputs/phase3/FINAL/01_zero_shot_species")
    p.add_argument("--hyrax-out", default="outputs/phase3/FINAL/02_zero_shot_hyrax")
    args = p.parse_args()

    root = Path(args.root)
    sp_dir, hy_dir = Path(args.out), Path(args.hyrax_out)
    sp_dir.mkdir(parents=True, exist_ok=True)
    hy_dir.mkdir(parents=True, exist_ok=True)

    sp = species_rows(root)
    write_csv(sp_dir / "species_zeroshot_f1_and_accuracy.csv", sp)
    fig_task(sp, "Species identification, frozen encoders (7-way, hyrax excluded)",
             sp[0]["chance"], sp_dir / "species_zeroshot_f1_and_accuracy.png",
             "Species ID (7-way)")

    hy = hyrax_rows(root)
    write_csv(hy_dir / "hyrax_zeroshot_f1_and_accuracy.csv", hy)
    for split, n in (("session_holdout", "8 individuals, split by session"),
                     ("by_file", "10 individuals, split by recording")):
        rs = [r for r in hy if r["split"] == split]
        if not rs:
            continue
        write_csv(hy_dir / f"hyrax_zeroshot_{split}.csv", rs)
        fig_task(rs, f"Hyrax individual identification, frozen encoders\n{n} — real bouts",
                 rs[0]["chance"], hy_dir / f"hyrax_zeroshot_{split}.png",
                 "Hyrax individual ID")

    print("SPECIES (frozen, 7-way):")
    for r in sp:
        print(f"  {r['label']:<30} F1 {r['f1_macro']:.4f}  acc {r['accuracy']:.4f}  "
              f"bal-acc {r['balanced_accuracy']:.4f}")

    for split in ("session_holdout", "by_file"):
        rs = sorted([r for r in hy if r["split"] == split], key=lambda r: -r["f1_macro"])
        if not rs:
            continue
        print(f"\nHYRAX ({split}, chance {rs[0]['chance']}):")
        for r in rs:
            print(f"  {r['label']:<30} L{r['best_layer']:<3} F1 {r['f1_macro']:.4f}  "
                  f"acc {r['accuracy']:.4f}  bal-acc {r['balanced_accuracy']:.4f}")

    mono = [r for r in hy if r["family"] == "monolingual" and r["split"] == "session_holdout"]
    mono.sort(key=lambda r: -r["f1_macro"])
    if mono:
        print("\nmonolingual ranking on hyrax (session holdout, the strict split):")
        for i, r in enumerate(mono, 1):
            print(f"  {i}. {r['label']:<30} F1 {r['f1_macro']:.4f}")

    print(f"\nwrote -> {sp_dir}\n         {hy_dir}")


if __name__ == "__main__":
    main()
