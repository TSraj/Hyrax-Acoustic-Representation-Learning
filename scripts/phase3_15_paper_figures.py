#!/usr/bin/env python3
"""
Phase 3 - Step 15: publication figures and CSVs for SQ1-SQ4 (ICASSP).

Reads existing results only; nothing is re-run. Every figure is emitted twice:
  *_column.png        ~3.3in wide, for a 2-column ICASSP page
  *_presentation.png  larger, for slides
both at 300 DPI, with one fixed colour per model reused across all figures.

Two things the source data does NOT support, handled explicitly rather than
faked:

  * Per-dataset macro-F1 for SQ1 does not exist. The Phase 2 per-dataset runs
    saved accuracy only - no per-class metrics, no cached embeddings. SQ1 uses
    per-dataset ACCURACY as primary, plus a macro-F1 panel derived from the
    pooled 69-way task (5 models; ECAPA has an empty embedding cache).

  * There is no frozen->fine-tuned reversal on hyrax. HuBERT leads XLS-R at
    both ends (zero-shot macro-F1 0.1735 vs 0.1017; fine-tuned 0.407 vs 0.317).
    XLS-R's frozen advantage is on the Phase 2 per-dataset task, which is a
    different task. The slope chart plots what the numbers say.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------- style

MODEL_COLORS = {
    "hubert_base":        "#DE8F05",
    "xls_r":              "#0173B2",
    "wavlm":              "#029E73",
    "wav2vec2_base":      "#CC78BC",
    "wav2vec2_base_960h": "#CA9161",
    "ecapa_tdnn":         "#949494",
}
MODEL_LABELS = {
    "hubert_base": "HuBERT", "xls_r": "XLS-R", "wavlm": "WavLM",
    "wav2vec2_base": "wav2vec2-base", "wav2vec2_base_960h": "wav2vec2-960h",
    "ecapa_tdnn": "ECAPA-TDNN",
}
MODEL_ORDER = ["hubert_base", "xls_r", "wavlm", "wav2vec2_base",
               "wav2vec2_base_960h", "ecapa_tdnn"]

SIZES = {
    "column":       dict(w=3.3, base=7,  dpi=300, lw=1.4, ms=3.5),
    "presentation": dict(w=9.0, base=13, dpi=300, lw=2.2, ms=7.0),
}

# Species ID was extended below its saturation ceiling (~0.977 for both models
# by 50%), so the two tasks no longer share a fraction grid.
TASK_FRACTIONS = {
    "hyrax_session_holdout": [10, 25, 50, 100],
    "species_id": [1, 2, 5, 10, 25, 50, 100],
}
FRACTIONS = sorted({f for v in TASK_FRACTIONS.values() for f in v})

# Species ID's hyrax class has 2 val / 2 test files, so one test item is worth
# 0.0625 of the 8-class macro-F1. Report macro-F1 without it as a robustness
# check. The hyrax individual-ID task has no such class.
ROBUSTNESS_DROP_CLASS = {"species_id": "hyrax"}

PROVENANCE = []


def rc(size):
    s = SIZES[size]
    plt.rcParams.update({
        "font.size": s["base"], "axes.titlesize": s["base"] + 1,
        "axes.labelsize": s["base"], "xtick.labelsize": s["base"] - 1,
        "ytick.labelsize": s["base"] - 1, "legend.fontsize": s["base"] - 1,
        "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
        "figure.dpi": 300, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return s


def save(fig, out_dir, name, size, sources):
    out = Path(out_dir) / f"{name}_{size}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    PROVENANCE.append({"output": out.name, "kind": "figure", "sources": "; ".join(sources)})
    return out


# ---------------------------------------------------------------- loaders

def load_sq1():
    p = "outputs/phase2V2/zero_shot/per_dataset_summary/accuracy_summary.csv"
    df = pd.read_csv(p)
    df = df[df["dataset"] != "MEAN"].set_index("dataset")
    return df, p


def load_sq1_macro_f1():
    """Per-dataset macro-F1 for the frozen per-dataset individual-ID task.

    The per-dataset summaries record accuracy only, but each layer's
    results.json carries a full sklearn classification_report, so macro-F1 at
    each model's best layer is recoverable exactly - no re-running.
    """
    root = Path("outputs/phase2V2/zero_shot/per_dataset")
    if not root.exists():
        return None, None
    rows = []
    for ds in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for m in MODEL_ORDER:
            d = root / ds / m
            sp = d / "summary.json"
            if not sp.exists():
                continue
            best = json.load(open(sp)).get("best_layer")
            cand = d / f"layer_{best}" / "results.json" if best is not None else None
            if cand is None or not cand.exists():
                hits = sorted(d.glob("**/results.json"))
                cand = hits[0] if hits else None
            if cand is None:
                continue
            cr = json.load(open(cand)).get("classification_report", {})
            mf = cr.get("macro avg", {}).get("f1-score")
            if mf is not None:
                rows.append({"dataset": ds, "model": m, "macro_f1": mf, "best_layer": best})
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    piv = df.pivot(index="dataset", columns="model", values="macro_f1")
    piv = piv[[m for m in MODEL_ORDER if m in piv.columns]]
    return piv, str(root / "<dataset>/<model>/layer_<best>/results.json")


def load_sq1_species_f1():
    """Per-species F1 from the frozen 8-class species-ID task (Phase 3)."""
    out, srcs = {}, []
    for m in MODEL_ORDER:
        p = Path(f"outputs/phase3/zero_shot/species_id/{m}/per_class_metrics.csv")
        if p.exists():
            out[m] = pd.read_csv(p).set_index("class")["f1-score"]
            srcs.append(str(p))
    if not out:
        return None, None
    return pd.DataFrame(out), "; ".join(srcs)


def robustness_f1(result, task):
    """Macro-F1 with the tiny hyrax class dropped (species_id only).

    None where nothing is dropped, so the column stays empty for the hyrax
    individual-ID task rather than silently duplicating the 8-class value.
    """
    drop = ROBUSTNESS_DROP_CLASS.get(task)
    if drop is None:
        return None
    per_class = result.get("test_per_class")
    if not per_class or drop not in per_class:
        return None
    # test_per_class is a sklearn classification_report dict, so it also carries
    # 'accuracy' / 'macro avg' / 'weighted avg' beside the real classes.
    scores = [v["f1-score"] for k, v in per_class.items()
              if k != drop and isinstance(v, dict) and "f1-score" in v
              and not k.endswith("avg")]
    return float(np.mean(scores)) if scores else None


def load_sweep(root="outputs/phase3/lora_sweep_V2"):
    """Load every run. lora_sweep_V2 is the single source of truth - it holds
    all 94 runs; lora_sweep_HPC is a strict 16-run subset of it."""
    rows, srcs = [], set()
    for task in ["hyrax_session_holdout", "species_id"]:
        for m in ["hubert_base", "xls_r"]:
            for fr in TASK_FRACTIONS[task]:
                base = Path(root) / task / m / f"frac{fr}"
                paths = ([base / "lora_fine_tuning_results.json"] +
                         sorted(base.glob("seed*/lora_fine_tuning_results.json")))
                for p in paths:
                    if not p.exists():
                        continue
                    r = json.load(open(p))
                    t = r["test_metrics"]
                    rows.append({"task": task, "model": m, "fraction": fr,
                                 "seed": r["config"]["seed"],
                                 "f1_macro": t["f1_macro"], "accuracy": t["accuracy"],
                                 "balanced_accuracy": t["balanced_accuracy"],
                                 "f1_macro_7cls": robustness_f1(r, task),
                                 "final_train_acc": r["history"]["train_acc"][-1]})
                    srcs.add(str(base.parent))
    return pd.DataFrame(rows), root


def zero_shot(task, model):
    p = (Path(f"outputs/phase3/zero_shot/species_id/{model}/results.json") if task == "species_id"
         else Path(f"outputs/phase3/zero_shot/hyrax_id/session_holdout/{model}/results.json"))
    if not p.exists():
        return None, None
    t = json.load(open(p))["test_metrics"]
    return {"f1_macro": t["f1_macro"], "accuracy": t["accuracy"],
            "balanced_accuracy": t["balanced_accuracy"]}, str(p)


def agg(df, metric):
    return (df.groupby(["task", "model", "fraction"])[metric]
              .agg(["mean", "std", "count", "min", "max"]).reset_index())


# ---------------------------------------------------------------- SQ1

def _grouped_bars(ax, table, models, s, ylabel, legend_loc="lower right"):
    datasets = list(table.index)
    x = np.arange(len(datasets)); w = 0.8 / len(models)
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * w, table[m], w,
               label=MODEL_LABELS[m], color=MODEL_COLORS[m], edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in datasets])
    ax.set_ylabel(ylabel); ax.set_ylim(0, 1.08)
    ax.legend(ncol=3, fontsize=s["base"] - 2, loc=legend_loc, framealpha=0.9)


def _heatmap(ax, table, models, s, cbar_label, fig):
    datasets = list(table.index)
    M = table[models].T.values
    im = ax.imshow(M, cmap="viridis", vmin=0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([d.replace("_", "\n") for d in datasets], fontsize=s["base"] - 2)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in models])
    for i in range(len(models)):
        for j in range(len(datasets)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=s["base"] - 2, color="white" if M[i, j] < 0.75 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.03)


def _model_means(ax, table, models, s, ylabel):
    means = [table[m].mean() for m in models]
    sds = [table[m].std(ddof=1) for m in models]
    order = np.argsort(means)[::-1]
    ax.bar(range(len(models)), [means[i] for i in order], yerr=[sds[i] for i in order],
           color=[MODEL_COLORS[models[i]] for i in order], capsize=3,
           edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS[models[i]] for i in order], rotation=35, ha="right")
    ax.set_ylabel(ylabel); ax.set_ylim(0, 1.08)


def fig_sq1(size, out_dir):
    s = rc(size)
    wide = s["w"] * (1.9 if size == "column" else 1.5)

    # ---- PRIMARY: per-dataset macro-F1, frozen individual-ID task
    mf, msrc = load_sq1_macro_f1()
    if mf is not None:
        mods = [m for m in MODEL_ORDER if m in mf.columns]

        fig, ax = plt.subplots(figsize=(wide, s["w"] * 0.62))
        _grouped_bars(ax, mf, mods, s, "Zero-shot macro-F1", "lower left")
        ax.set_title("Frozen per-dataset individual ID (macro-F1)")
        save(fig, out_dir, "sq1_a_per_dataset_macro_f1_bars", size, [msrc])

        fig, ax = plt.subplots(figsize=(s["w"] * 1.5, s["w"] * 0.75))
        _heatmap(ax, mf, mods, s, "Macro-F1", fig)
        ax.set_title("Frozen macro-F1, models x datasets")
        save(fig, out_dir, "sq1_b_heatmap_macro_f1", size, [msrc])

        fig, ax = plt.subplots(figsize=(s["w"], s["w"] * 0.78))
        _model_means(ax, mf, mods, s, "Mean macro-F1 across datasets")
        ax.set_title("Model ranking under a frozen encoder")
        save(fig, out_dir, "sq1_c_model_means_macro_f1", size, [msrc])

    # ---- SECONDARY: per-dataset accuracy (same task)
    acc, src = load_sq1()
    models = [m for m in MODEL_ORDER if m in acc.columns]
    fig, ax = plt.subplots(figsize=(wide, s["w"] * 0.62))
    _grouped_bars(ax, acc, models, s, "Zero-shot accuracy", "lower right")
    ax.set_title("Frozen per-dataset individual ID (accuracy)")
    save(fig, out_dir, "sq1_d_per_dataset_accuracy_bars", size, [src])

    # ---- SECONDARY: species-ID task, per-species F1 (different task, 8 classes)
    sf, ssrc = load_sq1_species_f1()
    if sf is not None:
        mods = [m for m in MODEL_ORDER if m in sf.columns]
        fig, ax = plt.subplots(figsize=(wide, s["w"] * 0.62))
        _grouped_bars(ax, sf, mods, s, "Per-class F1", "upper right")
        ax.set_title("Frozen species ID (8-class), per-species F1")
        save(fig, out_dir, "sq1_e_species_id_per_class_f1", size, [ssrc])


# ---------------------------------------------------------------- SQ2/SQ3

TASK_LABEL = {"hyrax_session_holdout": "Hyrax ID (session-holdout)",
              "species_id": "Species ID"}


def fig_dataeff(size, out_dir, df, metric, tag, ylabel):
    s = rc(size)
    a = agg(df, metric)
    tasks = ["hyrax_session_holdout", "species_id"]

    # line + SD band
    fig, axes = plt.subplots(1, 2, figsize=(s["w"] * 2.0, s["w"] * 0.72), sharey=True)
    for ax, task in zip(axes, tasks):
        fracs = TASK_FRACTIONS[task]
        n_by_frac = {}
        for m in ["hubert_base", "xls_r"]:
            sub = a[(a["task"] == task) & (a["model"] == m)].sort_values("fraction")
            if sub.empty:
                continue
            for _, cell in sub.iterrows():
                n_by_frac[cell["fraction"]] = max(n_by_frac.get(cell["fraction"], 0),
                                                  int(cell["count"]))
            n_lo, n_hi = int(sub["count"].min()), int(sub["count"].max())
            # Ragged grid (species: n=5 at 1-25%, n=1 at 50/100%) - state the
            # range rather than let the maximum stand for every point.
            suffix = ("" if n_hi == 1 else
                      f" (n={n_hi})" if n_lo == n_hi else f" (n={n_lo}-{n_hi})")
            ax.plot(sub["fraction"], sub["mean"], marker="o", color=MODEL_COLORS[m],
                    lw=s["lw"], ms=s["ms"], label=f"{MODEL_LABELS[m]}{suffix}")
            if (sub["count"] > 1).any():
                ax.fill_between(sub["fraction"], sub["mean"] - sub["std"],
                                sub["mean"] + sub["std"], color=MODEL_COLORS[m],
                                alpha=0.18, lw=0)
            zs, _ = zero_shot(task, m)
            if zs:
                ax.axhline(zs[metric], color=MODEL_COLORS[m], ls="--", lw=1.1, alpha=0.8)
        ax.axhline(1 / 8, color="grey", ls=":", lw=1.0)
        # Log-x where the grid spans a wide range, otherwise everything below
        # 25% collapses into the left margin - the region the extra runs exist
        # to resolve.
        if max(fracs) / min(fracs) >= 20:
            ax.set_xscale("log")
        ax.set_xticks(fracs); ax.set_xticklabels([f"{f}%" for f in fracs])
        ax.minorticks_off()
        ax.set_xlabel("Training data fraction")
        ax.set_title(TASK_LABEL[task])
        ax.set_ylim(0, 1.08)
        if len(set(n_by_frac.values())) > 1:
            for frac, n in n_by_frac.items():
                ax.text(frac, 0.015, f"n={n}", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=s["base"] - 3,
                        color="#444444")
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="best", fontsize=s["base"] - 2, framealpha=0.9)
    fig.suptitle(f"Data efficiency of LoRA fine-tuning ({ylabel}); "
                 f"dashed = frozen baseline", fontsize=s["base"] + 1)
    fig.tight_layout()
    save(fig, out_dir, f"sq2_a_dataeff_lines_{tag}", size,
         ["outputs/phase3/lora_sweep_V2", "outputs/phase3/zero_shot"])

    # grouped bars with SD error bars
    fig, axes = plt.subplots(1, 2, figsize=(s["w"] * 2.0, s["w"] * 0.72), sharey=True)
    for ax, task in zip(axes, tasks):
        fracs = TASK_FRACTIONS[task]
        # Categorical x here (unlike the line panel), so the ragged grid needs
        # no log scale - but a bar with n=1 has no SD and must not read as one.
        x = np.arange(len(fracs)); w = 0.36
        pos = {f: i for i, f in enumerate(fracs)}
        for i, m in enumerate(["hubert_base", "xls_r"]):
            sub = a[(a["task"] == task) & (a["model"] == m)].sort_values("fraction")
            if sub.empty:
                continue
            xs = np.array([pos[f] for f in sub["fraction"]]) + (i - 0.5) * w
            ax.bar(xs, sub["mean"], w, yerr=sub["std"].fillna(0),
                   capsize=3, color=MODEL_COLORS[m], label=MODEL_LABELS[m],
                   edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        # Mark the single-seed fractions in the tick label itself, so a bar
        # with no error bar cannot be mistaken for one with zero variance.
        n_by_frac = {int(c["fraction"]): int(c["count"])
                     for _, c in a[a["task"] == task].iterrows()}
        ragged = len(set(n_by_frac.values())) > 1
        ax.set_xticklabels([f"{f}%" + ("\n(n=1)" if ragged and n_by_frac.get(f) == 1 else "")
                            for f in fracs])
        ax.set_xlabel("Training data fraction"); ax.set_title(TASK_LABEL[task])
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="best", fontsize=s["base"] - 2)
    fig.suptitle(f"Data efficiency of LoRA fine-tuning ({ylabel}), "
                 f"error bars = ±1 SD", fontsize=s["base"] + 1)
    fig.tight_layout()
    save(fig, out_dir, f"sq2_b_dataeff_bars_{tag}", size, ["outputs/phase3/lora_sweep_V2"])


def species_per_class_table(root="outputs/phase3/lora_sweep_V2"):
    """Long-format per-class test F1 for every species run, one row per
    (model, fraction, seed, class).

    Species macro-F1 hides a specific, reproducible failure: XLS-R sometimes
    absorbs nearly all of bengalese_finch (n=292) into wetlands_bird (n=79),
    which flattens both classes at once. It happens in 2 of 5 seeds at both 10%
    and 25%, so it survives averaging as inflated SD but is invisible in any
    aggregate metric. This table is what makes it auditable.
    """
    rows = []
    for m in ["hubert_base", "xls_r"]:
        for fr in TASK_FRACTIONS["species_id"]:
            base = Path(root) / "species_id" / m / f"frac{fr}"
            for p in ([base / "lora_fine_tuning_results.json"] +
                      sorted(base.glob("seed*/lora_fine_tuning_results.json"))):
                if not p.exists():
                    continue
                r = json.load(open(p))
                for cls, v in (r.get("test_per_class") or {}).items():
                    if not isinstance(v, dict) or cls.endswith("avg"):
                        continue
                    rows.append({"model": m, "fraction": fr,
                                 "seed": r["config"]["seed"], "class": cls,
                                 "f1": v.get("f1-score"),
                                 "precision": v.get("precision"),
                                 "recall": v.get("recall"),
                                 "support": v.get("support")})
    if not rows:
        return None
    return (pd.DataFrame(rows)
              .sort_values(["model", "fraction", "class", "seed"])
              .reset_index(drop=True))


def fig_gap(size, out_dir, df):
    """Paired (HuBERT - XLS-R) macro-F1 vs fraction, one panel per task.

    Paired at the seed level, which is exact here: the window subsample is
    drawn from the seed alone and does not depend on the model, so both models
    at a given (fraction, seed) train on the identical window subset.

    Species now has 5 seeds at 1-25% so it gets the same treatment as hyrax;
    its 50/100% points remain single-seed and are drawn as bare markers with
    no error bar, since a paired difference of one run has no spread.
    """
    s = rc(size)
    tasks = ["hyrax_session_holdout", "species_id"]
    fig, axes = plt.subplots(1, 2, figsize=(s["w"] * 2.0, s["w"] * 0.78))

    for ax, task in zip(axes, tasks):
        d = df[df["task"] == task]
        fr_multi, mean_multi, sd_multi = [], [], []
        fr_single, mean_single = [], []
        for fr in TASK_FRACTIONS[task]:
            h = d[(d.model == "hubert_base") & (d.fraction == fr)].set_index("seed")["f1_macro"]
            x = d[(d.model == "xls_r") & (d.fraction == fr)].set_index("seed")["f1_macro"]
            common = sorted(set(h.index) & set(x.index))
            if not common:
                continue
            diff = h.loc[common].values - x.loc[common].values
            if len(diff) > 1:
                fr_multi.append(fr); mean_multi.append(diff.mean())
                sd_multi.append(diff.std(ddof=1))
            else:
                fr_single.append(fr); mean_single.append(diff.mean())

        if fr_multi:
            ax.errorbar(fr_multi, mean_multi, yerr=sd_multi, marker="o",
                        color="#333333", lw=s["lw"], ms=s["ms"], capsize=3)
            ax.fill_between(fr_multi, np.array(mean_multi) - np.array(sd_multi),
                            np.array(mean_multi) + np.array(sd_multi),
                            color="#333333", alpha=0.15, lw=0)
        if fr_single:
            ax.plot(fr_single, mean_single, marker="o", ls=":", color="#333333",
                    lw=s["lw"] * 0.8, ms=s["ms"], mfc="white", mew=s["lw"])

        ax.axhline(0, color="crimson", ls="--", lw=1.1)
        fracs = TASK_FRACTIONS[task]
        if max(fracs) / min(fracs) >= 20:
            ax.set_xscale("log")
        ax.set_xticks(fracs); ax.set_xticklabels([f"{f}%" for f in fracs])
        ax.minorticks_off()
        ax.set_xlabel("Training data fraction")
        n_multi = sorted({len(set(d[(d.model == 'hubert_base') & (d.fraction == fr)].seed) &
                              set(d[(d.model == 'xls_r') & (d.fraction == fr)].seed))
                          for fr in fr_multi}) if fr_multi else []
        note = f"paired, n={n_multi[0]} seeds" if len(n_multi) == 1 else "paired"
        if fr_single:
            note += "; open = n=1"
        ax.set_title(f"{TASK_LABEL[task]}\n({note})")
    axes[0].set_ylabel("Macro-F1 gap (HuBERT − XLS-R)")
    fig.suptitle("Monolingual advantage over the data-efficiency curve",
                 fontsize=s["base"] + 1)
    fig.tight_layout()
    save(fig, out_dir, "sq3_gap_curve", size, ["outputs/phase3/lora_sweep_V2"])


# ---------------------------------------------------------------- SQ2 frozen -> fine-tuned

def fig_reversal(size, out_dir, df):
    s = rc(size)
    ft = agg(df[df.task == "hyrax_session_holdout"], "f1_macro")
    ft = ft[ft.fraction == 100].set_index("model")

    zs = {}
    srcs = []
    for m in MODEL_ORDER:
        z, p = zero_shot("hyrax_session_holdout", m)
        if z:
            zs[m] = z["f1_macro"]; srcs.append(p)

    # slope chart
    fig, ax = plt.subplots(figsize=(s["w"], s["w"] * 0.95))
    for m, v in zs.items():
        if m in ft.index:
            y2 = ft.loc[m, "mean"]
            ax.plot([0, 1], [v, y2], "-o", color=MODEL_COLORS[m], lw=s["lw"], ms=s["ms"],
                    label=MODEL_LABELS[m])
            ax.errorbar([1], [y2], yerr=[ft.loc[m, "std"]], color=MODEL_COLORS[m], capsize=3)
            ax.annotate(f"{v:.3f}", (0, v), textcoords="offset points", xytext=(-4, 0),
                        ha="right", fontsize=s["base"] - 2, color=MODEL_COLORS[m])
            ax.annotate(f"{y2:.3f}", (1, y2), textcoords="offset points", xytext=(4, 0),
                        ha="left", fontsize=s["base"] - 2, color=MODEL_COLORS[m])
        else:
            ax.plot([0], [v], "o", color=MODEL_COLORS[m], ms=s["ms"], alpha=0.75)
            ax.annotate(MODEL_LABELS[m], (0, v), textcoords="offset points", xytext=(-4, 0),
                        ha="right", fontsize=s["base"] - 2, color=MODEL_COLORS[m])
    ax.set_xlim(-0.55, 1.45); ax.set_xticks([0, 1])
    ax.set_xticklabels(["frozen\n(zero-shot)", "fine-tuned\n(LoRA, 100%)"])
    ax.set_ylabel("Test macro-F1 (hyrax session-holdout)")
    ax.set_title("Frozen → fine-tuned on hyrax ID\n(HuBERT leads at both ends)")
    ax.legend(loc="upper left", fontsize=s["base"] - 2, framealpha=0.9)
    fig.tight_layout()
    save(fig, out_dir, "sq2_c_slope_frozen_to_finetuned", size,
         srcs + ["outputs/phase3/lora_sweep_V2"])

    # paired bars
    fig, ax = plt.subplots(figsize=(s["w"], s["w"] * 0.78))
    mods = [m for m in ["hubert_base", "xls_r"] if m in ft.index]
    x = np.arange(len(mods)); w = 0.36
    ax.bar(x - w / 2, [zs[m] for m in mods], w, label="frozen (zero-shot)",
           color=[MODEL_COLORS[m] for m in mods], alpha=0.45, edgecolor="white")
    ax.bar(x + w / 2, [ft.loc[m, "mean"] for m in mods], w,
           yerr=[ft.loc[m, "std"] for m in mods], capsize=3, label="fine-tuned (LoRA)",
           color=[MODEL_COLORS[m] for m in mods], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[m] for m in mods])
    ax.set_ylabel("Test macro-F1"); ax.set_title("Frozen vs fine-tuned (hyrax ID)")
    ax.legend(fontsize=s["base"] - 2)
    fig.tight_layout()
    save(fig, out_dir, "sq2_d_frozen_vs_finetuned_bars", size,
         srcs + ["outputs/phase3/lora_sweep_V2"])


# ---------------------------------------------------------------- SQ4

def fig_sq4(size, out_dir):
    s = rc(size)
    cp = "outputs/phase3/sq4/sq4_correlations.csv"
    fp = "outputs/phase3/sq4/per_individual_f1.csv"
    ap = "outputs/phase3/sq4/acoustic_predictors.csv"
    if not Path(cp).exists():
        print("  SQ4 inputs missing - skipping")
        return
    res = pd.read_csv(cp)

    # (a) drop-one-dataset collapse
    fig, ax = plt.subplots(figsize=(s["w"] * 1.35, s["w"] * 0.75))
    x = np.arange(len(res)); w = 0.38
    ax.bar(x - w / 2, res["naive_rho"].abs(), w, label="all 6 datasets",
           color="#0173B2", edgecolor="white")
    ax.bar(x + w / 2, res["loo_rho_when_worst_dropped"].abs(), w,
           label="wetlands_bird removed", color="#CC78BC", edgecolor="white")
    for i, r in res.iterrows():
        if r["naive_p"] < 0.05:
            ax.text(i - w / 2, abs(r["naive_rho"]) + 0.015, "*", ha="center",
                    fontsize=s["base"])
    ax.set_xticks(x)
    ax.set_xticklabels([l.split("(")[0].strip().replace(" ", "\n") for l in res["label"]],
                       fontsize=s["base"] - 2)
    ax.set_ylabel("|Spearman ρ| with transfer F1")
    ax.legend(fontsize=s["base"] - 2)
    ax.set_title("Every correlation collapses when\none dataset is removed (* p<0.05)")
    fig.tight_layout()
    save(fig, out_dir, "sq4_a_drop_one_dataset", size, [cp])

    # (b) scatter vs bandwidth
    if Path(fp).exists() and Path(ap).exists():
        d = pd.read_csv(fp).merge(pd.read_csv(ap), on=["individual", "dataset"])
        d = d[d["support"] >= 5]
        fig, ax = plt.subplots(figsize=(s["w"] * 1.2, s["w"] * 0.85))
        dss = sorted(d["dataset"].unique())
        cmap = dict(zip(dss, plt.cm.tab10(np.linspace(0, 1, len(dss)))))
        for ds in dss:
            g = d[d["dataset"] == ds]
            ax.scatter(g["bandwidth_hz"], g["f1_mean"], s=26 if size == "column" else 60,
                       color=cmap[ds], label=ds, edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xlabel("Bandwidth, rolloff95−05 (Hz)")
        ax.set_ylabel("Mean per-individual F1 (frozen)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=s["base"] - 3, loc="lower left", ncol=2, framealpha=0.9)
        ax.set_title("One corpus drives the trend")
        fig.tight_layout()
        save(fig, out_dir, "sq4_b_scatter_bandwidth", size, [fp, ap])

    # (c) ICC variance decomposition
    fig, ax = plt.subplots(figsize=(s["w"] * 0.85, s["w"] * 0.8))
    icc = 0.759
    ax.bar([0], [icc * 100], color="#0173B2", edgecolor="white", label="between datasets")
    ax.bar([0], [(1 - icc) * 100], bottom=[icc * 100], color="#CC78BC",
           edgecolor="white", label="within datasets")
    ax.text(0, icc * 50, f"{icc*100:.0f}%", ha="center", va="center",
            color="white", fontsize=s["base"] + 1, fontweight="bold")
    ax.text(0, icc * 100 + (1 - icc) * 50, f"{(1-icc)*100:.0f}%", ha="center",
            va="center", color="white", fontsize=s["base"])
    ax.set_xticks([]); ax.set_ylabel("% of variance in per-individual F1")
    ax.set_ylim(0, 100); ax.legend(fontsize=s["base"] - 2, loc="lower right")
    ax.set_title("Variance is between corpora,\nnot within them (ICC=0.76)")
    fig.tight_layout()
    save(fig, out_dir, "sq4_c_icc_variance", size, [cp])


# ---------------------------------------------------------------- CSVs

def write_csvs(out_dir, df):
    out_dir = Path(out_dir)

    mf, msrc = load_sq1_macro_f1()
    if mf is not None:
        t = mf.copy(); t.loc["MEAN"] = mf.mean()
        t.to_csv(out_dir / "sq1_per_dataset_zero_shot_macro_f1.csv")
        PROVENANCE.append({"output": "sq1_per_dataset_zero_shot_macro_f1.csv",
                           "kind": "csv", "sources": msrc})

    acc, src = load_sq1()
    t = acc.copy(); t.loc["MEAN"] = acc.mean()
    t.to_csv(out_dir / "sq1_per_dataset_zero_shot_accuracy.csv")
    PROVENANCE.append({"output": "sq1_per_dataset_zero_shot_accuracy.csv",
                       "kind": "csv", "sources": src})

    sf, ssrc = load_sq1_species_f1()
    if sf is not None:
        t = sf.copy(); t.loc["MEAN"] = sf.mean()
        t.to_csv(out_dir / "sq1_species_id_per_class_f1.csv")
        PROVENANCE.append({"output": "sq1_species_id_per_class_f1.csv",
                           "kind": "csv", "sources": ssrc})

    # SQ2/SQ3 data-efficiency table
    rows = []
    for task in ["hyrax_session_holdout", "species_id"]:
        for fr in TASK_FRACTIONS[task]:
            rec = {"task": task, "fraction": fr}
            paired = {}
            for m in ["hubert_base", "xls_r"]:
                sub = df[(df.task == task) & (df.model == m) & (df.fraction == fr)]
                if sub.empty:
                    continue
                paired[m] = sub.set_index("seed")
                # f1_7 is the robustness metric: macro-F1 without the 2-test-file
                # hyrax class. All-NaN for the hyrax task, which drops nothing.
                for met, short in [("f1_macro", "f1"), ("f1_macro_7cls", "f1_7"),
                                   ("balanced_accuracy", "bal"), ("accuracy", "acc")]:
                    vals = sub[met].dropna()
                    rec[f"{m}_{short}_mean"] = vals.mean() if len(vals) else np.nan
                    rec[f"{m}_{short}_sd"] = vals.std(ddof=1) if len(vals) > 1 else (
                        0.0 if len(vals) == 1 else np.nan)
                rec[f"{m}_n_seeds"] = sub["seed"].nunique()
                rec[f"{m}_train_acc_mean"] = sub["final_train_acc"].mean()
            if len(paired) == 2:
                common = sorted(set(paired["hubert_base"].index) & set(paired["xls_r"].index))
                for met, short in [("f1_macro", "f1"), ("f1_macro_7cls", "f1_7"),
                                   ("balanced_accuracy", "bal")]:
                    h = paired["hubert_base"].loc[common, met].values
                    x = paired["xls_r"].loc[common, met].values
                    if np.isnan(h).any() or np.isnan(x).any():
                        continue          # metric undefined for this task
                    rec[f"gap_{short}"] = h.mean() - x.mean()
                    if len(common) > 1:
                        rec[f"paired_p_{short}"] = stats.ttest_rel(h, x).pvalue
                        rec[f"welch_p_{short}"] = stats.ttest_ind(h, x, equal_var=False).pvalue
                        pooled = np.sqrt((h.var(ddof=1) + x.var(ddof=1)) / 2)
                        rec[f"cohens_d_{short}"] = (h.mean() - x.mean()) / pooled if pooled else np.nan
                    rec[f"hubert_wins_{short}"] = int((h > x).sum())
                    rec["n_paired"] = len(common)
            rows.append(rec)
    pd.DataFrame(rows).to_csv(out_dir / "sq2_sq3_data_efficiency.csv", index=False)
    PROVENANCE.append({"output": "sq2_sq3_data_efficiency.csv", "kind": "csv",
                       "sources": "outputs/phase3/lora_sweep_V2/**/lora_fine_tuning_results.json"})

    df.to_csv(out_dir / "sq2_sq3_per_seed_runs.csv", index=False)
    PROVENANCE.append({"output": "sq2_sq3_per_seed_runs.csv", "kind": "csv",
                       "sources": "outputs/phase3/lora_sweep_V2/**/lora_fine_tuning_results.json"})

    pc = species_per_class_table()
    if pc is not None:
        pc.to_csv(out_dir / "sq2_species_per_class_f1_by_seed.csv", index=False)
        PROVENANCE.append({
            "output": "sq2_species_per_class_f1_by_seed.csv", "kind": "csv",
            "sources": "outputs/phase3/lora_sweep_V2/species_id/**/"
                       "lora_fine_tuning_results.json"})

    # SQ2 reversal
    ft = agg(df[df.task == "hyrax_session_holdout"], "f1_macro")
    ft = ft[ft.fraction == 100].set_index("model")
    rows, srcs = [], []
    for m in MODEL_ORDER:
        z, p = zero_shot("hyrax_session_holdout", m)
        if not z:
            continue
        srcs.append(p)
        rows.append({"model": m, "zero_shot_f1_macro": z["f1_macro"],
                     "zero_shot_accuracy": z["accuracy"],
                     "zero_shot_balanced_acc": z["balanced_accuracy"],
                     "finetuned_f1_macro_mean": ft.loc[m, "mean"] if m in ft.index else np.nan,
                     "finetuned_f1_macro_sd": ft.loc[m, "std"] if m in ft.index else np.nan,
                     "delta": (ft.loc[m, "mean"] - z["f1_macro"]) if m in ft.index else np.nan})
    pd.DataFrame(rows).to_csv(out_dir / "sq2_frozen_vs_finetuned_hyrax.csv", index=False)
    PROVENANCE.append({"output": "sq2_frozen_vs_finetuned_hyrax.csv", "kind": "csv",
                       "sources": "; ".join(srcs) + "; outputs/phase3/lora_sweep_V2"})

    # SQ4
    for name, src in [("sq4_correlations.csv", "outputs/phase3/sq4/sq4_correlations.csv"),
                      ("sq4_dataset_means.csv", "outputs/phase3/sq4/sq4_dataset_means.csv"),
                      ("sq4_per_individual_f1.csv", "outputs/phase3/sq4/per_individual_f1.csv"),
                      ("sq4_acoustic_predictors.csv", "outputs/phase3/sq4/acoustic_predictors.csv")]:
        if Path(src).exists():
            pd.read_csv(src).to_csv(out_dir / name, index=False)
            PROVENANCE.append({"output": name, "kind": "csv", "sources": src})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/figures_paper")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    df, sweep_root = load_sweep()
    print(f"sweep runs loaded: {len(df)} from {sweep_root}")

    for size in ["column", "presentation"]:
        print(f"\n--- {size} ---")
        fig_sq1(size, out)
        fig_dataeff(size, out, df, "f1_macro", "macro_f1", "Test macro-F1")
        fig_dataeff(size, out, df, "balanced_accuracy", "balanced_acc", "Test balanced accuracy")
        fig_gap(size, out, df)
        fig_reversal(size, out, df)
        fig_sq4(size, out)

    write_csvs(out, df)

    prov = pd.DataFrame(PROVENANCE).drop_duplicates(subset=["output"])
    prov.to_csv(out / "PROVENANCE.csv", index=False)
    print(f"\nfigures: {sum(1 for p in PROVENANCE if p['kind']=='figure')}")
    print(f"csvs   : {sum(1 for p in PROVENANCE if p['kind']=='csv')}")
    print(f"wrote {out}/PROVENANCE.csv")


if __name__ == "__main__":
    main()
