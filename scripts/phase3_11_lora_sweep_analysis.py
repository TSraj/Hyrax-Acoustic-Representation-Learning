#!/usr/bin/env python3
"""
Phase 3 - Step 11: LoRA Sweep Analysis (data efficiency)

Aggregates every LoRA run under outputs/phase3/lora_sweep_V2:

    models    xls_r (multilingual) x hubert_base (monolingual)
    tasks     species_id x hyrax_id_session_holdout_ft
    fractions hyrax    10% / 25% / 50% / 100%          (5 seeds at each)
              species  1% / 2% / 5% / 10% / 25%        (5 seeds at each)
                       50% / 100%                      (1 seed - saturated)

and produces the data-efficiency curves - test macro-F1 and accuracy against
training fraction, one line per model, one panel per task - plus a per-run
table as CSV and markdown.

The two tasks no longer share a fraction grid: species ID saturates at >=50%
(both models ~0.977), so it was extended down to 1% to expose the region where
the models actually differ. The species panel therefore uses a log x-axis, and
because its grid is ragged the per-point seed count is annotated on the axis
rather than claimed once in the legend.

Species ID also reports macro-F1 over the 7 non-hyrax classes. Its hyrax class
has only 2 test files, so a single test item moves the 8-class macro-F1 by
0.0625 - large enough to read as curve structure when it is not.

Zero-shot baselines are read from outputs/phase3/zero_shot/ rather than
hardcoded, and drawn as dashed reference lines in each model's colour. The
fine-tuned hyrax test set is the same held-out session as the zero-shot
session-holdout manifest, so that comparison is like-for-like.

Runs that are missing or still queued are skipped and listed in the report, so
this can be run against a partially complete sweep.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger

TASKS = ["hyrax_session_holdout", "species_id"]
TASK_LABELS = {
    "hyrax_session_holdout": "Hyrax ID (8-class, session-holdout)",
    "species_id": "Species ID (8-class)",
}
MODELS = ["xls_r", "hubert_base"]
MODEL_LABELS = {
    "xls_r": "XLS-R (multilingual)",
    "hubert_base": "HuBERT (monolingual)",
}
MODEL_COLORS = {"xls_r": "#0173B2", "hubert_base": "#DE8F05"}

# Species ID was extended below the saturation ceiling (it reaches ~0.977 for
# both models by 50%), so the two tasks no longer share a fraction grid.
TASK_FRACTIONS = {
    "hyrax_session_holdout": [10, 25, 50, 100],
    "species_id": [1, 2, 5, 10, 25, 50, 100],
}
FRACTIONS = sorted({f for v in TASK_FRACTIONS.values() for f in v})

# Species ID carries a hyrax class with only 2 val and 2 test files, so a
# single test item is worth 0.0625 of the 8-class macro-F1. Every species
# figure and table therefore also reports macro-F1 over the 7 non-hyrax
# classes as a robustness check. The hyrax individual-ID task has no such
# class, so the 7-class column is undefined there.
ROBUSTNESS_DROP_CLASS = {"species_id": "hyrax"}

# ---------------------------------------------------------------------------
# DO NOT CONFUSE THIS COLUMN WITH THE 7-CLASS SPECIES TASK.
#
# 'macro-F1 (7)' / test_f1_macro_7cls here is 7 classes SCORED OUT OF AN 8-WAY
# MODEL: the hyrax class is dropped from the classification report after the
# fact. The classifier still had 8 outputs, chance was still 1/8, and its
# encoder was adapted on hyrax audio.
#
# outputs/phase3/manifests_species7/ defines a genuinely different task: a
# 7-OUTPUT classifier whose encoder never sees hyrax at all (chance 1/7), used
# for the staged adaptation. Its numbers land in zero_shot_species7/ and are
# NOT interchangeable with this column - different label space, different
# chance level, different training data. Never place the two in one column and
# never compute a delta between them.
# ---------------------------------------------------------------------------
ROBUSTNESS_CAVEAT = (
    "`macro-F1 (7)` is 7 classes **scored out of an 8-way model** (the hyrax "
    "class is dropped from the report after the fact; the classifier still had "
    "8 outputs, chance 1/8, and its encoder was adapted on hyrax audio). It is "
    "**not** comparable to the genuine 7-way species task in "
    "`manifests_species7/` + `zero_shot_species7/`, which has a 7-output "
    "classifier and an encoder that never sees hyrax (chance 1/7). Different "
    "label spaces - do not same-column them or delta them."
)

N_CLASSES = 8  # both tasks here are 8-class, so chance is the same


def zero_shot_baseline(task, model):
    """Read the frozen-encoder baseline for a (task, model) pair."""
    if task == "species_id":
        path = Path(f"outputs/phase3/zero_shot/species_id/{model}/results.json")
    else:
        path = Path(f"outputs/phase3/zero_shot/hyrax_id/session_holdout/{model}/results.json")

    if not path.exists():
        return None
    with open(path) as f:
        t = json.load(f)['test_metrics']
    return {'accuracy': t['accuracy'], 'f1_macro': t['f1_macro']}


def robustness_f1(result, task):
    """Macro-F1 recomputed with the tiny hyrax class dropped (species_id only).

    Returns None where no class is dropped, so the column stays empty for the
    hyrax individual-ID task rather than silently duplicating the 8-class value.
    """
    drop = ROBUSTNESS_DROP_CLASS.get(task)
    if drop is None:
        return None
    per_class = result.get('test_per_class')
    if not per_class or drop not in per_class:
        return None
    # test_per_class is a sklearn classification_report dict, so it also holds
    # 'accuracy' / 'macro avg' / 'weighted avg' alongside the real classes.
    scores = [v['f1-score'] for k, v in per_class.items()
              if k != drop and isinstance(v, dict) and 'f1-score' in v
              and not k.endswith('avg')]
    return float(np.mean(scores)) if scores else None


def collect(sweep_root, logger):
    """Read every run, across both directory layouts.

    Single-seed runs live at   <task>/<model>/frac<NN>/lora_fine_tuning_results.json
    multi-seed runs at         <task>/<model>/frac<NN>/seed<S>/lora_fine_tuning_results.json

    The seed is taken from each file's own config rather than from the path, so
    both layouts are handled identically.
    """
    rows, empty = [], []

    for task in TASKS:
        for model in MODELS:
            for frac in TASK_FRACTIONS[task]:
                frac_dir = Path(sweep_root) / task / model / f"frac{frac}"
                paths = sorted(frac_dir.glob("lora_fine_tuning_results.json")) + \
                        sorted(frac_dir.glob("seed*/lora_fine_tuning_results.json"))
                if not paths:
                    empty.append(f"{task}/{model}/frac{frac}")
                    continue

                for path in paths:
                    with open(path) as f:
                        r = json.load(f)

                    test = r.get('test_metrics', {})
                    val = r.get('val_metrics', {})
                    hist = r.get('history', {})
                    rows.append({
                        'task': task,
                        'model': model,
                        'fraction': frac,
                        'seed': r.get('config', {}).get('seed'),
                        'test_accuracy': test.get('accuracy'),
                        'test_f1_macro': test.get('f1_macro'),
                        'test_f1_macro_7cls': robustness_f1(r, task),
                        'test_balanced_accuracy': test.get('balanced_accuracy'),
                        'val_accuracy': val.get('accuracy'),
                        'val_f1_macro': val.get('f1_macro'),
                        'best_val_f1_macro': r.get('best_val_f1_macro'),
                        'best_epoch': r.get('best_epoch'),
                        'epochs_run': len(hist.get('train_acc', [])),
                        'final_train_acc': (hist.get('train_acc') or [None])[-1],
                        'path': str(path),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, df, empty

    dupes = df[df.duplicated(subset=['task', 'model', 'fraction', 'seed'], keep=False)]
    if not dupes.empty:
        logger.warning("Duplicate (task, model, fraction, seed) entries found:")
        for _, d in dupes.iterrows():
            logger.warning(f"  seed {d['seed']}: {d['path']}")

    # Aggregate across seeds
    agg = (df.groupby(['task', 'model', 'fraction'])
             .agg(n_seeds=('seed', 'nunique'),
                  seeds=('seed', lambda s: sorted(v for v in s.unique() if v is not None)),
                  f1_mean=('test_f1_macro', 'mean'),
                  f1_std=('test_f1_macro', lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
                  f1_min=('test_f1_macro', 'min'),
                  f1_max=('test_f1_macro', 'max'),
                  f1_7cls_mean=('test_f1_macro_7cls', 'mean'),
                  f1_7cls_std=('test_f1_macro_7cls',
                               lambda s: s.std(ddof=1) if s.notna().sum() > 1 else 0.0),
                  acc_mean=('test_accuracy', 'mean'),
                  acc_std=('test_accuracy', lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
                  train_acc_mean=('final_train_acc', 'mean'))
             .reset_index())

    logger.info(f"\nCollected {len(df)} runs across "
                f"{len(agg)} (task, model, fraction) cells")
    for task in TASKS:
        sub = agg[agg['task'] == task]
        if not sub.empty:
            counts = sorted(sub['n_seeds'].unique())
            logger.info(f"  {task}: seeds per cell = {counts}")
    if empty:
        logger.warning(f"Cells with no runs ({len(empty)}):")
        for m in empty:
            logger.warning(f"  {m}")

    return df, agg, empty


def plot_curves(agg, out_dir, logger):
    """2 rows (macro-F1, accuracy) x 2 cols (tasks), mean +/- std across seeds."""
    metrics = [("f1_mean", "f1_std", "Test macro-F1", "f1_macro"),
               ("acc_mean", "acc_std", "Test accuracy", "accuracy")]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))

    for r, (mcol, scol, ylabel, base_key) in enumerate(metrics):
        for c, task in enumerate(TASKS):
            ax = axes[r][c]

            series = {}
            multi_seed = False
            for model in MODELS:
                sub = agg[(agg['task'] == task) & (agg['model'] == model)].sort_values('fraction')
                sub = sub.dropna(subset=[mcol])
                if not sub.empty:
                    series[model] = dict(zip(sub['fraction'], sub[mcol]))
                    n_lo, n_hi = int(sub['n_seeds'].min()), int(sub['n_seeds'].max())
                    if n_hi == 1:
                        label = MODEL_LABELS[model]
                    elif n_lo == n_hi:
                        label = f"{MODEL_LABELS[model]} (n={n_hi} seeds)"
                    else:
                        # Ragged grid: state the range, not the maximum.
                        label = f"{MODEL_LABELS[model]} (n={n_lo}-{n_hi} seeds)"
                    ax.plot(sub['fraction'], sub[mcol], marker='o', linewidth=2,
                            markersize=7, color=MODEL_COLORS[model],
                            label=label, zorder=3)
                    if (sub['n_seeds'] > 1).any():
                        multi_seed = True
                        ax.fill_between(sub['fraction'],
                                        sub[mcol] - sub[scol],
                                        sub[mcol] + sub[scol],
                                        color=MODEL_COLORS[model], alpha=0.18,
                                        linewidth=0, zorder=1)

                base = zero_shot_baseline(task, model)
                if base is not None:
                    ax.axhline(base[base_key], color=MODEL_COLORS[model],
                               linestyle='--', linewidth=1.3, alpha=0.75, zorder=2)

            # Label each point on the side facing away from the other curve:
            # at every x the higher value is labelled above, the lower below.
            # A fixed above/below-per-model rule collides wherever the curves
            # cross or run close together.
            for frac in TASK_FRACTIONS[task]:
                present = {m: v[frac] for m, v in series.items() if frac in v}
                if not present:
                    continue
                top = max(present, key=present.get)
                for model, value in present.items():
                    above = (model == top) or len(present) == 1
                    dy, va = ((9, 'bottom') if above else (-11, 'top'))
                    ax.annotate(f"{value:.3f}", (frac, value),
                                textcoords="offset points", xytext=(0, dy),
                                ha='center', va=va, fontsize=7.5,
                                color=MODEL_COLORS[model])

            chance = 1.0 / N_CLASSES
            ax.axhline(chance, color='grey', linestyle=':', linewidth=1.2, zorder=1)
            ax.text(0.995, chance + 0.012, 'chance',
                    transform=ax.get_yaxis_transform(), ha='right', va='bottom',
                    fontsize=8, color='grey')

            # Species spans 1-100%, so a linear axis crushes everything below
            # 25% into the left margin - exactly the region the extra runs were
            # added to resolve. Log-x only where the grid actually needs it.
            fracs = TASK_FRACTIONS[task]
            if max(fracs) / min(fracs) >= 20:
                ax.set_xscale('log')
            ax.set_xticks(fracs)
            ax.set_xticklabels([f"{f}%" for f in fracs])
            ax.minorticks_off()
            ax.set_xlabel("Training data fraction", fontsize=10)

            # Per-point seed counts. The species grid is deliberately ragged
            # (n=5 at 1-25%, n=1 at 50/100%), so a single "n=5 seeds" legend
            # entry would overstate the runs behind the saturated points.
            n_by_frac = {}
            for model in MODELS:
                cells = agg[(agg['task'] == task) & (agg['model'] == model)]
                for _, cell in cells.iterrows():
                    n_by_frac[cell['fraction']] = max(n_by_frac.get(cell['fraction'], 0),
                                                      int(cell['n_seeds']))
            if n_by_frac and len(set(n_by_frac.values())) > 1:
                for frac, n in n_by_frac.items():
                    ax.text(frac, 0.012, f"n={n}", transform=ax.get_xaxis_transform(),
                            ha='center', va='bottom', fontsize=7, color='#444444')
            ax.set_ylabel(ylabel, fontsize=10)
            # Headroom so value labels on points near 1.0 stay inside the axes
            ax.set_ylim(0, 1.14)
            ax.grid(alpha=0.3)
            # 'best' keeps the box off the curves and off the chance label,
            # whose position varies a lot between the two tasks.
            ax.legend(fontsize=8, loc='best', framealpha=0.9)
            if r == 0:
                title = TASK_LABELS[task]
                if multi_seed:
                    title += "  (shaded: ±1 SD across seeds)"
                ax.set_title(title, fontsize=11)

    baseline_proxy = plt.Line2D([], [], color='grey', linestyle='--',
                                label='zero-shot baseline (frozen encoder)')
    fig.legend(handles=[baseline_proxy],
               labels=['zero-shot baseline (frozen encoder)'],
               loc='lower center', ncol=1, fontsize=9,
               bbox_to_anchor=(0.5, -0.01), frameon=False)

    fig.suptitle("LoRA fine-tuning data efficiency", fontsize=14)
    fig.tight_layout(rect=[0, 0.045, 1, 1])

    out = Path(out_dir) / "data_efficiency_curves.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"✓ Curves saved: {out}")


def write_tables(df, agg, empty, out_dir, logger):
    out_dir = Path(out_dir)

    per_run = out_dir / "lora_sweep_results.csv"
    df.to_csv(per_run, index=False)
    logger.info(f"✓ Per-run CSV saved: {per_run}")

    agg_csv = out_dir / "lora_sweep_seed_summary.csv"
    agg.to_csv(agg_csv, index=False)
    logger.info(f"✓ Seed-summary CSV saved: {agg_csv}")

    md = out_dir / "lora_sweep_report.md"
    with open(md, 'w') as f:
        f.write("# LoRA Fine-Tuning Sweep - Data Efficiency\n\n")
        f.write("LoRA r=16 alpha=32 on q/k/v/out_proj, frozen base encoder, "
                "Dropout(0.3)->Linear head, AdamW 1e-4/1e-3, "
                "ReduceLROnPlateau on val macro-F1, 5s/2.5s windows.\n\n")
        f.write(f"Total runs: **{len(df)}**\n\n")
        f.write("Values are mean +/- SD across seeds. Cells with one seed show "
                "the single value and SD 0.0000.\n\n")

        for task in TASKS:
            sub_agg = agg[agg['task'] == task]
            if sub_agg.empty:
                continue
            drop = ROBUSTNESS_DROP_CLASS.get(task)
            f.write(f"## {TASK_LABELS[task]}\n\n")
            if drop:
                f.write(f"`macro-F1 (7)` drops the `{drop}` class, which has only 2 test "
                        f"files - one test item there is worth 0.0625 of the 8-class "
                        f"macro-F1.\n\n")
                f.write(f"> {ROBUSTNESS_CAVEAT}\n\n")
            f.write("| Model | Fraction | Seeds | Test macro-F1 | vs zero-shot | ")
            if drop:
                f.write("macro-F1 (7) | ")
            f.write("Test acc | vs zero-shot | F1 min-max | Mean train acc |\n")
            f.write("|---|---|---|---|---|" + ("---|" if drop else "") + "---|---|---|---|\n")
            for model in MODELS:
                base = zero_shot_baseline(task, model)
                rows = sub_agg[sub_agg['model'] == model].sort_values('fraction')
                for _, r in rows.iterrows():
                    d_f1 = f"{r['f1_mean'] - base['f1_macro']:+.4f}" if base else "-"
                    d_acc = f"{r['acc_mean'] - base['accuracy']:+.4f}" if base else "-"
                    f.write(f"| {MODEL_LABELS[model]} | {int(r['fraction'])}% | "
                            f"{int(r['n_seeds'])} | "
                            f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f} | {d_f1} | ")
                    if drop:
                        c7 = (f"{r['f1_7cls_mean']:.4f} ± {r['f1_7cls_std']:.4f}"
                              if pd.notna(r['f1_7cls_mean']) else "-")
                        f.write(f"{c7} | ")
                    f.write(f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f} | {d_acc} | "
                            f"{r['f1_min']:.4f}-{r['f1_max']:.4f} | "
                            f"{r['train_acc_mean']:.4f} |\n")
                if base:
                    # Columns: Model | Fraction | Seeds | macro-F1 | vs zs
                    #          [| macro-F1(7)] | acc | vs zs | min-max | train acc
                    cells = [f"*{MODEL_LABELS[model]} zero-shot*", "-", "-",
                             f"*{base['f1_macro']:.4f}*", "-"]
                    if drop:
                        cells.append("-")
                    cells += [f"*{base['accuracy']:.4f}*", "-", "-", "-"]
                    f.write("| " + " | ".join(cells) + " |\n")
            f.write("\n")

        if empty:
            f.write("## Cells with no runs\n\n")
            for m in empty:
                f.write(f"- `{m}`\n")
            f.write("\n")

    logger.info(f"✓ Markdown saved: {md}")


def log_summary(agg, logger):
    if agg.empty:
        return
    logger.info("\n" + "=" * 80)
    logger.info("DATA EFFICIENCY (test macro-F1, mean ± SD across seeds)")
    logger.info("=" * 80)
    for task in TASKS:
        sub = agg[agg['task'] == task]
        if sub.empty:
            continue
        logger.info(f"\n{TASK_LABELS[task]}")
        for model in MODELS:
            rows = sub[sub['model'] == model].sort_values('fraction')
            if rows.empty:
                continue
            cells = "  ".join(
                f"{int(r['fraction']):>4}%: {r['f1_mean']:.4f}±{r['f1_std']:.4f}"
                f"(n={int(r['n_seeds'])})" for _, r in rows.iterrows())
            logger.info(f"  {MODEL_LABELS[model]:24s} {cells}")

        base_x = zero_shot_baseline(task, 'xls_r')
        base_h = zero_shot_baseline(task, 'hubert_base')
        for model, base in [('xls_r', base_x), ('hubert_base', base_h)]:
            rows = sub[(sub['model'] == model) & (sub['fraction'] == 100)]
            if base is not None and not rows.empty:
                r = rows.iloc[0]
                logger.info(f"  {MODEL_LABELS[model]}: zero-shot {base['f1_macro']:.4f} "
                            f"-> 100% {r['f1_mean']:.4f}±{r['f1_std']:.4f} "
                            f"({r['f1_mean'] - base['f1_macro']:+.4f})")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 - LoRA sweep analysis")
    # lora_sweep_V2 is the single source of truth: it holds all 94 runs
    # (hyrax 8 single-seed + 32 multi-seed, species 8 single-seed + 46
    # multi-seed/low-fraction). lora_sweep_HPC is a strict 16-run subset of it,
    # and a bare "outputs/phase3/lora_sweep" does not exist locally at all.
    parser.add_argument("--sweep-root", default="outputs/phase3/lora_sweep_V2")
    parser.add_argument("--output-dir", default="outputs/phase3/lora_sweep_V2/summary")
    args = parser.parse_args()

    log_dir = Path("outputs/phase3/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("Phase3_LoRASweepAnalysis",
                          log_file=str(log_dir / "lora_sweep_analysis.log"))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("PHASE 3 - LoRA SWEEP ANALYSIS")
    logger.info("=" * 80)

    df, agg, empty = collect(args.sweep_root, logger)
    if df.empty:
        logger.error("No results found - nothing to analyse yet.")
        return 1

    log_summary(agg, logger)
    plot_curves(agg, out_dir, logger)
    write_tables(df, agg, empty, out_dir, logger)

    logger.info("\n✓ Sweep analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
