#!/usr/bin/env python3
"""
Phase 3 - Step 11: LoRA Sweep Analysis (data efficiency)

Aggregates the 16 runs produced by run_phase3_lora_sweep.sh:

    models    xls_r (multilingual) x hubert_base (monolingual)
    tasks     species_id x hyrax_id_session_holdout_ft
    fractions 10% / 25% / 50% / 100% of training windows

and produces the data-efficiency curves - test macro-F1 and accuracy against
training fraction, one line per model, one panel per task - plus a per-run
table as CSV and markdown.

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
FRACTIONS = [10, 25, 50, 100]
N_CLASSES = 8  # both tasks are 8-class, so chance is the same


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
            for frac in FRACTIONS:
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
                    n_max = int(sub['n_seeds'].max())
                    label = (f"{MODEL_LABELS[model]} (n={n_max} seeds)" if n_max > 1
                             else MODEL_LABELS[model])
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
            for frac in FRACTIONS:
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

            ax.set_xticks(FRACTIONS)
            ax.set_xticklabels([f"{f}%" for f in FRACTIONS])
            ax.set_xlabel("Training data fraction", fontsize=10)
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
            f.write(f"## {TASK_LABELS[task]}\n\n")
            f.write("| Model | Fraction | Seeds | Test macro-F1 | vs zero-shot | "
                    "Test acc | vs zero-shot | F1 min-max | Mean train acc |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for model in MODELS:
                base = zero_shot_baseline(task, model)
                rows = sub_agg[sub_agg['model'] == model].sort_values('fraction')
                for _, r in rows.iterrows():
                    d_f1 = f"{r['f1_mean'] - base['f1_macro']:+.4f}" if base else "-"
                    d_acc = f"{r['acc_mean'] - base['accuracy']:+.4f}" if base else "-"
                    f.write(f"| {MODEL_LABELS[model]} | {int(r['fraction'])}% | "
                            f"{int(r['n_seeds'])} | "
                            f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f} | {d_f1} | "
                            f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f} | {d_acc} | "
                            f"{r['f1_min']:.4f}-{r['f1_max']:.4f} | "
                            f"{r['train_acc_mean']:.4f} |\n")
                if base:
                    f.write(f"| *{MODEL_LABELS[model]} zero-shot* | - | - | "
                            f"*{base['f1_macro']:.4f}* | - | *{base['accuracy']:.4f}* | "
                            f"- | - | - |\n")
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
    parser.add_argument("--sweep-root", default="outputs/phase3/lora_sweep")
    parser.add_argument("--output-dir", default="outputs/phase3/lora_sweep/summary")
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
