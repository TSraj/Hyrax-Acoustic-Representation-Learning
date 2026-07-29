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
    rows, missing = [], []

    for task in TASKS:
        for model in MODELS:
            for frac in FRACTIONS:
                path = (Path(sweep_root) / task / model / f"frac{frac}" /
                        "lora_fine_tuning_results.json")
                if not path.exists():
                    missing.append(f"{task}/{model}/frac{frac}")
                    continue

                with open(path) as f:
                    r = json.load(f)

                test = r.get('test_metrics', {})
                val = r.get('val_metrics', {})
                hist = r.get('history', {})
                rows.append({
                    'task': task,
                    'model': model,
                    'fraction': frac,
                    'test_accuracy': test.get('accuracy'),
                    'test_f1_macro': test.get('f1_macro'),
                    'test_balanced_accuracy': test.get('balanced_accuracy'),
                    'val_accuracy': val.get('accuracy'),
                    'val_f1_macro': val.get('f1_macro'),
                    'best_val_f1_macro': r.get('best_val_f1_macro'),
                    'best_epoch': r.get('best_epoch'),
                    'epochs_run': len(hist.get('train_acc', [])),
                    'final_train_acc': (hist.get('train_acc') or [None])[-1],
                })

    df = pd.DataFrame(rows)

    logger.info(f"\nCollected {len(df)} / {len(TASKS)*len(MODELS)*len(FRACTIONS)} runs")
    if missing:
        logger.warning(f"Missing or not yet finished ({len(missing)}):")
        for m in missing:
            logger.warning(f"  {m}")

    return df, missing


def plot_curves(df, out_dir, logger):
    """2 rows (macro-F1, accuracy) x 2 cols (tasks)."""
    metrics = [("test_f1_macro", "Test macro-F1", "f1_macro"),
               ("test_accuracy", "Test accuracy", "accuracy")]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))

    for r, (col, ylabel, base_key) in enumerate(metrics):
        for c, task in enumerate(TASKS):
            ax = axes[r][c]

            series = {}
            for model in MODELS:
                sub = df[(df['task'] == task) & (df['model'] == model)].sort_values('fraction')
                sub = sub.dropna(subset=[col])
                if not sub.empty:
                    series[model] = dict(zip(sub['fraction'], sub[col]))
                    ax.plot(sub['fraction'], sub[col], marker='o', linewidth=2,
                            markersize=7, color=MODEL_COLORS[model],
                            label=MODEL_LABELS[model], zorder=3)

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
            if r == 0:
                ax.set_title(TASK_LABELS[task], fontsize=12)

    # One legend for the whole figure, including the baseline line style
    handles, labels = axes[0][0].get_legend_handles_labels()
    baseline_proxy = plt.Line2D([], [], color='grey', linestyle='--',
                                label='zero-shot baseline (frozen encoder)')
    fig.legend(handles=handles + [baseline_proxy],
               labels=labels + ['zero-shot baseline (frozen encoder)'],
               loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01), frameon=False)

    fig.suptitle("LoRA fine-tuning data efficiency", fontsize=14)
    fig.tight_layout(rect=[0, 0.045, 1, 1])

    out = Path(out_dir) / "data_efficiency_curves.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"✓ Curves saved: {out}")


def write_tables(df, missing, out_dir, logger):
    out_dir = Path(out_dir)
    csv_file = out_dir / "lora_sweep_results.csv"
    df.to_csv(csv_file, index=False)
    logger.info(f"✓ CSV saved: {csv_file}")

    md = out_dir / "lora_sweep_report.md"
    with open(md, 'w') as f:
        f.write("# LoRA Fine-Tuning Sweep - Data Efficiency\n\n")
        f.write("LoRA r=16 alpha=32 on q/k/v/out_proj, frozen base encoder, "
                "Dropout(0.3)->Linear head, AdamW 1e-4/1e-3, "
                "ReduceLROnPlateau on val macro-F1, 5s/2.5s windows, seed 42.\n\n")
        f.write(f"Runs collected: **{len(df)} / "
                f"{len(TASKS)*len(MODELS)*len(FRACTIONS)}**\n\n")

        for task in TASKS:
            f.write(f"## {TASK_LABELS[task]}\n\n")
            f.write("| Model | Fraction | Test macro-F1 | vs zero-shot | Test acc | "
                    "vs zero-shot | Best val F1 | Best epoch | Final train acc |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for model in MODELS:
                base = zero_shot_baseline(task, model)
                sub = df[(df['task'] == task) & (df['model'] == model)].sort_values('fraction')
                for _, r in sub.iterrows():
                    d_f1 = (f"{r['test_f1_macro'] - base['f1_macro']:+.4f}"
                            if base and pd.notna(r['test_f1_macro']) else "-")
                    d_acc = (f"{r['test_accuracy'] - base['accuracy']:+.4f}"
                             if base and pd.notna(r['test_accuracy']) else "-")
                    f.write(f"| {MODEL_LABELS[model]} | {int(r['fraction'])}% | "
                            f"{r['test_f1_macro']:.4f} | {d_f1} | "
                            f"{r['test_accuracy']:.4f} | {d_acc} | "
                            f"{r['best_val_f1_macro']:.4f} | {int(r['best_epoch'])} | "
                            f"{r['final_train_acc']:.4f} |\n")
                if base:
                    f.write(f"| *{MODEL_LABELS[model]} zero-shot* | - | "
                            f"*{base['f1_macro']:.4f}* | - | *{base['accuracy']:.4f}* | "
                            f"- | - | - | - |\n")
            f.write("\n")

        if missing:
            f.write("## Missing runs\n\n")
            for m in missing:
                f.write(f"- `{m}`\n")
            f.write("\n")

    logger.info(f"✓ Markdown saved: {md}")


def log_summary(df, logger):
    if df.empty:
        return
    logger.info("\n" + "=" * 80)
    logger.info("DATA EFFICIENCY (test macro-F1)")
    logger.info("=" * 80)
    for task in TASKS:
        sub = df[df['task'] == task]
        if sub.empty:
            continue
        logger.info(f"\n{TASK_LABELS[task]}")
        pivot = sub.pivot(index='model', columns='fraction', values='test_f1_macro')
        logger.info("\n" + pivot.to_string(float_format=lambda x: f"{x:.4f}"))
        for model in MODELS:
            base = zero_shot_baseline(task, model)
            row = sub[sub['model'] == model].sort_values('fraction')
            if base is not None and not row.empty:
                full = row[row['fraction'] == 100]['test_f1_macro']
                if not full.empty:
                    logger.info(f"  {MODEL_LABELS[model]}: zero-shot {base['f1_macro']:.4f} "
                                f"-> 100% fine-tuned {full.iloc[0]:.4f} "
                                f"({full.iloc[0] - base['f1_macro']:+.4f})")


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

    df, missing = collect(args.sweep_root, logger)
    if df.empty:
        logger.error("No results found - nothing to analyse yet.")
        return 1

    log_summary(df, logger)
    plot_curves(df, out_dir, logger)
    write_tables(df, missing, out_dir, logger)

    logger.info("\n✓ Sweep analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
