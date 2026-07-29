#!/usr/bin/env python3
"""
Phase 3 - Denoiser Screen: aggregate and compare audio signal versions.

Screening experiment (NOT the full pipeline): compares three audio signal
versions - Original, BIODA, ACA - on the 8-individual hyrax session tasks
using XLS-R only, to decide which signal version to fine-tune on.

Two evaluation protocols per version, identical apart from the split rule:
  * hyrax_id_within_session  - random 80/20 bout split (sessions shared)
  * hyrax_id_session_holdout - held-out session is the test set

This script does not run models. It reads the results.json written by
phase3_03_zero_shot_evaluation.py for each (version, task) and:
  1. verifies all three versions used exactly the same bouts/individuals/sessions
  2. builds the comparison table (accuracy + macro-F1)
  3. writes CSV, markdown and a 300 DPI PNG summary
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

VERSIONS = ["original", "bioda", "aca"]
VERSION_LABELS = {"original": "Original", "bioda": "BIODA", "aca": "ACA"}
TASKS = ["hyrax_id_within_session", "hyrax_id_session_holdout"]
TASK_LABELS = {
    "hyrax_id_within_session": "Within-session",
    "hyrax_id_session_holdout": "Session-holdout",
}


def check_identical_data(screen_root, logger):
    """Confirm the three versions used the same bouts / individuals / sessions.

    Any accuracy difference is only attributable to the denoiser if the
    underlying bout inventory is identical across versions.
    """
    logger.info("\n" + "=" * 80)
    logger.info("DATA IDENTITY CHECK (same bouts/individuals/sessions across versions)")
    logger.info("=" * 80)

    all_ok = True
    details = {}

    for task in TASKS:
        inventories = {}
        for version in VERSIONS:
            manifest_file = screen_root / "manifests" / version / f"{task}.json"
            if not manifest_file.exists():
                logger.warning(f"  MISSING manifest: {manifest_file}")
                all_ok = False
                continue
            with open(manifest_file) as f:
                manifest = json.load(f)
            inventories[version] = {
                "individuals": sorted(manifest["individuals"]),
                "bout_inventory": manifest.get("bout_inventory", {}),
                "split_bouts": {
                    split: {i["individual"]: i["num_bouts"] for i in items}
                    for split, items in manifest["splits"].items()
                },
                "held_out_sessions": manifest.get("held_out_sessions"),
            }

        if len(inventories) < len(VERSIONS):
            continue

        reference = VERSIONS[0]
        task_ok = True
        for version in VERSIONS[1:]:
            for key in ["individuals", "bout_inventory", "split_bouts", "held_out_sessions"]:
                if inventories[version][key] != inventories[reference][key]:
                    logger.error(f"  MISMATCH [{task}] {key}: {reference} vs {version}")
                    task_ok = False
                    all_ok = False

        total_bouts = sum(v["total_valid_bouts"]
                          for v in inventories[reference]["bout_inventory"].values())
        n_sessions = sum(len(v["sessions"])
                         for v in inventories[reference]["bout_inventory"].values())
        status = "IDENTICAL" if task_ok else "MISMATCH"
        logger.info(f"  {TASK_LABELS[task]:16s}: {status} | "
                    f"{len(inventories[reference]['individuals'])} individuals, "
                    f"{total_bouts} valid bouts, {n_sessions} sessions")

        details[task] = {
            "identical": task_ok,
            "individuals": inventories[reference]["individuals"],
            "total_valid_bouts": total_bouts,
            "bout_inventory": inventories[reference]["bout_inventory"],
        }

    if all_ok:
        logger.info("\n  ✓ All three versions use exactly the same data. "
                    "Differences are attributable to the denoiser.")
    else:
        logger.error("\n  ✗ Data differs across versions - comparison is NOT clean.")

    return all_ok, details


def collect_results(screen_root, model, logger):
    """Read results.json for every (version, task) pair."""
    rows = []
    missing = []

    for version in VERSIONS:
        for task in TASKS:
            results_file = screen_root / "results" / version / task / model / "results.json"
            if not results_file.exists():
                missing.append(str(results_file))
                continue

            with open(results_file) as f:
                results = json.load(f)

            test = results["test_metrics"]
            train = results["train_metrics"]
            rows.append({
                "audio_version": VERSION_LABELS[version],
                "protocol": TASK_LABELS[task],
                "test_accuracy": test["accuracy"],
                "test_macro_f1": test["f1_macro"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "train_accuracy": train["accuracy"],
                "train_macro_f1": train["f1_macro"],
                "_version": version,
                "_task": task,
            })

    if missing:
        logger.warning(f"\nMissing {len(missing)} result file(s):")
        for m in missing:
            logger.warning(f"  {m}")

    return pd.DataFrame(rows)


def write_outputs(df, model, screen_root, data_ok, logger):
    """Write CSV, markdown table and PNG chart."""
    out_dir = screen_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_file = out_dir / "denoiser_screen_results.csv"
    df.drop(columns=["_version", "_task"]).to_csv(csv_file, index=False)
    logger.info(f"\n✓ CSV saved: {csv_file}")

    # Pivot: rows = audio version, cols = protocol x metric
    pivot = df.pivot(index="audio_version", columns="protocol",
                     values=["test_accuracy", "test_macro_f1"])
    pivot = pivot.reindex([VERSION_LABELS[v] for v in VERSIONS if VERSION_LABELS[v] in pivot.index])

    logger.info("\n" + "=" * 80)
    logger.info(f"DENOISER SCREEN RESULTS ({model})")
    logger.info("=" * 80)
    logger.info("\n" + pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    # Winner by session-holdout accuracy (the number that matters for fine-tuning)
    holdout = df[df["protocol"] == TASK_LABELS["hyrax_id_session_holdout"]]
    winner = None
    if not holdout.empty:
        winner_row = holdout.loc[holdout["test_accuracy"].idxmax()]
        winner = winner_row["audio_version"]
        logger.info(f"\nBest session-holdout accuracy: {winner} "
                    f"({winner_row['test_accuracy']:.4f} acc, "
                    f"{winner_row['test_macro_f1']:.4f} macro-F1)")

    # Markdown report
    md_file = out_dir / "denoiser_screen_report.md"
    with open(md_file, "w") as f:
        f.write("# Denoiser Screen - Audio Signal Version Comparison\n\n")
        f.write(f"**Model:** {model} (frozen encoder, mean-pool, linear head)  \n")
        f.write("**Task:** hyrax individual ID, 8 individuals "
                "(R3, Q7, P1, P8, Kashtan, O7, M9, U7)  \n")
        f.write("**Windowing:** 5.0 s / 2.5 s stride  \n")
        f.write(f"**Data identity check:** "
                f"{'PASS - identical bouts across all versions' if data_ok else 'FAIL - see log'}\n\n")

        f.write("## Results\n\n")
        f.write("| Audio version | Protocol | Test accuracy | Test macro-F1 | "
                "Balanced acc | Train accuracy |\n")
        f.write("|---|---|---|---|---|---|\n")
        for version in VERSIONS:
            for task in TASKS:
                sub = df[(df["_version"] == version) & (df["_task"] == task)]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                f.write(f"| {r['audio_version']} | {r['protocol']} | "
                        f"{r['test_accuracy']:.4f} | {r['test_macro_f1']:.4f} | "
                        f"{r['test_balanced_accuracy']:.4f} | {r['train_accuracy']:.4f} |\n")

        if winner:
            f.write(f"\n**Best session-holdout accuracy: {winner}**\n")
    logger.info(f"✓ Markdown report saved: {md_file}")

    # PNG chart (300 DPI, colorblind-safe)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"Original": "#0173B2", "BIODA": "#DE8F05", "ACA": "#029E73"}

    for ax, metric, title in zip(
        axes,
        ["test_accuracy", "test_macro_f1"],
        ["Test accuracy", "Test macro-F1"],
    ):
        protocols = [TASK_LABELS[t] for t in TASKS]
        x = np.arange(len(protocols))
        width = 0.26
        for i, version in enumerate(VERSIONS):
            label = VERSION_LABELS[version]
            vals = []
            for p in protocols:
                sub = df[(df["audio_version"] == label) & (df["protocol"] == p)]
                vals.append(sub.iloc[0][metric] if not sub.empty else np.nan)
            bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=colors[label])
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=8)

        chance = 1.0 / 8
        ax.axhline(chance, color="grey", linestyle="--", linewidth=1.2)
        # Anchor inside the axes (y in data coords, x in axes fraction)
        ax.text(0.995, chance + 0.012, "chance", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color="grey")
        ax.set_xticks(x)
        ax.set_xticklabels(protocols)
        ax.set_ylabel(title, fontsize=11)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(title, fontsize=12)

    axes[0].legend(title="Audio version", fontsize=9, title_fontsize=9)
    fig.suptitle(f"Denoiser screen: {model}, 8-individual hyrax ID", fontsize=13)
    fig.tight_layout()

    png_file = out_dir / "denoiser_screen_comparison.png"
    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✓ Chart saved: {png_file}")

    return winner


def main():
    parser = argparse.ArgumentParser(description="Phase 3 - Denoiser screen aggregation")
    parser.add_argument("--screen-root", default="outputs/phase3/denoiser_screen")
    parser.add_argument("--model", default="xls_r")
    parser.add_argument("--check-only", action="store_true",
                        help="Only run the data identity check (before models are run)")
    args = parser.parse_args()

    log_dir = Path("outputs/phase3/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("Phase3_DenoiserScreen",
                          log_file=str(log_dir / "denoiser_screen.log"))

    screen_root = Path(args.screen_root)

    logger.info("=" * 80)
    logger.info("PHASE 3 - DENOISER SCREEN")
    logger.info(f"Versions: {', '.join(VERSION_LABELS[v] for v in VERSIONS)}")
    logger.info(f"Model: {args.model}")
    logger.info("=" * 80)

    data_ok, _ = check_identical_data(screen_root, logger)

    if args.check_only:
        return 0 if data_ok else 1

    df = collect_results(screen_root, args.model, logger)
    if df.empty:
        logger.error("No results found - run the evaluations first.")
        return 1

    write_outputs(df, args.model, screen_root, data_ok, logger)
    logger.info("\n✓ Denoiser screen aggregation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
