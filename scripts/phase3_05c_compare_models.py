#!/usr/bin/env python3
"""
Phase 3 - Step 5c: Compare Monolingual vs Multilingual Fine-Tuning
Generates gap curve showing mono vs multi performance difference across data fractions.
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger


def plot_gap_curve(mono_results, multi_results, mono_name, multi_name, task, output_dir):
    """
    Plot gap curve: monolingual - multilingual performance across data fractions.
    Shows whether multilingual advantage grows as data shrinks.
    IEEE publication ready: 300 DPI PNG, colorblind-safe.
    """
    fractions = sorted([float(k) for k in mono_results.keys()])

    # Extract data
    fraction_labels = [f'{int(f*100)}%' for f in fractions]
    mono_accs = [mono_results[str(f)]['test_metrics']['accuracy'] for f in fractions]
    multi_accs = [multi_results[str(f)]['test_metrics']['accuracy'] for f in fractions]

    gaps = [m - mu for m, mu in zip(mono_accs, multi_accs)]

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Colorblind-safe colors
    color_mono = '#0173B2'   # Blue
    color_multi = '#DE8F05'  # Orange
    color_gap = '#029E73'    # Green

    # Panel 1: Both models
    x = np.arange(len(fraction_labels))
    width = 0.35

    bars1 = ax1.bar(x - width/2, mono_accs, width, label=f'Monolingual ({mono_name})',
                    color=color_mono)
    bars2 = ax1.bar(x + width/2, multi_accs, width, label=f'Multilingual ({multi_name})',
                    color=color_multi)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax1.set_xlabel('Training Data Fraction', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title(f'Monolingual vs Multilingual Performance\n{task.replace("_", " ").title()}',
                 fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(fraction_labels)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)

    # Panel 2: Gap curve (positive = mono better, negative = multi better)
    ax2.plot(x, gaps, 'o-', color=color_gap, linewidth=3, markersize=10,
            label='Performance Gap')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    # Fill area
    ax2.fill_between(x, 0, gaps, where=np.array(gaps) > 0,
                     color=color_mono, alpha=0.3, label='Mono advantage')
    ax2.fill_between(x, 0, gaps, where=np.array(gaps) <= 0,
                     color=color_multi, alpha=0.3, label='Multi advantage')

    # Annotations
    for i, (xi, gap) in enumerate(zip(x, gaps)):
        ax2.annotate(f'{gap:+.3f}',
                    xy=(xi, gap), xytext=(0, 10 if gap > 0 else -15),
                    textcoords='offset points', ha='center',
                    fontsize=9, fontweight='bold')

    ax2.set_xlabel('Training Data Fraction', fontsize=12)
    ax2.set_ylabel('Accuracy Gap (Mono - Multi)', fontsize=12)
    ax2.set_title('Performance Gap Across Data Fractions\n(Positive = Monolingual Better)',
                 fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(fraction_labels)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f"gap_curve_{task}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def plot_combined_data_efficiency(mono_results, multi_results, mono_name, multi_name,
                                  task, output_dir, mono_zero_shot, multi_zero_shot):
    """
    Plot combined data efficiency curve for both models.
    IEEE publication ready: 300 DPI PNG, colorblind-safe.
    """
    fractions = sorted([float(k) for k in mono_results.keys()])

    # Extract data
    mono_samples = [mono_results[str(f)]['n_train_samples'] for f in fractions]
    multi_samples = [multi_results[str(f)]['n_train_samples'] for f in fractions]

    mono_accs = [mono_results[str(f)]['test_metrics']['accuracy'] for f in fractions]
    multi_accs = [multi_results[str(f)]['test_metrics']['accuracy'] for f in fractions]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Colorblind-safe colors
    color_mono = '#0173B2'   # Blue
    color_multi = '#DE8F05'  # Orange

    ax.plot(mono_samples, mono_accs, 'o-', color=color_mono,
           label=f'Monolingual ({mono_name})', linewidth=3, markersize=10)
    ax.plot(multi_samples, multi_accs, 's-', color=color_multi,
           label=f'Multilingual ({multi_name})', linewidth=3, markersize=10)

    # Zero-shot baselines
    if mono_zero_shot > 0:
        ax.axhline(mono_zero_shot, color=color_mono, linestyle='--',
                  linewidth=2, alpha=0.7, label=f'{mono_name} zero-shot')
    if multi_zero_shot > 0:
        ax.axhline(multi_zero_shot, color=color_multi, linestyle='--',
                  linewidth=2, alpha=0.7, label=f'{multi_name} zero-shot')

    # Annotations
    for x, y, frac in zip(mono_samples, mono_accs, fractions):
        ax.annotate(f'{int(frac*100)}%',
                   xy=(x, y), xytext=(0, 10),
                   textcoords='offset points', ha='center',
                   fontsize=8, color=color_mono, fontweight='bold')

    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'Data Efficiency Comparison: Monolingual vs Multilingual\n{task.replace("_", " ").title()}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    output_file = output_dir / f"combined_data_efficiency_{task}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def create_comparison_table(mono_results, multi_results, mono_name, multi_name, task, output_dir):
    """Create comparison table with both models."""
    fractions = sorted([float(k) for k in mono_results.keys()])

    rows = []
    for fraction in fractions:
        mono_data = mono_results[str(fraction)]
        multi_data = multi_results[str(fraction)]

        mono_acc = mono_data['test_metrics']['accuracy']
        multi_acc = multi_data['test_metrics']['accuracy']
        gap = mono_acc - multi_acc

        rows.append({
            'data_fraction': f'{int(fraction*100)}%',
            'n_samples': mono_data['n_train_samples'],
            f'{mono_name}_accuracy': mono_acc,
            f'{multi_name}_accuracy': multi_acc,
            'gap_mono_minus_multi': gap,
            f'{mono_name}_f1_macro': mono_data['test_metrics']['f1_macro'],
            f'{multi_name}_f1_macro': multi_data['test_metrics']['f1_macro']
        })

    df = pd.DataFrame(rows)

    # Save CSV
    csv_file = output_dir / f"model_comparison_{task}.csv"
    df.to_csv(csv_file, index=False)

    return csv_file, df


def create_comparison_report(mono_results, multi_results, mono_name, multi_name,
                            task, output_dir, df, mono_zero_shot, multi_zero_shot):
    """Create comparison report."""
    report_file = output_dir / f"model_comparison_report_{task}.txt"

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MONOLINGUAL VS MULTILINGUAL COMPARISON\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Task: {task}\n")
        f.write(f"Monolingual: {mono_name}\n")
        f.write(f"Multilingual: {multi_name}\n\n")

        f.write(f"Zero-shot baselines:\n")
        f.write(f"  {mono_name}: {mono_zero_shot:.4f}\n")
        f.write(f"  {multi_name}: {multi_zero_shot:.4f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("RESULTS BY DATA FRACTION\n")
        f.write("=" * 80 + "\n\n")

        for _, row in df.iterrows():
            f.write(f"{row['data_fraction']} Training Data:\n")
            f.write(f"  {mono_name}:  {row[f'{mono_name}_accuracy']:.4f} accuracy\n")
            f.write(f"  {multi_name}: {row[f'{multi_name}_accuracy']:.4f} accuracy\n")
            f.write(f"  Gap:          {row['gap_mono_minus_multi']:+.4f} ")
            f.write(f"({'mono better' if row['gap_mono_minus_multi'] > 0 else 'multi better'})\n\n")

        f.write("=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n\n")

        # Best model at each fraction
        f.write("Winner at each data fraction:\n")
        for _, row in df.iterrows():
            winner = mono_name if row['gap_mono_minus_multi'] > 0 else multi_name
            f.write(f"  {row['data_fraction']}: {winner}\n")

        # Data efficiency
        f.write(f"\nData efficiency (10% vs 100%):\n")
        acc_10_mono = df[df['data_fraction'] == '10%'][f'{mono_name}_accuracy'].values[0]
        acc_100_mono = df[df['data_fraction'] == '100%'][f'{mono_name}_accuracy'].values[0]
        acc_10_multi = df[df['data_fraction'] == '10%'][f'{multi_name}_accuracy'].values[0]
        acc_100_multi = df[df['data_fraction'] == '100%'][f'{multi_name}_accuracy'].values[0]

        f.write(f"  {mono_name}: 10% captures {acc_10_mono/acc_100_mono*100:.1f}% of full performance\n")
        f.write(f"  {multi_name}: 10% captures {acc_10_multi/acc_100_multi*100:.1f}% of full performance\n")

        # Gap trend
        gaps = df['gap_mono_minus_multi'].values
        if gaps[0] > gaps[-1]:
            f.write(f"\nGap trend: Multilingual advantage INCREASES with more data\n")
        elif gaps[0] < gaps[-1]:
            f.write(f"\nGap trend: Monolingual advantage INCREASES with more data\n")
        else:
            f.write(f"\nGap trend: Stable across data fractions\n")

    return report_file


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare Monolingual vs Multilingual")
    parser.add_argument("--task", required=True, choices=["species_id", "hyrax_id"])
    parser.add_argument("--mono", required=True, help="Monolingual model name")
    parser.add_argument("--multi", required=True, help="Multilingual model name")
    args = parser.parse_args()

    # Setup logging
    log_dir = Path("outputs/phase3/logs")
    logger = setup_logger(f"Phase3_CompareModels_{args.task}")

    logger.info("=" * 80)
    logger.info("COMPARING MONOLINGUAL VS MULTILINGUAL")
    logger.info("=" * 80)

    # Load results
    mono_file = Path(f"outputs/phase3/fine_tuning/{args.task}/{args.mono}/fine_tuning_results.json")
    multi_file = Path(f"outputs/phase3/fine_tuning/{args.task}/{args.multi}/fine_tuning_results.json")

    with open(mono_file, 'r') as f:
        mono_results = json.load(f)
    with open(multi_file, 'r') as f:
        multi_results = json.load(f)

    # Load zero-shot baselines
    mono_zero_shot_file = Path(f"outputs/phase3/zero_shot/{args.task}/{args.mono}/results.json")
    multi_zero_shot_file = Path(f"outputs/phase3/zero_shot/{args.task}/{args.multi}/results.json")

    if mono_zero_shot_file.exists():
        with open(mono_zero_shot_file, 'r') as f:
            mono_zero_shot = json.load(f)['test_metrics']['accuracy']
    else:
        mono_zero_shot = 0.0

    if multi_zero_shot_file.exists():
        with open(multi_zero_shot_file, 'r') as f:
            multi_zero_shot = json.load(f)['test_metrics']['accuracy']
    else:
        multi_zero_shot = 0.0

    # Output directory
    output_dir = Path(f"outputs/phase3/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Task: {args.task}")
    logger.info(f"Monolingual: {args.mono}")
    logger.info(f"Multilingual: {args.multi}\n")

    # Generate visualizations
    logger.info("Generating gap curve...")
    gap_file = plot_gap_curve(
        mono_results, multi_results, args.mono, args.multi, args.task, output_dir
    )
    logger.info(f"✓ Gap curve saved: {gap_file}")

    logger.info("\nGenerating combined data efficiency plot...")
    combined_file = plot_combined_data_efficiency(
        mono_results, multi_results, args.mono, args.multi, args.task, output_dir,
        mono_zero_shot, multi_zero_shot
    )
    logger.info(f"✓ Combined efficiency plot saved: {combined_file}")

    # Create comparison table
    logger.info("\nCreating comparison table...")
    csv_file, df = create_comparison_table(
        mono_results, multi_results, args.mono, args.multi, args.task, output_dir
    )
    logger.info(f"✓ Comparison table saved: {csv_file}")

    # Create report
    logger.info("\nCreating comparison report...")
    report_file = create_comparison_report(
        mono_results, multi_results, args.mono, args.multi, args.task, output_dir,
        df, mono_zero_shot, multi_zero_shot
    )
    logger.info(f"✓ Comparison report saved: {report_file}")

    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
