#!/usr/bin/env python3
"""
Phase 3 - Step 5b: Analyze Fine-Tuning Results
Generates learning curves and data efficiency visualizations.
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


def plot_learning_curves(results, model_name, task, output_dir, zero_shot_acc):
    """
    Plot learning curves for all data fractions.
    IEEE publication ready: 300 DPI PNG, colorblind-safe.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Colorblind-safe palette
    colors = plt.cm.viridis(np.linspace(0, 0.9, 4))

    fractions = sorted([float(k) for k in results.keys()])

    for idx, fraction in enumerate(fractions):
        data = results[str(fraction)]
        history = data['history']

        # Plot training curves
        ax = axes[idx // 2, idx % 2]

        epochs = range(1, len(history['train_loss']) + 1)

        # Accuracy on primary axis
        ax.plot(epochs, history['train_acc'], 'o-', color=colors[idx],
               label='Train Acc', linewidth=2, markersize=4)
        ax.plot(epochs, history['val_acc'], 's-', color=colors[(idx+1)%4],
               label='Val Acc', linewidth=2, markersize=4)

        # Zero-shot baseline
        if zero_shot_acc > 0:
            ax.axhline(zero_shot_acc, color='red', linestyle='--',
                      linewidth=2, label='Zero-shot baseline')

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{int(fraction*100)}% Training Data ({data["n_train_samples"]} samples)',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.suptitle(f'Learning Curves: {model_name} on {task.replace("_", " ").title()}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / "learning_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def plot_data_efficiency(results, model_name, task, output_dir, zero_shot_acc):
    """
    Plot data efficiency: accuracy vs. training data amount.
    IEEE publication ready: 300 DPI PNG, colorblind-safe.
    """
    fractions = sorted([float(k) for k in results.keys()])

    # Extract data
    train_samples = []
    test_accs = []
    val_accs = []

    for fraction in fractions:
        data = results[str(fraction)]
        train_samples.append(data['n_train_samples'])
        test_accs.append(data['test_metrics']['accuracy'])
        val_accs.append(data['best_val_acc'])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colorblind-safe colors
    color_test = '#0173B2'  # Blue
    color_val = '#DE8F05'   # Orange

    ax.plot(train_samples, test_accs, 'o-', color=color_test,
           label='Test Accuracy', linewidth=3, markersize=10)
    ax.plot(train_samples, val_accs, 's--', color=color_val,
           label='Val Accuracy', linewidth=2, markersize=8)

    # Zero-shot baseline
    if zero_shot_acc > 0:
        ax.axhline(zero_shot_acc, color='red', linestyle='--',
                  linewidth=2, label='Zero-shot baseline')

    # Annotations on test accuracy points
    for x, y, frac in zip(train_samples, test_accs, fractions):
        ax.annotate(f'{int(frac*100)}%\n{y:.3f}',
                   xy=(x, y), xytext=(0, 10),
                   textcoords='offset points', ha='center',
                   fontsize=9, fontweight='bold')

    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Data Efficiency: {model_name} on {task.replace("_", " ").title()}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    output_file = output_dir / "data_efficiency.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def plot_grouped_metrics_by_fraction(results, model_name, task, output_dir):
    """
    Plot grouped bar chart: Accuracy and F1-Macro for each data fraction.
    Clearer comparison than line chart.
    IEEE publication ready: 300 DPI PNG, colorblind-safe.
    """
    fractions = sorted([float(k) for k in results.keys()])

    # Extract data
    accuracies = []
    f1_macros = []
    fraction_labels = []

    for fraction in fractions:
        data = results[str(fraction)]
        accuracies.append(data['test_metrics']['accuracy'])
        f1_macros.append(data['test_metrics']['f1_macro'])
        fraction_labels.append(f'{int(fraction*100)}%')

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(fraction_labels))
    width = 0.35

    # Colorblind-safe colors
    color_acc = '#0173B2'   # Blue
    color_f1 = '#DE8F05'    # Orange

    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy',
                   color=color_acc)
    bars2 = ax.bar(x + width/2, f1_macros, width, label='F1-Macro',
                   color=color_f1)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Training Data Fraction', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Accuracy & F1-Macro by Data Fraction\n{model_name} on {task.replace("_", " ").title()}',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(fraction_labels)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "metrics_by_fraction.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def create_summary_table(results, model_name, task, output_dir, zero_shot_acc):
    """Create summary table with all data fractions."""
    fractions = sorted([float(k) for k in results.keys()])

    rows = []
    for fraction in fractions:
        data = results[str(fraction)]

        improvement = data['test_metrics']['accuracy'] - zero_shot_acc

        rows.append({
            'data_fraction': f'{int(fraction*100)}%',
            'n_train_samples': data['n_train_samples'],
            'best_val_acc': data['best_val_acc'],
            'test_accuracy': data['test_metrics']['accuracy'],
            'test_balanced_acc': data['test_metrics']['balanced_accuracy'],
            'test_f1_macro': data['test_metrics']['f1_macro'],
            'improvement_over_zero_shot': improvement
        })

    df = pd.DataFrame(rows)

    # Save CSV
    csv_file = output_dir / "fine_tuning_summary.csv"
    df.to_csv(csv_file, index=False)

    return csv_file, df


def create_summary_report(results, model_name, task, output_dir, zero_shot_acc, df):
    """Create human-readable summary report."""
    report_file = output_dir / "fine_tuning_report.txt"

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FINE-TUNING SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Model: {model_name}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Zero-shot baseline: {zero_shot_acc:.4f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("RESULTS BY DATA FRACTION\n")
        f.write("=" * 80 + "\n\n")

        for _, row in df.iterrows():
            f.write(f"{row['data_fraction']} Training Data ({row['n_train_samples']} samples):\n")
            f.write(f"  Test Accuracy:     {row['test_accuracy']:.4f}\n")
            f.write(f"  Balanced Accuracy: {row['test_balanced_acc']:.4f}\n")
            f.write(f"  F1-Macro:          {row['test_f1_macro']:.4f}\n")
            f.write(f"  Improvement:       {row['improvement_over_zero_shot']:+.4f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n\n")

        # Best result
        best_idx = df['test_accuracy'].idxmax()
        best_row = df.loc[best_idx]

        f.write(f"Best performance: {best_row['data_fraction']} training data\n")
        f.write(f"  Accuracy: {best_row['test_accuracy']:.4f}\n")
        f.write(f"  Improvement: {best_row['improvement_over_zero_shot']:+.4f}\n\n")

        # Data efficiency
        acc_10 = df[df['data_fraction'] == '10%']['test_accuracy'].values[0]
        acc_100 = df[df['data_fraction'] == '100%']['test_accuracy'].values[0]

        f.write(f"Data efficiency:\n")
        f.write(f"  10% data achieves: {acc_10:.4f} accuracy\n")
        f.write(f"  100% data achieves: {acc_100:.4f} accuracy\n")
        f.write(f"  10% captures {acc_10/acc_100*100:.1f}% of full performance\n")

    return report_file


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Fine-Tuning Results")
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    # Setup logging
    log_dir = Path("outputs/phase3/logs")
    logger = setup_logger(f"Phase3_AnalyzeFineTune_{args.task}_{args.model}")

    logger.info("=" * 80)
    logger.info("ANALYZING FINE-TUNING RESULTS")
    logger.info("=" * 80)

    # Paths
    results_dir = Path(f"outputs/phase3/fine_tuning/{args.task}/{args.model}")
    results_file = results_dir / "fine_tuning_results.json"

    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Load zero-shot baseline
    zero_shot_file = Path(f"outputs/phase3/zero_shot/{args.task}/{args.model}/results.json")
    if zero_shot_file.exists():
        with open(zero_shot_file, 'r') as f:
            zero_shot_data = json.load(f)
            zero_shot_acc = zero_shot_data['test_metrics']['accuracy']
    else:
        zero_shot_acc = 0.0

    logger.info(f"Model: {args.model}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Zero-shot baseline: {zero_shot_acc:.4f}\n")

    # Generate visualizations
    logger.info("Generating learning curves...")
    learning_curves_file = plot_learning_curves(
        results, args.model, args.task, results_dir, zero_shot_acc
    )
    logger.info(f"✓ Learning curves saved: {learning_curves_file}")

    logger.info("\nGenerating data efficiency plot...")
    efficiency_file = plot_data_efficiency(
        results, args.model, args.task, results_dir, zero_shot_acc
    )
    logger.info(f"✓ Data efficiency plot saved: {efficiency_file}")

    logger.info("\nGenerating grouped metrics by fraction...")
    grouped_file = plot_grouped_metrics_by_fraction(
        results, args.model, args.task, results_dir
    )
    logger.info(f"✓ Grouped metrics chart saved: {grouped_file}")

    # Create summary table
    logger.info("\nCreating summary table...")
    csv_file, df = create_summary_table(
        results, args.model, args.task, results_dir, zero_shot_acc
    )
    logger.info(f"✓ Summary table saved: {csv_file}")

    # Create report
    logger.info("\nCreating summary report...")
    report_file = create_summary_report(
        results, args.model, args.task, results_dir, zero_shot_acc, df
    )
    logger.info(f"✓ Summary report saved: {report_file}")

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutput directory: {results_dir}")
    logger.info("\nGenerated files:")
    logger.info("  - fine_tuning_results.json")
    logger.info("  - learning_curves.png (4 subplots: train/val per fraction)")
    logger.info("  - metrics_by_fraction.png (grouped bar: accuracy + F1)")
    logger.info("  - data_efficiency.png (line: accuracy vs. samples)")
    logger.info("  - data_efficiency.png")
    logger.info("  - fine_tuning_summary.csv")
    logger.info("  - fine_tuning_report.txt")


if __name__ == "__main__":
    main()
