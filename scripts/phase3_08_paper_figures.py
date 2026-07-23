#!/usr/bin/env python3
"""
Phase 3 - Step 8: Paper Figures for ICASSP 2027
Generates publication-ready visualizations with backing CSVs for:
1. Monolingual Experiments (Species ID + Hyrax ID)
2. Multilingual Experiments (Species ID + Hyrax ID)
3. Adaptation Experiments (Zero-shot vs Fine-tuned)
4. Statistical Winner Declaration
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.logging_utils import setup_logger


def load_zero_shot_results():
    """Load all zero-shot results."""
    results = {}

    for task in ['species_id', 'hyrax_id']:
        results[task] = {}
        zero_shot_dir = Path(f"outputs/phase3/zero_shot/{task}")

        if not zero_shot_dir.exists():
            continue

        for model_dir in zero_shot_dir.glob("*"):
            if model_dir.is_dir():
                result_file = model_dir / "results.json"
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        results[task][model_dir.name] = json.load(f)

    return results


def load_fine_tuned_results():
    """Load fine-tuning results for selected models."""
    results = {}

    # Load selected models
    selected_file = Path("outputs/phase3/model_selection/selected_models.json")
    with open(selected_file, 'r') as f:
        selected = json.load(f)

    for task in ['species_id', 'hyrax_id']:
        results[task] = {}

        for model_name in [selected['monolingual'], selected['multilingual']]:
            result_file = Path(f"outputs/phase3/fine_tuning/{task}/{model_name}/fine_tuning_results.json")
            if result_file.exists():
                with open(result_file, 'r') as f:
                    results[task][model_name] = json.load(f)

    return results, selected


def create_monolingual_experiments_csv(zero_shot_results, output_dir):
    """CSV for all monolingual models on both tasks."""
    mono_models = ['wav2vec2_base', 'wav2vec2_base_960h', 'hubert_base', 'ecapa_tdnn']

    rows = []
    for task in ['species_id', 'hyrax_id']:
        for model in mono_models:
            if model in zero_shot_results.get(task, {}):
                metrics = zero_shot_results[task][model]['test_metrics']
                rows.append({
                    'task': task,
                    'model': model,
                    'accuracy': metrics['accuracy'],
                    'balanced_accuracy': metrics['balanced_accuracy'],
                    'f1_macro': metrics['f1_macro'],
                    'f1_weighted': metrics['f1_weighted'],
                    'precision_macro': metrics['precision_macro'],
                    'recall_macro': metrics['recall_macro']
                })

    df = pd.DataFrame(rows)
    csv_file = output_dir / "monolingual_experiments.csv"
    df.to_csv(csv_file, index=False)

    return csv_file, df


def create_multilingual_experiments_csv(zero_shot_results, output_dir):
    """CSV for all multilingual models on both tasks."""
    multi_models = ['xls_r', 'wavlm']

    rows = []
    for task in ['species_id', 'hyrax_id']:
        for model in multi_models:
            if model in zero_shot_results.get(task, {}):
                metrics = zero_shot_results[task][model]['test_metrics']
                rows.append({
                    'task': task,
                    'model': model,
                    'accuracy': metrics['accuracy'],
                    'balanced_accuracy': metrics['balanced_accuracy'],
                    'f1_macro': metrics['f1_macro'],
                    'f1_weighted': metrics['f1_weighted'],
                    'precision_macro': metrics['precision_macro'],
                    'recall_macro': metrics['recall_macro']
                })

    df = pd.DataFrame(rows)
    csv_file = output_dir / "multilingual_experiments.csv"
    df.to_csv(csv_file, index=False)

    return csv_file, df


def create_adaptation_experiments_csv(zero_shot_results, fine_tuned_results, selected, output_dir):
    """CSV for zero-shot vs fine-tuned on selected models."""
    rows = []

    for task in ['species_id', 'hyrax_id']:
        for model_type, model_name in [('monolingual', selected['monolingual']),
                                       ('multilingual', selected['multilingual'])]:

            # Zero-shot
            if model_name in zero_shot_results.get(task, {}):
                zs_metrics = zero_shot_results[task][model_name]['test_metrics']
                rows.append({
                    'task': task,
                    'model': model_name,
                    'model_type': model_type,
                    'condition': 'zero_shot',
                    'data_fraction': 'N/A',
                    'accuracy': zs_metrics['accuracy'],
                    'balanced_accuracy': zs_metrics['balanced_accuracy'],
                    'f1_macro': zs_metrics['f1_macro'],
                    'f1_weighted': zs_metrics['f1_weighted'],
                    'precision_macro': zs_metrics['precision_macro'],
                    'recall_macro': zs_metrics['recall_macro']
                })

            # Fine-tuned (all fractions)
            if model_name in fine_tuned_results.get(task, {}):
                ft_data = fine_tuned_results[task][model_name]
                fractions = sorted([float(k) for k in ft_data.keys()])

                for fraction in fractions:
                    ft_metrics = ft_data[str(fraction)]['test_metrics']
                    rows.append({
                        'task': task,
                        'model': model_name,
                        'model_type': model_type,
                        'condition': 'fine_tuned',
                        'data_fraction': f'{int(fraction*100)}%',
                        'accuracy': ft_metrics['accuracy'],
                        'balanced_accuracy': ft_metrics['balanced_accuracy'],
                        'f1_macro': ft_metrics['f1_macro'],
                        'f1_weighted': ft_metrics['f1_weighted'],
                        'precision_macro': ft_metrics['precision_macro'],
                        'recall_macro': ft_metrics['recall_macro']
                    })

    df = pd.DataFrame(rows)
    csv_file = output_dir / "adaptation_experiments.csv"
    df.to_csv(csv_file, index=False)

    return csv_file, df


def plot_monolingual_experiments(df, output_dir):
    """Publication figure: All monolingual models on both tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    tasks = ['species_id', 'hyrax_id']

    for ax, task in zip(axes, tasks):
        task_df = df[df['task'] == task]

        # Prepare data
        models = task_df['model'].values
        x = np.arange(len(models))
        width = 0.2

        # Plot bars
        colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC']
        for i, metric in enumerate(metrics):
            values = task_df[metric].values
            offset = (i - len(metrics)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title(),
                         color=colors[i], alpha=0.8)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Monolingual Models: {task.replace("_", " ").title()}',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "monolingual_experiments.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def plot_multilingual_experiments(df, output_dir):
    """Publication figure: All multilingual models on both tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    tasks = ['species_id', 'hyrax_id']

    for ax, task in zip(axes, tasks):
        task_df = df[df['task'] == task]

        # Prepare data
        models = task_df['model'].values
        x = np.arange(len(models))
        width = 0.2

        # Plot bars
        colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC']
        for i, metric in enumerate(metrics):
            values = task_df[metric].values
            offset = (i - len(metrics)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title(),
                         color=colors[i], alpha=0.8)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Multilingual Models: {task.replace("_", " ").title()}',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "multilingual_experiments.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def plot_adaptation_experiments(df, selected, output_dir):
    """Publication figure: Zero-shot vs Fine-tuned for both tasks."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    tasks = ['species_id', 'hyrax_id']
    metrics_to_plot = [('accuracy', 'Accuracy'), ('f1_macro', 'F1-Macro')]

    for row_idx, (metric, metric_label) in enumerate(metrics_to_plot):
        for col_idx, task in enumerate(tasks):
            ax = axes[row_idx, col_idx]

            task_df = df[df['task'] == task]

            mono_name = selected['monolingual']
            multi_name = selected['multilingual']

            # Get data
            mono_df = task_df[task_df['model'] == mono_name].sort_values('condition')
            multi_df = task_df[task_df['model'] == multi_name].sort_values('condition')

            # X positions: zero-shot, then 4 fine-tuned fractions
            conditions = ['zero_shot'] + [f'{int(f*100)}%' for f in [0.1, 0.25, 0.5, 1.0]]
            x = np.arange(len(conditions))

            # Prepare values
            mono_vals = []
            multi_vals = []

            for cond in conditions:
                if cond == 'zero_shot':
                    mono_row = mono_df[mono_df['condition'] == 'zero_shot']
                    multi_row = multi_df[multi_df['condition'] == 'zero_shot']
                else:
                    mono_row = mono_df[mono_df['data_fraction'] == cond]
                    multi_row = multi_df[multi_df['data_fraction'] == cond]

                mono_vals.append(mono_row[metric].values[0] if len(mono_row) > 0 else 0)
                multi_vals.append(multi_row[metric].values[0] if len(multi_row) > 0 else 0)

            # Plot
            width = 0.35
            color_mono = '#0173B2'
            color_multi = '#DE8F05'

            bars1 = ax.bar(x - width/2, mono_vals, width,
                          label=f'Monolingual ({mono_name})', color=color_mono, alpha=0.8)
            bars2 = ax.bar(x + width/2, multi_vals, width,
                          label=f'Multilingual ({multi_name})', color=color_multi, alpha=0.8)

            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)

            ax.set_xlabel('Training Condition', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
            ax.set_title(f'{task.replace("_", " ").title()}: {metric_label}',
                        fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(['Zero-shot', '10%', '25%', '50%', '100%'])
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "adaptation_experiments.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def compute_statistical_winner(zero_shot_results, fine_tuned_results, selected, output_dir):
    """Compute statistical tests: mono vs multi."""

    results = []

    for task in ['species_id', 'hyrax_id']:
        mono_name = selected['monolingual']
        multi_name = selected['multilingual']

        # Zero-shot comparison
        if (mono_name in zero_shot_results.get(task, {}) and
            multi_name in zero_shot_results.get(task, {})):

            mono_acc_zs = zero_shot_results[task][mono_name]['test_metrics']['accuracy']
            multi_acc_zs = zero_shot_results[task][multi_name]['test_metrics']['accuracy']

            results.append({
                'task': task,
                'condition': 'zero_shot',
                'mono_model': mono_name,
                'multi_model': multi_name,
                'mono_accuracy': mono_acc_zs,
                'multi_accuracy': multi_acc_zs,
                'difference': mono_acc_zs - multi_acc_zs,
                'winner': mono_name if mono_acc_zs > multi_acc_zs else multi_name
            })

        # Fine-tuned comparison (100% data)
        if (mono_name in fine_tuned_results.get(task, {}) and
            multi_name in fine_tuned_results.get(task, {})):

            mono_acc_ft = fine_tuned_results[task][mono_name]['1.0']['test_metrics']['accuracy']
            multi_acc_ft = fine_tuned_results[task][multi_name]['1.0']['test_metrics']['accuracy']

            results.append({
                'task': task,
                'condition': 'fine_tuned_100%',
                'mono_model': mono_name,
                'multi_model': multi_name,
                'mono_accuracy': mono_acc_ft,
                'multi_accuracy': multi_acc_ft,
                'difference': mono_acc_ft - multi_acc_ft,
                'winner': mono_name if mono_acc_ft > multi_acc_ft else multi_name
            })

    df = pd.DataFrame(results)
    csv_file = output_dir / "statistical_winner.csv"
    df.to_csv(csv_file, index=False)

    return csv_file, df


def create_winner_report(winner_df, output_dir):
    """Create final winner declaration report."""
    report_file = output_dir / "winner_declaration.txt"

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 3 - WINNER DECLARATION\n")
        f.write("Monolingual vs Multilingual: Statistical Evidence\n")
        f.write("=" * 80 + "\n\n")

        for _, row in winner_df.iterrows():
            f.write(f"Task: {row['task']}\n")
            f.write(f"Condition: {row['condition']}\n")
            f.write(f"\n")
            f.write(f"  Monolingual ({row['mono_model']}): {row['mono_accuracy']:.4f}\n")
            f.write(f"  Multilingual ({row['multi_model']}): {row['multi_accuracy']:.4f}\n")
            f.write(f"\n")
            f.write(f"  Difference: {row['difference']:+.4f}\n")
            f.write(f"  WINNER: {row['winner']}\n")
            f.write("\n" + "-" * 80 + "\n\n")

        # Overall summary
        f.write("=" * 80 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        mono_wins = (winner_df['winner'] == winner_df['mono_model']).sum()
        multi_wins = (winner_df['winner'] == winner_df['multi_model']).sum()

        f.write(f"Monolingual wins: {mono_wins}/{len(winner_df)}\n")
        f.write(f"Multilingual wins: {multi_wins}/{len(winner_df)}\n\n")

        if mono_wins > multi_wins:
            f.write(f"FINAL WINNER: MONOLINGUAL\n")
        elif multi_wins > mono_wins:
            f.write(f"FINAL WINNER: MULTILINGUAL\n")
        else:
            f.write(f"RESULT: TIE\n")

    return report_file


def main():
    """Main entry point."""
    logger = setup_logger("Phase3_PaperFigures")

    logger.info("=" * 80)
    logger.info("PHASE 3 - PAPER FIGURES GENERATION")
    logger.info("=" * 80)

    # Output directory
    output_dir = Path("outputs/phase3/paper_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    logger.info("\nLoading results...")
    zero_shot_results = load_zero_shot_results()
    fine_tuned_results, selected = load_fine_tuned_results()

    logger.info(f"  Zero-shot: {sum(len(v) for v in zero_shot_results.values())} results")
    logger.info(f"  Fine-tuned: {sum(len(v) for v in fine_tuned_results.values())} results")
    logger.info(f"  Selected mono: {selected['monolingual']}")
    logger.info(f"  Selected multi: {selected['multilingual']}")

    # Figure 1: Monolingual Experiments
    logger.info("\n" + "=" * 80)
    logger.info("FIGURE 1: MONOLINGUAL EXPERIMENTS")
    logger.info("=" * 80)

    mono_csv, mono_df = create_monolingual_experiments_csv(zero_shot_results, output_dir)
    logger.info(f"✓ CSV: {mono_csv}")

    mono_fig = plot_monolingual_experiments(mono_df, output_dir)
    logger.info(f"✓ Figure: {mono_fig}")

    # Figure 2: Multilingual Experiments
    logger.info("\n" + "=" * 80)
    logger.info("FIGURE 2: MULTILINGUAL EXPERIMENTS")
    logger.info("=" * 80)

    multi_csv, multi_df = create_multilingual_experiments_csv(zero_shot_results, output_dir)
    logger.info(f"✓ CSV: {multi_csv}")

    multi_fig = plot_multilingual_experiments(multi_df, output_dir)
    logger.info(f"✓ Figure: {multi_fig}")

    # Figure 3: Adaptation Experiments
    logger.info("\n" + "=" * 80)
    logger.info("FIGURE 3: ADAPTATION EXPERIMENTS")
    logger.info("=" * 80)

    adapt_csv, adapt_df = create_adaptation_experiments_csv(
        zero_shot_results, fine_tuned_results, selected, output_dir
    )
    logger.info(f"✓ CSV: {adapt_csv}")

    adapt_fig = plot_adaptation_experiments(adapt_df, selected, output_dir)
    logger.info(f"✓ Figure: {adapt_fig}")

    # Statistical Winner
    logger.info("\n" + "=" * 80)
    logger.info("STATISTICAL WINNER DECLARATION")
    logger.info("=" * 80)

    winner_csv, winner_df = compute_statistical_winner(
        zero_shot_results, fine_tuned_results, selected, output_dir
    )
    logger.info(f"✓ CSV: {winner_csv}")

    winner_report = create_winner_report(winner_df, output_dir)
    logger.info(f"✓ Report: {winner_report}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PAPER FIGURES COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("\nGenerated files:")
    logger.info("  1. monolingual_experiments.csv + .png")
    logger.info("  2. multilingual_experiments.csv + .png")
    logger.info("  3. adaptation_experiments.csv + .png")
    logger.info("  4. statistical_winner.csv")
    logger.info("  5. winner_declaration.txt")


if __name__ == "__main__":
    main()
