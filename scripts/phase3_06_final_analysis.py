#!/usr/bin/env python3
"""
Phase 3 - Step 6: Final Analysis & Paper Figures
Comprehensive analysis combining all Phase 3 results for ICASSP 2027 paper.
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


def load_all_results(task):
    """Load all results from Steps 3, 4, 5."""
    results = {
        'zero_shot': {},
        'fine_tuned': {},
        'selected_models': None
    }

    # Load selected models
    selected_file = Path("outputs/phase3/model_selection/selected_models.json")
    with open(selected_file, 'r') as f:
        results['selected_models'] = json.load(f)

    # Load zero-shot results (all 6 models)
    zero_shot_dir = Path(f"outputs/phase3/zero_shot/{task}")
    for model_dir in zero_shot_dir.glob("*"):
        if model_dir.is_dir():
            result_file = model_dir / "results.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    results['zero_shot'][model_dir.name] = json.load(f)

    # Load fine-tuning results (selected models only)
    ft_dir = Path(f"outputs/phase3/fine_tuning/{task}")
    for model_name in [results['selected_models']['monolingual'],
                       results['selected_models']['multilingual']]:
        result_file = ft_dir / model_name / "fine_tuning_results.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                results['fine_tuned'][model_name] = json.load(f)

    return results


def plot_zero_shot_comparison(results, task, output_dir):
    """Compare all 6 models on zero-shot performance."""
    models = []
    accuracies = []
    model_types = []

    mono_models = ['wav2vec2_base', 'wav2vec2_base_960h', 'hubert_base', 'ecapa_tdnn']
    multi_models = ['xls_r', 'wavlm']

    for model, data in results['zero_shot'].items():
        models.append(model)
        accuracies.append(data['test_metrics']['accuracy'])
        model_types.append('monolingual' if model in mono_models else 'multilingual')

    # Sort by accuracy
    sorted_idx = np.argsort(accuracies)[::-1]
    models = [models[i] for i in sorted_idx]
    accuracies = [accuracies[i] for i in sorted_idx]
    model_types = [model_types[i] for i in sorted_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#0173B2' if t == 'monolingual' else '#DE8F05' for t in model_types]
    bars = ax.barh(models, accuracies, color=colors)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
               f'{width:.3f}', va='center', ha='left', fontsize=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#0173B2', label='Monolingual'),
        Patch(facecolor='#DE8F05', label='Multilingual')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.set_xlabel('Test Accuracy', fontsize=12)
    ax.set_title(f'Zero-Shot Performance Comparison\n{task.replace("_", " ").title()}',
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f"zero_shot_comparison_{task}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def plot_full_pipeline_comparison(results, task, output_dir):
    """Compare zero-shot vs fine-tuned for selected models."""
    mono_name = results['selected_models']['monolingual']
    multi_name = results['selected_models']['multilingual']

    # Extract data
    mono_zero = results['zero_shot'][mono_name]['test_metrics']['accuracy']
    multi_zero = results['zero_shot'][multi_name]['test_metrics']['accuracy']

    fractions = sorted([float(k) for k in results['fine_tuned'][mono_name].keys()])
    fraction_labels = ['Zero-shot'] + [f'{int(f*100)}%' for f in fractions]

    mono_accs = [mono_zero] + [results['fine_tuned'][mono_name][str(f)]['test_metrics']['accuracy']
                                for f in fractions]
    multi_accs = [multi_zero] + [results['fine_tuned'][multi_name][str(f)]['test_metrics']['accuracy']
                                  for f in fractions]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(fraction_labels))
    width = 0.35

    color_mono = '#0173B2'
    color_multi = '#DE8F05'

    bars1 = ax.bar(x - width/2, mono_accs, width,
                   label=f'Monolingual ({mono_name})', color=color_mono)
    bars2 = ax.bar(x + width/2, multi_accs, width,
                   label=f'Multilingual ({multi_name})', color=color_multi)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Training Data', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'Full Pipeline: Zero-Shot vs Fine-Tuned\n{task.replace("_", " ").title()}',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(fraction_labels)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f"full_pipeline_{task}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def create_master_table(results, task, output_dir):
    """Create master table with all results."""
    rows = []

    mono_name = results['selected_models']['monolingual']
    multi_name = results['selected_models']['multilingual']

    # Zero-shot row
    rows.append({
        'condition': 'Zero-shot',
        'data_fraction': 'N/A',
        'n_train_samples': 0,
        f'{mono_name}_accuracy': results['zero_shot'][mono_name]['test_metrics']['accuracy'],
        f'{multi_name}_accuracy': results['zero_shot'][multi_name]['test_metrics']['accuracy'],
        'gap': results['zero_shot'][mono_name]['test_metrics']['accuracy'] -
               results['zero_shot'][multi_name]['test_metrics']['accuracy']
    })

    # Fine-tuned rows
    fractions = sorted([float(k) for k in results['fine_tuned'][mono_name].keys()])
    for fraction in fractions:
        mono_data = results['fine_tuned'][mono_name][str(fraction)]
        multi_data = results['fine_tuned'][multi_name][str(fraction)]

        rows.append({
            'condition': 'Fine-tuned',
            'data_fraction': f'{int(fraction*100)}%',
            'n_train_samples': mono_data['n_train_samples'],
            f'{mono_name}_accuracy': mono_data['test_metrics']['accuracy'],
            f'{multi_name}_accuracy': multi_data['test_metrics']['accuracy'],
            'gap': mono_data['test_metrics']['accuracy'] - multi_data['test_metrics']['accuracy']
        })

    df = pd.DataFrame(rows)

    csv_file = output_dir / f"master_results_{task}.csv"
    df.to_csv(csv_file, index=False)

    return csv_file, df


def create_final_report(results, task, output_dir, df):
    """Create comprehensive final report."""
    report_file = output_dir / f"final_report_{task}.txt"

    mono_name = results['selected_models']['monolingual']
    multi_name = results['selected_models']['multilingual']

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 3 - FINAL ANALYSIS REPORT\n")
        f.write("ICASSP 2027: Monolingual vs Multilingual Speech Models for Hyrax Acoustics\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Task: {task}\n")
        f.write(f"Selected Monolingual Model: {mono_name}\n")
        f.write(f"Selected Multilingual Model: {multi_name}\n\n")

        f.write("=" * 80 + "\n")
        f.write("ZERO-SHOT EVALUATION (ALL 6 MODELS)\n")
        f.write("=" * 80 + "\n\n")

        zero_shot_sorted = sorted(results['zero_shot'].items(),
                                 key=lambda x: x[1]['test_metrics']['accuracy'],
                                 reverse=True)

        for i, (model, data) in enumerate(zero_shot_sorted, 1):
            acc = data['test_metrics']['accuracy']
            f.write(f"{i}. {model}: {acc:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("FINE-TUNING RESULTS (SELECTED MODELS)\n")
        f.write("=" * 80 + "\n\n")

        for _, row in df.iterrows():
            if row['condition'] == 'Fine-tuned':
                f.write(f"{row['data_fraction']} Training Data ({row['n_train_samples']} samples):\n")
                f.write(f"  {mono_name}:  {row[f'{mono_name}_accuracy']:.4f}\n")
                f.write(f"  {multi_name}: {row[f'{multi_name}_accuracy']:.4f}\n")
                f.write(f"  Gap:          {row['gap']:+.4f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n\n")

        # Best zero-shot
        best_zero = max(results['zero_shot'].items(),
                       key=lambda x: x[1]['test_metrics']['accuracy'])
        f.write(f"Best zero-shot model: {best_zero[0]} ({best_zero[1]['test_metrics']['accuracy']:.4f})\n\n")

        # Fine-tuning improvements
        mono_zero = results['zero_shot'][mono_name]['test_metrics']['accuracy']
        multi_zero = results['zero_shot'][multi_name]['test_metrics']['accuracy']

        mono_100 = results['fine_tuned'][mono_name]['1.0']['test_metrics']['accuracy']
        multi_100 = results['fine_tuned'][multi_name]['1.0']['test_metrics']['accuracy']

        f.write(f"Fine-tuning improvement (100% data):\n")
        f.write(f"  {mono_name}: {mono_zero:.4f} → {mono_100:.4f} ({mono_100-mono_zero:+.4f})\n")
        f.write(f"  {multi_name}: {multi_zero:.4f} → {multi_100:.4f} ({multi_100-multi_zero:+.4f})\n\n")

        # Data efficiency
        mono_10 = results['fine_tuned'][mono_name]['0.1']['test_metrics']['accuracy']
        multi_10 = results['fine_tuned'][multi_name]['0.1']['test_metrics']['accuracy']

        f.write(f"Data efficiency (10% vs 100%):\n")
        f.write(f"  {mono_name}: 10% achieves {mono_10/mono_100*100:.1f}% of full performance\n")
        f.write(f"  {multi_name}: 10% achieves {multi_10/multi_100*100:.1f}% of full performance\n\n")

        # Overall winner
        if mono_100 > multi_100:
            f.write(f"Overall best model: {mono_name} (monolingual)\n")
        else:
            f.write(f"Overall best model: {multi_name} (multilingual)\n")

    return report_file


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3 Final Analysis")
    parser.add_argument("--task", required=True, choices=["species_id", "hyrax_id"])
    args = parser.parse_args()

    # Setup logging
    log_dir = Path("outputs/phase3/logs")
    logger = setup_logger(f"Phase3_FinalAnalysis_{args.task}")

    logger.info("=" * 80)
    logger.info("PHASE 3 - FINAL ANALYSIS & PAPER FIGURES")
    logger.info("=" * 80)

    # Load all results
    logger.info(f"\nLoading results for task: {args.task}")
    results = load_all_results(args.task)

    logger.info(f"  Zero-shot models: {len(results['zero_shot'])}")
    logger.info(f"  Fine-tuned models: {len(results['fine_tuned'])}")
    logger.info(f"  Selected mono: {results['selected_models']['monolingual']}")
    logger.info(f"  Selected multi: {results['selected_models']['multilingual']}")

    # Output directory
    output_dir = Path(f"outputs/phase3/final_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    logger.info("\nGenerating zero-shot comparison...")
    zero_shot_file = plot_zero_shot_comparison(results, args.task, output_dir)
    logger.info(f"✓ {zero_shot_file}")

    logger.info("\nGenerating full pipeline comparison...")
    pipeline_file = plot_full_pipeline_comparison(results, args.task, output_dir)
    logger.info(f"✓ {pipeline_file}")

    # Create master table
    logger.info("\nCreating master results table...")
    csv_file, df = create_master_table(results, args.task, output_dir)
    logger.info(f"✓ {csv_file}")

    # Create final report
    logger.info("\nCreating final report...")
    report_file = create_final_report(results, args.task, output_dir, df)
    logger.info(f"✓ {report_file}")

    logger.info("\n" + "=" * 80)
    logger.info("FINAL ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("\nGenerated files:")
    logger.info(f"  - zero_shot_comparison_{args.task}.png")
    logger.info(f"  - full_pipeline_{args.task}.png")
    logger.info(f"  - master_results_{args.task}.csv")
    logger.info(f"  - final_report_{args.task}.txt")


if __name__ == "__main__":
    main()
