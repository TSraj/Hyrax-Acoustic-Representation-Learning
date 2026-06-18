#!/usr/bin/env python3
"""
Phase 2 - Stage 3: Aggregate Pooled Evaluation Results
Summarizes pooled results and bird clustering analysis across all 5 models.
"""

import json
import yaml
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger


def collect_pooled_results(pooled_dir, logger):
    """Collect pooled results from all models."""
    pooled_dir = Path(pooled_dir)
    all_results = []

    for model_dir in sorted(pooled_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        summary_path = model_dir / "summary.json"

        if not summary_path.exists():
            logger.warning(f"Missing summary for {model_name}")
            continue

        with open(summary_path, 'r') as f:
            summary = json.load(f)

        result = {
            'model': model_name,
            'test_accuracy': summary['test_accuracy'],
            'per_dataset_accuracy': summary.get('per_dataset_accuracy', {}),
            'bird_clustering_metric': summary.get('bird_clustering_metric', {})
        }

        all_results.append(result)

    return all_results


def create_pooled_accuracy_comparison(results, output_path, logger):
    """Compare pooled accuracy across models."""
    models = [r['model'] for r in results]
    accuracies = [r['test_accuracy'] for r in results]

    # Sort by accuracy
    sorted_data = sorted(zip(models, accuracies), key=lambda x: x[1], reverse=True)
    models, accuracies = zip(*sorted_data)

    plt.figure(figsize=(12, 6))

    bars = plt.bar(
        range(len(models)),
        [acc * 100 for acc in accuracies],
        color=sns.color_palette('husl', len(models)),
        edgecolor='black',
        linewidth=1.5
    )

    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.title('Pooled Zero-Shot Accuracy (All 7 Datasets Combined, ~100+ Classes)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 100)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(
            i,
            acc * 100 + 1,
            f'{acc*100:.2f}%',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Pooled accuracy comparison saved: {output_path}")


def create_per_dataset_breakdown(results, output_path, logger):
    """Create heatmap showing per-dataset accuracy for each model in pooled task."""
    # Extract per-dataset accuracies
    models = []
    dataset_accs = []

    for r in results:
        models.append(r['model'])
        dataset_accs.append(r['per_dataset_accuracy'])

    # Get all datasets
    all_datasets = set()
    for acc_dict in dataset_accs:
        all_datasets.update(acc_dict.keys())

    all_datasets = sorted(all_datasets)

    # Build matrix
    matrix = []
    for acc_dict in dataset_accs:
        row = [acc_dict.get(ds, 0.0) for ds in all_datasets]
        matrix.append(row)

    matrix = np.array(matrix) * 100  # Convert to percentage

    # Create heatmap
    plt.figure(figsize=(12, 8))

    sns.heatmap(
        matrix,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        vmin=0,
        vmax=100,
        xticklabels=all_datasets,
        yticklabels=models,
        cbar_kws={'label': 'Accuracy (%)'},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title('Pooled Task: Per-Dataset Accuracy Breakdown', fontsize=14, fontweight='bold')
    plt.xlabel('Source Dataset', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Per-dataset breakdown heatmap saved: {output_path}")


def create_bird_clustering_comparison(results, output_path, logger):
    """Compare bird clustering metrics across models."""
    models = []
    silhouette_scores = []

    for r in results:
        if r['bird_clustering_metric'] and 'silhouette_by_dataset' in r['bird_clustering_metric']:
            models.append(r['model'])
            silhouette_scores.append(r['bird_clustering_metric']['silhouette_by_dataset'])

    if not models:
        logger.warning("No bird clustering metrics found")
        return

    plt.figure(figsize=(12, 6))

    bars = plt.bar(
        range(len(models)),
        silhouette_scores,
        color=['red' if s > 0.3 else 'orange' if s > 0.2 else 'green' for s in silhouette_scores],
        edgecolor='black',
        linewidth=1.5
    )

    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylabel('Silhouette Score (by Dataset)', fontsize=12, fontweight='bold')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.title('Bird Clustering Analysis: Silhouette Score\n(Lower = Better, means birds cluster by individual, not dataset)', fontsize=14, fontweight='bold')
    plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='High clustering (bad)')
    plt.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Moderate clustering')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, silhouette_scores)):
        plt.text(
            i,
            score + 0.01,
            f'{score:.3f}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Bird clustering comparison saved: {output_path}")


def generate_pooled_report(results, output_path, logger):
    """Generate text report summarizing pooled evaluation."""
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("PHASE 2 - STAGE 3: POOLED ZERO-SHOT EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Task: All 7 datasets combined (~100+ individual classes)")
    report_lines.append("Goal: Test whether models identify animals or dataset artifacts")
    report_lines.append("")

    # Overall accuracy ranking
    report_lines.append("-" * 80)
    report_lines.append("OVERALL POOLED ACCURACY RANKING")
    report_lines.append("-" * 80)

    sorted_results = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)

    for i, r in enumerate(sorted_results, 1):
        report_lines.append(f"{i}. {r['model']:<25} {r['test_accuracy']*100:>6.2f}%")

    report_lines.append("")

    # Bird clustering analysis
    report_lines.append("-" * 80)
    report_lines.append("BIRD CLUSTERING ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append("Silhouette Score (by dataset) - Lower is better")
    report_lines.append("  High (>0.3): Birds cluster by dataset → artifact (BAD)")
    report_lines.append("  Low  (<0.2): Birds mix across datasets → identifying birds (GOOD)")
    report_lines.append("")

    bird_results = [(r['model'], r['bird_clustering_metric'].get('silhouette_by_dataset', None))
                    for r in results if r['bird_clustering_metric']]

    bird_results = [(m, s) for m, s in bird_results if s is not None]
    bird_results.sort(key=lambda x: x[1])  # Sort by score (lower = better)

    for model, score in bird_results:
        status = "✓ GOOD" if score < 0.2 else "⚠ MODERATE" if score < 0.3 else "✗ BAD"
        report_lines.append(f"{model:<25} {score:>6.3f}  {status}")

    report_lines.append("")

    # Per-dataset breakdown
    report_lines.append("-" * 80)
    report_lines.append("PER-DATASET ACCURACY BREAKDOWN (in pooled task)")
    report_lines.append("-" * 80)

    # Get all datasets
    all_datasets = set()
    for r in results:
        all_datasets.update(r['per_dataset_accuracy'].keys())

    all_datasets = sorted(all_datasets)

    # Header
    header = f"{'Model':<25}"
    for ds in all_datasets:
        header += f" {ds[:10]:>10}"
    report_lines.append(header)
    report_lines.append("-" * 80)

    for r in sorted_results:
        line = f"{r['model']:<25}"
        for ds in all_datasets:
            acc = r['per_dataset_accuracy'].get(ds, 0.0)
            line += f" {acc*100:>9.1f}%"
        report_lines.append(line)

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("INTERPRETATION")
    report_lines.append("=" * 80)
    report_lines.append("1. If per-dataset accuracies are very different → model is sensitive to dataset")
    report_lines.append("2. If bird silhouette score is high → birds cluster by dataset (artifact)")
    report_lines.append("3. Best model = high pooled accuracy + low bird clustering score")
    report_lines.append("=" * 80)

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"✓ Pooled evaluation report saved: {output_path}")

    # Also print to console
    logger.info("\n" + '\n'.join(report_lines))


def main():
    """Main function."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    logger = setup_logger("Phase2_AggregatePooled", config['experiment']['log_level'])

    logger.info("="*80)
    logger.info("PHASE 2 - STAGE 3: AGGREGATE POOLED RESULTS")
    logger.info("="*80)

    # Get pooled results directory
    pooled_dir = Path(config['paths']['output_dir']) / "phase2" / "zero_shot" / "pooled"

    if not pooled_dir.exists():
        logger.error(f"Pooled results directory not found: {pooled_dir}")
        logger.error("Run phase2_03_zero_shot_pooled.py first")
        return

    # Collect results
    logger.info("\nCollecting pooled results from all models...")
    results = collect_pooled_results(pooled_dir, logger)

    if not results:
        logger.error("No pooled results found!")
        return

    logger.info(f"✓ Collected results from {len(results)} models")

    # Create output directory
    output_dir = Path(config['paths']['output_dir']) / "phase2" / "zero_shot" / "pooled_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    with open(output_dir / "pooled_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Create visualizations
    logger.info("\nCreating visualizations...")
    create_pooled_accuracy_comparison(results, output_dir / "pooled_accuracy_comparison.png", logger)
    create_per_dataset_breakdown(results, output_dir / "per_dataset_breakdown_heatmap.png", logger)
    create_bird_clustering_comparison(results, output_dir / "bird_clustering_comparison.png", logger)

    # Generate report
    logger.info("\nGenerating pooled evaluation report...")
    generate_pooled_report(results, output_dir / "pooled_evaluation_report.txt", logger)

    logger.info(f"\n{'='*80}")
    logger.info("POOLED AGGREGATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"\nAll summaries saved to: {output_dir}")
    logger.info("\n✓ Ready to proceed to Stage 4 (Model selection)")


if __name__ == "__main__":
    main()
