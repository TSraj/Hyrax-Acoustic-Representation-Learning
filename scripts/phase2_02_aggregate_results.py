#!/usr/bin/env python3
"""
Phase 2 - Stage 2: Aggregate Per-Dataset Zero-Shot Results
Collects results from all model×dataset combinations and creates summary tables/figures.
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


def collect_results(results_dir, logger):
    """Collect all results from model×dataset combinations."""
    results_dir = Path(results_dir)

    all_results = []

    # Iterate through datasets
    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        # Iterate through models
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            summary_path = model_dir / "summary.json"

            if not summary_path.exists():
                logger.warning(f"Missing summary for {dataset_name} × {model_name}")
                continue

            # Load summary
            with open(summary_path, 'r') as f:
                summary = json.load(f)

            # Extract key metrics
            result = {
                'dataset': dataset_name,
                'model': model_name,
                'model_type': summary.get('model_type', 'unknown'),
                'accuracy': summary['best_accuracy'],
            }

            if summary['model_type'] == 'transformer':
                result['best_layer'] = summary['best_layer']
                result['num_layers'] = summary['num_layers']
            else:
                result['best_layer'] = 'N/A'
                result['num_layers'] = 'N/A'

            all_results.append(result)

    return pd.DataFrame(all_results)


def create_summary_table(df, output_path, logger):
    """Create summary table with accuracy for each model×dataset."""
    # Pivot table
    pivot = df.pivot(index='dataset', columns='model', values='accuracy')

    # Add mean row
    pivot.loc['MEAN'] = pivot.mean()

    # Save CSV
    pivot.to_csv(output_path)
    logger.info(f"✓ Summary table saved: {output_path}")

    return pivot


def plot_heatmap(pivot_table, output_path, logger):
    """Plot heatmap of model×dataset accuracies."""
    plt.figure(figsize=(14, 8))

    # Create heatmap
    sns.heatmap(
        pivot_table * 100,  # Convert to percentage
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Accuracy (%)'},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title('Zero-Shot Accuracy: 5 Models × 7 Datasets', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Dataset', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Heatmap saved: {output_path}")


def plot_model_comparison(df, output_path, logger):
    """Plot bar chart comparing models across all datasets."""
    # Calculate mean accuracy per model
    model_means = df.groupby('model')['accuracy'].agg(['mean', 'std']).reset_index()
    model_means = model_means.sort_values('mean', ascending=False)

    plt.figure(figsize=(12, 6))

    bars = plt.bar(
        range(len(model_means)),
        model_means['mean'] * 100,
        yerr=model_means['std'] * 100,
        capsize=5,
        color=sns.color_palette('husl', len(model_means)),
        edgecolor='black',
        linewidth=1.5
    )

    plt.xticks(range(len(model_means)), model_means['model'], rotation=45, ha='right')
    plt.ylabel('Mean Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.title('Model Comparison (Mean Accuracy Across All Datasets)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 100)

    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, model_means['mean'])):
        plt.text(
            i,
            mean_val * 100 + 2,
            f'{mean_val*100:.2f}%',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Model comparison plot saved: {output_path}")


def plot_dataset_comparison(df, output_path, logger):
    """Plot bar chart comparing datasets across all models."""
    # Calculate mean accuracy per dataset
    dataset_means = df.groupby('dataset')['accuracy'].agg(['mean', 'std']).reset_index()
    dataset_means = dataset_means.sort_values('mean', ascending=False)

    plt.figure(figsize=(12, 6))

    bars = plt.bar(
        range(len(dataset_means)),
        dataset_means['mean'] * 100,
        yerr=dataset_means['std'] * 100,
        capsize=5,
        color=sns.color_palette('muted', len(dataset_means)),
        edgecolor='black',
        linewidth=1.5
    )

    plt.xticks(range(len(dataset_means)), dataset_means['dataset'], rotation=45, ha='right')
    plt.ylabel('Mean Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Dataset', fontsize=12, fontweight='bold')
    plt.title('Dataset Comparison (Mean Accuracy Across All Models)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 100)

    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, dataset_means['mean'])):
        plt.text(
            i,
            mean_val * 100 + 2,
            f'{mean_val*100:.2f}%',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Dataset comparison plot saved: {output_path}")


def identify_best_model(df, logger):
    """Identify the single best model based on mean accuracy."""
    model_means = df.groupby('model')['accuracy'].mean().sort_values(ascending=False)

    best_model = model_means.index[0]
    best_accuracy = model_means.iloc[0]

    logger.info(f"\n{'='*60}")
    logger.info("BEST MODEL IDENTIFIED")
    logger.info(f"{'='*60}")
    logger.info(f"Model: {best_model}")
    logger.info(f"Mean accuracy: {best_accuracy*100:.2f}%")
    logger.info(f"\nThis model will be used for:")
    logger.info(f"  - Fine-tuning (Stage 5)")
    logger.info(f"  - Hyrax evaluation")
    logger.info(f"  - Sampling rate experiment (Stage 6)")

    return {
        'best_model': best_model,
        'mean_accuracy': best_accuracy,
        'all_model_means': model_means.to_dict()
    }


def create_best_layer_analysis(df, output_path, logger):
    """Analyze which layers work best for transformer models."""
    transformer_results = df[df['model_type'] == 'transformer'].copy()

    if transformer_results.empty:
        logger.warning("No transformer results found for layer analysis")
        return

    # Group by model and get best layer statistics
    layer_summary = []

    for model in transformer_results['model'].unique():
        model_data = transformer_results[transformer_results['model'] == model]

        layer_summary.append({
            'model': model,
            'mean_best_layer': model_data['best_layer'].mean(),
            'std_best_layer': model_data['best_layer'].std(),
            'min_best_layer': model_data['best_layer'].min(),
            'max_best_layer': model_data['best_layer'].max()
        })

    layer_df = pd.DataFrame(layer_summary)
    layer_df.to_csv(output_path, index=False)

    logger.info(f"✓ Best layer analysis saved: {output_path}")
    logger.info("\nBest Layer Summary:")
    for _, row in layer_df.iterrows():
        logger.info(f"  {row['model']}: mean={row['mean_best_layer']:.1f}, range=[{row['min_best_layer']:.0f}, {row['max_best_layer']:.0f}]")


def main():
    """Main function."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    logger = setup_logger("Phase2_AggregateResults", config['experiment']['log_level'])

    logger.info("="*80)
    logger.info("PHASE 2 - STAGE 2: AGGREGATE ZERO-SHOT RESULTS")
    logger.info("="*80)

    # Get results directory
    results_dir = Path(config['paths']['output_dir']) / "phase2" / "zero_shot" / "per_dataset"

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        logger.error("Run phase2_02_zero_shot_per_dataset.py first")
        return

    # Collect results
    logger.info("\nCollecting results from all model×dataset combinations...")
    df = collect_results(results_dir, logger)

    if df.empty:
        logger.error("No results found!")
        return

    logger.info(f"✓ Collected {len(df)} results")
    logger.info(f"  Models: {df['model'].nunique()}")
    logger.info(f"  Datasets: {df['dataset'].nunique()}")

    # Create output directory
    output_dir = Path(config['paths']['output_dir']) / "phase2" / "zero_shot" / "per_dataset_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    df.to_csv(output_dir / "all_results.csv", index=False)
    logger.info(f"\n✓ Raw results saved: {output_dir / 'all_results.csv'}")

    # Create summary table
    logger.info("\nCreating summary table...")
    pivot = create_summary_table(df, output_dir / "accuracy_summary.csv", logger)

    # Create visualizations
    logger.info("\nCreating visualizations...")
    plot_heatmap(pivot[:-1], output_dir / "accuracy_heatmap.png", logger)  # Exclude MEAN row
    plot_model_comparison(df, output_dir / "model_comparison.png", logger)
    plot_dataset_comparison(df, output_dir / "dataset_comparison.png", logger)

    # Best layer analysis
    logger.info("\nAnalyzing best layers for transformer models...")
    create_best_layer_analysis(df, output_dir / "best_layer_analysis.csv", logger)

    # Identify best model
    best_model_info = identify_best_model(df, logger)

    # Save best model info
    with open(output_dir / "best_model.json", 'w') as f:
        json.dump(best_model_info, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info("AGGREGATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"\nAll summaries saved to: {output_dir}")
    logger.info("\n✓ Ready to proceed to Stage 3 (Pooled evaluation)")


if __name__ == "__main__":
    main()
