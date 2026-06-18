#!/usr/bin/env python3
"""
Phase 2 - Stage 4: Model Comparison and Selection
Synthesizes results from Stage 2 (per-dataset) and Stage 3 (pooled) to identify the best model.
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


class ModelSelector:
    """Selects the best model based on comprehensive evaluation criteria."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config['paths']['output_dir']) / "phase2" / "model_selection"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load per-dataset results
        self.per_dataset_summary = self._load_per_dataset_summary()

        # Load pooled results
        self.pooled_results = self._load_pooled_results()

    def _load_per_dataset_summary(self):
        """Load per-dataset zero-shot results from Stage 2."""
        summary_path = Path(self.config['paths']['output_dir']) / "phase2" / "zero_shot" / "per_dataset_summary" / "all_results.csv"

        if not summary_path.exists():
            self.logger.error(f"Per-dataset summary not found: {summary_path}")
            self.logger.error("Run phase2_02_aggregate_results.py first")
            return None

        df = pd.read_csv(summary_path)
        self.logger.info(f"✓ Loaded per-dataset results: {len(df)} entries")
        return df

    def _load_pooled_results(self):
        """Load pooled zero-shot results from Stage 3."""
        pooled_path = Path(self.config['paths']['output_dir']) / "phase2" / "zero_shot" / "pooled_summary" / "pooled_results.json"

        if not pooled_path.exists():
            self.logger.error(f"Pooled results not found: {pooled_path}")
            self.logger.error("Run phase2_03_aggregate_pooled_results.py first")
            return None

        with open(pooled_path, 'r') as f:
            results = json.load(f)

        self.logger.info(f"✓ Loaded pooled results: {len(results)} models")
        return results

    def compute_model_scores(self):
        """
        Compute composite scores for each model based on multiple criteria.

        Criteria:
        1. Per-dataset mean accuracy (40%)
        2. Pooled accuracy (30%)
        3. Bird clustering quality (20%) - lower silhouette = better
        4. Consistency across datasets (10%) - lower std = better
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("COMPUTING MODEL SCORES")
        self.logger.info("="*80)

        models = self.per_dataset_summary['model'].unique()
        scores = []

        for model in models:
            # 1. Per-dataset mean accuracy
            model_per_dataset = self.per_dataset_summary[self.per_dataset_summary['model'] == model]
            per_dataset_mean = model_per_dataset['accuracy'].mean()
            per_dataset_std = model_per_dataset['accuracy'].std()

            # 2. Pooled accuracy
            pooled_result = next((r for r in self.pooled_results if r['model'] == model), None)
            pooled_acc = pooled_result['test_accuracy'] if pooled_result else 0.0

            # 3. Bird clustering quality (lower silhouette = better)
            bird_silhouette = None
            if pooled_result and pooled_result.get('bird_clustering_metric'):
                bird_silhouette = pooled_result['bird_clustering_metric'].get('silhouette_by_dataset')

            # Normalize bird silhouette: high silhouette is bad, so invert
            # Scale: 0.0 (worst clustering) -> 1.0, 0.5 (best no clustering) -> 0.0
            bird_score = 0.0
            if bird_silhouette is not None:
                # Lower silhouette = better, so invert: 1 - normalized_silhouette
                # Assume silhouette range is [-1, 1], typical range is [0, 0.5]
                bird_score = max(0, 1 - (bird_silhouette / 0.5))  # 0.0 silhouette -> 1.0 score
                bird_score = min(1.0, bird_score)  # Cap at 1.0

            # 4. Consistency (lower std = better)
            # Normalize: std of 0.0 (perfect consistency) = 1.0, std of 0.2 (poor) = 0.0
            consistency_score = max(0, 1 - (per_dataset_std / 0.2))
            consistency_score = min(1.0, consistency_score)

            # Weighted composite score
            weights = {
                'per_dataset': 0.40,
                'pooled': 0.30,
                'bird_clustering': 0.20,
                'consistency': 0.10
            }

            composite_score = (
                per_dataset_mean * weights['per_dataset'] +
                pooled_acc * weights['pooled'] +
                bird_score * weights['bird_clustering'] +
                consistency_score * weights['consistency']
            )

            scores.append({
                'model': model,
                'per_dataset_mean': per_dataset_mean,
                'per_dataset_std': per_dataset_std,
                'pooled_accuracy': pooled_acc,
                'bird_silhouette': bird_silhouette if bird_silhouette is not None else np.nan,
                'bird_score': bird_score,
                'consistency_score': consistency_score,
                'composite_score': composite_score
            })

        scores_df = pd.DataFrame(scores)
        scores_df = scores_df.sort_values('composite_score', ascending=False)

        self.logger.info("\nModel Scores:")
        self.logger.info("-" * 80)
        for _, row in scores_df.iterrows():
            self.logger.info(f"{row['model']:<25} Composite: {row['composite_score']:.4f}")
            self.logger.info(f"  Per-dataset: {row['per_dataset_mean']:.4f} ± {row['per_dataset_std']:.4f}")
            self.logger.info(f"  Pooled:      {row['pooled_accuracy']:.4f}")
            self.logger.info(f"  Bird:        {row['bird_silhouette']:.4f} (lower is better)")
            self.logger.info("")

        # Save scores
        scores_df.to_csv(self.output_dir / "model_scores.csv", index=False)
        self.logger.info(f"✓ Model scores saved: {self.output_dir / 'model_scores.csv'}")

        return scores_df

    def select_best_model(self, scores_df):
        """Select the best model based on composite score."""
        best_model = scores_df.iloc[0]

        self.logger.info("\n" + "="*80)
        self.logger.info("BEST MODEL SELECTED")
        self.logger.info("="*80)
        self.logger.info(f"\nModel: {best_model['model']}")
        self.logger.info(f"Composite Score: {best_model['composite_score']:.4f}")
        self.logger.info(f"\nBreakdown:")
        self.logger.info(f"  Per-dataset mean accuracy: {best_model['per_dataset_mean']*100:.2f}%")
        self.logger.info(f"  Pooled accuracy:           {best_model['pooled_accuracy']*100:.2f}%")
        self.logger.info(f"  Bird silhouette:           {best_model['bird_silhouette']:.4f} (lower = better)")
        self.logger.info(f"  Consistency (std):         {best_model['per_dataset_std']:.4f} (lower = better)")

        selection = {
            'best_model': best_model['model'],
            'composite_score': best_model['composite_score'],
            'per_dataset_mean': best_model['per_dataset_mean'],
            'per_dataset_std': best_model['per_dataset_std'],
            'pooled_accuracy': best_model['pooled_accuracy'],
            'bird_silhouette': best_model['bird_silhouette'] if not np.isnan(best_model['bird_silhouette']) else None,
            'ranking': scores_df['model'].tolist(),
            'selection_criteria': {
                'per_dataset_weight': 0.40,
                'pooled_weight': 0.30,
                'bird_clustering_weight': 0.20,
                'consistency_weight': 0.10
            }
        }

        # Save selection
        with open(self.output_dir / "best_model_selection.json", 'w') as f:
            json.dump(selection, f, indent=2)

        self.logger.info(f"\n✓ Best model selection saved: {self.output_dir / 'best_model_selection.json'}")

        return selection

    def create_comparison_visualizations(self, scores_df):
        """Create comprehensive comparison visualizations."""
        self.logger.info("\nCreating comparison visualizations...")

        # 1. Composite score comparison
        self._plot_composite_scores(scores_df)

        # 2. Multi-metric radar chart
        self._plot_radar_chart(scores_df)

        # 3. Per-dataset vs Pooled scatter
        self._plot_per_dataset_vs_pooled(scores_df)

        # 4. Summary table
        self._create_summary_table(scores_df)

    def _plot_composite_scores(self, scores_df):
        """Plot composite score comparison."""
        fig, ax = plt.subplots(figsize=(12, 6))

        models = scores_df['model'].values
        scores = scores_df['composite_score'].values

        colors = ['green' if i == 0 else 'skyblue' for i in range(len(models))]

        bars = ax.bar(range(len(models)), scores, color=colors, edgecolor='black', linewidth=1.5)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Composite Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison: Composite Score\n(40% Per-Dataset + 30% Pooled + 20% Bird Clustering + 10% Consistency)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(
                i,
                score + 0.01,
                f'{score:.3f}',
                ha='center',
                va='bottom',
                fontweight='bold'
            )

        # Highlight winner
        ax.text(0, scores[0] + 0.05, '★ WINNER ★', ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='green')

        plt.tight_layout()
        plt.savefig(self.output_dir / "composite_score_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"  ✓ Composite score plot saved")

    def _plot_radar_chart(self, scores_df):
        """Create radar chart comparing top 3 models across all metrics."""
        from math import pi

        # Select top 3 models
        top3 = scores_df.head(3)

        categories = ['Per-Dataset\nAccuracy', 'Pooled\nAccuracy', 'Bird\nClustering', 'Consistency']
        num_vars = len(categories)

        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        colors = ['green', 'blue', 'orange']

        for idx, (_, row) in enumerate(top3.iterrows()):
            values = [
                row['per_dataset_mean'],
                row['pooled_accuracy'],
                row['bird_score'],  # Already normalized
                row['consistency_score']  # Already normalized
            ]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.set_title('Top 3 Models: Multi-Metric Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / "radar_chart_top3.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"  ✓ Radar chart saved")

    def _plot_per_dataset_vs_pooled(self, scores_df):
        """Scatter plot: per-dataset accuracy vs pooled accuracy."""
        fig, ax = plt.subplots(figsize=(10, 8))

        x = scores_df['per_dataset_mean'].values
        y = scores_df['pooled_accuracy'].values
        models = scores_df['model'].values

        colors = ['green' if i == 0 else 'skyblue' for i in range(len(models))]

        ax.scatter(x * 100, y * 100, s=200, c=colors, edgecolors='black', linewidths=2, alpha=0.7)

        # Add model labels
        for i, model in enumerate(models):
            ax.annotate(
                model,
                (x[i] * 100, y[i] * 100),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold' if i == 0 else 'normal'
            )

        # Add diagonal line (perfect correlation)
        lims = [0, 100]
        ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0, label='Perfect correlation')

        ax.set_xlabel('Per-Dataset Mean Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pooled Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Per-Dataset vs Pooled Accuracy', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(self.output_dir / "per_dataset_vs_pooled_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"  ✓ Scatter plot saved")

    def _create_summary_table(self, scores_df):
        """Create a summary table comparing all models."""
        fig, ax = plt.subplots(figsize=(14, len(scores_df) * 0.8 + 2))
        ax.axis('tight')
        ax.axis('off')

        # Prepare data
        table_data = []
        headers = ['Rank', 'Model', 'Composite\nScore', 'Per-Dataset\nMean', 'Pooled\nAccuracy', 'Bird\nSilhouette', 'Consistency\n(Std)']

        for idx, (_, row) in enumerate(scores_df.iterrows(), 1):
            table_data.append([
                f"{idx}",
                row['model'],
                f"{row['composite_score']:.3f}",
                f"{row['per_dataset_mean']*100:.2f}%",
                f"{row['pooled_accuracy']*100:.2f}%",
                f"{row['bird_silhouette']:.3f}" if not np.isnan(row['bird_silhouette']) else "N/A",
                f"{row['per_dataset_std']:.3f}"
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            colWidths=[0.08, 0.25, 0.12, 0.12, 0.12, 0.12, 0.12]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Highlight winner row
        for i in range(len(headers)):
            table[(1, i)].set_facecolor('#C8E6C9')
            table[(1, i)].set_text_props(weight='bold')

        plt.title('Model Selection Summary Table', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / "summary_table.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"  ✓ Summary table saved")

    def generate_final_report(self, selection, scores_df):
        """Generate comprehensive final report."""
        report_lines = []

        report_lines.append("=" * 80)
        report_lines.append("PHASE 2 - STAGE 4: MODEL SELECTION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Selection result
        report_lines.append("-" * 80)
        report_lines.append("SELECTED MODEL")
        report_lines.append("-" * 80)
        report_lines.append(f"★ {selection['best_model'].upper()} ★")
        report_lines.append("")
        report_lines.append(f"Composite Score: {selection['composite_score']:.4f}")
        report_lines.append(f"Per-Dataset Mean Accuracy: {selection['per_dataset_mean']*100:.2f}%")
        report_lines.append(f"Pooled Accuracy: {selection['pooled_accuracy']*100:.2f}%")
        if selection['bird_silhouette'] is not None:
            report_lines.append(f"Bird Silhouette Score: {selection['bird_silhouette']:.4f} (lower = better)")
        report_lines.append(f"Consistency (Std Dev): {selection['per_dataset_std']:.4f} (lower = better)")
        report_lines.append("")

        # Selection criteria
        report_lines.append("-" * 80)
        report_lines.append("SELECTION CRITERIA")
        report_lines.append("-" * 80)
        report_lines.append("Composite score is weighted combination of:")
        report_lines.append(f"  • Per-Dataset Accuracy (40%): Generalization across individual datasets")
        report_lines.append(f"  • Pooled Accuracy (30%): Multi-dataset task performance")
        report_lines.append(f"  • Bird Clustering (20%): Quality of embeddings (lower silhouette = better)")
        report_lines.append(f"  • Consistency (10%): Stable performance across datasets (lower std = better)")
        report_lines.append("")

        # Full ranking
        report_lines.append("-" * 80)
        report_lines.append("COMPLETE RANKING")
        report_lines.append("-" * 80)
        for idx, (_, row) in enumerate(scores_df.iterrows(), 1):
            report_lines.append(f"{idx}. {row['model']:<25} Score: {row['composite_score']:.4f}")
        report_lines.append("")

        # Next steps
        report_lines.append("-" * 80)
        report_lines.append("NEXT STEPS")
        report_lines.append("-" * 80)
        report_lines.append(f"The selected model ({selection['best_model']}) will be used for:")
        report_lines.append(f"  1. Fine-tuning (Stage 5)")
        report_lines.append(f"  2. Hyrax identification (zero-shot + fine-tuned)")
        report_lines.append(f"  3. Sampling rate experiment (Stage 6)")
        report_lines.append("")

        # Interpretation notes
        report_lines.append("-" * 80)
        report_lines.append("INTERPRETATION")
        report_lines.append("-" * 80)
        report_lines.append("Per-Dataset vs Pooled:")
        per_pool_diff = abs(selection['per_dataset_mean'] - selection['pooled_accuracy'])
        if per_pool_diff < 0.05:
            report_lines.append("  ✓ Very consistent (diff < 5%) - model generalizes well")
        elif per_pool_diff < 0.10:
            report_lines.append("  ⚠ Moderate difference (5-10%) - acceptable generalization")
        else:
            report_lines.append("  ✗ Large difference (>10%) - may struggle with multi-dataset tasks")
        report_lines.append("")

        if selection['bird_silhouette'] is not None:
            report_lines.append("Bird Clustering:")
            if selection['bird_silhouette'] < 0.2:
                report_lines.append("  ✓ Low silhouette (<0.2) - embeddings mix well across datasets")
            elif selection['bird_silhouette'] < 0.3:
                report_lines.append("  ⚠ Moderate silhouette (0.2-0.3) - some dataset clustering")
            else:
                report_lines.append("  ✗ High silhouette (>0.3) - strong dataset clustering (artifact)")
            report_lines.append("")

        report_lines.append("=" * 80)

        # Write report
        report_path = self.output_dir / "model_selection_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        self.logger.info(f"\n✓ Final report saved: {report_path}")

        # Print to console
        self.logger.info("\n" + '\n'.join(report_lines))

    def run(self):
        """Run complete model selection process."""
        if self.per_dataset_summary is None or self.pooled_results is None:
            self.logger.error("Cannot proceed - missing input data")
            return None

        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 2 - STAGE 4: MODEL SELECTION")
        self.logger.info("="*80)

        # Compute scores
        scores_df = self.compute_model_scores()

        # Select best model
        selection = self.select_best_model(scores_df)

        # Create visualizations
        self.create_comparison_visualizations(scores_df)

        # Generate report
        self.generate_final_report(selection, scores_df)

        self.logger.info(f"\n{'='*80}")
        self.logger.info("MODEL SELECTION COMPLETE")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"\nBest model: {selection['best_model']}")
        self.logger.info(f"All results saved to: {self.output_dir}")

        return selection


def main():
    """Main function."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    logger = setup_logger("Phase2_ModelSelection", config['experiment']['log_level'])

    # Run model selection
    selector = ModelSelector(config, logger)
    selection = selector.run()

    if selection:
        logger.info("\n✓ Ready to proceed to Stage 5 (Fine-tuning)")
    else:
        logger.error("\n✗ Model selection failed")


if __name__ == "__main__":
    main()
