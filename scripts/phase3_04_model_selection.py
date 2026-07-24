#!/usr/bin/env python3
"""
Phase 3 - Step 4: Model Selection
Compares all 6 models across both tasks and selects best monolingual + multilingual.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger


class ModelSelector:
    """Compare models and select best candidates for fine-tuning."""

    def __init__(self, zero_shot_dir, output_dir, logger):
        """
        Initialize selector.

        Args:
            zero_shot_dir: Directory with zero-shot results
            output_dir: Output directory for comparison results
            logger: Logger instance
        """
        self.zero_shot_dir = Path(zero_shot_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

        # Model categorization
        self.monolingual_models = [
            'wav2vec2_base', 'wav2vec2_base_960h', 'hubert_base',
            'wavlm', 'ecapa_tdnn'
        ]
        self.multilingual_models = ['xls_r']
        self.all_models = self.monolingual_models + self.multilingual_models

        self.tasks = ['species_id', 'hyrax_id']

        # Load all results
        self.results = self._load_all_results()

    def _load_all_results(self):
        """Load results from all model/task combinations."""
        self.logger.info("\nLoading zero-shot results...")

        results = {}

        for task in self.tasks:
            results[task] = {}

            # Map task to folder structure
            task_folder = self.zero_shot_dir / task

            for model in self.all_models:
                result_file = task_folder / model / "results.json"

                if not result_file.exists():
                    self.logger.warning(f"Missing results: {task}/{model}")
                    continue

                with open(result_file, 'r') as f:
                    results[task][model] = json.load(f)

                self.logger.info(f"  ✓ Loaded {task}/{model}")

        return results

    def create_comparison_table(self):
        """Create comparison table with all models and tasks."""
        self.logger.info("\nCreating comparison table...")

        rows = []

        for task in self.tasks:
            for model in self.all_models:
                if model not in self.results[task]:
                    continue

                metrics = self.results[task][model]['test_metrics']

                row = {
                    'task': task,
                    'model': model,
                    'model_type': 'multilingual' if model in self.multilingual_models else 'monolingual',
                    'accuracy': metrics['accuracy'],
                    'balanced_accuracy': metrics['balanced_accuracy'],
                    'f1_macro': metrics['f1_macro'],
                    'f1_weighted': metrics['f1_weighted'],
                    'precision_macro': metrics['precision_macro'],
                    'recall_macro': metrics['recall_macro'],
                    'roc_auc_macro': metrics.get('roc_auc_macro', 0.0)
                }

                rows.append(row)

        df = pd.DataFrame(rows)

        # Save CSV
        csv_file = self.output_dir / "model_comparison.csv"
        df.to_csv(csv_file, index=False)
        self.logger.info(f"✓ Comparison table saved: {csv_file}")

        return df

    def plot_accuracy_comparison(self, df):
        """
        Plot accuracy comparison across models for both tasks.
        IEEE publication ready: 300 DPI PNG, colorblind-safe.
        """
        self.logger.info("\nPlotting accuracy comparison...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Colorblind-safe palette
        colors = {
            'monolingual': '#0173B2',  # Blue
            'multilingual': '#DE8F05'  # Orange
        }

        for idx, task in enumerate(self.tasks):
            ax = axes[idx]

            task_data = df[df['task'] == task].sort_values('accuracy', ascending=False)

            # Color by model type
            bar_colors = [colors[mt] for mt in task_data['model_type']]

            bars = ax.barh(task_data['model'], task_data['accuracy'], color=bar_colors)

            ax.set_xlabel('Test Accuracy', fontsize=12)
            ax.set_ylabel('Model', fontsize=12)
            ax.set_title(f'{task.replace("_", " ").title()}', fontsize=13, fontweight='bold')
            ax.set_xlim(0, 1.0)
            ax.grid(axis='x', alpha=0.3, linestyle='--')

            # Add value labels at end of bars (no overlap, cleaner)
            for i, (bar, acc) in enumerate(zip(bars, task_data['accuracy'])):
                width = bar.get_width()
                # Always place outside bar to avoid overlap
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', va='center', ha='left', fontsize=10)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors['monolingual'], label='Monolingual'),
            Patch(facecolor=colors['multilingual'], label='Multilingual')
        ]
        fig.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)

        plt.tight_layout(rect=[0, 0.05, 1, 1])

        acc_file = self.output_dir / "accuracy_comparison.png"
        plt.savefig(acc_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ Accuracy comparison saved: {acc_file}")

    def save_accuracy_comparison_csv(self, df):
        """
        Save accuracy comparison as CSV for concrete numbers.
        """
        self.logger.info("\nSaving accuracy comparison CSV...")

        # Create pivot table: models as rows, tasks as columns
        accuracy_pivot = df.pivot_table(
            index='model',
            columns='task',
            values='accuracy',
            aggfunc='first'
        )

        # Add model type column
        model_types = df.drop_duplicates('model').set_index('model')['model_type']
        accuracy_pivot.insert(0, 'model_type', accuracy_pivot.index.map(model_types))

        # Reorder columns: model_type, species_id, hyrax_id
        cols = ['model_type'] + [t for t in self.tasks if t in accuracy_pivot.columns]
        accuracy_pivot = accuracy_pivot[cols]

        # Sort by model type then by species_id accuracy
        accuracy_pivot = accuracy_pivot.sort_values(
            by=['model_type', 'species_id'] if 'species_id' in accuracy_pivot.columns else ['model_type'],
            ascending=[True, False]
        )

        # Save
        csv_file = self.output_dir / "accuracy_comparison.csv"
        accuracy_pivot.to_csv(csv_file)

        self.logger.info(f"✓ Accuracy comparison CSV saved: {csv_file}")

    def plot_metrics_heatmap(self, df):
        """
        Plot heatmap of model performance across tasks.
        IEEE publication ready: 300 DPI PNG, colorblind-safe.
        """
        self.logger.info("\nPlotting performance heatmap...")

        # Pivot for heatmap (models × tasks, values = accuracy)
        pivot_acc = df.pivot(index='model', columns='task', values='accuracy')

        # Reorder models (monolingual first, then multilingual)
        model_order = [m for m in self.monolingual_models if m in pivot_acc.index] + \
                     [m for m in self.multilingual_models if m in pivot_acc.index]
        pivot_acc = pivot_acc.loc[model_order]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlGnBu',
                   cbar_kws={'label': 'Test Accuracy'}, ax=ax,
                   vmin=0, vmax=1, linewidths=0.5, linecolor='gray')

        ax.set_xlabel('Task', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        ax.set_title('Model Performance Across Tasks', fontsize=14)

        # Rotate task labels
        ax.set_xticklabels([t.replace('_', ' ').title() for t in pivot_acc.columns])

        plt.tight_layout()

        heatmap_file = self.output_dir / "performance_heatmap.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ Performance heatmap saved: {heatmap_file}")

    def plot_grouped_metrics(self, df):
        """
        Plot grouped bar chart: precision/recall/F1 for each model (averaged across tasks).
        IEEE publication ready: 300 DPI PNG, colorblind-safe.
        """
        self.logger.info("\nPlotting grouped metrics comparison...")

        # Average metrics across tasks
        avg_metrics = df.groupby('model')[['precision_macro', 'recall_macro', 'f1_macro']].mean()

        # Reorder
        model_order = [m for m in self.monolingual_models if m in avg_metrics.index] + \
                     [m for m in self.multilingual_models if m in avg_metrics.index]
        avg_metrics = avg_metrics.loc[model_order]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(avg_metrics))
        width = 0.25

        # Colorblind-safe palette
        colors = ['#0173B2', '#DE8F05', '#029E73']  # Blue, Orange, Green

        bars1 = ax.bar(x - width, avg_metrics['precision_macro'], width,
                      label='Precision', color=colors[0])
        bars2 = ax.bar(x, avg_metrics['recall_macro'], width,
                      label='Recall', color=colors[1])
        bars3 = ax.bar(x + width, avg_metrics['f1_macro'], width,
                      label='F1-Score', color=colors[2])

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Average Performance Metrics Across Tasks', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(avg_metrics.index, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        grouped_file = self.output_dir / "grouped_metrics_comparison.png"
        plt.savefig(grouped_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ Grouped metrics comparison saved: {grouped_file}")

    def select_best_models(self, df):
        """
        Select best monolingual and multilingual models.

        Selection criteria:
        1. Average accuracy across both tasks
        2. Balanced accuracy (for fairness across classes)
        3. F1-macro (for class-imbalanced tasks)

        Returns:
            dict with 'monolingual' and 'multilingual' model names
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("MODEL SELECTION")
        self.logger.info("=" * 80)

        # Multilingual: only xls_r
        multilingual_best = 'xls_r'
        self.logger.info(f"\nMultilingual model: {multilingual_best} (only candidate)")

        # Monolingual: rank by composite score
        mono_df = df[df['model_type'] == 'monolingual'].copy()

        # Composite score: average of accuracy, balanced_accuracy, f1_macro
        mono_df['composite_score'] = (
            mono_df['accuracy'] +
            mono_df['balanced_accuracy'] +
            mono_df['f1_macro']
        ) / 3.0

        # Average composite score across tasks
        mono_avg = mono_df.groupby('model')['composite_score'].mean().sort_values(ascending=False)

        self.logger.info(f"\nMonolingual model rankings (by composite score):")
        for rank, (model, score) in enumerate(mono_avg.items(), 1):
            self.logger.info(f"  {rank}. {model}: {score:.4f}")

        monolingual_best = mono_avg.index[0]

        self.logger.info(f"\n✓ Best monolingual model: {monolingual_best}")

        # Get detailed metrics for selected models
        selected_models = {
            'monolingual': monolingual_best,
            'multilingual': multilingual_best
        }

        self.logger.info("\n" + "=" * 80)
        self.logger.info("SELECTED MODELS PERFORMANCE")
        self.logger.info("=" * 80)

        for model_type, model_name in selected_models.items():
            self.logger.info(f"\n{model_type.upper()}: {model_name}")

            for task in self.tasks:
                task_data = df[(df['model'] == model_name) & (df['task'] == task)]

                if task_data.empty:
                    continue

                row = task_data.iloc[0]

                self.logger.info(f"\n  {task.replace('_', ' ').title()}:")
                self.logger.info(f"    Accuracy:          {row['accuracy']:.4f}")
                self.logger.info(f"    Balanced Accuracy: {row['balanced_accuracy']:.4f}")
                self.logger.info(f"    F1-Macro:          {row['f1_macro']:.4f}")
                self.logger.info(f"    Precision-Macro:   {row['precision_macro']:.4f}")
                self.logger.info(f"    Recall-Macro:      {row['recall_macro']:.4f}")

        return selected_models

    def save_selection_report(self, selected_models, df):
        """Save selection report with rationale."""
        report_file = self.output_dir / "model_selection_report.txt"

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PHASE 3 - MODEL SELECTION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("SELECTED MODELS FOR FINE-TUNING:\n\n")

            for model_type, model_name in selected_models.items():
                f.write(f"  {model_type.upper()}: {model_name}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("SELECTION CRITERIA\n")
            f.write("=" * 80 + "\n\n")

            f.write("Monolingual models ranked by composite score:\n")
            f.write("  Composite = (Accuracy + Balanced Accuracy + F1-Macro) / 3\n")
            f.write("  Averaged across both tasks (Species ID + Hyrax ID)\n\n")

            f.write("Multilingual model:\n")
            f.write("  xls_r (only multilingual candidate)\n\n")

            f.write("=" * 80 + "\n")
            f.write("SELECTED MODELS PERFORMANCE\n")
            f.write("=" * 80 + "\n\n")

            for model_type, model_name in selected_models.items():
                f.write(f"{model_type.upper()}: {model_name}\n")
                f.write("-" * 40 + "\n")

                for task in self.tasks:
                    task_data = df[(df['model'] == model_name) & (df['task'] == task)]

                    if task_data.empty:
                        continue

                    row = task_data.iloc[0]

                    f.write(f"\n{task.replace('_', ' ').title()}:\n")
                    f.write(f"  Accuracy:          {row['accuracy']:.4f}\n")
                    f.write(f"  Balanced Accuracy: {row['balanced_accuracy']:.4f}\n")
                    f.write(f"  F1-Macro:          {row['f1_macro']:.4f}\n")
                    f.write(f"  F1-Weighted:       {row['f1_weighted']:.4f}\n")
                    f.write(f"  Precision-Macro:   {row['precision_macro']:.4f}\n")
                    f.write(f"  Recall-Macro:      {row['recall_macro']:.4f}\n")

                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("NEXT STEP\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Fine-tune {selected_models['monolingual']} and {selected_models['multilingual']}\n")
            f.write("with 10%, 25%, 50%, and 100% of training data\n")
            f.write("to evaluate data efficiency and animal adaptation.\n")

        self.logger.info(f"\n✓ Selection report saved: {report_file}")

    def run(self):
        """Run full model selection pipeline."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PHASE 3 - STEP 4: MODEL SELECTION")
        self.logger.info("=" * 80)

        # Create comparison table
        df = self.create_comparison_table()

        # Generate visualizations
        self.plot_accuracy_comparison(df)
        self.save_accuracy_comparison_csv(df)
        self.plot_metrics_heatmap(df)
        self.plot_grouped_metrics(df)

        # Select best models
        selected_models = self.select_best_models(df)

        # Save selection report
        self.save_selection_report(selected_models, df)

        # Save selection as JSON for Step 5
        selection_file = self.output_dir / "selected_models.json"
        with open(selection_file, 'w') as f:
            json.dump(selected_models, f, indent=2)

        self.logger.info(f"\n✓ Selected models saved: {selection_file}")

        self.logger.info("\n" + "=" * 80)
        self.logger.info("MODEL SELECTION COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"\nOutput directory: {self.output_dir}")
        self.logger.info("\nGenerated files:")
        self.logger.info("  - model_comparison.csv (all metrics)")
        self.logger.info("  - accuracy_comparison.csv (accuracy table)")
        self.logger.info("  - accuracy_comparison.png")
        self.logger.info("  - performance_heatmap.png")
        self.logger.info("  - grouped_metrics_comparison.png")
        self.logger.info("  - model_selection_report.txt")
        self.logger.info("  - selected_models.json")

        return selected_models


def main():
    """Main entry point."""

    # Setup logging
    log_dir = Path("outputs/phase3/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("Phase3_ModelSelection",
                         log_file=str(log_dir / "model_selection.log"))

    # Paths
    zero_shot_dir = Path("outputs/phase3/zero_shot")
    output_dir = Path("outputs/phase3/model_selection")

    # Run selection
    selector = ModelSelector(zero_shot_dir, output_dir, logger)
    selected_models = selector.run()

    logger.info(f"\n✓ Selected for fine-tuning:")
    logger.info(f"  Monolingual: {selected_models['monolingual']}")
    logger.info(f"  Multilingual: {selected_models['multilingual']}")


if __name__ == "__main__":
    main()
