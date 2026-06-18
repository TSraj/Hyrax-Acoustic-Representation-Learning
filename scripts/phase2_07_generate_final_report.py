#!/usr/bin/env python3
"""
Phase 2 - Final Report Generator
Synthesizes all results from Stages 1-6 into a comprehensive final report.
"""

import json
import yaml
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger


class FinalReportGenerator:
    """Generates comprehensive final report for Phase 2."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config['paths']['output_dir']) / "phase2" / "final_report"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase2_dir = Path(config['paths']['output_dir']) / "phase2"

    def load_all_results(self):
        """Load results from all stages."""
        results = {}

        # Stage 2: Per-dataset results
        stage2_summary = self.phase2_dir / "zero_shot" / "per_dataset_summary" / "all_results.csv"
        if stage2_summary.exists():
            results['stage2'] = pd.read_csv(stage2_summary)
            self.logger.info("✓ Loaded Stage 2 results")
        else:
            self.logger.warning("⚠ Stage 2 results not found")
            results['stage2'] = None

        # Stage 3: Pooled results
        stage3_summary = self.phase2_dir / "zero_shot" / "pooled_summary" / "pooled_results.json"
        if stage3_summary.exists():
            with open(stage3_summary, 'r') as f:
                results['stage3'] = json.load(f)
            self.logger.info("✓ Loaded Stage 3 results")
        else:
            self.logger.warning("⚠ Stage 3 results not found")
            results['stage3'] = None

        # Stage 4: Model selection
        stage4_summary = self.phase2_dir / "model_selection" / "best_model_selection.json"
        if stage4_summary.exists():
            with open(stage4_summary, 'r') as f:
                results['stage4'] = json.load(f)
            self.logger.info("✓ Loaded Stage 4 results")
        else:
            self.logger.warning("⚠ Stage 4 results not found")
            results['stage4'] = None

        # Stage 5: Fine-tuning
        if results['stage4']:
            best_model = results['stage4']['best_model']
            stage5_summary = self.phase2_dir / "fine_tuning" / best_model / "results" / "fine_tuning_summary.json"
            if stage5_summary.exists():
                with open(stage5_summary, 'r') as f:
                    results['stage5'] = json.load(f)
                self.logger.info("✓ Loaded Stage 5 results")
            else:
                self.logger.warning("⚠ Stage 5 results not found")
                results['stage5'] = None
        else:
            results['stage5'] = None

        # Stage 6: Sampling rate experiment
        # Check for picidae or wetlands_bird
        for dataset in ['picidae', 'wetlands_bird']:
            stage6_summary = self.phase2_dir / "sampling_rate_experiment" / dataset / "comparison" / "results.json"
            if stage6_summary.exists():
                with open(stage6_summary, 'r') as f:
                    results['stage6'] = json.load(f)
                    results['stage6']['dataset'] = dataset
                self.logger.info(f"✓ Loaded Stage 6 results ({dataset})")
                break
        else:
            self.logger.warning("⚠ Stage 6 results not found")
            results['stage6'] = None

        return results

    def generate_executive_summary(self, results):
        """Generate executive summary section."""
        summary = []
        summary.append("=" * 80)
        summary.append("PHASE 2: EXECUTIVE SUMMARY")
        summary.append("=" * 80)
        summary.append("")
        summary.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")

        # Best model
        if results['stage4']:
            best_model = results['stage4']['best_model']
            summary.append(f"★ SELECTED MODEL: {best_model.upper()} ★")
            summary.append("")

        # Key metrics
        summary.append("-" * 80)
        summary.append("KEY METRICS")
        summary.append("-" * 80)

        if results['stage2']:
            per_dataset_mean = results['stage2']['accuracy'].mean()
            summary.append(f"Per-Dataset Zero-Shot (mean):  {per_dataset_mean*100:.2f}%")

        if results['stage3'] and results['stage4']:
            best_model = results['stage4']['best_model']
            pooled_result = next((r for r in results['stage3'] if r['model'] == best_model), None)
            if pooled_result:
                summary.append(f"Pooled Zero-Shot:              {pooled_result['test_accuracy']*100:.2f}%")

        if results['stage5']:
            summary.append(f"Fine-Tuned Pooled:             {results['stage5']['test_accuracy']*100:.2f}%")

            # Calculate improvement
            if results['stage3'] and results['stage4']:
                zero_shot_acc = pooled_result['test_accuracy']
                fine_tuned_acc = results['stage5']['test_accuracy']
                improvement = (fine_tuned_acc - zero_shot_acc) * 100
                summary.append(f"Fine-Tuning Improvement:       +{improvement:.2f}%")

        if results['stage6']:
            orig_acc = results['stage6']['original']['test_accuracy'] * 100
            khz16_acc = results['stage6']['16khz']['test_accuracy'] * 100
            diff = results['stage6']['comparison']['accuracy_difference'] * 100
            summary.append("")
            summary.append(f"Sampling Rate Experiment ({results['stage6']['dataset']}):")
            summary.append(f"  Original: {orig_acc:.2f}% | 16kHz: {khz16_acc:.2f}% | Δ = {diff:.2f}%")

        summary.append("")
        return summary

    def generate_stage_summaries(self, results):
        """Generate detailed summaries for each stage."""
        summaries = []

        # Stage 1
        summaries.append("-" * 80)
        summaries.append("STAGE 1: DATA MANIFESTS")
        summaries.append("-" * 80)
        manifest_dir = self.phase2_dir / "manifests"
        if manifest_dir.exists():
            manifests = list(manifest_dir.glob("*_manifest.json"))
            summaries.append(f"Created {len(manifests)} manifests (7 datasets + 1 pooled)")
            summaries.append("Split ratio: 80% train / 10% validation / 10% test")
            summaries.append("Stratification: By individual (all individuals in all splits)")
        else:
            summaries.append("❌ Not completed")
        summaries.append("")

        # Stage 2
        summaries.append("-" * 80)
        summaries.append("STAGE 2: PER-DATASET ZERO-SHOT EVALUATION")
        summaries.append("-" * 80)
        if results['stage2']:
            num_models = results['stage2']['model'].nunique()
            num_datasets = results['stage2']['dataset'].nunique()
            summaries.append(f"Models evaluated: {num_models}")
            summaries.append(f"Datasets: {num_datasets}")
            summaries.append(f"Total evaluations: {len(results['stage2'])}")
            summaries.append("")
            summaries.append("Top 3 models (by mean accuracy):")
            model_means = results['stage2'].groupby('model')['accuracy'].mean().sort_values(ascending=False)
            for i, (model, acc) in enumerate(model_means.head(3).items(), 1):
                summaries.append(f"  {i}. {model}: {acc*100:.2f}%")
        else:
            summaries.append("❌ Not completed")
        summaries.append("")

        # Stage 3
        summaries.append("-" * 80)
        summaries.append("STAGE 3: POOLED ZERO-SHOT EVALUATION")
        summaries.append("-" * 80)
        if results['stage3']:
            summaries.append(f"Models evaluated: {len(results['stage3'])}")
            summaries.append("Task: All 7 datasets combined (~100+ classes)")
            summaries.append("")
            summaries.append("Results (by pooled accuracy):")
            pooled_sorted = sorted(results['stage3'], key=lambda x: x['test_accuracy'], reverse=True)
            for i, r in enumerate(pooled_sorted, 1):
                acc = r['test_accuracy'] * 100
                bird_sil = r.get('bird_clustering_metric', {}).get('silhouette_by_dataset', 'N/A')
                if bird_sil != 'N/A':
                    summaries.append(f"  {i}. {r['model']}: {acc:.2f}% (bird silhouette: {bird_sil:.3f})")
                else:
                    summaries.append(f"  {i}. {r['model']}: {acc:.2f}%")
        else:
            summaries.append("❌ Not completed")
        summaries.append("")

        # Stage 4
        summaries.append("-" * 80)
        summaries.append("STAGE 4: MODEL SELECTION")
        summaries.append("-" * 80)
        if results['stage4']:
            summaries.append(f"Selected Model: {results['stage4']['best_model']}")
            summaries.append(f"Composite Score: {results['stage4']['composite_score']:.4f}")
            summaries.append("")
            summaries.append("Selection criteria (weighted):")
            summaries.append("  • Per-dataset mean accuracy: 40%")
            summaries.append("  • Pooled accuracy: 30%")
            summaries.append("  • Bird clustering quality: 20%")
            summaries.append("  • Consistency: 10%")
            summaries.append("")
            summaries.append("Full ranking:")
            for i, model in enumerate(results['stage4']['ranking'], 1):
                summaries.append(f"  {i}. {model}")
        else:
            summaries.append("❌ Not completed")
        summaries.append("")

        # Stage 5
        summaries.append("-" * 80)
        summaries.append("STAGE 5: FINE-TUNING")
        summaries.append("-" * 80)
        if results['stage5']:
            summaries.append(f"Model: {results['stage5']['model']}")
            summaries.append(f"Fine-tuned layers: First {results['stage5']['fine_tuned_layers']} layers")
            summaries.append(f"Num classes: {results['stage5']['num_classes']}")
            summaries.append(f"Training epochs: {results['stage5']['training_epochs']}")
            summaries.append("")
            summaries.append(f"Best validation accuracy: {results['stage5']['best_val_accuracy']*100:.2f}%")
            summaries.append(f"Test accuracy: {results['stage5']['test_accuracy']*100:.2f}%")

            # Compare with zero-shot
            if results['stage3'] and results['stage4']:
                best_model = results['stage4']['best_model']
                pooled_result = next((r for r in results['stage3'] if r['model'] == best_model), None)
                if pooled_result:
                    zero_shot = pooled_result['test_accuracy'] * 100
                    fine_tuned = results['stage5']['test_accuracy'] * 100
                    improvement = fine_tuned - zero_shot
                    summaries.append("")
                    summaries.append(f"Comparison: Zero-Shot vs Fine-Tuned")
                    summaries.append(f"  Zero-shot:  {zero_shot:.2f}%")
                    summaries.append(f"  Fine-tuned: {fine_tuned:.2f}%")
                    summaries.append(f"  Improvement: +{improvement:.2f}%")
        else:
            summaries.append("❌ Not completed")
        summaries.append("")

        # Stage 6
        summaries.append("-" * 80)
        summaries.append("STAGE 6: SAMPLING RATE EXPERIMENT")
        summaries.append("-" * 80)
        if results['stage6']:
            summaries.append(f"Dataset: {results['stage6']['dataset']}")
            summaries.append(f"Original SR: {results['stage6']['original']['sampling_rate']} Hz")
            summaries.append(f"Comparison SR: 16000 Hz")
            summaries.append("")
            summaries.append("Results:")
            summaries.append(f"  Original rate:  {results['stage6']['original']['test_accuracy']*100:.2f}%")
            summaries.append(f"  16kHz:          {results['stage6']['16khz']['test_accuracy']*100:.2f}%")
            summaries.append(f"  Difference:     {results['stage6']['comparison']['accuracy_difference']*100:.2f}%")
            summaries.append(f"  Info loss:      {results['stage6']['comparison']['percent_information_loss']:.2f}%")
            summaries.append("")
            if results['stage6']['comparison']['original_better']:
                summaries.append("  ✓ Original sampling rate performs better")
            else:
                summaries.append("  ⚠ No information loss detected (16kHz sufficient)")
        else:
            summaries.append("❌ Not completed")
        summaries.append("")

        return summaries

    def create_comparison_figure(self, results):
        """Create comprehensive comparison figure."""
        if not all([results['stage2'], results['stage3'], results['stage4'], results['stage5']]):
            self.logger.warning("Skipping comparison figure - missing required results")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Model comparison across stages
        best_model = results['stage4']['best_model']

        # Per-dataset mean
        per_dataset_mean = results['stage2'][results['stage2']['model'] == best_model]['accuracy'].mean()

        # Pooled zero-shot
        pooled_result = next((r for r in results['stage3'] if r['model'] == best_model), None)
        pooled_zero_shot = pooled_result['test_accuracy'] if pooled_result else 0

        # Fine-tuned
        fine_tuned = results['stage5']['test_accuracy']

        stages = ['Per-Dataset\nZero-Shot', 'Pooled\nZero-Shot', 'Pooled\nFine-Tuned']
        accuracies = [per_dataset_mean * 100, pooled_zero_shot * 100, fine_tuned * 100]

        axes[0, 0].bar(stages, accuracies, color=['skyblue', 'orange', 'green'], edgecolor='black', linewidth=2)
        axes[0, 0].set_ylabel('Accuracy (%)', fontweight='bold')
        axes[0, 0].set_title(f'Best Model Performance Across Stages\n({best_model})', fontweight='bold')
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].grid(axis='y', alpha=0.3)

        for i, acc in enumerate(accuracies):
            axes[0, 0].text(i, acc + 2, f'{acc:.2f}%', ha='center', fontweight='bold')

        # 2. All models ranking (Stage 2)
        model_means = results['stage2'].groupby('model')['accuracy'].mean().sort_values(ascending=False)
        axes[0, 1].barh(range(len(model_means)), model_means.values * 100,
                       color=['green' if m == best_model else 'skyblue' for m in model_means.index],
                       edgecolor='black', linewidth=1.5)
        axes[0, 1].set_yticks(range(len(model_means)))
        axes[0, 1].set_yticklabels(model_means.index)
        axes[0, 1].set_xlabel('Mean Accuracy (%)', fontweight='bold')
        axes[0, 1].set_title('All Models Ranking (Per-Dataset Mean)', fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        axes[0, 1].invert_yaxis()

        # 3. Bird clustering analysis
        if results['stage3']:
            models = []
            silhouettes = []
            for r in results['stage3']:
                if r.get('bird_clustering_metric') and 'silhouette_by_dataset' in r['bird_clustering_metric']:
                    models.append(r['model'])
                    silhouettes.append(r['bird_clustering_metric']['silhouette_by_dataset'])

            if models:
                colors = ['red' if s > 0.3 else 'orange' if s > 0.2 else 'green' for s in silhouettes]
                axes[1, 0].bar(range(len(models)), silhouettes, color=colors, edgecolor='black', linewidth=1.5)
                axes[1, 0].set_xticks(range(len(models)))
                axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
                axes[1, 0].set_ylabel('Silhouette Score', fontweight='bold')
                axes[1, 0].set_title('Bird Clustering Quality\n(Lower = Better)', fontweight='bold')
                axes[1, 0].axhline(y=0.3, color='red', linestyle='--', alpha=0.5)
                axes[1, 0].axhline(y=0.2, color='orange', linestyle='--', alpha=0.5)
                axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. Summary statistics table
        axes[1, 1].axis('off')

        table_data = [
            ['Metric', 'Value'],
            ['Best Model', best_model],
            ['Datasets', str(results['stage2']['dataset'].nunique())],
            ['Total Individuals', str(results['stage5']['num_classes'])],
            ['Per-Dataset Acc', f"{per_dataset_mean*100:.2f}%"],
            ['Pooled Zero-Shot', f"{pooled_zero_shot*100:.2f}%"],
            ['Pooled Fine-Tuned', f"{fine_tuned*100:.2f}%"],
            ['Improvement', f"+{(fine_tuned - pooled_zero_shot)*100:.2f}%"],
        ]

        if results['stage6']:
            table_data.append(['SR Experiment', results['stage6']['dataset']])
            table_data.append(['Info Loss', f"{results['stage6']['comparison']['percent_information_loss']:.2f}%"])

        table = axes[1, 1].table(cellText=table_data, cellLoc='left', loc='center',
                                colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.suptitle('Phase 2: Comprehensive Results Summary', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / "comprehensive_results_summary.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ Comparison figure saved")

    def generate_text_report(self, results):
        """Generate comprehensive text report."""
        report = []

        # Executive summary
        report.extend(self.generate_executive_summary(results))

        # Stage summaries
        report.extend(self.generate_stage_summaries(results))

        # Conclusions
        report.append("=" * 80)
        report.append("CONCLUSIONS")
        report.append("=" * 80)
        report.append("")

        if results['stage4']:
            report.append(f"1. Best Model: {results['stage4']['best_model']}")
            report.append("   Selected based on composite score (per-dataset + pooled + bird clustering + consistency)")
            report.append("")

        if results['stage5'] and results['stage3'] and results['stage4']:
            best_model = results['stage4']['best_model']
            pooled_result = next((r for r in results['stage3'] if r['model'] == best_model), None)
            if pooled_result:
                improvement = (results['stage5']['test_accuracy'] - pooled_result['test_accuracy']) * 100
                report.append(f"2. Fine-Tuning Impact: +{improvement:.2f}% improvement on pooled task")
                report.append("   Fine-tuning first 4 layers significantly improves multi-dataset generalization")
                report.append("")

        if results['stage6']:
            if results['stage6']['comparison']['original_better']:
                report.append("3. Sampling Rate: Information loss detected")
                report.append(f"   {results['stage6']['comparison']['percent_information_loss']:.2f}% accuracy loss when resampling to 16kHz")
                report.append("   Higher frequencies contain discriminative information for bird identification")
            else:
                report.append("3. Sampling Rate: No significant information loss")
                report.append("   16kHz sampling is sufficient for animal identification")
            report.append("")

        # Next steps
        report.append("=" * 80)
        report.append("NEXT STEPS")
        report.append("=" * 80)
        report.append("")
        report.append("1. Hyrax Evaluation (pending data):")
        report.append("   • Zero-shot: Use fine-tuned model with frozen backbone + new FC head")
        report.append("   • Fine-tuned: Continue training on hyrax dataset")
        report.append("   • Target: >80% accuracy (previous baseline: ~80%)")
        report.append("")
        report.append("2. Thesis Writing:")
        report.append("   • Compile all Phase 2 results")
        report.append("   • Compare zero-shot vs fine-tuned approaches")
        report.append("   • Discuss dataset artifacts and bird clustering findings")
        report.append("   • Report sampling rate experiment conclusions")
        report.append("")

        report.append("=" * 80)
        report.append(f"END OF REPORT - Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)

        return report

    def save_report(self, report):
        """Save text report to file."""
        report_path = self.output_dir / "phase2_final_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))

        self.logger.info(f"✓ Final report saved: {report_path}")

        # Also print to console
        print("\n" + '\n'.join(report))

    def generate(self):
        """Generate complete final report."""
        self.logger.info("\n" + "="*80)
        self.logger.info("GENERATING FINAL COMPREHENSIVE REPORT")
        self.logger.info("="*80)

        # Load all results
        results = self.load_all_results()

        # Generate text report
        report = self.generate_text_report(results)
        self.save_report(report)

        # Generate comparison figure
        self.create_comparison_figure(results)

        self.logger.info("\n" + "="*80)
        self.logger.info("FINAL REPORT GENERATION COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"\nOutputs saved to: {self.output_dir}")
        self.logger.info(f"  • phase2_final_report.txt")
        self.logger.info(f"  • comprehensive_results_summary.png")


def main():
    """Main function."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    logger = setup_logger("Phase2_FinalReport", config['experiment']['log_level'])

    # Generate report
    generator = FinalReportGenerator(config, logger)
    generator.generate()


if __name__ == "__main__":
    main()
