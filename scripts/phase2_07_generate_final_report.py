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

        if results['stage2'] is not None:
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
        if results['stage2'] is not None:
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
                bird_metric = r.get('bird_clustering_metric')
                if bird_metric and 'silhouette_by_dataset' in bird_metric:
                    bird_sil = bird_metric['silhouette_by_dataset']
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

    def create_comparison_figures(self, results):
        """Create individual comparison figures."""
        if not all([results['stage2'] is not None, results['stage3'] is not None,
                    results['stage4'] is not None, results['stage5'] is not None]):
            self.logger.warning("Skipping comparison figures - missing required results")
            return

        best_model = results['stage4']['best_model']

        # Per-dataset mean
        per_dataset_mean = results['stage2'][results['stage2']['model'] == best_model]['accuracy'].mean()

        # Pooled zero-shot
        pooled_result = next((r for r in results['stage3'] if r['model'] == best_model), None)
        pooled_zero_shot = pooled_result['test_accuracy'] if pooled_result else 0

        # Fine-tuned
        fine_tuned = results['stage5']['test_accuracy']

        # Figure 1: Best Model Performance Across Stages
        fig, ax = plt.subplots(figsize=(10, 6))
        stages = ['Per-Dataset\nZero-Shot', 'Pooled\nZero-Shot', 'Pooled\nFine-Tuned']
        accuracies = [per_dataset_mean * 100, pooled_zero_shot * 100, fine_tuned * 100]

        ax.bar(stages, accuracies, color=['skyblue', 'orange', 'green'], edgecolor='black', linewidth=2)
        ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
        ax.set_title(f'Best Model Performance Across Stages\n({best_model})', fontweight='bold', fontsize=14)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        for i, acc in enumerate(accuracies):
            ax.text(i, acc + 2, f'{acc:.2f}%', ha='center', fontweight='bold', fontsize=11)

        plt.tight_layout()
        plt.savefig(self.output_dir / "1_best_model_performance_stages.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  ✓ Figure 1: Best model performance across stages")

        # Figure 2: All Models Ranking
        fig, ax = plt.subplots(figsize=(10, 6))
        model_means = results['stage2'].groupby('model')['accuracy'].mean().sort_values(ascending=False)
        ax.barh(range(len(model_means)), model_means.values * 100,
                color=['green' if m == best_model else 'skyblue' for m in model_means.index],
                edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(len(model_means)))
        ax.set_yticklabels(model_means.index, fontsize=11)
        ax.set_xlabel('Mean Accuracy (%)', fontweight='bold', fontsize=12)
        ax.set_title('All Models Ranking (Per-Dataset Mean)', fontweight='bold', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(self.output_dir / "2_all_models_ranking.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  ✓ Figure 2: All models ranking")

        # Figure 3: Bird Clustering Analysis
        if results['stage3']:
            models = []
            silhouettes = []
            for r in results['stage3']:
                bird_metric = r.get('bird_clustering_metric')
                if bird_metric and 'silhouette_by_dataset' in bird_metric:
                    models.append(r['model'])
                    silhouettes.append(bird_metric['silhouette_by_dataset'])

            if models:
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['red' if s > 0.3 else 'orange' if s > 0.2 else 'green' for s in silhouettes]
                ax.bar(range(len(models)), silhouettes, color=colors, edgecolor='black', linewidth=1.5)
                ax.set_xticks(range(len(models)))
                ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
                ax.set_ylabel('Silhouette Score', fontweight='bold', fontsize=12)
                ax.set_title('Bird Clustering Quality\n(Lower = Better)', fontweight='bold', fontsize=14)
                ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Poor (>0.3)')
                ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.2-0.3)')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)

                plt.tight_layout()
                plt.savefig(self.output_dir / "3_bird_clustering_quality.png", dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"  ✓ Figure 3: Bird clustering quality")

        # Figure 4: Summary Statistics Table
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')

        table_data = [
            ['Metric', 'Value'],
            ['Best Model', best_model],
            ['Datasets', str(results['stage2']['dataset'].nunique())],
            ['Total Individuals', str(results['stage5']['num_classes'])],
            ['Per-Dataset Accuracy', f"{per_dataset_mean*100:.2f}%"],
            ['Pooled Zero-Shot', f"{pooled_zero_shot*100:.2f}%"],
            ['Pooled Fine-Tuned', f"{fine_tuned*100:.2f}%"],
            ['Fine-Tuning Improvement', f"+{(fine_tuned - pooled_zero_shot)*100:.2f}%"],
        ]

        if results['stage6']:
            table_data.append(['SR Experiment Dataset', results['stage6']['dataset']])
            table_data.append(['Information Loss', f"{results['stage6']['comparison']['percent_information_loss']:.2f}%"])

        table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                        colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.scale(1, 3)

        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=14)

        # Highlight best model row
        table[(1, 0)].set_facecolor('#E8F5E9')
        table[(1, 1)].set_facecolor('#E8F5E9')
        table[(1, 1)].set_text_props(weight='bold')

        ax.set_title('Phase 2: Summary Statistics', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / "4_summary_statistics.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  ✓ Figure 4: Summary statistics table")

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

    def generate_csv_reports(self, results):
        """Generate CSV reports for easy analysis."""
        self.logger.info("\nGenerating CSV reports...")

        # CSV 1: Executive Summary
        if results['stage4'] and results['stage5']:
            best_model = results['stage4']['best_model']
            per_dataset_mean = results['stage2'][results['stage2']['model'] == best_model]['accuracy'].mean() if results['stage2'] is not None else None
            pooled_result = next((r for r in results['stage3'] if r['model'] == best_model), None) if results['stage3'] else None
            pooled_zero_shot = pooled_result['test_accuracy'] if pooled_result else None
            fine_tuned = results['stage5']['test_accuracy']

            summary_data = {
                'Metric': ['Best Model', 'Per-Dataset Zero-Shot Mean', 'Pooled Zero-Shot', 'Pooled Fine-Tuned', 'Fine-Tuning Improvement'],
                'Value': [
                    best_model,
                    f"{per_dataset_mean*100:.2f}%" if per_dataset_mean else "N/A",
                    f"{pooled_zero_shot*100:.2f}%" if pooled_zero_shot else "N/A",
                    f"{fine_tuned*100:.2f}%",
                    f"+{(fine_tuned - pooled_zero_shot)*100:.2f}%" if pooled_zero_shot else "N/A"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.output_dir / "executive_summary.csv", index=False)
            self.logger.info(f"  ✓ executive_summary.csv")

        # CSV 2: All Models Comparison (Stage 2)
        if results['stage2'] is not None:
            model_comparison = results['stage2'].groupby('model').agg({
                'accuracy': ['mean', 'std', 'min', 'max']
            }).round(4)
            model_comparison.columns = ['Mean_Accuracy', 'Std_Accuracy', 'Min_Accuracy', 'Max_Accuracy']
            model_comparison = model_comparison.sort_values('Mean_Accuracy', ascending=False)
            model_comparison.to_csv(self.output_dir / "all_models_comparison.csv")
            self.logger.info(f"  ✓ all_models_comparison.csv")

        # CSV 3: Per-Dataset Results (Stage 2)
        if results['stage2'] is not None:
            results['stage2'].to_csv(self.output_dir / "per_dataset_results.csv", index=False)
            self.logger.info(f"  ✓ per_dataset_results.csv")

        # CSV 4: Pooled Results (Stage 3)
        if results['stage3']:
            pooled_data = []
            for r in results['stage3']:
                bird_metric = r.get('bird_clustering_metric')
                pooled_data.append({
                    'Model': r['model'],
                    'Test_Accuracy': r['test_accuracy'],
                    'Val_Accuracy': r.get('val_accuracy', 'N/A'),
                    'Bird_Silhouette': bird_metric['silhouette_by_dataset'] if bird_metric and 'silhouette_by_dataset' in bird_metric else 'N/A'
                })
            pooled_df = pd.DataFrame(pooled_data)
            pooled_df = pooled_df.sort_values('Test_Accuracy', ascending=False)
            pooled_df.to_csv(self.output_dir / "pooled_results.csv", index=False)
            self.logger.info(f"  ✓ pooled_results.csv")

        # CSV 5: Model Selection Details (Stage 4)
        if results['stage4']:
            selection_data = {
                'Rank': list(range(1, len(results['stage4']['ranking']) + 1)),
                'Model': results['stage4']['ranking'],
                'Selected': [m == results['stage4']['best_model'] for m in results['stage4']['ranking']]
            }
            selection_df = pd.DataFrame(selection_data)
            selection_df.to_csv(self.output_dir / "model_selection_ranking.csv", index=False)
            self.logger.info(f"  ✓ model_selection_ranking.csv")

        # CSV 6: Fine-Tuning Results (Stage 5)
        if results['stage5']:
            finetuning_data = {
                'Metric': ['Model', 'Fine-Tuned Layers', 'Num Classes', 'Training Epochs', 'Batch Size', 'Learning Rate',
                          'Best Val Accuracy', 'Test Accuracy'],
                'Value': [
                    results['stage5']['model'],
                    results['stage5']['fine_tuned_layers'],
                    results['stage5']['num_classes'],
                    results['stage5']['training_epochs'],
                    results['stage5']['batch_size'],
                    results['stage5']['learning_rate'],
                    f"{results['stage5']['best_val_accuracy']*100:.2f}%",
                    f"{results['stage5']['test_accuracy']*100:.2f}%"
                ]
            }
            finetuning_df = pd.DataFrame(finetuning_data)
            finetuning_df.to_csv(self.output_dir / "fine_tuning_results.csv", index=False)
            self.logger.info(f"  ✓ fine_tuning_results.csv")

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

        # Generate individual figures
        self.create_comparison_figures(results)

        # Generate CSV reports
        self.generate_csv_reports(results)

        self.logger.info("\n" + "="*80)
        self.logger.info("FINAL REPORT GENERATION COMPLETE")
        self.logger.info("="*80)
        self.logger.info(f"\nOutputs saved to: {self.output_dir}")
        self.logger.info(f"  Text Report:")
        self.logger.info(f"    • phase2_final_report.txt")
        self.logger.info(f"  Figures:")
        self.logger.info(f"    • 1_best_model_performance_stages.png")
        self.logger.info(f"    • 2_all_models_ranking.png")
        self.logger.info(f"    • 3_bird_clustering_quality.png")
        self.logger.info(f"    • 4_summary_statistics.png")
        self.logger.info(f"  CSV Reports:")
        self.logger.info(f"    • executive_summary.csv")
        self.logger.info(f"    • all_models_comparison.csv")
        self.logger.info(f"    • per_dataset_results.csv")
        self.logger.info(f"    • pooled_results.csv")
        self.logger.info(f"    • model_selection_ranking.csv")
        self.logger.info(f"    • fine_tuning_results.csv")


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
