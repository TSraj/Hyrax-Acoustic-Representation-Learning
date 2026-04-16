"""
Advanced Visualizer Module

Creates publication-quality comparison visualizations for thesis work.

This module generates:
- Grouped bar charts for multi-way comparisons
- Heatmaps for layer × classifier performance
- Feature comparison plots
- Scaling curves
- Confusion matrices with better formatting

Author: Raj
Date: 2026-04-11
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AdvancedVisualizer:
    """
    Create advanced comparison visualizations.

    Generates publication-quality plots for thesis work,
    including multi-way comparisons and performance heatmaps.
    """

    def __init__(
        self,
        style: str = 'seaborn-v0_8-darkgrid',
        dpi: int = 300,
        figsize: Tuple[int, int] = (12, 6),
        color_palette: str = 'Set2'
    ):
        """
        Initialize advanced visualizer.

        Args:
            style: Matplotlib style
            dpi: Resolution for saved figures
            figsize: Default figure size
            color_palette: Seaborn color palette
        """
        self.style = style
        self.dpi = dpi
        self.figsize = figsize
        self.color_palette = color_palette

        # Set style
        plt.style.use(style)
        sns.set_palette(color_palette)

        logger.info(f"Initialized AdvancedVisualizer with style: {style}")

    def plot_feature_comparison(
        self,
        results_dict: Dict[str, Dict[str, float]],
        metrics: List[str] = ['accuracy', 'balanced_accuracy', 'macro_f1'],
        title: str = "Feature Type Comparison",
        save_path: Optional[str] = None
    ) -> None:
        """
        Create grouped bar chart comparing different feature types.

        Args:
            results_dict: Nested dict with structure:
                         {feature_type: {metric_name: value}}
                         e.g., {'MFCC': {'accuracy': 0.85, 'macro_f1': 0.82}, ...}
            metrics: Which metrics to plot
            title: Plot title
            save_path: Path to save figure
        """
        feature_types = list(results_dict.keys())
        n_features = len(feature_types)
        n_metrics = len(metrics)

        # Prepare data
        data = np.zeros((n_features, n_metrics))
        for i, feat_type in enumerate(feature_types):
            for j, metric in enumerate(metrics):
                data[i, j] = results_dict[feat_type].get(metric, 0.0)

        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)

        x = np.arange(n_features)
        width = 0.25
        multiplier = 0

        metric_labels = {
            'accuracy': 'Accuracy',
            'balanced_accuracy': 'Balanced Accuracy',
            'macro_f1': 'Macro F1'
        }

        for j, metric in enumerate(metrics):
            offset = width * multiplier
            bars = ax.bar(x + offset, data[:, j], width,
                         label=metric_labels.get(metric, metric))

            # Add value labels on bars with smaller font to avoid overlap
            ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=7)
            multiplier += 1

        ax.set_xlabel('Feature Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x + width * (n_metrics - 1) / 2)
        ax.set_xticklabels(feature_types, rotation=45, ha='right')
        ax.set_ylim([0, 1.15])  # Adjusted for bar labels
        ax.grid(True, alpha=0.3, axis='y')

        # Position legend outside plot area on the right to avoid overlap with bar labels
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                 framealpha=0.95, fontsize=10)

        plt.tight_layout(rect=[0, 0, 0.88, 1])  # Leave space on right for legend

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved feature comparison plot to: {save_path}")

        plt.close()

    def plot_classifier_comparison(
        self,
        results_dict: Dict[str, Dict[str, float]],
        metrics: List[str] = ['accuracy', 'balanced_accuracy', 'macro_f1'],
        title: str = "Classifier Comparison",
        save_path: Optional[str] = None
    ) -> None:
        """
        Create grouped bar chart comparing different classifiers.

        Args:
            results_dict: Nested dict with structure:
                         {classifier_name: {metric_name: value}}
            metrics: Which metrics to plot
            title: Plot title
            save_path: Path to save figure
        """
        # Same implementation as plot_feature_comparison but with different context
        self.plot_feature_comparison(results_dict, metrics, title, save_path)

    def plot_layer_heatmap(
        self,
        layer_results: Dict[int, Dict[str, float]],
        metrics: List[str] = ['accuracy', 'balanced_accuracy', 'macro_f1'],
        title: str = "Layer Performance Heatmap",
        save_path: Optional[str] = None
    ) -> None:
        """
        Create heatmap showing performance across layers and metrics.

        Args:
            layer_results: Nested dict with structure:
                          {layer_idx: {metric_name: value}}
            metrics: Which metrics to include
            title: Plot title
            save_path: Path to save figure
        """
        layers = sorted(layer_results.keys())
        n_layers = len(layers)
        n_metrics = len(metrics)

        # Prepare data
        data = np.zeros((n_layers, n_metrics))
        for i, layer in enumerate(layers):
            for j, metric in enumerate(metrics):
                data[i, j] = layer_results[layer].get(metric, 0.0)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(n_metrics * 2, n_layers * 0.5))

        metric_labels = {
            'accuracy': 'Accuracy',
            'balanced_accuracy': 'Balanced Acc',
            'macro_f1': 'Macro F1'
        }

        sns.heatmap(
            data,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=[metric_labels.get(m, m) for m in metrics],
            yticklabels=[f"Layer {l}" for l in layers],
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Score'},
            ax=ax
        )

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout(pad=1.0)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved layer heatmap to: {save_path}")

        plt.close()

    def plot_multi_way_comparison(
        self,
        results_dict: Dict[Tuple[str, str], Dict[str, float]],
        metric: str = 'accuracy',
        x_label: str = "Feature Type",
        group_label: str = "Classifier",
        title: str = "Multi-Way Comparison",
        save_path: Optional[str] = None
    ) -> None:
        """
        Create grouped bar chart for multi-way comparisons.

        Args:
            results_dict: Dict with structure:
                         {(feature_type, classifier): {metric: value}}
                         e.g., {('MFCC', 'kNN'): {'accuracy': 0.85}, ...}
            metric: Which metric to plot
            x_label: Label for x-axis (grouping variable)
            group_label: Label for bar groups
            title: Plot title
            save_path: Path to save figure
        """
        # Extract unique feature types and classifiers
        feature_types = sorted(set(k[0] for k in results_dict.keys()))
        classifiers = sorted(set(k[1] for k in results_dict.keys()))

        n_features = len(feature_types)
        n_classifiers = len(classifiers)

        # Prepare data
        data = np.zeros((n_features, n_classifiers))
        for i, feat_type in enumerate(feature_types):
            for j, clf in enumerate(classifiers):
                key = (feat_type, clf)
                if key in results_dict:
                    data[i, j] = results_dict[key].get(metric, 0.0)

        # Create plot
        fig, ax = plt.subplots(figsize=(max(12, n_features * 2), 6))

        x = np.arange(n_features)
        width = 0.8 / n_classifiers

        for j, clf in enumerate(classifiers):
            offset = width * j - (width * n_classifiers / 2 - width / 2)
            bars = ax.bar(x + offset, data[:, j], width, label=clf)
            ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=6, rotation=90)

        ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_types, rotation=45, ha='right')
        ax.set_ylim([0, 1.2])  # Adjusted for rotated bar labels
        ax.grid(True, alpha=0.3, axis='y')

        # Position legend outside plot area on the right
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                 title=group_label, framealpha=0.95, fontsize=9)

        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on right for legend

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved multi-way comparison to: {save_path}")

        plt.close()

    def plot_confusion_matrix_enhanced(
        self,
        confusion_matrix: np.ndarray,
        class_labels: List[str],
        title: str = "Confusion Matrix",
        normalize: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create enhanced confusion matrix visualization.

        Args:
            confusion_matrix: Confusion matrix array
            class_labels: Labels for classes
            title: Plot title
            normalize: Whether to normalize by row
            save_path: Path to save figure
        """
        cm = confusion_matrix.copy()

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'

        fig, ax = plt.subplots(figsize=(max(8, len(class_labels)), max(6, len(class_labels))))

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels,
            vmin=0,
            vmax=1 if normalize else None,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'},
            ax=ax,
            square=True
        )

        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout(pad=1.0)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to: {save_path}")

        plt.close()

    def plot_pooling_comparison(
        self,
        results_dict: Dict[str, Dict[str, float]],
        metrics: List[str] = ['accuracy', 'balanced_accuracy', 'macro_f1'],
        title: str = "Pooling Method Comparison",
        save_path: Optional[str] = None
    ) -> None:
        """
        Create bar chart comparing pooling methods.

        Args:
            results_dict: {pooling_method: {metric: value}}
            metrics: Which metrics to plot
            title: Plot title
            save_path: Path to save figure
        """
        self.plot_feature_comparison(results_dict, metrics, title, save_path)

    def create_summary_dashboard(
        self,
        all_results: Dict,
        save_dir: Path
    ) -> None:
        """
        Create comprehensive summary dashboard with multiple subplots.

        Args:
            all_results: Dictionary containing all experimental results
            save_dir: Directory to save dashboard
        """
        logger.info("Creating summary dashboard...")

        # Create multi-panel figure
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # This would be customized based on available results
        # For now, just log that it's a placeholder
        logger.info("Summary dashboard creation - to be customized based on results")

        plt.close()

    def __repr__(self) -> str:
        return f"AdvancedVisualizer(style={self.style}, dpi={self.dpi})"


def create_default_visualizer() -> AdvancedVisualizer:
    """Create visualizer with default publication-quality settings."""
    return AdvancedVisualizer(
        style='seaborn-v0_8-darkgrid',
        dpi=300,
        figsize=(12, 6),
        color_palette='Set2'
    )
