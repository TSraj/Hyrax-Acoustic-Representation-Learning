"""
Layer comparator for analyzing which wav2vec layers provide best representations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder

from src.utils.logging_utils import setup_logger


class LayerComparator:
    """Compares representations across different wav2vec layers."""

    def __init__(self, config: dict, log_level: str = "INFO"):
        """
        Initialize layer comparator.

        Args:
            config: Configuration dictionary
            log_level: Logging level
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__, log_level)

        self.metrics_config = config['layer_comparison']['metrics']

    def compute_clustering_metrics(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute clustering quality metrics.

        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)

        Returns:
            Dictionary of metric name to value
        """
        # Encode labels
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)

        metrics = {}

        try:
            # Silhouette score (higher is better, range [-1, 1])
            if 'silhouette_score' in self.metrics_config:
                metrics['silhouette_score'] = silhouette_score(features, labels_encoded)
        except Exception as e:
            self.logger.warning(f"Error computing silhouette score: {e}")
            metrics['silhouette_score'] = np.nan

        try:
            # Calinski-Harabasz score (higher is better)
            if 'calinski_harabasz_score' in self.metrics_config:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, labels_encoded)
        except Exception as e:
            self.logger.warning(f"Error computing Calinski-Harabasz score: {e}")
            metrics['calinski_harabasz_score'] = np.nan

        try:
            # Davies-Bouldin score (lower is better)
            if 'davies_bouldin_score' in self.metrics_config:
                metrics['davies_bouldin_score'] = davies_bouldin_score(features, labels_encoded)
        except Exception as e:
            self.logger.warning(f"Error computing Davies-Bouldin score: {e}")
            metrics['davies_bouldin_score'] = np.nan

        return metrics

    def compare_layers(
        self,
        features_per_layer: Dict[int, np.ndarray],
        labels: np.ndarray,
        knn_results: Dict[int, Dict] = None,
        linear_probe_results: Dict[int, Dict] = None,
        logreg_results: Dict[int, Dict] = None
    ) -> Dict[int, Dict]:
        """
        Compare all layers using clustering metrics and classifier results.

        Args:
            features_per_layer: Dictionary mapping layer index to features
            labels: Label array
            knn_results: Optional k-NN evaluation results per layer
            linear_probe_results: Optional linear probe evaluation results per layer
            logreg_results: Optional logistic regression evaluation results per layer

        Returns:
            Dictionary mapping layer index to metrics
        """
        self.logger.info("Comparing layers...")

        layer_metrics = {}

        for layer_idx in sorted(features_per_layer.keys()):
            features = features_per_layer[layer_idx]

            # Compute clustering metrics
            metrics = self.compute_clustering_metrics(features, labels)

            # Add k-NN metrics if available
            if knn_results and layer_idx in knn_results:
                best_k_result = max(knn_results[layer_idx].values(),
                                   key=lambda x: x['cv_mean'])
                metrics['knn_accuracy'] = best_k_result.get('accuracy', best_k_result['cv_mean'])
                metrics['knn_balanced_accuracy'] = best_k_result.get('balanced_accuracy', 0)
                metrics['knn_macro_f1'] = best_k_result.get('macro_f1', 0)
                metrics['knn_best_k'] = best_k_result['k']

            # Add Linear Probe metrics if available
            if linear_probe_results and layer_idx in linear_probe_results:
                best_C_result = max(linear_probe_results[layer_idx].values(),
                                   key=lambda x: x['cv_mean'])
                metrics['linear_probe_accuracy'] = best_C_result.get('accuracy', best_C_result['cv_mean'])
                metrics['linear_probe_balanced_accuracy'] = best_C_result.get('balanced_accuracy', 0)
                metrics['linear_probe_macro_f1'] = best_C_result.get('macro_f1', 0)
                metrics['linear_probe_best_C'] = best_C_result['C']

            # Add Logistic Regression metrics if available
            if logreg_results and layer_idx in logreg_results:
                best_C_result = max(logreg_results[layer_idx].values(),
                                   key=lambda x: x['cv_mean'])
                metrics['logreg_accuracy'] = best_C_result.get('accuracy', best_C_result['cv_mean'])
                metrics['logreg_balanced_accuracy'] = best_C_result.get('balanced_accuracy', 0)
                metrics['logreg_macro_f1'] = best_C_result.get('macro_f1', 0)
                metrics['logreg_best_C'] = best_C_result['C']

            layer_metrics[layer_idx] = metrics

            # Log summary
            log_msg = f"  Layer {layer_idx}: "
            log_msg += f"Silhouette={metrics.get('silhouette_score', 0):.3f}, "
            log_msg += f"k-NN={metrics.get('knn_accuracy', 0):.3f}, "
            log_msg += f"LinearProbe={metrics.get('linear_probe_accuracy', 0):.3f}, "
            log_msg += f"LogReg={metrics.get('logreg_accuracy', 0):.3f}"
            self.logger.info(log_msg)

        return layer_metrics

    def find_best_layers(
        self,
        layer_metrics: Dict[int, Dict],
        metric: str = 'knn_accuracy'
    ) -> List[int]:
        """
        Find best performing layers based on a specific metric.

        Args:
            layer_metrics: Dictionary of layer metrics
            metric: Metric to use for ranking

        Returns:
            List of layer indices sorted by performance (best first)
        """
        # For Davies-Bouldin, lower is better
        reverse = metric != 'davies_bouldin_score'

        ranked = sorted(
            layer_metrics.items(),
            key=lambda x: x[1].get(metric, float('-inf') if reverse else float('inf')),
            reverse=reverse
        )

        return [layer_idx for layer_idx, _ in ranked]

    def plot_layer_comparison(
        self,
        layer_metrics: Dict[int, Dict],
        output_path: str,
        title: str = "Layer Comparison"
    ):
        """
        Plot comparison of layers across metrics.

        Args:
            layer_metrics: Dictionary of layer metrics
            output_path: Path to save figure
            title: Plot title
        """
        layers = sorted(layer_metrics.keys())

        # Prepare data
        metrics_data = {}
        for metric in ['silhouette_score', 'calinski_harabasz_score',
                      'davies_bouldin_score', 'knn_accuracy']:
            values = [layer_metrics[l].get(metric, np.nan) for l in layers]
            if not all(np.isnan(values)):
                metrics_data[metric] = values

        if not metrics_data:
            self.logger.warning("No metrics available for plotting")
            return

        # Create subplots
        n_metrics = len(metrics_data)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for idx, (metric, values) in enumerate(metrics_data.items()):
            ax = axes[idx]

            ax.plot(layers, values, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Layer Index', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Highlight best layer
            if metric == 'davies_bouldin_score':
                # Lower is better
                best_idx = np.nanargmin(values)
            else:
                # Higher is better
                best_idx = np.nanargmax(values)

            if not np.isnan(values[best_idx]):
                ax.scatter(
                    [layers[best_idx]],
                    [values[best_idx]],
                    color='red',
                    s=200,
                    marker='*',
                    zorder=5,
                    label=f'Best: Layer {layers[best_idx]}'
                )
                ax.legend()

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        self.logger.info(f"Layer comparison plot saved to {output_path}")

        plt.close(fig)

    def plot_metric_heatmap(
        self,
        layer_metrics: Dict[int, Dict],
        output_path: str,
        title: str = "Layer Metrics Heatmap"
    ):
        """
        Create a heatmap of all metrics across layers.

        Args:
            layer_metrics: Dictionary of layer metrics
            output_path: Path to save figure
            title: Plot title
        """
        layers = sorted(layer_metrics.keys())

        # Collect all metrics
        all_metrics = set()
        for metrics in layer_metrics.values():
            all_metrics.update(metrics.keys())

        # Remove non-numeric metrics
        all_metrics = sorted([m for m in all_metrics if m != 'knn_best_k'])

        # Build data matrix
        data_matrix = []
        for metric in all_metrics:
            row = [layer_metrics[l].get(metric, np.nan) for l in layers]
            data_matrix.append(row)

        data_matrix = np.array(data_matrix)

        # Normalize each metric to [0, 1] for visualization
        normalized_matrix = data_matrix.copy()
        for i in range(len(all_metrics)):
            row = normalized_matrix[i]
            valid_mask = ~np.isnan(row)
            if valid_mask.any():
                if all_metrics[i] == 'davies_bouldin_score':
                    # Lower is better, invert
                    row[valid_mask] = 1 - (row[valid_mask] - np.nanmin(row)) / (np.nanmax(row) - np.nanmin(row))
                else:
                    # Higher is better
                    row[valid_mask] = (row[valid_mask] - np.nanmin(row)) / (np.nanmax(row) - np.nanmin(row))

        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(layers)), max(6, len(all_metrics) * 0.5)))

        sns.heatmap(
            normalized_matrix,
            annot=data_matrix,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=[f'Layer {l}' for l in layers],
            yticklabels=[m.replace('_', ' ').title() for m in all_metrics],
            ax=ax,
            cbar_kws={'label': 'Normalized Score (0=worst, 1=best)'}
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        self.logger.info(f"Metric heatmap saved to {output_path}")

        plt.close(fig)

    def generate_comparison_report(
        self,
        layer_metrics: Dict[int, Dict],
        output_path: str,
        model_name: str = "",
        dataset_name: str = ""
    ):
        """
        Generate a text report comparing layers.

        Args:
            layer_metrics: Dictionary of layer metrics
            output_path: Path to save report
            model_name: Name of the model
            dataset_name: Name of the dataset
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("LAYER COMPARISON REPORT")
        if model_name:
            report_lines.append(f"Model: {model_name}")
        if dataset_name:
            report_lines.append(f"Dataset: {dataset_name}")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Metrics table
        layers = sorted(layer_metrics.keys())

        # Collect all metrics
        all_metrics = set()
        for metrics in layer_metrics.values():
            all_metrics.update(metrics.keys())
        all_metrics = sorted([m for m in all_metrics if m != 'knn_best_k'])

        # Header
        header = f"{'Layer':<8}"
        for metric in all_metrics:
            header += f"{metric[:20]:<22}"
        report_lines.append(header)
        report_lines.append("-" * len(header))

        # Rows
        for layer in layers:
            row = f"{layer:<8}"
            for metric in all_metrics:
                value = layer_metrics[layer].get(metric, np.nan)
                if np.isnan(value):
                    row += f"{'N/A':<22}"
                else:
                    row += f"{value:<22.4f}"
            report_lines.append(row)

        report_lines.append("-" * len(header))
        report_lines.append("")

        # Best layers
        report_lines.append("Best Performing Layers:")
        report_lines.append("-" * 80)

        for metric in all_metrics:
            best_layers = self.find_best_layers(layer_metrics, metric)[:3]
            report_lines.append(f"\n{metric.replace('_', ' ').title()}:")
            for rank, layer in enumerate(best_layers, 1):
                value = layer_metrics[layer].get(metric, np.nan)
                report_lines.append(f"  {rank}. Layer {layer}: {value:.4f}")

        report_lines.append("")
        report_lines.append("=" * 80)

        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))

        self.logger.info(f"Comparison report saved to {output_path}")

        # Also print to console
        print('\n'.join(report_lines))
