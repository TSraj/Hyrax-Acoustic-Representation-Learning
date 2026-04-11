"""
k-NN classifier for evaluating representation quality.
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils.logging_utils import setup_logger
from src.evaluation.metrics import compute_all_metrics


class KNNClassifier:
    """k-NN classifier for downstream evaluation of representations."""

    def __init__(self, config: dict, log_level: str = "INFO"):
        """
        Initialize k-NN classifier.

        Args:
            config: Configuration dictionary
            log_level: Logging level
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__, log_level)

        # k-NN parameters
        self.knn_config = config['knn']
        self.k_values = self.knn_config['n_neighbors']
        self.metric = self.knn_config['metric']
        self.cv_folds = self.knn_config['cv_folds']
        self.test_size = self.knn_config['test_size']
        self.random_state = self.knn_config['random_state']

        self.label_encoder = LabelEncoder()

    def evaluate_single_k(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        k: int
    ) -> Dict:
        """
        Evaluate k-NN with a single k value using cross-validation.

        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            k: Number of neighbors

        Returns:
            Dictionary containing evaluation metrics
        """
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)

        # Create k-NN classifier
        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric=self.metric,
            n_jobs=self.config['computation']['n_jobs']
        )

        # Cross-validation
        cv_scores = cross_val_score(
            knn,
            features,
            labels_encoded,
            cv=self.cv_folds,
            scoring='accuracy'
        )

        # Train-test split evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels_encoded
        )

        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # Compute comprehensive metrics
        metrics = compute_all_metrics(y_test, y_pred, self.label_encoder)

        return {
            'k': k,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': metrics['accuracy'],  # Keep for backward compatibility
            'accuracy': metrics['accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'macro_f1': metrics['macro_f1'],
            'macro_precision': metrics['macro_precision'],
            'macro_recall': metrics['macro_recall'],
            'per_class_f1': metrics['per_class_f1'],
            'cv_scores': cv_scores,
            'y_test': y_test,
            'y_pred': y_pred,
            'label_encoder': self.label_encoder
        }

    def evaluate_multiple_k(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        k_values: List[int] = None
    ) -> Dict[int, Dict]:
        """
        Evaluate k-NN with multiple k values.

        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            k_values: List of k values to test

        Returns:
            Dictionary mapping k to evaluation results
        """
        if k_values is None:
            k_values = self.k_values

        results = {}

        for k in k_values:
            self.logger.info(f"  Evaluating k={k}...")
            results[k] = self.evaluate_single_k(features, labels, k)

        # Find best k
        best_k = max(results.items(), key=lambda x: x[1]['cv_mean'])[0]
        self.logger.info(f"  Best k: {best_k} (CV accuracy: {results[best_k]['cv_mean']:.4f})")

        return results

    def evaluate_layer(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        layer_idx: int
    ) -> Dict:
        """
        Evaluate a single layer.

        Args:
            features: Feature array for this layer
            labels: Label array
            layer_idx: Layer index

        Returns:
            Evaluation results
        """
        self.logger.info(f"Evaluating layer {layer_idx}...")

        results = self.evaluate_multiple_k(features, labels)

        # Add layer index to results
        for k in results:
            results[k]['layer'] = layer_idx

        return results

    def evaluate_all_layers(
        self,
        features_per_layer: Dict[int, np.ndarray],
        labels: np.ndarray
    ) -> Dict[int, Dict]:
        """
        Evaluate all layers.

        Args:
            features_per_layer: Dictionary mapping layer index to features
            labels: Label array

        Returns:
            Dictionary mapping layer index to evaluation results
        """
        all_results = {}

        for layer_idx in sorted(features_per_layer.keys()):
            features = features_per_layer[layer_idx]
            all_results[layer_idx] = self.evaluate_layer(features, labels, layer_idx)

        return all_results

    def plot_k_comparison(
        self,
        results: Dict[int, Dict],
        output_path: str,
        title: str = "k-NN Performance Comparison"
    ):
        """
        Plot comparison of different k values.

        Args:
            results: Results from evaluate_multiple_k
            output_path: Path to save figure
            title: Plot title
        """
        k_values = sorted(results.keys())
        cv_means = [results[k]['cv_mean'] for k in k_values]
        cv_stds = [results[k]['cv_std'] for k in k_values]
        test_accs = [results[k]['test_accuracy'] for k in k_values]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.errorbar(k_values, cv_means, yerr=cv_stds, label='CV Accuracy', marker='o', capsize=5)
        ax.plot(k_values, test_accs, label='Test Accuracy', marker='s')

        ax.set_xlabel('k (number of neighbors)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        self.logger.info(f"k comparison plot saved to {output_path}")

        plt.close(fig)

    def plot_confusion_matrix(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        label_encoder: LabelEncoder,
        output_path: str,
        title: str = "Confusion Matrix"
    ):
        """
        Plot confusion matrix.

        Args:
            y_test: True labels (encoded)
            y_pred: Predicted labels (encoded)
            label_encoder: Label encoder
            output_path: Path to save figure
            title: Plot title
        """
        cm = confusion_matrix(y_test, y_pred)

        # Get original label names
        labels = label_encoder.classes_

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Count'}
        )

        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        self.logger.info(f"Confusion matrix saved to {output_path}")

        plt.close(fig)

    def generate_report(
        self,
        results: Dict[int, Dict],
        best_k: int,
        output_path: str
    ):
        """
        Generate a text report of classification results.

        Args:
            results: Evaluation results
            best_k: Best k value
            output_path: Path to save report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("k-NN CLASSIFICATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summary table
        report_lines.append("Performance Summary:")
        report_lines.append("-" * 80)
        report_lines.append(f"{'k':<5} {'CV Mean':<12} {'CV Std':<12} {'Test Accuracy':<15}")
        report_lines.append("-" * 80)

        for k in sorted(results.keys()):
            result = results[k]
            report_lines.append(
                f"{k:<5} {result['cv_mean']:.4f}      {result['cv_std']:.4f}      "
                f"{result['test_accuracy']:.4f}"
            )

        report_lines.append("-" * 80)
        report_lines.append(f"\nBest k: {best_k}")
        report_lines.append(f"Best CV Accuracy: {results[best_k]['cv_mean']:.4f} ± {results[best_k]['cv_std']:.4f}")
        report_lines.append(f"Test Accuracy: {results[best_k]['test_accuracy']:.4f}")
        report_lines.append("")
        report_lines.append("=" * 80)

        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))

        self.logger.info(f"Report saved to {output_path}")

        # Also print to console
        print('\n'.join(report_lines))
