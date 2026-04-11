"""
Linear classifiers for evaluating representation quality.
Includes Linear Probe and Logistic Regression classifiers.
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils.logging_utils import setup_logger
from src.evaluation.metrics import compute_all_metrics, generate_metrics_report


class LinearProbe:
    """
    Linear probe classifier for frozen representation evaluation.

    Uses a single linear layer (logistic regression) on top of frozen embeddings
    to evaluate representation quality.
    """

    def __init__(self, config: dict, log_level: str = "INFO"):
        """
        Initialize linear probe classifier.

        Args:
            config: Configuration dictionary
            log_level: Logging level
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__, log_level)

        # Linear probe parameters
        self.probe_config = config.get('linear_probe', {})
        self.C_values = self.probe_config.get('regularization_strengths', [0.001, 0.01, 0.1, 1.0, 10.0])
        self.max_iter = self.probe_config.get('max_iter', 1000)
        self.solver = self.probe_config.get('solver', 'lbfgs')
        self.cv_folds = self.probe_config.get('cv_folds', 5)
        self.test_size = self.probe_config.get('test_size', 0.2)
        self.random_state = self.probe_config.get('random_state', 42)
        self.multi_class = self.probe_config.get('multi_class', 'multinomial')
        self.n_jobs = config['computation']['n_jobs']

        self.label_encoder = LabelEncoder()

    def evaluate_single_config(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        C: float = 1.0
    ) -> Dict:
        """
        Evaluate linear probe with a single regularization strength.

        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            C: Regularization strength (inverse of lambda)

        Returns:
            Dictionary containing evaluation metrics
        """
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)

        # Create linear probe (logistic regression)
        probe = LogisticRegression(
            C=C,
            max_iter=self.max_iter,
            solver=self.solver,
            multi_class=self.multi_class,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        # Cross-validation
        cv_scores = cross_val_score(
            probe,
            features,
            labels_encoded,
            cv=self.cv_folds,
            scoring='accuracy',
            n_jobs=self.n_jobs
        )

        # Train-test split evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels_encoded
        )

        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)

        # Compute comprehensive metrics
        metrics = compute_all_metrics(y_test, y_pred, self.label_encoder)

        return {
            'C': C,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'accuracy': metrics['accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'macro_f1': metrics['macro_f1'],
            'macro_precision': metrics['macro_precision'],
            'macro_recall': metrics['macro_recall'],
            'per_class_f1': metrics['per_class_f1'],
            'y_test': y_test,
            'y_pred': y_pred,
            'label_encoder': self.label_encoder,
            'model': probe
        }

    def evaluate_multiple_configs(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        C_values: List[float] = None
    ) -> Dict[float, Dict]:
        """
        Evaluate linear probe with multiple regularization strengths.

        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            C_values: List of C values to test

        Returns:
            Dictionary mapping C to evaluation results
        """
        if C_values is None:
            C_values = self.C_values

        results = {}

        for C in C_values:
            self.logger.info(f"  Evaluating C={C}...")
            results[C] = self.evaluate_single_config(features, labels, C)

        # Find best C
        best_C = max(results.items(), key=lambda x: x[1]['cv_mean'])[0]
        self.logger.info(f"  Best C: {best_C} (CV accuracy: {results[best_C]['cv_mean']:.4f})")

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
        self.logger.info(f"Evaluating layer {layer_idx} with Linear Probe...")

        results = self.evaluate_multiple_configs(features, labels)

        # Add layer index to results
        for C in results:
            results[C]['layer'] = layer_idx

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

    def plot_regularization_comparison(
        self,
        results: Dict[float, Dict],
        output_path: str,
        title: str = "Linear Probe Regularization Comparison"
    ):
        """
        Plot comparison of different regularization strengths.

        Args:
            results: Results from evaluate_multiple_configs
            output_path: Path to save figure
            title: Plot title
        """
        C_values = sorted(results.keys())
        cv_means = [results[C]['cv_mean'] for C in C_values]
        cv_stds = [results[C]['cv_std'] for C in C_values]
        test_accs = [results[C]['accuracy'] for C in C_values]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.errorbar(C_values, cv_means, yerr=cv_stds, label='CV Accuracy', marker='o', capsize=5)
        ax.plot(C_values, test_accs, label='Test Accuracy', marker='s')

        ax.set_xlabel('Regularization Strength (C)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xscale('log')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        self.logger.info(f"Regularization comparison plot saved to {output_path}")

        plt.close(fig)

    def plot_confusion_matrix(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        label_encoder: LabelEncoder,
        output_path: str,
        title: str = "Confusion Matrix - Linear Probe"
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
        results: Dict[float, Dict],
        best_C: float,
        output_path: str
    ):
        """
        Generate a text report of classification results.

        Args:
            results: Evaluation results
            best_C: Best C value
            output_path: Path to save report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("LINEAR PROBE CLASSIFICATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summary table
        report_lines.append("Performance Summary:")
        report_lines.append("-" * 80)
        report_lines.append(f"{'C':<12} {'CV Mean':<12} {'CV Std':<12} {'Accuracy':<12} {'Bal. Acc':<12} {'Macro F1':<12}")
        report_lines.append("-" * 80)

        for C in sorted(results.keys()):
            result = results[C]
            report_lines.append(
                f"{C:<12.4f} {result['cv_mean']:<12.4f} {result['cv_std']:<12.4f} "
                f"{result['accuracy']:<12.4f} {result['balanced_accuracy']:<12.4f} {result['macro_f1']:<12.4f}"
            )

        report_lines.append("-" * 80)
        report_lines.append(f"\nBest C: {best_C}")
        report_lines.append(f"Best CV Accuracy: {results[best_C]['cv_mean']:.4f} ± {results[best_C]['cv_std']:.4f}")
        report_lines.append(f"Test Accuracy: {results[best_C]['accuracy']:.4f}")
        report_lines.append(f"Balanced Accuracy: {results[best_C]['balanced_accuracy']:.4f}")
        report_lines.append(f"Macro F1: {results[best_C]['macro_f1']:.4f}")
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


class LogisticRegressionClassifier:
    """
    Logistic regression classifier wrapper.
    Similar to LinearProbe but with a distinct class for clarity.
    """

    def __init__(self, config: dict, log_level: str = "INFO"):
        """
        Initialize logistic regression classifier.

        Args:
            config: Configuration dictionary
            log_level: Logging level
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__, log_level)

        # Logistic regression parameters
        self.logreg_config = config.get('logistic_regression', {})
        self.C_values = self.logreg_config.get('regularization_strengths', [0.001, 0.01, 0.1, 1.0, 10.0])
        self.max_iter = self.logreg_config.get('max_iter', 1000)
        self.solver = self.logreg_config.get('solver', 'lbfgs')
        self.cv_folds = self.logreg_config.get('cv_folds', 5)
        self.test_size = self.logreg_config.get('test_size', 0.2)
        self.random_state = self.logreg_config.get('random_state', 42)
        self.multi_class = self.logreg_config.get('multi_class', 'multinomial')
        self.n_jobs = config['computation']['n_jobs']

        self.label_encoder = LabelEncoder()

    def evaluate_single_config(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        C: float = 1.0
    ) -> Dict:
        """
        Evaluate logistic regression with a single regularization strength.

        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            C: Regularization strength

        Returns:
            Dictionary containing evaluation metrics
        """
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)

        # Create logistic regression model
        logreg = LogisticRegression(
            C=C,
            max_iter=self.max_iter,
            solver=self.solver,
            multi_class=self.multi_class,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        # Cross-validation
        cv_scores = cross_val_score(
            logreg,
            features,
            labels_encoded,
            cv=self.cv_folds,
            scoring='accuracy',
            n_jobs=self.n_jobs
        )

        # Train-test split evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels_encoded
        )

        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)

        # Compute comprehensive metrics
        metrics = compute_all_metrics(y_test, y_pred, self.label_encoder)

        return {
            'C': C,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'accuracy': metrics['accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'macro_f1': metrics['macro_f1'],
            'macro_precision': metrics['macro_precision'],
            'macro_recall': metrics['macro_recall'],
            'per_class_f1': metrics['per_class_f1'],
            'y_test': y_test,
            'y_pred': y_pred,
            'label_encoder': self.label_encoder,
            'model': logreg
        }

    def evaluate_multiple_configs(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        C_values: List[float] = None
    ) -> Dict[float, Dict]:
        """
        Evaluate logistic regression with multiple regularization strengths.

        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            C_values: List of C values to test

        Returns:
            Dictionary mapping C to evaluation results
        """
        if C_values is None:
            C_values = self.C_values

        results = {}

        for C in C_values:
            self.logger.info(f"  Evaluating C={C}...")
            results[C] = self.evaluate_single_config(features, labels, C)

        # Find best C
        best_C = max(results.items(), key=lambda x: x[1]['cv_mean'])[0]
        self.logger.info(f"  Best C: {best_C} (CV accuracy: {results[best_C]['cv_mean']:.4f})")

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
        self.logger.info(f"Evaluating layer {layer_idx} with Logistic Regression...")

        results = self.evaluate_multiple_configs(features, labels)

        # Add layer index to results
        for C in results:
            results[C]['layer'] = layer_idx

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

    def plot_regularization_comparison(
        self,
        results: Dict[float, Dict],
        output_path: str,
        title: str = "Logistic Regression Regularization Comparison"
    ):
        """
        Plot comparison of different regularization strengths.

        Args:
            results: Results from evaluate_multiple_configs
            output_path: Path to save figure
            title: Plot title
        """
        C_values = sorted(results.keys())
        cv_means = [results[C]['cv_mean'] for C in C_values]
        cv_stds = [results[C]['cv_std'] for C in C_values]
        test_accs = [results[C]['accuracy'] for C in C_values]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.errorbar(C_values, cv_means, yerr=cv_stds, label='CV Accuracy', marker='o', capsize=5)
        ax.plot(C_values, test_accs, label='Test Accuracy', marker='s')

        ax.set_xlabel('Regularization Strength (C)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xscale('log')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        self.logger.info(f"Regularization comparison plot saved to {output_path}")

        plt.close(fig)

    def plot_confusion_matrix(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        label_encoder: LabelEncoder,
        output_path: str,
        title: str = "Confusion Matrix - Logistic Regression"
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
        results: Dict[float, Dict],
        best_C: float,
        output_path: str
    ):
        """
        Generate a text report of classification results.

        Args:
            results: Evaluation results
            best_C: Best C value
            output_path: Path to save report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("LOGISTIC REGRESSION CLASSIFICATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summary table
        report_lines.append("Performance Summary:")
        report_lines.append("-" * 80)
        report_lines.append(f"{'C':<12} {'CV Mean':<12} {'CV Std':<12} {'Accuracy':<12} {'Bal. Acc':<12} {'Macro F1':<12}")
        report_lines.append("-" * 80)

        for C in sorted(results.keys()):
            result = results[C]
            report_lines.append(
                f"{C:<12.4f} {result['cv_mean']:<12.4f} {result['cv_std']:<12.4f} "
                f"{result['accuracy']:<12.4f} {result['balanced_accuracy']:<12.4f} {result['macro_f1']:<12.4f}"
            )

        report_lines.append("-" * 80)
        report_lines.append(f"\nBest C: {best_C}")
        report_lines.append(f"Best CV Accuracy: {results[best_C]['cv_mean']:.4f} ± {results[best_C]['cv_std']:.4f}")
        report_lines.append(f"Test Accuracy: {results[best_C]['accuracy']:.4f}")
        report_lines.append(f"Balanced Accuracy: {results[best_C]['balanced_accuracy']:.4f}")
        report_lines.append(f"Macro F1: {results[best_C]['macro_f1']:.4f}")
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
