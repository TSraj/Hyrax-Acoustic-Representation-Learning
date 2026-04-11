"""
Centralized metrics computation for evaluation.
Provides consistent metric calculations across all classifiers.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    classification_report
)
from sklearn.preprocessing import LabelEncoder


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute standard accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score
    """
    return accuracy_score(y_true, y_pred)


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute balanced accuracy (handles class imbalance).

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Balanced accuracy score
    """
    return balanced_accuracy_score(y_true, y_pred)


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute macro-averaged F1 score.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Macro F1 score
    """
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def compute_macro_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute macro-averaged precision.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Macro precision score
    """
    return precision_score(y_true, y_pred, average='macro', zero_division=0)


def compute_macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute macro-averaged recall.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Macro recall score
    """
    return recall_score(y_true, y_pred, average='macro', zero_division=0)


def compute_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute F1 score for each class.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Array of per-class F1 scores
    """
    _, _, f1_scores, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    return f1_scores


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: Optional[LabelEncoder] = None
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels (encoded)
        y_pred: Predicted labels (encoded)
        label_encoder: Optional label encoder for class names

    Returns:
        Dictionary containing:
            - accuracy: Standard accuracy
            - balanced_accuracy: Balanced accuracy (handles imbalance)
            - macro_f1: Macro-averaged F1 score
            - macro_precision: Macro-averaged precision
            - macro_recall: Macro-averaged recall
            - per_class_f1: Array of per-class F1 scores
            - n_classes: Number of classes
            - n_samples: Number of samples
    """
    metrics = {
        'accuracy': compute_accuracy(y_true, y_pred),
        'balanced_accuracy': compute_balanced_accuracy(y_true, y_pred),
        'macro_f1': compute_macro_f1(y_true, y_pred),
        'macro_precision': compute_macro_precision(y_true, y_pred),
        'macro_recall': compute_macro_recall(y_true, y_pred),
        'per_class_f1': compute_per_class_f1(y_true, y_pred),
        'n_classes': len(np.unique(y_true)),
        'n_samples': len(y_true)
    }

    return metrics


def generate_metrics_report(
    metrics: Dict,
    label_encoder: Optional[LabelEncoder] = None,
    title: str = "Classification Metrics"
) -> str:
    """
    Generate a formatted text report of classification metrics.

    Args:
        metrics: Dictionary of metrics from compute_all_metrics()
        label_encoder: Optional label encoder for class names
        title: Report title

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append(title.upper())
    lines.append("=" * 80)
    lines.append("")

    # Overall metrics
    lines.append("Overall Performance:")
    lines.append("-" * 80)
    lines.append(f"  Accuracy:          {metrics['accuracy']:.4f}")
    lines.append(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    lines.append(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    lines.append(f"  Macro Precision:   {metrics['macro_precision']:.4f}")
    lines.append(f"  Macro Recall:      {metrics['macro_recall']:.4f}")
    lines.append("")

    # Dataset info
    lines.append(f"Dataset Information:")
    lines.append("-" * 80)
    lines.append(f"  Number of samples: {metrics['n_samples']}")
    lines.append(f"  Number of classes: {metrics['n_classes']}")
    lines.append("")

    # Per-class F1 scores
    if 'per_class_f1' in metrics and len(metrics['per_class_f1']) > 0:
        lines.append("Per-Class F1 Scores:")
        lines.append("-" * 80)

        per_class_f1 = metrics['per_class_f1']

        if label_encoder is not None:
            class_names = label_encoder.classes_
        else:
            class_names = [f"Class {i}" for i in range(len(per_class_f1))]

        for class_name, f1 in zip(class_names, per_class_f1):
            lines.append(f"  {class_name:<30} {f1:.4f}")

        lines.append("")

    lines.append("=" * 80)

    return '\n'.join(lines)


def print_metrics_summary(
    metrics: Dict,
    classifier_name: str = "Classifier"
):
    """
    Print a concise metrics summary to console.

    Args:
        metrics: Dictionary of metrics
        classifier_name: Name of the classifier
    """
    print(f"\n{classifier_name} Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
