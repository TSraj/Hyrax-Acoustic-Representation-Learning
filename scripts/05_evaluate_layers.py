#!/usr/bin/env python3
"""
Script 05: Evaluate Layers
Evaluates layers using multiple classifiers and clustering metrics.
"""

import yaml
import sys
import numpy as np
import gc
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.feature_pooling import FeaturePooler
from src.evaluation.knn_classifier import KNNClassifier
from src.evaluation.linear_classifiers import LinearProbe, LogisticRegressionClassifier
from src.evaluation.layer_comparator import LayerComparator
from src.utils.logging_utils import setup_logger, get_timestamp

# Memory monitoring (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def log_memory_usage(logger, label=""):
    """
    Log current memory usage.

    Args:
        logger: Logger instance
        label: Optional label for the log message
    """
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"  Memory usage {label}: {mem_mb:.1f} MB")
    else:
        logger.debug("  psutil not available for memory monitoring")


def evaluate_model_dataset(
    config,
    model_name,
    dataset_name,
    embeddings_path,
    output_dir,
    knn_classifier,
    linear_probe,
    logreg_classifier,
    layer_comparator,
    pooler,
    logger
):
    """
    Evaluate embeddings for a specific model and dataset.

    Args:
        config: Configuration dictionary
        model_name: Name of the model
        dataset_name: Name of the dataset
        embeddings_path: Path to pooled embeddings
        output_dir: Output directory for results
        knn_classifier: KNNClassifier instance
        linear_probe: LinearProbe instance
        logreg_classifier: LogisticRegressionClassifier instance
        layer_comparator: LayerComparator instance
        pooler: FeaturePooler instance
        logger: Logger instance
    """
    logger.info(f"\nEvaluating {model_name} on {dataset_name}...")
    logger.info("-" * 80)

    # Log memory before loading
    if config.get('memory', {}).get('enable_monitoring', True):
        log_memory_usage(logger, "before loading embeddings")

    # Load pooled features
    pooled_features = pooler.load_pooled_features(str(embeddings_path))

    # Get metadata
    metadata = pooled_features['mean']['metadata']
    num_layers = len(metadata['layers'])

    logger.info(f"  Loaded {len(metadata['file_paths'])} samples")
    logger.info(f"  Number of layers: {num_layers}")

    # Create output directory
    model_output_dir = output_dir / dataset_name / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Use mean pooling for evaluation
    pooling_method = 'mean'

    # Prepare features per layer
    features_per_layer = {}
    for layer_idx in metadata['layers']:
        features, labels = pooler.get_pooled_arrays(
            pooled_features,
            layer_idx,
            pooling_method
        )
        features_per_layer[layer_idx] = features

    # Get labels (same for all layers)
    _, labels = pooler.get_pooled_arrays(
        pooled_features,
        metadata['layers'][0],
        pooling_method
    )

    # Log memory after loading
    if config.get('memory', {}).get('enable_monitoring', True):
        log_memory_usage(logger, "after loading embeddings")

    # k-NN evaluation
    logger.info(f"\n  Running k-NN evaluation...")
    knn_results = knn_classifier.evaluate_all_layers(features_per_layer, labels)

    # Find best k across all layers
    best_layer_k = {}
    for layer_idx, layer_results in knn_results.items():
        best_k = max(layer_results.items(), key=lambda x: x[1]['cv_mean'])[0]
        best_layer_k[layer_idx] = layer_results[best_k]

    # Plot k-NN results for each layer
    logger.info(f"\n  Generating k-NN plots...")
    for layer_idx, layer_results in knn_results.items():
        knn_classifier.plot_k_comparison(
            layer_results,
            str(model_output_dir / "knn_plots" / f"layer_{layer_idx}_k_comparison.png"),
            title=f"{model_name} - Layer {layer_idx} - k-NN Performance\n{dataset_name}"
        )

        # Plot confusion matrix for best k
        best_k = max(layer_results.items(), key=lambda x: x[1]['cv_mean'])[0]
        best_result = layer_results[best_k]

        if len(np.unique(labels)) <= 20:  # Only plot confusion matrix if not too many classes
            knn_classifier.plot_confusion_matrix(
                best_result['y_test'],
                best_result['y_pred'],
                best_result['label_encoder'],
                str(model_output_dir / "confusion_matrices" / f"layer_{layer_idx}_knn_confusion_k{best_k}.png"),
                title=f"{model_name} - Layer {layer_idx} - k-NN Confusion Matrix (k={best_k})\n{dataset_name}"
            )

    # Generate k-NN report for best layer
    best_layer_idx = max(best_layer_k.items(), key=lambda x: x[1]['cv_mean'])[0]
    best_k = best_layer_k[best_layer_idx]['k']

    knn_classifier.generate_report(
        knn_results[best_layer_idx],
        best_k,
        str(model_output_dir / f"knn_report_layer{best_layer_idx}.txt")
    )

    # Linear Probe evaluation
    linear_probe_results = None
    if config.get('linear_probe', {}).get('enabled', True):
        logger.info(f"\n  Running Linear Probe evaluation...")
        linear_probe_results = linear_probe.evaluate_all_layers(features_per_layer, labels)

        # Find best C across all layers
        best_layer_C = {}
        for layer_idx, layer_results in linear_probe_results.items():
            best_C = max(layer_results.items(), key=lambda x: x[1]['cv_mean'])[0]
            best_layer_C[layer_idx] = layer_results[best_C]

        # Plot regularization comparison and confusion matrices
        logger.info(f"\n  Generating Linear Probe plots...")
        for layer_idx, layer_results in linear_probe_results.items():
            linear_probe.plot_regularization_comparison(
                layer_results,
                str(model_output_dir / "linear_probe_plots" / f"layer_{layer_idx}_regularization.png"),
                title=f"{model_name} - Layer {layer_idx} - Linear Probe Performance\n{dataset_name}"
            )

            # Plot confusion matrix for best C
            best_C = max(layer_results.items(), key=lambda x: x[1]['cv_mean'])[0]
            best_result = layer_results[best_C]

            if len(np.unique(labels)) <= 20:
                linear_probe.plot_confusion_matrix(
                    best_result['y_test'],
                    best_result['y_pred'],
                    best_result['label_encoder'],
                    str(model_output_dir / "confusion_matrices" / f"layer_{layer_idx}_linear_probe_confusion_C{best_C}.png"),
                    title=f"{model_name} - Layer {layer_idx} - Linear Probe Confusion (C={best_C})\n{dataset_name}"
                )

        # Generate report for best layer
        best_layer_idx_lp = max(best_layer_C.items(), key=lambda x: x[1]['cv_mean'])[0]
        best_C_lp = best_layer_C[best_layer_idx_lp]['C']

        linear_probe.generate_report(
            linear_probe_results[best_layer_idx_lp],
            best_C_lp,
            str(model_output_dir / f"linear_probe_report_layer{best_layer_idx_lp}.txt")
        )

    # Logistic Regression evaluation
    logreg_results = None
    if config.get('logistic_regression', {}).get('enabled', True):
        logger.info(f"\n  Running Logistic Regression evaluation...")
        logreg_results = logreg_classifier.evaluate_all_layers(features_per_layer, labels)

        # Find best C across all layers
        best_layer_C_lr = {}
        for layer_idx, layer_results in logreg_results.items():
            best_C = max(layer_results.items(), key=lambda x: x[1]['cv_mean'])[0]
            best_layer_C_lr[layer_idx] = layer_results[best_C]

        # Plot regularization comparison and confusion matrices
        logger.info(f"\n  Generating Logistic Regression plots...")
        for layer_idx, layer_results in logreg_results.items():
            logreg_classifier.plot_regularization_comparison(
                layer_results,
                str(model_output_dir / "logreg_plots" / f"layer_{layer_idx}_regularization.png"),
                title=f"{model_name} - Layer {layer_idx} - Logistic Regression Performance\n{dataset_name}"
            )

            # Plot confusion matrix for best C
            best_C = max(layer_results.items(), key=lambda x: x[1]['cv_mean'])[0]
            best_result = layer_results[best_C]

            if len(np.unique(labels)) <= 20:
                logreg_classifier.plot_confusion_matrix(
                    best_result['y_test'],
                    best_result['y_pred'],
                    best_result['label_encoder'],
                    str(model_output_dir / "confusion_matrices" / f"layer_{layer_idx}_logreg_confusion_C{best_C}.png"),
                    title=f"{model_name} - Layer {layer_idx} - LogReg Confusion (C={best_C})\n{dataset_name}"
                )

        # Generate report for best layer
        best_layer_idx_lr = max(best_layer_C_lr.items(), key=lambda x: x[1]['cv_mean'])[0]
        best_C_lr = best_layer_C_lr[best_layer_idx_lr]['C']

        logreg_classifier.generate_report(
            logreg_results[best_layer_idx_lr],
            best_C_lr,
            str(model_output_dir / f"logreg_report_layer{best_layer_idx_lr}.txt")
        )

    # Layer comparison with all classifiers
    logger.info(f"\n  Running layer comparison...")
    layer_metrics = layer_comparator.compare_layers(
        features_per_layer,
        labels,
        knn_results=knn_results,
        linear_probe_results=linear_probe_results,
        logreg_results=logreg_results
    )

    # Plot layer comparison
    layer_comparator.plot_layer_comparison(
        layer_metrics,
        str(model_output_dir / "layer_comparison.png"),
        title=f"{model_name} - Layer Comparison\n{dataset_name}"
    )

    # Plot metric heatmap
    layer_comparator.plot_metric_heatmap(
        layer_metrics,
        str(model_output_dir / "layer_metrics_heatmap.png"),
        title=f"{model_name} - Layer Metrics Heatmap\n{dataset_name}"
    )

    # Generate layer comparison report
    layer_comparator.generate_comparison_report(
        layer_metrics,
        str(model_output_dir / "layer_comparison_report.txt"),
        model_name=model_name,
        dataset_name=dataset_name
    )

    logger.info(f"  Evaluation complete. Results saved to: {model_output_dir}")

    return layer_metrics


def main():
    """Main function to evaluate layers."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    logger = setup_logger("EvaluateLayers", config['experiment']['log_level'])
    logger.info("=" * 80)
    logger.info("SCRIPT 05: EVALUATE LAYERS")
    logger.info("=" * 80)

    # Create evaluators
    knn_classifier = KNNClassifier(config, config['experiment']['log_level'])
    linear_probe = LinearProbe(config, config['experiment']['log_level'])
    logreg_classifier = LogisticRegressionClassifier(config, config['experiment']['log_level'])
    layer_comparator = LayerComparator(config, config['experiment']['log_level'])
    pooler = FeaturePooler(config, config['experiment']['log_level'])

    # Log initial memory
    if config.get('memory', {}).get('enable_monitoring', True):
        log_memory_usage(logger, "at start")

    # Set paths
    embeddings_dir = Path(config['paths']['embeddings_dir'])
    output_dir = Path(config['paths']['reports_dir']) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate all combinations of models and datasets
    models = ['wav2vec2_base', 'wav2vec2_xlsr']
    datasets = ['macaque', 'zebra_finch']

    all_results = {}

    for model_name in models:
        for dataset_name in datasets:
            logger.info("\n" + "=" * 80)
            logger.info(f"EVALUATING: {model_name} - {dataset_name}")
            logger.info("=" * 80)

            embeddings_path = embeddings_dir / f"{dataset_name}_{model_name}_pooled.npz"

            if not embeddings_path.exists():
                logger.warning(f"Embeddings not found: {embeddings_path}")
                continue

            try:
                layer_metrics = evaluate_model_dataset(
                    config,
                    model_name,
                    dataset_name,
                    embeddings_path,
                    output_dir,
                    knn_classifier,
                    linear_probe,
                    logreg_classifier,
                    layer_comparator,
                    pooler,
                    logger
                )
                all_results[f"{model_name}_{dataset_name}"] = layer_metrics

                # Memory cleanup
                if config.get('memory', {}).get('enable_garbage_collection', True):
                    gc.collect()
                    if config.get('memory', {}).get('enable_monitoring', True):
                        log_memory_usage(logger, "after GC")

            except Exception as e:
                logger.error(f"Error evaluating {model_name} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()

    # Generate summary comparison
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: Model Comparison")
    logger.info("=" * 80)

    summary_lines = []
    summary_lines.append("\n" + "=" * 80)
    summary_lines.append("OVERALL SUMMARY - ALL CLASSIFIERS")
    summary_lines.append("=" * 80)

    for key, metrics in all_results.items():
        model, dataset = key.rsplit('_', 1)

        summary_lines.append(f"\n{model.upper()} on {dataset.upper()}:")
        summary_lines.append("-" * 80)

        # k-NN results
        best_layers_knn = layer_comparator.find_best_layers(metrics, 'knn_accuracy')
        if best_layers_knn:
            best_layer = best_layers_knn[0]
            knn_acc = metrics[best_layer].get('knn_accuracy', 0)
            knn_bal_acc = metrics[best_layer].get('knn_balanced_accuracy', 0)
            knn_f1 = metrics[best_layer].get('knn_macro_f1', 0)
            summary_lines.append(f"  k-NN (Layer {best_layer}):")
            summary_lines.append(f"    Accuracy: {knn_acc:.4f}")
            summary_lines.append(f"    Balanced Accuracy: {knn_bal_acc:.4f}")
            summary_lines.append(f"    Macro F1: {knn_f1:.4f}")

        # Linear Probe results
        best_layers_lp = layer_comparator.find_best_layers(metrics, 'linear_probe_accuracy')
        if best_layers_lp:
            best_layer = best_layers_lp[0]
            lp_acc = metrics[best_layer].get('linear_probe_accuracy', 0)
            lp_bal_acc = metrics[best_layer].get('linear_probe_balanced_accuracy', 0)
            lp_f1 = metrics[best_layer].get('linear_probe_macro_f1', 0)
            summary_lines.append(f"  Linear Probe (Layer {best_layer}):")
            summary_lines.append(f"    Accuracy: {lp_acc:.4f}")
            summary_lines.append(f"    Balanced Accuracy: {lp_bal_acc:.4f}")
            summary_lines.append(f"    Macro F1: {lp_f1:.4f}")

        # Logistic Regression results
        best_layers_lr = layer_comparator.find_best_layers(metrics, 'logreg_accuracy')
        if best_layers_lr:
            best_layer = best_layers_lr[0]
            lr_acc = metrics[best_layer].get('logreg_accuracy', 0)
            lr_bal_acc = metrics[best_layer].get('logreg_balanced_accuracy', 0)
            lr_f1 = metrics[best_layer].get('logreg_macro_f1', 0)
            summary_lines.append(f"  Logistic Regression (Layer {best_layer}):")
            summary_lines.append(f"    Accuracy: {lr_acc:.4f}")
            summary_lines.append(f"    Balanced Accuracy: {lr_bal_acc:.4f}")
            summary_lines.append(f"    Macro F1: {lr_f1:.4f}")

    summary_lines.append("\n" + "=" * 80)

    summary_text = '\n'.join(summary_lines)
    print(summary_text)

    # Save summary
    summary_path = output_dir / "overall_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary_text)

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
