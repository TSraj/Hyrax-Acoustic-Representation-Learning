#!/usr/bin/env python3
"""
Script 05 (Comprehensive): Comprehensive Feature Evaluation

Evaluates and compares:
- Handcrafted features (MFCC, Spectral, Combined)
- Wav2vec embeddings (all layers, both models)
- 7 different classifiers
- Generates bar chart comparisons

This is the enhanced version with all new features integrated.

Author: Raj
Date: 2026-04-11
"""

import yaml
import sys
import numpy as np
import gc
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.feature_pooling import FeaturePooler
from src.evaluation.svm_classifier import create_linear_svm, create_rbf_svm
from src.evaluation.ensemble_classifiers import create_random_forest, create_xgboost, XGBOOST_AVAILABLE
from src.evaluation.advanced_visualizer import create_default_visualizer
from src.utils.logging_utils import setup_logger, get_timestamp
from sklearn.model_selection import train_test_split

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def log_memory_usage(logger, label=""):
    """Log current memory usage."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"  Memory usage {label}: {mem_mb:.1f} MB")


def load_handcrafted_features(embeddings_dir: Path, dataset_name: str, pooling: str = 'mean'):
    """
    Load MFCC, spectral, and combined handcrafted features.

    Returns:
        Dict with 'mfcc', 'spectral', 'combined' keys, or None if not found
    """
    dataset_dir = embeddings_dir / dataset_name
    features_dict = {}

    for feature_type in ['mfcc', 'spectral', 'handcrafted_combined']:
        file_path = dataset_dir / f"{feature_type}_features_{pooling}.npz"
        if file_path.exists():
            data = np.load(file_path, allow_pickle=True)
            key = 'combined' if feature_type == 'handcrafted_combined' else feature_type
            features_dict[key] = {
                'features': data['features'],
                'labels': data['labels'],
                'file_paths': data['file_paths']
            }

    return features_dict if features_dict else None


def evaluate_with_all_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
    logger
) -> Dict:
    """
    Evaluate with all 7 classifiers.

    Returns dict with results for each classifier.
    """
    from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        f1_score, precision_score, recall_score
    )

    results = {}

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # 1. k-NN
    logger.info("  Evaluating k-NN...")
    knn = SklearnKNN(n_neighbors=5, metric='cosine', n_jobs=-1)
    knn.fit(X_train, y_train_enc)
    y_pred = knn.predict(X_test)

    results['knn'] = {
        'accuracy': accuracy_score(y_test_enc, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test_enc, y_pred),
        'macro_f1': f1_score(y_test_enc, y_pred, average='macro'),
        'macro_precision': precision_score(y_test_enc, y_pred, average='macro'),
        'macro_recall': recall_score(y_test_enc, y_pred, average='macro')
    }

    # 2. Linear Probe (Logistic Regression with limited iterations)
    logger.info("  Evaluating Linear Probe...")
    linear_probe = LogisticRegression(
        C=1.0, max_iter=1000, solver='lbfgs',
        random_state=42, n_jobs=-1
    )
    linear_probe.fit(X_train, y_train_enc)
    y_pred = linear_probe.predict(X_test)

    results['linear_probe'] = {
        'accuracy': accuracy_score(y_test_enc, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test_enc, y_pred),
        'macro_f1': f1_score(y_test_enc, y_pred, average='macro'),
        'macro_precision': precision_score(y_test_enc, y_pred, average='macro'),
        'macro_recall': recall_score(y_test_enc, y_pred, average='macro')
    }

    # 3. Logistic Regression
    logger.info("  Evaluating Logistic Regression...")
    logreg = LogisticRegression(
        C=1.0, max_iter=1000, solver='lbfgs',
        random_state=42, n_jobs=-1
    )
    logreg.fit(X_train, y_train_enc)
    y_pred = logreg.predict(X_test)

    results['logistic_regression'] = {
        'accuracy': accuracy_score(y_test_enc, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test_enc, y_pred),
        'macro_f1': f1_score(y_test_enc, y_pred, average='macro'),
        'macro_precision': precision_score(y_test_enc, y_pred, average='macro'),
        'macro_recall': recall_score(y_test_enc, y_pred, average='macro')
    }

    # 4. Linear SVM
    logger.info("  Evaluating Linear SVM...")
    svm_linear = create_linear_svm(C_values=[0.1, 1.0])
    svm_lin_results = svm_linear.train_and_evaluate(X_train, y_train, X_test, y_test, perform_grid_search=True)
    results['svm_linear'] = svm_lin_results

    # 5. RBF SVM
    logger.info("  Evaluating RBF SVM...")
    svm_rbf = create_rbf_svm(C_values=[0.1, 1.0], gamma_values=['scale'])
    svm_rbf_results = svm_rbf.train_and_evaluate(X_train, y_train, X_test, y_test, perform_grid_search=True)
    results['svm_rbf'] = svm_rbf_results

    # 6. Random Forest
    logger.info("  Evaluating Random Forest...")
    rf = create_random_forest(n_estimators_values=[100])
    rf_results = rf.train_and_evaluate(X_train, y_train, X_test, y_test, perform_grid_search=False)
    results['random_forest'] = rf_results

    # 7. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        logger.info("  Evaluating XGBoost...")
        xgb = create_xgboost(n_estimators_values=[100])
        xgb_results = xgb.train_and_evaluate(X_train, y_train, X_test, y_test, perform_grid_search=False)
        results['xgboost'] = xgb_results
    else:
        logger.warning("  XGBoost not available, skipping...")

    return results


def evaluate_feature_type(
    feature_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    random_state: int,
    config: dict,
    logger
) -> Dict:
    """
    Evaluate one feature type with all classifiers.

    Returns dict with classifier results.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {feature_name}")
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"{'='*60}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Evaluate with all classifiers
    results = evaluate_with_all_classifiers(X_train, y_train, X_test, y_test, config, logger)

    # Log summary
    logger.info(f"\nResults for {feature_name}:")
    for clf_name, clf_results in results.items():
        logger.info(f"  {clf_name:20} Acc={clf_results['accuracy']:.4f}, "
                   f"Bal_Acc={clf_results['balanced_accuracy']:.4f}, "
                   f"F1={clf_results['macro_f1']:.4f}")

    return results


def compare_wav2vec_layers(
    embeddings_path: Path,
    pooling_method: str,
    test_size: float,
    random_state: int,
    config: dict,
    logger
) -> Dict:
    """
    Compare different wav2vec layers to find best one.

    Returns dict with best layer info.
    """
    logger.info("\nComparing wav2vec layers...")

    # Load pooled features
    pooler = FeaturePooler(config=config, log_level="WARNING")
    pooled_features = pooler.load_pooled_features(str(embeddings_path))

    if pooling_method not in pooled_features:
        logger.warning(f"Pooling method {pooling_method} not found")
        return {}

    features_data = pooled_features[pooling_method]
    labels = features_data['metadata']['labels']
    layers = features_data['metadata']['layers']

    # Reorganize features by layer (currently organized by file)
    # Convert from {file: {layer: features}} to {layer: [features]}
    features_by_layer = {}
    file_keys = sorted(features_data['features'].keys())

    for layer_idx in layers:
        layer_features = []
        for file_key in file_keys:
            if layer_idx in features_data['features'][file_key]:
                layer_features.append(features_data['features'][file_key][layer_idx])
        features_by_layer[layer_idx] = np.array(layer_features)

    # Evaluate each layer with k-NN (fast baseline)
    from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    layer_results = {}
    le = LabelEncoder()

    for layer_idx in layers:  # Evaluate all layers
        features = features_by_layer[layer_idx]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )

        # Encode labels
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # Quick k-NN evaluation
        knn = SklearnKNN(n_neighbors=5, metric='cosine', n_jobs=-1)
        knn.fit(X_train, y_train_enc)
        y_pred = knn.predict(X_test)

        layer_results[layer_idx] = {
            'accuracy': accuracy_score(y_test_enc, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test_enc, y_pred),
            'macro_f1': f1_score(y_test_enc, y_pred, average='macro')
        }

        logger.info(f"  Layer {layer_idx:2}: Acc={layer_results[layer_idx]['accuracy']:.4f}, "
                   f"F1={layer_results[layer_idx]['macro_f1']:.4f}")

    # Find best layer
    best_layer = max(layer_results.keys(), key=lambda k: layer_results[k]['balanced_accuracy'])
    logger.info(f"\n✓ Best layer: {best_layer} (Bal_Acc={layer_results[best_layer]['balanced_accuracy']:.4f})")

    return {
        'best_layer': best_layer,
        'layer_results': layer_results,
        'best_features': features_by_layer[best_layer],
        'labels': labels
    }


def plot_combined_layer_comparison(
    layer_data: Dict,
    output_path: Path,
    dataset_name: str,
    logger
):
    """
    Create combined layer comparison BAR CHART for both wav2vec models.

    Args:
        layer_data: Dict with model names as keys, layer_results as values
        output_path: Path to save the plot
        dataset_name: Name of dataset for title
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Define colors and labels
    colors = {'wav2vec2_base': '#2E86AB', 'wav2vec2_xlsr': '#A23B72'}
    labels = {'wav2vec2_base': 'Wav2Vec 2.0 Base', 'wav2vec2_xlsr': 'Wav2Vec 2.0 XLSR'}

    # Collect data from both models
    models_data = {}
    for model_name, data in layer_data.items():
        if data is not None:
            layer_results = data['layer_results']
            layers = sorted(layer_results.keys())
            balanced_accs = [layer_results[l]['balanced_accuracy'] for l in layers]
            models_data[model_name] = {
                'layers': layers,
                'accuracies': balanced_accs,
                'best_layer': data['best_layer'],
                'best_acc': layer_results[data['best_layer']]['balanced_accuracy']
            }

    if not models_data:
        logger.warning("No layer data to plot")
        return

    # Create figure with appropriate size for all layers
    fig, ax = plt.subplots(figsize=(16, 8))

    # Determine all unique layers (union of both models)
    all_layers = sorted(set().union(*[set(d['layers']) for d in models_data.values()]))

    # Set up bar positions
    x = np.arange(len(all_layers))
    width = 0.35

    # Plot bars for each model
    for i, (model_name, data) in enumerate(models_data.items()):
        # Create accuracy array aligned with all_layers
        accuracies = []
        for layer in all_layers:
            if layer in data['layers']:
                idx = data['layers'].index(layer)
                accuracies.append(data['accuracies'][idx])
            else:
                accuracies.append(0)  # This layer doesn't exist for this model

        # Calculate offset for grouped bars
        offset = width * (i - 0.5) if len(models_data) == 2 else 0

        # Plot bars
        bars = ax.bar(x + offset, accuracies, width,
                     label=labels[model_name],
                     color=colors[model_name],
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=0.5)

        # Highlight best layer with a gold edge
        best_layer_idx = all_layers.index(data['best_layer'])
        bars[best_layer_idx].set_edgecolor('gold')
        bars[best_layer_idx].set_linewidth(3)

        # Add text annotation on best layer bar
        ax.text(best_layer_idx + offset, data['best_acc'] + 0.01,
                f"Best\n{data['best_acc']:.3f}",
                ha='center', va='bottom',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[model_name], alpha=0.3))

    # Styling
    ax.set_xlabel('Layer Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Balanced Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f'Layer-wise Performance Comparison (Bar Chart)\n{dataset_name.title()} Dataset',
                fontsize=16, fontweight='bold', pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels(all_layers)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Move legend outside plot area to avoid overlap with bars
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12, framealpha=0.9)
    ax.set_ylim([0, 1.0])

    # Add horizontal line at 0.5 for reference
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved: {output_path.name}")


def generate_comparison_visualizations(
    all_results: Dict,
    layer_data: Dict,
    output_dir: Path,
    dataset_name: str,
    logger
):
    """Generate bar chart comparisons."""
    logger.info("\nGenerating comparison visualizations...")

    viz = create_default_visualizer()

    # 1. Feature Type Comparison (for each classifier)
    for clf_name in ['knn', 'linear_probe', 'svm_linear', 'random_forest']:
        feature_comparison = {}

        for feat_name, feat_results in all_results.items():
            if clf_name in feat_results:
                feature_comparison[feat_name] = feat_results[clf_name]

        if feature_comparison:
            save_path = output_dir / f"feature_comparison_{clf_name}.png"
            viz.plot_feature_comparison(
                feature_comparison,
                metrics=['accuracy', 'balanced_accuracy', 'macro_f1'],
                title=f"Feature Comparison ({clf_name.replace('_', ' ').title()})",
                save_path=str(save_path)
            )
            logger.info(f"  Saved: {save_path.name}")

    # 2. Classifier Comparison (for each feature type)
    for feat_name, feat_results in all_results.items():
        classifier_comparison = {}

        for clf_name, clf_results in feat_results.items():
            classifier_comparison[clf_name] = clf_results

        if classifier_comparison:
            save_path = output_dir / f"classifier_comparison_{feat_name}.png"
            viz.plot_classifier_comparison(
                classifier_comparison,
                metrics=['accuracy', 'balanced_accuracy', 'macro_f1'],
                title=f"Classifier Comparison ({feat_name.upper()})",
                save_path=str(save_path)
            )
            logger.info(f"  Saved: {save_path.name}")

    # 3. Combined Layer Comparison
    if layer_data:
        save_path = output_dir / "layer_comparison_combined.png"
        plot_combined_layer_comparison(layer_data, save_path, dataset_name, logger)


def save_summary_report(all_results: Dict, output_path: Path, logger):
    """Save comprehensive text summary."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE EVALUATION SUMMARY\n")
        f.write(f"Generated: {get_timestamp()}\n")
        f.write("="*80 + "\n\n")

        for feat_name, feat_results in all_results.items():
            f.write(f"\n{feat_name.upper()}\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Classifier':<20} {'Accuracy':<12} {'Balanced Acc':<15} {'Macro F1':<12}\n")
            f.write("-"*80 + "\n")

            for clf_name, clf_results in feat_results.items():
                f.write(f"{clf_name:<20} "
                       f"{clf_results['accuracy']:<12.4f} "
                       f"{clf_results['balanced_accuracy']:<15.4f} "
                       f"{clf_results['macro_f1']:<12.4f}\n")
            f.write("\n")

    logger.info(f"Saved summary report: {output_path}")


def main():
    """Main execution."""
    # Setup logging
    logger = setup_logger("comprehensive_evaluation", log_level="INFO")

    logger.info("="*80)
    logger.info("Script 05 (Comprehensive): Complete Feature Evaluation")
    logger.info("="*80)

    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    embeddings_dir = Path(config['paths']['embeddings_dir'])
    output_dir = Path(config['paths']['reports_dir']) / "comprehensive_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_size = config['knn']['test_size']
    random_state = config['random_seed']
    pooling_method = 'mean'

    # Process each dataset
    for dataset_key in ['macaque', 'zebra_finch']:
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"# DATASET: {dataset_key.upper()}")
        logger.info(f"{'#'*80}\n")

        dataset_output_dir = output_dir / dataset_key
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        all_results = {}

        # 1. Load and evaluate handcrafted features
        logger.info("\n[1/2] Evaluating Handcrafted Features...")
        handcrafted = load_handcrafted_features(embeddings_dir, dataset_key, pooling_method)

        if handcrafted:
            for feat_type in ['mfcc', 'spectral', 'combined']:
                if feat_type in handcrafted:
                    results = evaluate_feature_type(
                        feature_name=f"{feat_type}_features",
                        features=handcrafted[feat_type]['features'],
                        labels=handcrafted[feat_type]['labels'],
                        test_size=test_size,
                        random_state=random_state,
                        config=config,
                        logger=logger
                    )
                    all_results[feat_type] = results
        else:
            logger.warning("No handcrafted features found, skipping...")

        # 2. Load and evaluate wav2vec features (best layer only)
        logger.info("\n[2/2] Evaluating Wav2vec Features...")

        layer_data = {}  # Store layer comparison data for visualization

        for model_name in ['wav2vec2_base', 'wav2vec2_xlsr']:
            embeddings_file = embeddings_dir / f"{dataset_key}_{model_name}_pooled.npz"

            if embeddings_file.exists():
                logger.info(f"\nModel: {model_name}")

                # Find best layer
                layer_info = compare_wav2vec_layers(
                    embeddings_file,
                    pooling_method,
                    test_size,
                    random_state,
                    config,
                    logger
                )

                if layer_info:
                    # Store layer data for visualization
                    layer_data[model_name] = layer_info

                    # Evaluate best layer with all classifiers
                    best_layer = layer_info['best_layer']
                    logger.info(f"\nEvaluating best layer ({best_layer}) with all classifiers...")

                    results = evaluate_feature_type(
                        feature_name=f"wav2vec_{model_name}_layer{best_layer}",
                        features=layer_info['best_features'],
                        labels=layer_info['labels'],
                        test_size=test_size,
                        random_state=random_state,
                        config=config,
                        logger=logger
                    )
                    all_results[f"wav2vec_{model_name}"] = results
            else:
                logger.warning(f"Embeddings not found: {embeddings_file}")

        # 3. Generate visualizations
        if all_results:
            generate_comparison_visualizations(all_results, layer_data, dataset_output_dir, dataset_key, logger)

            # Save summary report
            summary_path = dataset_output_dir / "evaluation_summary.txt"
            save_summary_report(all_results, summary_path, logger)

        # Clean up
        gc.collect()
        if PSUTIL_AVAILABLE:
            log_memory_usage(logger, "after dataset")

    logger.info("\n" + "="*80)
    logger.info("✓ Comprehensive evaluation complete!")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("\nGenerated files:")
    logger.info("  - feature_comparison_*.png (bar charts)")
    logger.info("  - classifier_comparison_*.png (bar charts)")
    logger.info("  - evaluation_summary.txt (text report)")


if __name__ == "__main__":
    main()
