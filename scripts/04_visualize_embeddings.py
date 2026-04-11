#!/usr/bin/env python3
"""
Script 04: Visualize Embeddings
Creates visualizations of embeddings using PCA, LDA, t-SNE, and UMAP.
"""

import yaml
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.feature_pooling import FeaturePooler
from src.evaluation.visualizer import EmbeddingVisualizer
from src.utils.logging_utils import setup_logger, get_timestamp


def visualize_model_dataset(
    config,
    model_name,
    dataset_name,
    embeddings_path,
    output_dir,
    visualizer,
    pooler,
    logger
):
    """
    Visualize embeddings for a specific model and dataset.

    Args:
        config: Configuration dictionary
        model_name: Name of the model
        dataset_name: Name of the dataset
        embeddings_path: Path to pooled embeddings
        output_dir: Output directory for figures
        visualizer: EmbeddingVisualizer instance
        pooler: FeaturePooler instance
        logger: Logger instance
    """
    logger.info(f"\nVisualizing {model_name} on {dataset_name}...")
    logger.info("-" * 80)

    # Load pooled features
    pooled_features = pooler.load_pooled_features(str(embeddings_path))

    # Get metadata
    metadata = pooled_features['mean']['metadata']
    num_layers = len(metadata['layers'])

    logger.info(f"  Loaded {len(metadata['file_paths'])} samples")
    logger.info(f"  Number of layers: {num_layers}")
    logger.info(f"  Pooling methods: {list(pooled_features.keys())}")

    # Create output directory
    model_output_dir = output_dir / dataset_name / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize with mean pooling (default)
    pooling_method = 'mean'

    # Visualize each layer individually
    logger.info(f"\n  Generating individual layer visualizations...")
    for layer_idx in metadata['layers']:
        features, labels = pooler.get_pooled_arrays(
            pooled_features,
            layer_idx,
            pooling_method
        )

        visualizer.visualize_layer(
            features,
            labels,
            layer_idx,
            model_name,
            dataset_name,
            str(model_output_dir / "individual_layers")
        )

    # Create layer comparison grids
    logger.info(f"\n  Generating layer comparison grids...")

    # Prepare features per layer
    features_per_layer = {}
    for layer_idx in metadata['layers']:
        features, labels = pooler.get_pooled_arrays(
            pooled_features,
            layer_idx,
            pooling_method
        )
        features_per_layer[layer_idx] = features

    # Create grids for each visualization method
    for method in ['pca', 'lda', 'tsne', 'umap']:
        if config['visualization']['methods'][method]['enabled']:
            try:
                grid_path = model_output_dir / "comparison_grids" / f"layer_comparison_{method}.{config['visualization']['format']}"
                visualizer.create_layer_comparison_grid(
                    features_per_layer,
                    labels,
                    model_name,
                    dataset_name,
                    str(grid_path),
                    method=method
                )
            except Exception as e:
                logger.error(f"    Error creating {method} grid: {e}")

    logger.info(f"  Visualizations saved to: {model_output_dir}")


def main():
    """Main function to visualize embeddings."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    logger = setup_logger("VisualizeEmbeddings", config['experiment']['log_level'])
    logger.info("=" * 80)
    logger.info("SCRIPT 04: VISUALIZE EMBEDDINGS")
    logger.info("=" * 80)

    # Create visualizer and pooler
    visualizer = EmbeddingVisualizer(config, config['experiment']['log_level'])
    pooler = FeaturePooler(config, config['experiment']['log_level'])

    # Set paths
    embeddings_dir = Path(config['paths']['embeddings_dir'])
    output_dir = Path(config['paths']['figures_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize all combinations of models and datasets
    models = ['wav2vec2_base', 'wav2vec2_xlsr']
    datasets = ['macaque', 'zebra_finch']

    for model_name in models:
        for dataset_name in datasets:
            logger.info("\n" + "=" * 80)
            logger.info(f"VISUALIZING: {model_name} - {dataset_name}")
            logger.info("=" * 80)

            embeddings_path = embeddings_dir / f"{dataset_name}_{model_name}_pooled.npz"

            if not embeddings_path.exists():
                logger.warning(f"Embeddings not found: {embeddings_path}")
                continue

            try:
                visualize_model_dataset(
                    config,
                    model_name,
                    dataset_name,
                    embeddings_path,
                    output_dir,
                    visualizer,
                    pooler,
                    logger
                )
            except Exception as e:
                logger.error(f"Error visualizing {model_name} on {dataset_name}: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("Visualization complete!")
    logger.info(f"Figures saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
