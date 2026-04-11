#!/usr/bin/env python3
"""
Script 03: Extract Embeddings
Extracts wav2vec embeddings from preprocessed audio using both base and XLSR models.
"""

import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.wav2vec_extractor import Wav2VecFeatureExtractor
from src.models.feature_pooling import FeaturePooler
from src.utils.logging_utils import setup_logger, get_timestamp


def extract_and_pool_features(config, model_name, data_dir, output_dir, logger):
    """
    Extract and pool features for a given model.

    Args:
        config: Configuration dictionary
        model_name: Name of the model ('wav2vec2_base' or 'wav2vec2_xlsr')
        data_dir: Directory containing preprocessed audio
        output_dir: Directory to save embeddings
        logger: Logger instance
    """
    logger.info(f"\nExtracting features with {model_name}...")
    logger.info("-" * 80)

    # Create feature extractor
    extractor = Wav2VecFeatureExtractor(
        model_name,
        config,
        device=config['feature_extraction']['device'],
        log_level=config['experiment']['log_level']
    )

    # Extract features from Macaque subset
    logger.info("\nProcessing Macaque dataset...")
    macaque_features = extractor.extract_features_from_dataset(
        str(data_dir / "macaque"),
        extract_all_layers=config['feature_extraction']['extract_all_layers']
    )

    # Save raw features
    macaque_output = output_dir / f"macaque_{model_name}_features.npz"
    extractor.save_features(macaque_features, str(macaque_output))

    # Extract features from Zebra Finch subset
    logger.info("\nProcessing Zebra Finch dataset...")
    zebra_features = extractor.extract_features_from_dataset(
        str(data_dir / "zebra_finch" / "AdultVocalizations"),
        extract_all_layers=config['feature_extraction']['extract_all_layers']
    )

    # Save raw features
    zebra_output = output_dir / f"zebra_finch_{model_name}_features.npz"
    extractor.save_features(zebra_features, str(zebra_output))

    # Pool features
    logger.info("\nPooling features...")
    pooler = FeaturePooler(config, config['experiment']['log_level'])

    # Pool Macaque features
    macaque_pooled = pooler.pool_dataset_features(macaque_features)
    pooler.save_pooled_features(
        macaque_pooled,
        str(output_dir / f"macaque_{model_name}_pooled.npz")
    )

    # Pool Zebra Finch features
    zebra_pooled = pooler.pool_dataset_features(zebra_features)
    pooler.save_pooled_features(
        zebra_pooled,
        str(output_dir / f"zebra_finch_{model_name}_pooled.npz")
    )

    logger.info(f"\nFeature extraction complete for {model_name}")
    return macaque_pooled, zebra_pooled


def main():
    """Main function to extract embeddings."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    logger = setup_logger("ExtractEmbeddings", config['experiment']['log_level'])
    logger.info("=" * 80)
    logger.info("SCRIPT 03: EXTRACT EMBEDDINGS")
    logger.info("=" * 80)

    # Set paths
    if config['subset']['enabled']:
        data_dir = Path(config['paths']['processed_dir']) / "preprocessed_subsets"
    else:
        data_dir = Path(config['paths']['processed_dir']) / "preprocessed_full"

    output_dir = Path(config['paths']['embeddings_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = get_timestamp()

    # Extract features with wav2vec2-base
    logger.info("\n" + "=" * 80)
    logger.info("MODEL 1: wav2vec2-base")
    logger.info("=" * 80)

    base_macaque, base_zebra = extract_and_pool_features(
        config,
        'wav2vec2_base',
        data_dir,
        output_dir,
        logger
    )

    # Extract features with wav2vec2-xlsr
    logger.info("\n" + "=" * 80)
    logger.info("MODEL 2: wav2vec2-xls-r")
    logger.info("=" * 80)

    xlsr_macaque, xlsr_zebra = extract_and_pool_features(
        config,
        'wav2vec2_xlsr',
        data_dir,
        output_dir,
        logger
    )

    logger.info("\n" + "=" * 80)
    logger.info("Feature extraction complete for both models!")
    logger.info(f"Embeddings saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
