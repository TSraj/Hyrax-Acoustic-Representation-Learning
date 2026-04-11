#!/usr/bin/env python3
"""
Script 02: Create Subsets and Preprocess
Creates small test subsets and preprocesses audio to standard format.
"""

import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.subset_creator import SubsetCreator
from src.data.audio_preprocessor import AudioPreprocessor
from src.utils.logging_utils import setup_logger, get_timestamp


def main():
    """Main function to create subsets and preprocess audio."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    logger = setup_logger("CreateSubsetsPreprocess", config['experiment']['log_level'])
    logger.info("=" * 80)
    logger.info("SCRIPT 02: CREATE SUBSETS AND PREPROCESS")
    logger.info("=" * 80)

    # Create subsets
    if config['subset']['enabled']:
        logger.info("\nSTEP 1: Creating subsets...")
        logger.info("-" * 80)

        subset_creator = SubsetCreator(config, config['experiment']['log_level'])
        subset_output_dir = Path(config['paths']['processed_dir']) / "subsets"

        subset_metadata = subset_creator.create_all_subsets(str(subset_output_dir))

        logger.info(f"\nSubsets created:")
        logger.info(f"  Macaque: {subset_metadata['macaque']['total_files']} files")
        logger.info(f"  Zebra Finch: {subset_metadata['zebra_finch']['total_files']} files")
    else:
        logger.info("\nSubset creation disabled in config")
        subset_output_dir = Path(config['paths']['data_dir'])

    # Preprocess audio
    logger.info("\n\nSTEP 2: Preprocessing audio...")
    logger.info("-" * 80)

    preprocessor = AudioPreprocessor(config, config['experiment']['log_level'])

    if config['subset']['enabled']:
        # Preprocess subsets
        preprocess_output_dir = Path(config['paths']['processed_dir']) / "preprocessed_subsets"
        stats = preprocessor.preprocess_all_subsets(
            str(subset_output_dir),
            str(preprocess_output_dir)
        )
    else:
        # Preprocess full datasets
        preprocess_output_dir = Path(config['paths']['processed_dir']) / "preprocessed_full"
        stats = {}

        # Macaque
        stats['macaque'] = preprocessor.preprocess_dataset(
            config['datasets']['macaque']['path'],
            str(preprocess_output_dir / "macaque")
        )

        # Zebra Finch
        stats['zebra_finch'] = preprocessor.preprocess_dataset(
            config['datasets']['zebra_finch']['path'],
            str(preprocess_output_dir / "zebra_finch")
        )

    logger.info(f"\nPreprocessing complete:")
    logger.info(f"  Macaque: {stats['macaque']['successful']}/{stats['macaque']['total_files']} successful")
    logger.info(f"  Zebra Finch: {stats['zebra_finch']['successful']}/{stats['zebra_finch']['total_files']} successful")

    logger.info("\n" + "=" * 80)
    logger.info("Subset creation and preprocessing complete!")
    logger.info(f"Preprocessed data saved to: {preprocess_output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
