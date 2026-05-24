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

    # Get active datasets from config
    active_datasets = config['datasets'].get('active', [])
    logger.info(f"\nProcessing {len(active_datasets)} datasets: {', '.join(active_datasets)}")

    # Determine input/output directories
    if config['subset']['enabled']:
        logger.info("\nSubset mode: ENABLED")
        logger.info(f"  Samples per individual: {config['subset']['samples_per_individual']}")
        subset_output_dir = Path(config['paths']['processed_dir']) / "subsets"
        preprocess_output_dir = Path(config['paths']['processed_dir']) / "preprocessed_subsets"
    else:
        logger.info("\nSubset mode: DISABLED (using full datasets)")
        subset_output_dir = None
        preprocess_output_dir = Path(config['paths']['processed_dir']) / "preprocessed_full"

    # Initialize preprocessor
    preprocessor = AudioPreprocessor(config, config['experiment']['log_level'])
    subset_creator = SubsetCreator(config, config['experiment']['log_level']) if config['subset']['enabled'] else None

    # Process each dataset
    all_stats = {}
    for dataset_key in active_datasets:
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING: {dataset_key.upper()}")
        logger.info(f"{'='*80}")

        dataset_config = config['datasets'].get(dataset_key)
        if not dataset_config:
            logger.warning(f"Dataset '{dataset_key}' not found in config, skipping...")
            continue

        dataset_path = Path(dataset_config['path'])
        if not dataset_path.exists():
            logger.warning(f"Dataset path does not exist: {dataset_path}, skipping...")
            continue

        try:
            if config['subset']['enabled']:
                # Create subset first
                logger.info(f"\nSTEP 1: Creating subset for {dataset_key}...")
                dataset_subset_dir = subset_output_dir / dataset_key
                subset_meta = subset_creator.create_subset_generic(
                    str(dataset_path),
                    str(dataset_subset_dir),
                    dataset_name=dataset_config['name']
                )
                logger.info(f"  Created subset: {subset_meta.get('total_files', 0)} files")

                # Preprocess subset
                logger.info(f"\nSTEP 2: Preprocessing subset for {dataset_key}...")
                dataset_preprocess_dir = preprocess_output_dir / dataset_key
                stats = preprocessor.preprocess_dataset(
                    str(dataset_subset_dir),
                    str(dataset_preprocess_dir)
                )
            else:
                # Preprocess full dataset
                logger.info(f"\nPreprocessing full dataset for {dataset_key}...")
                dataset_preprocess_dir = preprocess_output_dir / dataset_key
                stats = preprocessor.preprocess_dataset(
                    str(dataset_path),
                    str(dataset_preprocess_dir)
                )

            all_stats[dataset_key] = stats
            logger.info(f"✓ {dataset_key}: {stats['successful']}/{stats['total_files']} files successful")

        except Exception as e:
            logger.error(f"Error processing {dataset_key}: {e}")
            continue

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 80)
    total_successful = sum(s['successful'] for s in all_stats.values())
    total_files = sum(s['total_files'] for s in all_stats.values())
    logger.info(f"Total: {total_successful}/{total_files} files successfully preprocessed")
    logger.info(f"Datasets processed: {len(all_stats)}/{len(active_datasets)}")

    for dataset_key, stats in all_stats.items():
        logger.info(f"  {dataset_key}: {stats['successful']}/{stats['total_files']}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Preprocessing complete!")
    logger.info(f"Preprocessed data saved to: {preprocess_output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
