"""
Script 03b: Extract Handcrafted Features

Extract MFCC and spectral features from preprocessed audio as baselines
for comparison with wav2vec learned representations.

This script extracts traditional acoustic features that have been used
for decades in audio classification tasks.

Usage:
    python scripts/03b_extract_handcrafted_features.py

Input:
    - Preprocessed audio files from Script 02

Output:
    - MFCC features saved as .npz files
    - Spectral features saved as .npz files
    - Combined handcrafted features saved as .npz files

Author: Raj
Date: 2026-04-11
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import logging
from typing import Dict, List, Tuple
from tqdm import tqdm
import gc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.mfcc_extractor import create_default_mfcc_extractor
from src.models.spectral_extractor import create_default_spectral_extractor
from src.models.prosodic_extractor import create_default_prosodic_extractor
from src.utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def collect_audio_files(
    data_dir: Path,
    dataset_key: str,
    dataset_config: dict,
    subset_enabled: bool
) -> Tuple[List[Path], List[str]]:
    """
    Collect audio files and their labels.

    Args:
        data_dir: Root data directory
        dataset_key: Config key for the dataset (e.g. 'macaque', 'wetlands_bird')
        dataset_config: Dataset configuration
        subset_enabled: Whether subset mode is enabled

    Returns:
        audio_paths: List of audio file paths
        labels: List of corresponding labels
    """
    audio_paths = []
    labels = []

    # Use dataset_key directly — matches the folder name created by script 02
    if subset_enabled:
        subset_dir = data_dir / "processed" / "preprocessed_subsets" / dataset_key
    else:
        subset_dir = data_dir / "processed" / "preprocessed_full" / dataset_key

    if not subset_dir.exists():
        logger.warning(f"Directory not found: {subset_dir}")
        return [], []

    if 'individuals' in dataset_config:
        # Explicit individual list in config (macaque-style)
        logger.info(f"Using explicit individual-folder structure for {dataset_config.get('name', 'dataset')}")

        for individual in dataset_config['individuals']:
            individual_dir = subset_dir / individual
            if not individual_dir.exists():
                logger.warning(f"Individual directory not found: {individual_dir}")
                continue

            for audio_file in sorted(individual_dir.glob("**/*.wav")):
                audio_paths.append(audio_file)
                labels.append(individual)

    elif 'subdirs' in dataset_config:
        # Subdirectory with individual IDs in filenames (zebra_finch-style)
        logger.info(f"Using subdirectory structure for {dataset_config.get('name', 'dataset')}")

        subdir_name = dataset_config['subdirs'].get('adult', 'AdultVocalizations')
        files_dir = subset_dir / subdir_name

        if not files_dir.exists():
            logger.warning(f"Subdirectory not found: {files_dir}")
            return [], []

        for audio_file in sorted(files_dir.glob("**/*.wav")):
            individual = audio_file.stem.split('_')[0] if '_' in audio_file.stem else audio_file.stem
            audio_paths.append(audio_file)
            labels.append(individual)

    else:
        # Generic: each subdirectory is an individual (anuraset, bengalese_finch, etc.)
        logger.info(f"Using generic subdirectory structure for {dataset_config.get('name', 'dataset')}")

        subdirs = sorted([d for d in subset_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
        if not subdirs:
            logger.warning(f"No subdirectories found in {subset_dir}")
            return [], []

        for individual_dir in subdirs:
            for audio_file in sorted(individual_dir.glob("**/*.wav")):
                audio_paths.append(audio_file)
                labels.append(individual_dir.name)

    logger.info(f"Collected {len(audio_paths)} audio files with {len(set(labels))} unique labels")

    return audio_paths, labels


def extract_features_for_dataset(
    dataset_key: str,
    dataset_name: str,
    dataset_config: dict,
    data_dir: Path,
    output_dir: Path,
    sample_rate: int,
    subset_enabled: bool,
    pooling_methods: List[str],
    max_audio_duration: float = None,
) -> None:
    """
    Extract handcrafted features for one dataset.

    Args:
        dataset_key: Config key for the dataset (e.g. 'macaque')
        dataset_name: Human-readable name of the dataset
        dataset_config: Dataset configuration
        data_dir: Root data directory
        output_dir: Output directory for features
        sample_rate: Target sample rate
        subset_enabled: Whether subset mode is enabled
        pooling_methods: List of pooling methods to use
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"{'='*60}")

    # Collect audio files
    audio_paths, labels = collect_audio_files(data_dir, dataset_key, dataset_config, subset_enabled)

    if len(audio_paths) == 0:
        logger.warning(f"No audio files found for {dataset_name}, skipping...")
        return

    # Create extractors
    logger.info("Initializing feature extractors...")
    mfcc_extractor = create_default_mfcc_extractor(sample_rate=sample_rate, max_duration=max_audio_duration)
    spectral_extractor = create_default_spectral_extractor(sample_rate=sample_rate, max_duration=max_audio_duration)
    prosodic_extractor = create_default_prosodic_extractor(sample_rate=sample_rate, max_duration=max_audio_duration)

    # Extract features with different pooling methods
    for pooling_method in pooling_methods:
        logger.info(f"\nExtracting features with pooling method: {pooling_method}")

        # Extract MFCC features
        logger.info("Extracting MFCC features...")
        mfcc_features, mfcc_successful_paths = mfcc_extractor.extract_batch(
            audio_paths,
            pooling_method=pooling_method,
            verbose=True
        )

        # Extract spectral features
        logger.info("Extracting spectral features...")
        spectral_features, spectral_successful_paths = spectral_extractor.extract_batch(
            audio_paths,
            pooling_method=pooling_method,
            verbose=True
        )

        # Extract prosodic features
        logger.info("Extracting prosodic features...")
        prosodic_features, prosodic_successful_paths = prosodic_extractor.extract_batch(
            audio_paths,
            pooling_method=pooling_method,
            verbose=True
        )

        # Find common successful paths (files that succeeded in ALL extractors)
        common_paths = set(mfcc_successful_paths) & set(spectral_successful_paths) & set(prosodic_successful_paths)
        common_paths = sorted(common_paths)  # Keep consistent order

        logger.info(f"Files successful in all extractors: {len(common_paths)}/{len(audio_paths)}")

        # Filter features to only include common successful files
        mfcc_indices = [i for i, path in enumerate(mfcc_successful_paths) if path in common_paths]
        spectral_indices = [i for i, path in enumerate(spectral_successful_paths) if path in common_paths]
        prosodic_indices = [i for i, path in enumerate(prosodic_successful_paths) if path in common_paths]

        mfcc_features_filtered = mfcc_features[mfcc_indices]
        spectral_features_filtered = spectral_features[spectral_indices]
        prosodic_features_filtered = prosodic_features[prosodic_indices]

        # Combine features (MFCC + Spectral + Prosodic)
        combined_features = np.concatenate([
            mfcc_features_filtered,
            spectral_features_filtered,
            prosodic_features_filtered
        ], axis=1)

        # Filter labels to match common successful paths
        successful_labels = [labels[i] for i, path in enumerate(audio_paths)
                            if str(path) in common_paths]
        successful_paths = common_paths

        # Save features
        output_subdir = output_dir / dataset_name
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Save MFCC features
        mfcc_output = output_subdir / f"mfcc_features_{pooling_method}.npz"
        np.savez_compressed(
            mfcc_output,
            features=mfcc_features_filtered,
            labels=np.array(successful_labels),
            file_paths=successful_paths,
            feature_names=mfcc_extractor.get_feature_names(),
            pooling_method=pooling_method,
            feature_type='mfcc'
        )
        logger.info(f"Saved MFCC features to: {mfcc_output}")

        # Save spectral features
        spectral_output = output_subdir / f"spectral_features_{pooling_method}.npz"
        np.savez_compressed(
            spectral_output,
            features=spectral_features_filtered,
            labels=np.array(successful_labels),
            file_paths=successful_paths,
            feature_names=spectral_extractor.get_feature_names(),
            pooling_method=pooling_method,
            feature_type='spectral'
        )
        logger.info(f"Saved spectral features to: {spectral_output}")

        # Save prosodic features
        prosodic_output = output_subdir / f"prosodic_features_{pooling_method}.npz"
        np.savez_compressed(
            prosodic_output,
            features=prosodic_features_filtered,
            labels=np.array(successful_labels),
            file_paths=successful_paths,
            feature_names=prosodic_extractor.get_feature_names(),
            pooling_method=pooling_method,
            feature_type='prosodic'
        )
        logger.info(f"Saved prosodic features to: {prosodic_output}")

        # Save combined features
        combined_output = output_subdir / f"handcrafted_combined_{pooling_method}.npz"
        all_feature_names = (mfcc_extractor.get_feature_names() +
                            spectral_extractor.get_feature_names() +
                            prosodic_extractor.get_feature_names())
        np.savez_compressed(
            combined_output,
            features=combined_features,
            labels=np.array(successful_labels),
            file_paths=successful_paths,
            feature_names=all_feature_names,
            pooling_method=pooling_method,
            feature_type='handcrafted_combined'
        )
        logger.info(f"Saved combined features to: {combined_output}")

        # Log feature dimensions
        logger.info(f"\nFeature dimensions:")
        logger.info(f"  MFCC: {mfcc_features_filtered.shape}")
        logger.info(f"  Spectral: {spectral_features_filtered.shape}")
        logger.info(f"  Prosodic: {prosodic_features_filtered.shape}")
        logger.info(f"  Combined: {combined_features.shape}")

        # Clean up
        del mfcc_features, spectral_features, mfcc_features_filtered, spectral_features_filtered, combined_features
        gc.collect()

    logger.info(f"\nCompleted feature extraction for {dataset_name}")


def main():
    """Main execution function."""
    # Setup logging
    logger = setup_logger("extract_handcrafted_features", log_level="INFO")

    logger.info("="*60)
    logger.info("Script 03b: Extract Handcrafted Features")
    logger.info("="*60)

    # Load configuration
    config_path = project_root / "config" / "config.yaml"
    config = load_config(config_path)

    # Extract relevant config
    data_dir = Path(config['paths']['data_dir'])
    output_dir = Path(config['paths']['embeddings_dir'])
    sample_rate = config['preprocessing']['target_sample_rate']
    subset_enabled = config['subset']['enabled']
    pooling_methods = [method for method in config['pooling']['methods']]
    max_audio_duration = config.get('feature_extraction', {}).get('max_audio_duration', None)

    logger.info(f"\nConfiguration:")
    logger.info(f"  Data directory: {data_dir}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Sample rate: {sample_rate}")
    logger.info(f"  Subset enabled: {subset_enabled}")
    logger.info(f"  Pooling methods: {pooling_methods}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all active datasets from config
    active_datasets = config['datasets'].get('active', [])
    datasets = config['datasets']

    logger.info(f"\nProcessing {len(active_datasets)} datasets: {', '.join(active_datasets)}")

    for dataset_key in active_datasets:
        dataset_config = datasets.get(dataset_key)
        if not dataset_config:
            logger.warning(f"Dataset '{dataset_key}' not found in config, skipping...")
            continue

        extract_features_for_dataset(
            dataset_key=dataset_key,
            dataset_name=dataset_config.get('name', dataset_key),
            dataset_config=dataset_config,
            data_dir=data_dir,
            output_dir=output_dir,
            sample_rate=sample_rate,
            subset_enabled=subset_enabled,
            pooling_methods=pooling_methods,
            max_audio_duration=max_audio_duration,
        )
        gc.collect()

    logger.info("\n" + "="*60)
    logger.info("Handcrafted feature extraction complete!")
    logger.info("="*60)
    logger.info(f"\nFeatures saved to: {output_dir}")
    logger.info("\nNext steps:")
    logger.info("  1. Run Script 04: Visualize embeddings (including handcrafted features)")
    logger.info("  2. Run Script 05: Evaluate and compare all features")


if __name__ == "__main__":
    main()
