#!/usr/bin/env python3
"""
Phase 2 - Stage 1: Create Train/Validation/Test Manifests
Creates stratified 80/10/10 splits for all datasets + pooled manifest.
"""

import json
import yaml
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger


class ManifestCreator:
    """Creates train/val/test manifests for Phase 2 experiments."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.data_dir = Path(config['paths']['data_dir'])
        self.processed_dir = Path(config['paths']['processed_dir'])
        self.manifest_dir = Path(config['paths']['output_dir']) / "phase2" / "manifests"
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        # Split ratios
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

    def get_audio_files_by_individual(self, dataset_key, dataset_config):
        """
        Scan dataset and organize audio files by individual.

        Returns:
            dict: {individual_id: [list of audio file paths]}
        """
        dataset_path = Path(dataset_config['path'])
        individual_files = defaultdict(list)

        self.logger.info(f"\nScanning dataset: {dataset_key}")
        self.logger.info(f"  Path: {dataset_path}")

        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']

        # Check if individuals are in subdirectories
        subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]

        if subdirs:
            # Assume each subdirectory is an individual
            for individual_dir in subdirs:
                individual_id = individual_dir.name
                audio_files = []

                for ext in audio_extensions:
                    audio_files.extend(individual_dir.rglob(f"*{ext}"))

                if audio_files:
                    individual_files[f"{dataset_key}_{individual_id}"] = [
                        str(f.relative_to(self.data_dir)) for f in audio_files
                    ]
        else:
            # Flat structure - use filename patterns to infer individuals
            self.logger.warning(f"  Flat structure detected for {dataset_key}")
            self.logger.warning("  Assuming all files belong to one individual")
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(dataset_path.rglob(f"*{ext}"))

            if audio_files:
                individual_files[f"{dataset_key}_individual1"] = [
                    str(f.relative_to(self.data_dir)) for f in audio_files
                ]

        # Log summary
        total_files = sum(len(files) for files in individual_files.values())
        self.logger.info(f"  Found {len(individual_files)} individuals, {total_files} total files")
        for ind_id, files in individual_files.items():
            self.logger.info(f"    {ind_id}: {len(files)} files")

        return dict(individual_files)

    def create_stratified_split(self, individual_files):
        """
        Create stratified 80/10/10 split ensuring each individual is represented.

        Args:
            individual_files: dict mapping individual_id -> list of file paths

        Returns:
            tuple: (train_files, val_files, test_files, class_weights)
        """
        train_files = []
        val_files = []
        test_files = []

        # Track samples per individual for class weighting
        individual_counts = {}

        for individual_id, files in individual_files.items():
            n_files = len(files)
            individual_counts[individual_id] = n_files

            if n_files < 3:
                # Too few samples - put all in training
                self.logger.warning(f"  {individual_id}: only {n_files} files, adding all to train")
                train_files.extend([(f, individual_id) for f in files])
                continue

            # Calculate split sizes
            n_val = max(1, int(n_files * self.val_ratio))
            n_test = max(1, int(n_files * self.test_ratio))
            n_train = n_files - n_val - n_test

            # Shuffle files
            files_shuffled = np.random.permutation(files).tolist()

            # Split
            train = files_shuffled[:n_train]
            val = files_shuffled[n_train:n_train + n_val]
            test = files_shuffled[n_train + n_val:]

            train_files.extend([(f, individual_id) for f in train])
            val_files.extend([(f, individual_id) for f in val])
            test_files.extend([(f, individual_id) for f in test])

        # Calculate class weights (inverse frequency)
        total_samples = sum(individual_counts.values())
        class_weights = {}
        for individual_id, count in individual_counts.items():
            class_weights[individual_id] = total_samples / (len(individual_counts) * count)

        self.logger.info(f"\n  Split summary:")
        self.logger.info(f"    Train: {len(train_files)} files ({len(train_files)/sum([len(train_files), len(val_files), len(test_files)])*100:.1f}%)")
        self.logger.info(f"    Val:   {len(val_files)} files ({len(val_files)/sum([len(train_files), len(val_files), len(test_files)])*100:.1f}%)")
        self.logger.info(f"    Test:  {len(test_files)} files ({len(test_files)/sum([len(train_files), len(val_files), len(test_files)])*100:.1f}%)")

        return train_files, val_files, test_files, class_weights

    def create_dataset_manifest(self, dataset_key, dataset_config):
        """Create manifest for a single dataset."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Creating manifest: {dataset_key}")
        self.logger.info(f"{'='*80}")

        # Get audio files organized by individual
        individual_files = self.get_audio_files_by_individual(dataset_key, dataset_config)

        if not individual_files:
            self.logger.warning(f"No audio files found for {dataset_key}, skipping...")
            return None

        # Create stratified split
        train_files, val_files, test_files, class_weights = self.create_stratified_split(
            individual_files
        )

        # Build manifest
        manifest = {
            "dataset": dataset_key,
            "dataset_name": dataset_config['name'],
            "split_ratio": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio
            },
            "num_individuals": len(individual_files),
            "individuals": list(individual_files.keys()),
            "class_weights": class_weights,
            "train": [{"file": f, "individual": ind} for f, ind in train_files],
            "val": [{"file": f, "individual": ind} for f, ind in val_files],
            "test": [{"file": f, "individual": ind} for f, ind in test_files],
            "statistics": {
                "train_samples": len(train_files),
                "val_samples": len(val_files),
                "test_samples": len(test_files),
                "total_samples": len(train_files) + len(val_files) + len(test_files)
            }
        }

        # Save manifest
        manifest_path = self.manifest_dir / f"{dataset_key}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        self.logger.info(f"\n✓ Manifest saved: {manifest_path}")

        return manifest

    def create_pooled_manifest(self, dataset_manifests):
        """Create pooled manifest combining all datasets."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Creating POOLED manifest (all datasets combined)")
        self.logger.info(f"{'='*80}")

        pooled_train = []
        pooled_val = []
        pooled_test = []
        all_individuals = []
        combined_class_weights = {}

        for manifest in dataset_manifests:
            if manifest is None:
                continue

            pooled_train.extend(manifest['train'])
            pooled_val.extend(manifest['val'])
            pooled_test.extend(manifest['test'])
            all_individuals.extend(manifest['individuals'])
            combined_class_weights.update(manifest['class_weights'])

        # Shuffle pooled splits
        np.random.shuffle(pooled_train)
        np.random.shuffle(pooled_val)
        np.random.shuffle(pooled_test)

        pooled_manifest = {
            "dataset": "pooled",
            "dataset_name": "Pooled Multi-Dataset (All 7 Datasets)",
            "source_datasets": [m['dataset'] for m in dataset_manifests if m is not None],
            "split_ratio": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio
            },
            "num_individuals": len(all_individuals),
            "individuals": all_individuals,
            "class_weights": combined_class_weights,
            "train": pooled_train,
            "val": pooled_val,
            "test": pooled_test,
            "statistics": {
                "train_samples": len(pooled_train),
                "val_samples": len(pooled_val),
                "test_samples": len(pooled_test),
                "total_samples": len(pooled_train) + len(pooled_val) + len(pooled_test)
            }
        }

        # Save pooled manifest
        manifest_path = self.manifest_dir / "pooled_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(pooled_manifest, f, indent=2)

        self.logger.info(f"\n  Pooled dataset summary:")
        self.logger.info(f"    Source datasets: {len(pooled_manifest['source_datasets'])}")
        self.logger.info(f"    Total individuals: {len(all_individuals)}")
        self.logger.info(f"    Train: {len(pooled_train)} files")
        self.logger.info(f"    Val:   {len(pooled_val)} files")
        self.logger.info(f"    Test:  {len(pooled_test)} files")
        self.logger.info(f"\n✓ Pooled manifest saved: {manifest_path}")

        return pooled_manifest

    def create_all_manifests(self):
        """Create manifests for all active datasets + pooled."""
        active_datasets = self.config['datasets'].get('active', [])

        self.logger.info("="*80)
        self.logger.info("PHASE 2 - STAGE 1: MANIFEST CREATION")
        self.logger.info("="*80)
        self.logger.info(f"\nCreating manifests for {len(active_datasets)} datasets")
        self.logger.info(f"Split ratio: {self.train_ratio*100:.0f}% train / {self.val_ratio*100:.0f}% val / {self.test_ratio*100:.0f}% test")
        self.logger.info(f"Output directory: {self.manifest_dir}")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Create manifest for each dataset
        dataset_manifests = []
        for dataset_key in active_datasets:
            dataset_config = self.config['datasets'].get(dataset_key)
            if not dataset_config:
                self.logger.warning(f"Dataset '{dataset_key}' not found in config, skipping...")
                continue

            manifest = self.create_dataset_manifest(dataset_key, dataset_config)
            if manifest:
                dataset_manifests.append(manifest)

        # Create pooled manifest
        pooled_manifest = self.create_pooled_manifest(dataset_manifests)

        # Summary
        self.logger.info(f"\n{'='*80}")
        self.logger.info("MANIFEST CREATION COMPLETE")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"\nCreated {len(dataset_manifests)} dataset manifests + 1 pooled manifest")
        self.logger.info(f"Total manifests: {len(dataset_manifests) + 1}")
        self.logger.info(f"\nAll manifests saved to: {self.manifest_dir}")

        return dataset_manifests, pooled_manifest


def main():
    """Main function."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    logger = setup_logger("Phase2_ManifestCreator", config['experiment']['log_level'])

    # Create manifests
    creator = ManifestCreator(config, logger)
    dataset_manifests, pooled_manifest = creator.create_all_manifests()

    logger.info("\n✓ Stage 1 complete - Manifests ready for zero-shot evaluation")


if __name__ == "__main__":
    main()
