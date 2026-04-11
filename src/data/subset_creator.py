"""
Subset creator for generating small test subsets from datasets.
"""

import random
import shutil
import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

from src.utils.logging_utils import setup_logger


class SubsetCreator:
    """Creates balanced subsets from audio datasets for rapid prototyping."""

    def __init__(self, config: dict, log_level: str = "INFO"):
        """
        Initialize subset creator.

        Args:
            config: Configuration dictionary
            log_level: Logging level
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__, log_level)
        self.random_seed = config.get('random_seed', 42)
        random.seed(self.random_seed)

    def create_macaque_subset(self, output_dir: str, samples_per_individual: int = 20) -> Dict:
        """
        Create a balanced subset of Macaque vocalizations.

        Args:
            output_dir: Directory to save subset
            samples_per_individual: Number of samples per individual

        Returns:
            Dictionary containing subset metadata
        """
        self.logger.info(f"Creating Macaque subset with {samples_per_individual} samples per individual...")

        dataset_path = Path(self.config['datasets']['macaque']['path'])
        individuals = self.config['datasets']['macaque']['individuals']
        output_path = Path(output_dir) / "macaque"
        output_path.mkdir(parents=True, exist_ok=True)

        subset_info = {
            'dataset': 'macaque',
            'samples_per_individual': samples_per_individual,
            'individuals': {},
            'total_files': 0,
            'random_seed': self.random_seed
        }

        for individual in individuals:
            individual_dir = dataset_path / individual
            if not individual_dir.exists():
                self.logger.warning(f"Individual directory not found: {individual}")
                continue

            # Get all audio files
            audio_files = list(individual_dir.glob("*.wav"))

            if len(audio_files) == 0:
                self.logger.warning(f"No audio files found for {individual}")
                continue

            # Sample files
            num_samples = min(samples_per_individual, len(audio_files))
            selected_files = random.sample(audio_files, num_samples)

            # Create output directory for this individual
            individual_output_dir = output_path / individual
            individual_output_dir.mkdir(parents=True, exist_ok=True)

            # Copy selected files
            copied_files = []
            for file_path in selected_files:
                dest_path = individual_output_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                copied_files.append(file_path.name)

            subset_info['individuals'][individual] = {
                'num_files': num_samples,
                'files': copied_files
            }
            subset_info['total_files'] += num_samples

            self.logger.info(f"  {individual}: {num_samples} files copied")

        # Save subset metadata
        metadata_path = output_path / "subset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(subset_info, f, indent=2)

        self.logger.info(f"Macaque subset created: {subset_info['total_files']} total files")
        return subset_info

    def create_zebra_finch_subset(self, output_dir: str, samples_per_individual: int = 20) -> Dict:
        """
        Create a balanced subset of Zebra Finch vocalizations.

        Args:
            output_dir: Directory to save subset
            samples_per_individual: Number of samples per individual

        Returns:
            Dictionary containing subset metadata
        """
        self.logger.info(f"Creating Zebra Finch subset with {samples_per_individual} samples per individual...")

        dataset_path = Path(self.config['datasets']['zebra_finch']['path'])
        adult_path = dataset_path / self.config['datasets']['zebra_finch']['subdirs']['adult']
        output_path = Path(output_dir) / "zebra_finch"
        output_path.mkdir(parents=True, exist_ok=True)

        subset_info = {
            'dataset': 'zebra_finch',
            'samples_per_individual': samples_per_individual,
            'individuals': {},
            'total_files': 0,
            'random_seed': self.random_seed
        }

        if not adult_path.exists():
            self.logger.error(f"Adult vocalization directory not found: {adult_path}")
            return subset_info

        # Get all audio files
        audio_files = list(adult_path.glob("*.wav"))

        # Group files by individual (first part of filename before underscore)
        individual_files = defaultdict(list)
        for file_path in audio_files:
            individual_id = file_path.stem.split('_')[0]
            individual_files[individual_id].append(file_path)

        # Select top individuals with most samples
        sorted_individuals = sorted(individual_files.items(), key=lambda x: len(x[1]), reverse=True)
        top_individuals = sorted_individuals[:10]  # Select top 10 individuals

        # Create subset directory
        adult_output_dir = output_path / "AdultVocalizations"
        adult_output_dir.mkdir(parents=True, exist_ok=True)

        for individual_id, files in top_individuals:
            # Sample files
            num_samples = min(samples_per_individual, len(files))
            selected_files = random.sample(files, num_samples)

            # Copy selected files
            copied_files = []
            for file_path in selected_files:
                dest_path = adult_output_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                copied_files.append(file_path.name)

            subset_info['individuals'][individual_id] = {
                'num_files': num_samples,
                'files': copied_files
            }
            subset_info['total_files'] += num_samples

            self.logger.info(f"  {individual_id}: {num_samples} files copied")

        # Save subset metadata
        metadata_path = output_path / "subset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(subset_info, f, indent=2)

        self.logger.info(f"Zebra Finch subset created: {subset_info['total_files']} total files")
        return subset_info

    def create_all_subsets(self, output_dir: str = None) -> Dict:
        """
        Create subsets for all datasets.

        Args:
            output_dir: Base directory for subsets (optional)

        Returns:
            Dictionary containing all subset metadata
        """
        if output_dir is None:
            output_dir = self.config['paths']['processed_dir'] + "/subsets"

        samples_per_individual = self.config['subset']['samples_per_individual']

        all_subsets = {
            'macaque': self.create_macaque_subset(output_dir, samples_per_individual),
            'zebra_finch': self.create_zebra_finch_subset(output_dir, samples_per_individual)
        }

        # Save combined metadata
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metadata_path = output_path / "all_subsets_metadata.json"

        with open(metadata_path, 'w') as f:
            json.dump(all_subsets, f, indent=2)

        self.logger.info(f"All subsets created in {output_dir}")
        return all_subsets
