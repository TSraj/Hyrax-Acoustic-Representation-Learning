"""
Dataset analyzer for inspecting audio datasets and generating statistics.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from src.utils.audio_utils import get_audio_info
from src.utils.logging_utils import setup_logger


class DatasetAnalyzer:
    """Analyzes audio datasets and generates summary statistics."""

    def __init__(self, config: dict, log_level: str = "INFO"):
        """
        Initialize dataset analyzer.

        Args:
            config: Configuration dictionary
            log_level: Logging level
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__, log_level)

    def analyze_macaque_dataset(self) -> Dict:
        """
        Analyze the Macaque vocalization dataset.

        Returns:
            Dictionary containing dataset statistics
        """
        self.logger.info("Analyzing Macaque dataset...")

        dataset_path = Path(self.config['datasets']['macaque']['path'])
        individuals = self.config['datasets']['macaque']['individuals']

        stats = {
            'dataset_name': 'Macaque Vocalizations',
            'dataset_path': str(dataset_path),
            'individuals': {},
            'total_files': 0,
            'total_duration': 0.0,
            'sample_rates': set(),
            'channels': set()
        }

        for individual in individuals:
            individual_path = dataset_path / individual
            if not individual_path.exists():
                self.logger.warning(f"Individual directory not found: {individual}")
                continue

            audio_files = list(individual_path.glob("*.wav"))
            individual_stats = {
                'num_files': len(audio_files),
                'durations': [],
                'sample_rates': [],
                'channels': []
            }

            # Sample analysis (first 10 files for speed)
            for audio_file in tqdm(audio_files[:10], desc=f"Sampling {individual}", leave=False):
                try:
                    info = get_audio_info(str(audio_file))
                    individual_stats['durations'].append(info['duration'])
                    individual_stats['sample_rates'].append(info['sample_rate'])
                    individual_stats['channels'].append(info['channels'])

                    stats['sample_rates'].add(info['sample_rate'])
                    stats['channels'].add(info['channels'])
                except Exception as e:
                    self.logger.warning(f"Error analyzing {audio_file}: {e}")

            # Compute summary statistics
            individual_stats['avg_duration'] = sum(individual_stats['durations']) / len(individual_stats['durations']) if individual_stats['durations'] else 0
            individual_stats['min_duration'] = min(individual_stats['durations']) if individual_stats['durations'] else 0
            individual_stats['max_duration'] = max(individual_stats['durations']) if individual_stats['durations'] else 0

            stats['individuals'][individual] = individual_stats
            stats['total_files'] += individual_stats['num_files']
            stats['total_duration'] += individual_stats['avg_duration'] * individual_stats['num_files']

        # Convert sets to lists for JSON serialization
        stats['sample_rates'] = sorted(list(stats['sample_rates']))
        stats['channels'] = sorted(list(stats['channels']))

        self.logger.info(f"Macaque dataset: {stats['total_files']} files across {len(individuals)} individuals")
        return stats

    def analyze_zebra_finch_dataset(self) -> Dict:
        """
        Analyze the Zebra Finch vocalization dataset.

        Returns:
            Dictionary containing dataset statistics
        """
        self.logger.info("Analyzing Zebra Finch dataset...")

        dataset_path = Path(self.config['datasets']['zebra_finch']['path'])
        adult_path = dataset_path / self.config['datasets']['zebra_finch']['subdirs']['adult']
        chick_path = dataset_path / self.config['datasets']['zebra_finch']['subdirs']['chick']

        stats = {
            'dataset_name': 'Zebra Finch Vocalizations',
            'dataset_path': str(dataset_path),
            'adult': {},
            'chick': {},
            'total_files': 0,
            'total_duration': 0.0,
            'sample_rates': set(),
            'channels': set()
        }

        # Analyze adult vocalizations
        if adult_path.exists():
            adult_files = list(adult_path.glob("*.wav"))
            stats['adult']['num_files'] = len(adult_files)

            # Extract individual IDs and call types from filenames
            individuals = set()
            call_types = set()

            for f in adult_files:
                name = f.stem
                parts = name.split('_')
                if len(parts) >= 1:
                    individuals.add(parts[0])
                if '-' in name:
                    call_type = name.split('-')[-1]
                    call_types.add(call_type)

            stats['adult']['num_individuals'] = len(individuals)
            stats['adult']['num_call_types'] = len(call_types)
            stats['adult']['individuals'] = sorted(list(individuals))[:20]  # Top 20
            stats['adult']['call_types'] = sorted(list(call_types))

            # Sample analysis
            sample_files = adult_files[:20]
            durations = []
            for audio_file in tqdm(sample_files, desc="Sampling adult", leave=False):
                try:
                    info = get_audio_info(str(audio_file))
                    durations.append(info['duration'])
                    stats['sample_rates'].add(info['sample_rate'])
                    stats['channels'].add(info['channels'])
                except Exception as e:
                    self.logger.warning(f"Error analyzing {audio_file}: {e}")

            stats['adult']['avg_duration'] = sum(durations) / len(durations) if durations else 0
            stats['total_files'] += len(adult_files)
            stats['total_duration'] += stats['adult']['avg_duration'] * len(adult_files)

        # Analyze chick vocalizations
        if chick_path.exists():
            chick_files = list(chick_path.glob("*.wav"))
            stats['chick']['num_files'] = len(chick_files)

            # Sample analysis
            sample_files = chick_files[:20]
            durations = []
            for audio_file in tqdm(sample_files, desc="Sampling chick", leave=False):
                try:
                    info = get_audio_info(str(audio_file))
                    durations.append(info['duration'])
                    stats['sample_rates'].add(info['sample_rate'])
                    stats['channels'].add(info['channels'])
                except Exception as e:
                    self.logger.warning(f"Error analyzing {audio_file}: {e}")

            stats['chick']['avg_duration'] = sum(durations) / len(durations) if durations else 0
            stats['total_files'] += len(chick_files)
            stats['total_duration'] += stats['chick']['avg_duration'] * len(chick_files)

        # Convert sets to lists
        stats['sample_rates'] = sorted(list(stats['sample_rates']))
        stats['channels'] = sorted(list(stats['channels']))

        self.logger.info(f"Zebra Finch dataset: {stats['total_files']} files (adult + chick)")
        return stats

    def save_analysis(self, stats: Dict, output_path: str):
        """
        Save analysis results to JSON file.

        Args:
            stats: Statistics dictionary
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"Analysis saved to {output_path}")

    def generate_report(self, macaque_stats: Dict, zebra_stats: Dict, output_path: str):
        """
        Generate a human-readable analysis report.

        Args:
            macaque_stats: Macaque dataset statistics
            zebra_stats: Zebra Finch dataset statistics
            output_path: Path to save report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATASET ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Macaque section
        report_lines.append("MACAQUE VOCALIZATIONS")
        report_lines.append("-" * 80)
        report_lines.append(f"Total files: {macaque_stats['total_files']}")
        report_lines.append(f"Total duration: {macaque_stats['total_duration'] / 3600:.2f} hours")
        report_lines.append(f"Sample rates: {macaque_stats['sample_rates']}")
        report_lines.append(f"Channels: {macaque_stats['channels']}")
        report_lines.append(f"\nIndividual breakdown:")
        for ind, info in macaque_stats['individuals'].items():
            report_lines.append(f"  {ind}: {info['num_files']} files, avg duration {info['avg_duration']:.2f}s")
        report_lines.append("")

        # Zebra Finch section
        report_lines.append("ZEBRA FINCH VOCALIZATIONS")
        report_lines.append("-" * 80)
        report_lines.append(f"Total files: {zebra_stats['total_files']}")
        report_lines.append(f"Total duration: {zebra_stats['total_duration'] / 3600:.2f} hours")
        report_lines.append(f"Sample rates: {zebra_stats['sample_rates']}")
        report_lines.append(f"Channels: {zebra_stats['channels']}")
        report_lines.append(f"\nAdult vocalizations:")
        report_lines.append(f"  Files: {zebra_stats['adult']['num_files']}")
        report_lines.append(f"  Individuals: {zebra_stats['adult']['num_individuals']}")
        report_lines.append(f"  Call types: {zebra_stats['adult']['num_call_types']}")
        report_lines.append(f"  Avg duration: {zebra_stats['adult']['avg_duration']:.2f}s")
        report_lines.append(f"\nChick vocalizations:")
        report_lines.append(f"  Files: {zebra_stats['chick']['num_files']}")
        report_lines.append(f"  Avg duration: {zebra_stats['chick']['avg_duration']:.2f}s")
        report_lines.append("")

        report_lines.append("=" * 80)

        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))

        self.logger.info(f"Report saved to {output_path}")

        # Also print to console
        print('\n'.join(report_lines))
