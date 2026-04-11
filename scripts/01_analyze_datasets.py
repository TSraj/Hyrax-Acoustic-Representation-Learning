#!/usr/bin/env python3
"""
Script 01: Analyze Datasets
Analyzes both Macaque and Zebra Finch datasets and generates summary reports.
"""

import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_analyzer import DatasetAnalyzer
from src.utils.logging_utils import setup_logger, get_timestamp


def main():
    """Main function to analyze datasets."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    logger = setup_logger("AnalyzeDatasets", config['experiment']['log_level'])
    logger.info("=" * 80)
    logger.info("SCRIPT 01: DATASET ANALYSIS")
    logger.info("=" * 80)

    # Create analyzer
    analyzer = DatasetAnalyzer(config, config['experiment']['log_level'])

    # Analyze Macaque dataset
    macaque_stats = analyzer.analyze_macaque_dataset()

    # Analyze Zebra Finch dataset
    zebra_stats = analyzer.analyze_zebra_finch_dataset()

    # Save individual analyses
    output_dir = Path(config['paths']['reports_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = get_timestamp()

    analyzer.save_analysis(
        macaque_stats,
        str(output_dir / f"macaque_analysis_{timestamp}.json")
    )

    analyzer.save_analysis(
        zebra_stats,
        str(output_dir / f"zebra_finch_analysis_{timestamp}.json")
    )

    # Generate combined report
    analyzer.generate_report(
        macaque_stats,
        zebra_stats,
        str(output_dir / f"dataset_analysis_report_{timestamp}.txt")
    )

    logger.info("=" * 80)
    logger.info("Dataset analysis complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
