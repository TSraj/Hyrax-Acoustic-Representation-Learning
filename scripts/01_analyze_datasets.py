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

    # Get active datasets from config
    active_datasets = config['datasets'].get('active', [])
    logger.info(f"\nAnalyzing {len(active_datasets)} datasets: {', '.join(active_datasets)}")

    # Create analyzer
    analyzer = DatasetAnalyzer(config, config['experiment']['log_level'])

    # Output directory
    output_dir = Path(config['paths']['reports_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()

    # Analyze each dataset
    all_stats = {}
    for dataset_key in active_datasets:
        logger.info(f"\n{'='*80}")
        logger.info(f"ANALYZING: {dataset_key.upper()}")
        logger.info(f"{'='*80}")

        dataset_config = config['datasets'].get(dataset_key)
        if not dataset_config:
            logger.warning(f"Dataset '{dataset_key}' not found in config, skipping...")
            continue

        dataset_path = Path(dataset_config['path'])
        if not dataset_path.exists():
            logger.warning(f"Dataset path does not exist: {dataset_path}, skipping...")
            continue

        # Analyze dataset using generic method
        try:
            stats = analyzer.analyze_generic_dataset(
                dataset_path=str(dataset_path),
                dataset_name=dataset_config['name']
            )
            all_stats[dataset_key] = stats

            # Save individual analysis
            analyzer.save_analysis(
                stats,
                str(output_dir / f"{dataset_key}_analysis_{timestamp}.json")
            )
            logger.info(f"✓ Analysis saved for {dataset_key}")

        except Exception as e:
            logger.error(f"Error analyzing {dataset_key}: {e}")
            continue

    # Generate combined report for all datasets
    if all_stats:
        report_path = output_dir / f"dataset_analysis_report_{timestamp}.txt"
        analyzer.generate_combined_report(all_stats, str(report_path))
        logger.info(f"\n✓ Combined report saved to: {report_path}")

    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Dataset analysis complete! Analyzed {len(all_stats)}/{len(active_datasets)} datasets")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
