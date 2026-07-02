#!/usr/bin/env python3
"""
Phase 2 - Stage 1b: Validate Manifests
Validates created manifests without loading any models.
"""

import json
import yaml
import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger


def validate_manifest(manifest_path, data_dir, config, logger):
    """Validate a single manifest file."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Validating: {manifest_path.name}")
    logger.info(f"{'='*60}")

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Check required fields
    required_fields = ['dataset', 'train', 'val', 'test', 'individuals', 'class_weights']
    missing_fields = [f for f in required_fields if f not in manifest]

    if missing_fields:
        logger.error(f"❌ Missing required fields: {missing_fields}")
        return False

    # Check split ratios
    total_samples = len(manifest['train']) + len(manifest['val']) + len(manifest['test'])
    train_ratio = len(manifest['train']) / total_samples
    val_ratio = len(manifest['val']) / total_samples
    test_ratio = len(manifest['test']) / total_samples

    logger.info(f"\n📊 Split Statistics:")
    logger.info(f"  Train: {len(manifest['train'])} samples ({train_ratio*100:.1f}%)")
    logger.info(f"  Val:   {len(manifest['val'])} samples ({val_ratio*100:.1f}%)")
    logger.info(f"  Test:  {len(manifest['test'])} samples ({test_ratio*100:.1f}%)")
    logger.info(f"  Total: {total_samples} samples")

    # Check if ratios are approximately 80/10/10
    if not (0.75 <= train_ratio <= 0.85):
        logger.warning(f"⚠️  Train ratio {train_ratio*100:.1f}% is outside 75-85% range")
    if not (0.05 <= val_ratio <= 0.15):
        logger.warning(f"⚠️  Val ratio {val_ratio*100:.1f}% is outside 5-15% range")
    if not (0.05 <= test_ratio <= 0.15):
        logger.warning(f"⚠️  Test ratio {test_ratio*100:.1f}% is outside 5-15% range")

    # Check individual representation in each split
    logger.info(f"\n👥 Individual Representation:")
    logger.info(f"  Total individuals: {len(manifest['individuals'])}")

    train_individuals = set(item['individual'] for item in manifest['train'])
    val_individuals = set(item['individual'] for item in manifest['val'])
    test_individuals = set(item['individual'] for item in manifest['test'])

    logger.info(f"  Individuals in train: {len(train_individuals)}")
    logger.info(f"  Individuals in val:   {len(val_individuals)}")
    logger.info(f"  Individuals in test:  {len(test_individuals)}")

    # Check if any individual is missing from any split
    only_in_train = train_individuals - val_individuals - test_individuals
    if only_in_train:
        logger.warning(f"⚠️  {len(only_in_train)} individuals only in train split")

    # Check class distribution
    logger.info(f"\n📈 Class Distribution:")
    train_dist = Counter(item['individual'] for item in manifest['train'])
    val_dist = Counter(item['individual'] for item in manifest['val'])
    test_dist = Counter(item['individual'] for item in manifest['test'])

    # Find min/max samples per individual
    all_counts = list(train_dist.values()) + list(val_dist.values()) + list(test_dist.values())
    logger.info(f"  Min samples per individual: {min(all_counts) if all_counts else 0}")
    logger.info(f"  Max samples per individual: {max(all_counts) if all_counts else 0}")
    logger.info(f"  Mean samples per individual: {sum(all_counts)/len(manifest['individuals']):.1f}")

    # Check for severe class imbalance
    if max(all_counts) / min(all_counts) > 100:
        logger.warning(f"⚠️  Severe class imbalance detected (ratio: {max(all_counts)/min(all_counts):.1f}x)")
        logger.warning(f"     Class weighting is CRITICAL for this dataset")

    # Verify file paths exist AND can be loaded
    logger.info(f"\n📁 File Path & Audio Loading Validation:")
    missing_files = []
    corrupt_files = []
    short_files = []
    long_files = []
    silent_files = []
    checked_files = 0

    from src.utils.audio_utils import load_audio

    for split_name, split_data in [('train', manifest['train']), ('val', manifest['val']), ('test', manifest['test'])]:
        for item in split_data[:10]:  # Check first 10 from each split (increased from 5)
            file_path = data_dir / item['file']

            # Check if file exists
            if not file_path.exists():
                missing_files.append((split_name, item['file']))
                checked_files += 1
                continue

            # Try loading the audio file
            try:
                audio, sr = load_audio(str(file_path), target_sr=16000, mono=True)

                # Check duration
                duration = len(audio) / sr
                min_duration = config.get('preprocessing', {}).get('min_duration', 0.05)
                max_duration = config.get('preprocessing', {}).get('max_duration', 3600.0)

                if duration < min_duration:
                    short_files.append((split_name, item['file'], f"{duration:.3f}s"))
                elif duration > max_duration:
                    long_files.append((split_name, item['file'], f"{duration:.1f}s"))

                # Check amplitude (detect near-silent files)
                max_amplitude = abs(audio).max()
                if max_amplitude < 0.01:  # Less than 1% of full scale
                    silent_files.append((split_name, item['file'], f"max_amp={max_amplitude:.4f}"))

                checked_files += 1

            except Exception as e:
                corrupt_files.append((split_name, item['file'], str(e)))
                checked_files += 1

    # Report results
    logger.info(f"  Checked: {checked_files} samples (10 per split)")

    if missing_files:
        logger.error(f"❌ Missing files: {len(missing_files)}")
        for split_name, file_path in missing_files[:3]:
            logger.error(f"     [{split_name}] {file_path}")

    if corrupt_files:
        logger.error(f"❌ Corrupt/unreadable files: {len(corrupt_files)}")
        for split_name, file_path, error in corrupt_files[:3]:
            logger.error(f"     [{split_name}] {file_path}")
            logger.error(f"              Error: {error}")

    if short_files:
        logger.warning(f"⚠️  Very short files (< {config.get('preprocessing', {}).get('min_duration', 0.05)}s): {len(short_files)}")
        for split_name, file_path, duration in short_files[:3]:
            logger.warning(f"     [{split_name}] {file_path} ({duration})")

    if long_files:
        logger.info(f"ℹ️  Long files (> {config.get('preprocessing', {}).get('max_duration', 3600.0)}s): {len(long_files)}")
        for split_name, file_path, duration in long_files[:3]:
            logger.info(f"     [{split_name}] {file_path} ({duration})")
        logger.info(f"     Note: Long files will be truncated during extraction")

    if silent_files:
        logger.warning(f"⚠️  Near-silent files (max amplitude < 0.01): {len(silent_files)}")
        for split_name, file_path, amp_info in silent_files[:3]:
            logger.warning(f"     [{split_name}] {file_path} ({amp_info})")

    # Summary
    issues_found = len(missing_files) + len(corrupt_files)
    warnings_found = len(short_files) + len(silent_files)

    if issues_found == 0 and warnings_found == 0:
        logger.info(f"✓ All checked files are valid and loadable")
    elif issues_found == 0:
        logger.warning(f"⚠️  {warnings_found} warnings (files may still work but review recommended)")
    else:
        logger.error(f"❌ {issues_found} critical issues found")
        return False

    # Check class weights
    logger.info(f"\n⚖️  Class Weights:")
    weights = list(manifest['class_weights'].values())
    logger.info(f"  Min weight: {min(weights):.4f}")
    logger.info(f"  Max weight: {max(weights):.4f}")
    logger.info(f"  Mean weight: {sum(weights)/len(weights):.4f}")

    if max(weights) / min(weights) > 10:
        logger.info(f"  ℹ️  Weight ratio: {max(weights)/min(weights):.1f}x (high variance)")

    logger.info(f"\n✅ Manifest validation complete")

    return True


def main():
    """Main function."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    logger = setup_logger("Phase2_ManifestValidator", config['experiment']['log_level'])

    logger.info("="*80)
    logger.info("PHASE 2 - STAGE 1B: MANIFEST VALIDATION")
    logger.info("="*80)

    # Get manifest directory
    manifest_dir = Path(config['paths']['output_dir']) / "phase2" / "manifests"
    data_dir = Path(config['paths']['data_dir'])

    if not manifest_dir.exists():
        logger.error(f"❌ Manifest directory not found: {manifest_dir}")
        logger.error("   Run phase2_01_create_manifests.py first")
        return

    # Find all manifest files
    manifest_files = list(manifest_dir.glob("*_manifest.json"))

    if not manifest_files:
        logger.error(f"❌ No manifest files found in {manifest_dir}")
        return

    logger.info(f"\nFound {len(manifest_files)} manifest files")

    # Validate each manifest
    results = {}
    for manifest_path in sorted(manifest_files):
        try:
            success = validate_manifest(manifest_path, data_dir, config, logger)
            results[manifest_path.name] = success
        except Exception as e:
            logger.error(f"❌ Error validating {manifest_path.name}: {e}")
            results[manifest_path.name] = False

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*80}")

    successful = sum(results.values())
    logger.info(f"\n✓ {successful}/{len(results)} manifests validated successfully")

    if successful == len(results):
        logger.info("\n✅ All manifests are valid and ready for Stage 2 (zero-shot evaluation)")
    else:
        logger.warning("\n⚠️  Some manifests have issues - review warnings above")

    # Special check for pooled manifest
    pooled_path = manifest_dir / "pooled_manifest.json"
    if pooled_path.exists():
        logger.info(f"\n📦 Pooled manifest detected")
        with open(pooled_path, 'r') as f:
            pooled = json.load(f)
        logger.info(f"   Combines {len(pooled['source_datasets'])} datasets")
        logger.info(f"   Total individuals: {len(pooled['individuals'])}")
        logger.info(f"   Total samples: {pooled['statistics']['total_samples']}")


if __name__ == "__main__":
    main()
