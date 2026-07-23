#!/usr/bin/env python3
"""
Phase 3 - Step 2: Create Manifests for Species ID and Hyrax ID
Creates train/val/test splits (80/10/10) and manifests for both tasks.
"""

import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
from src.utils.audio_utils import get_audio_info


def load_hyrax_data(hyrax_dir):
    """Load extracted hyrax concatenated files."""
    hyrax_files = sorted(hyrax_dir.glob("*_concatenated.wav"))

    hyrax_data = {}
    for audio_file in hyrax_files:
        individual_id = audio_file.stem.replace("_concatenated", "")

        # Get audio info
        info = get_audio_info(str(audio_file))

        # Store relative path from project root
        rel_path = str(audio_file).replace(str(Path.cwd()) + '/', '')

        hyrax_data[individual_id] = {
            'file': rel_path,
            'duration': info['duration'],
            'sample_rate': info['sample_rate']
        }

    return hyrax_data


def split_individuals(individuals, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split individuals into train/val/test sets.

    Args:
        individuals: List of individual IDs
        train_ratio: Training set ratio (default 0.8)
        val_ratio: Validation set ratio (default 0.1)
        seed: Random seed for reproducibility

    Returns:
        train_ids, val_ids, test_ids
    """
    np.random.seed(seed)

    individuals = sorted(individuals)  # Consistent ordering
    n_total = len(individuals)

    # Calculate splits with proper rounding
    # For 18: train=14, val=2, test=2
    n_train = int(n_total * train_ratio)
    n_val = max(1, round(n_total * val_ratio))  # At least 1, round to nearest
    n_test = n_total - n_train - n_val

    # Ensure valid split
    if n_test < 1:
        n_val = n_total - n_train - 1
        n_test = 1

    # Shuffle and split
    shuffled = np.random.permutation(individuals)

    train_ids = sorted(shuffled[:n_train].tolist())
    val_ids = sorted(shuffled[n_train:n_train + n_val].tolist())
    test_ids = sorted(shuffled[n_train + n_val:].tolist())

    return train_ids, val_ids, test_ids


def create_hyrax_id_manifest(hyrax_data, train_ids, val_ids, test_ids, output_dir, logger):
    """
    Create manifest for Hyrax ID task (18-individual classification).

    Args:
        hyrax_data: Dict of {individual_id: {file, duration, sample_rate}}
        train_ids, val_ids, test_ids: Individual ID lists for each split
        output_dir: Output directory
        logger: Logger instance
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREATING HYRAX ID MANIFEST (18-class individual identification)")
    logger.info("=" * 80)

    # All individuals
    all_individuals = sorted(hyrax_data.keys())

    # Create class mapping
    class_to_idx = {ind: idx for idx, ind in enumerate(all_individuals)}

    # Build splits
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    for split_name, individual_ids in splits.items():
        items = []

        for individual_id in individual_ids:
            items.append({
                'file': hyrax_data[individual_id]['file'],
                'individual': individual_id,
                'duration': hyrax_data[individual_id]['duration']
            })

        logger.info(f"\n{split_name.upper()} split:")
        logger.info(f"  Individuals: {len(individual_ids)}")
        logger.info(f"  Files: {len(items)}")
        logger.info(f"  Total duration: {sum(item['duration'] for item in items)/60:.2f} min")

    # Class weights (inverse frequency)
    class_counts = {ind: 1 for ind in all_individuals}  # Each individual has 1 concatenated file
    total_samples = len(all_individuals)
    class_weights = {ind: total_samples / (len(all_individuals) * count)
                     for ind, count in class_counts.items()}

    # Create manifest
    manifest = {
        'task': 'hyrax_id',
        'description': '18-class hyrax individual identification',
        'num_classes': len(all_individuals),
        'individuals': all_individuals,
        'class_to_idx': class_to_idx,
        'class_weights': class_weights,
        'splits': {
            'train': [{'file': hyrax_data[ind]['file'],
                      'individual': ind,
                      'duration': hyrax_data[ind]['duration']}
                     for ind in train_ids],
            'val': [{'file': hyrax_data[ind]['file'],
                    'individual': ind,
                    'duration': hyrax_data[ind]['duration']}
                   for ind in val_ids],
            'test': [{'file': hyrax_data[ind]['file'],
                     'individual': ind,
                     'duration': hyrax_data[ind]['duration']}
                    for ind in test_ids]
        },
        'split_counts': {
            'train': len(train_ids),
            'val': len(val_ids),
            'test': len(test_ids)
        }
    }

    # Save manifest
    manifest_file = output_dir / "hyrax_id_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\n✓ Hyrax ID manifest saved: {manifest_file}")

    return manifest


def create_species_id_manifest(hyrax_data, train_ids, val_ids, test_ids,
                                phase2_manifests_dir, output_dir, logger):
    """
    Create manifest for Species ID task (8-class: 7 bird/animal species + hyrax).

    Args:
        hyrax_data: Dict of hyrax data
        train_ids, val_ids, test_ids: Hyrax individual ID lists
        phase2_manifests_dir: Phase 2 manifests directory
        output_dir: Output directory
        logger: Logger instance
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREATING SPECIES ID MANIFEST (8-class: 7 species + hyrax)")
    logger.info("=" * 80)

    # Load Phase 2 pooled manifest (has all 7 bird/animal species)
    phase2_pooled = phase2_manifests_dir / "pooled_manifest.json"

    if not phase2_pooled.exists():
        logger.error(f"Phase 2 pooled manifest not found: {phase2_pooled}")
        logger.error("Run Phase 2 first to generate bird/animal data manifests")
        return None

    with open(phase2_pooled, 'r') as f:
        phase2_data = json.load(f)

    # Extract species from Phase 2 individuals
    # Phase 2 individual format: "species_individualname"
    phase2_species = set()
    species_to_individuals = defaultdict(list)

    for individual in phase2_data['individuals']:
        # Extract species (handles multi-word like "bengalese_finch")
        known_species = ['anuraset', 'bengalese_finch', 'macaque', 'marmoset',
                        'picidae', 'wetlands_bird', 'zebra_finch']

        species = None
        for sp in sorted(known_species, key=len, reverse=True):
            if individual.startswith(sp + '_'):
                species = sp
                break

        if species is None:
            species = individual.split('_')[0]

        phase2_species.add(species)
        species_to_individuals[species].append(individual)

    phase2_species = sorted(phase2_species)

    logger.info(f"\nPhase 2 species: {phase2_species}")
    logger.info(f"Adding hyrax as 8th species")

    # All species (7 + hyrax)
    all_species = phase2_species + ['hyrax']
    species_to_idx = {sp: idx for idx, sp in enumerate(all_species)}

    # Build splits
    splits = {}

    for split_name in ['train', 'val', 'test']:
        items = []

        # Add Phase 2 data for this split
        if split_name in phase2_data:
            for item in phase2_data[split_name]:
                individual = item['individual']

                # Find species
                species = None
                for sp in sorted(phase2_species, key=len, reverse=True):
                    if individual.startswith(sp + '_'):
                        species = sp
                        break

                if species is None:
                    species = individual.split('_')[0]

                items.append({
                    'file': item['file'],
                    'individual': individual,
                    'species': species,
                    'duration': item.get('duration', 0.0)  # Phase 2 doesn't have duration
                })

        # Add hyrax data for this split
        if split_name == 'train':
            hyrax_ids = train_ids
        elif split_name == 'val':
            hyrax_ids = val_ids
        else:
            hyrax_ids = test_ids

        for individual_id in hyrax_ids:
            items.append({
                'file': hyrax_data[individual_id]['file'],
                'individual': individual_id,
                'species': 'hyrax',
                'duration': hyrax_data[individual_id]['duration']
            })

        splits[split_name] = items

        # Log split stats
        species_counts = defaultdict(int)
        for item in items:
            species_counts[item['species']] += 1

        logger.info(f"\n{split_name.upper()} split:")
        logger.info(f"  Total files: {len(items)}")
        logger.info(f"  Total duration: {sum(item['duration'] for item in items)/60:.2f} min")
        logger.info(f"  Species distribution:")
        for sp in sorted(species_counts.keys()):
            logger.info(f"    {sp}: {species_counts[sp]} files")

    # Class weights (inverse frequency based on train split)
    train_species_counts = defaultdict(int)
    for item in splits['train']:
        train_species_counts[item['species']] += 1

    total_train = len(splits['train'])
    class_weights = {sp: total_train / (len(all_species) * train_species_counts[sp])
                     for sp in all_species}

    # Create manifest
    manifest = {
        'task': 'species_id',
        'description': '8-class species identification (7 bird/animal + hyrax)',
        'num_classes': len(all_species),
        'species': all_species,
        'species_to_idx': species_to_idx,
        'class_weights': class_weights,
        'splits': splits,
        'split_counts': {
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test'])
        },
        'phase2_species': phase2_species,
        'hyrax_individuals': {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
    }

    # Save manifest
    manifest_file = output_dir / "species_id_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\n✓ Species ID manifest saved: {manifest_file}")

    return manifest


def main():
    """Main pipeline."""

    # Setup logging
    log_dir = Path("outputs/phase3/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("Phase3_Manifests", log_file=str(log_dir / "manifest_creation.log"))

    logger.info("=" * 80)
    logger.info("PHASE 3 - STEP 2: MANIFEST CREATION")
    logger.info("=" * 80)

    # Paths
    hyrax_dir = Path("outputs/phase3/hyrax_data")
    phase2_manifests_dir = Path("outputs/phase2/manifests")
    output_dir = Path("outputs/phase3/manifests")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load hyrax data
    logger.info("\nLoading extracted hyrax data...")
    hyrax_data = load_hyrax_data(hyrax_dir)

    logger.info(f"✓ Loaded {len(hyrax_data)} individuals")

    # Split individuals (80/10/10)
    logger.info("\nSplitting individuals (80% train / 10% val / 10% test)...")
    train_ids, val_ids, test_ids = split_individuals(list(hyrax_data.keys()))

    logger.info(f"\nSplit sizes:")
    logger.info(f"  Train: {len(train_ids)} individuals - {train_ids}")
    logger.info(f"  Val:   {len(val_ids)} individuals - {val_ids}")
    logger.info(f"  Test:  {len(test_ids)} individuals - {test_ids}")

    # Create Hyrax ID manifest
    hyrax_id_manifest = create_hyrax_id_manifest(
        hyrax_data, train_ids, val_ids, test_ids, output_dir, logger
    )

    # Create Species ID manifest
    species_id_manifest = create_species_id_manifest(
        hyrax_data, train_ids, val_ids, test_ids,
        phase2_manifests_dir, output_dir, logger
    )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("MANIFEST CREATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"  - hyrax_id_manifest.json (18-class individual ID)")
    logger.info(f"  - species_id_manifest.json (8-class species ID)")
    logger.info("\n✓ Ready for Phase 3 experiments!")


if __name__ == "__main__":
    main()
