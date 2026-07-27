#!/usr/bin/env python3
"""
Phase 3 - Step 2: Create Manifests for Species ID and Hyrax ID

Creates:
1. Species ID manifest (8 classes)
2. Hyrax ID Option A: Per-individual 80/10/10 splits (all individuals ≥10 bouts)
3. Hyrax ID Option C: Session-stratified splits (R3, Q7, P1, P8)
4. Session profile log
"""

import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
from src.utils.audio_utils import get_audio_info, load_audio, save_audio


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
    manifest_file = output_dir / "species_id.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\n✓ Species ID manifest saved: {manifest_file}")

    return manifest


def parse_hyrax_id_labels(data_dir, logger):
    """Parse GTLabels to extract bout info per individual from BIODA."""
    logger.info("\nParsing GTLabels bout annotations...")

    bouts_per_individual = defaultdict(list)
    session_profile = defaultdict(lambda: defaultdict(int))

    label_files = list(data_dir.glob("*/GTLabels/*.txt"))
    logger.info(f"Found {len(label_files)} label files")

    for label_file in sorted(label_files):
        # Find corresponding BIODA denoised audio
        bioda_dir = label_file.parent.parent / "BIODA" / "denoised"

        # Match audio file (same base name without _labels suffix)
        audio_name = label_file.stem.replace('_labels', '')
        audio_file = bioda_dir / f"{audio_name}.wav"

        if not audio_file.exists():
            # Try without .txt extension variations
            audio_matches = list(bioda_dir.glob(f"{audio_name}*.wav"))
            if audio_matches:
                audio_file = audio_matches[0]
            else:
                continue

        # Extract individual and session from filename
        filename = label_file.stem.replace('_labels', '')
        parts = filename.split('_')
        if len(parts) < 2:
            continue

        individual = parts[0]

        # Extract session (date part - usually last or second to last)
        session = parts[-1] if len(parts) >= 4 else parts[-1]

        # Parse bouts from GTLabels (format: start end bout_X)
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('\t')
                if len(parts) < 3:
                    continue

                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    label = parts[2]

                    # GTLabels use lowercase "bout_" prefix
                    if not label.lower().startswith('bout_'):
                        continue

                    bouts_per_individual[individual].append({
                        'audio_file': audio_file,
                        'start': start,
                        'end': end,
                        'session': session
                    })
                    session_profile[individual][session] += 1

                except (ValueError, IndexError):
                    continue

    logger.info(f"✓ Parsed bouts for {len(bouts_per_individual)} individuals")
    for individual in sorted(bouts_per_individual.keys()):
        logger.info(f"  {individual}: {len(bouts_per_individual[individual])} bouts")

    return dict(bouts_per_individual), dict(session_profile)


def concatenate_bouts(bout_list, target_sr=16000):
    """Concatenate bout segments without silence."""
    segments = []
    for bout in bout_list:
        try:
            audio, sr = load_audio(str(bout['audio_file']), target_sr=target_sr, mono=True)
            start_sample = int(bout['start'] * sr)
            end_sample = int(bout['end'] * sr)
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            segments.append(audio[start_sample:end_sample])
        except Exception:
            continue
    return np.concatenate(segments) if segments else np.array([])


def create_hyrax_id_manifest(bouts_per_individual, session_profile, output_dir, logger, min_bouts=10, seed=42):
    """Hyrax ID: 18-class individual recognition with per-individual 80/10/10 bout splits."""
    logger.info("\n" + "=" * 80)
    logger.info("HYRAX ID: 18-class individual recognition")
    logger.info("=" * 80)

    np.random.seed(seed)

    # Filter by min_bouts
    included = [ind for ind, bouts in bouts_per_individual.items() if len(bouts) >= min_bouts]
    excluded = [(ind, len(bouts)) for ind, bouts in bouts_per_individual.items() if len(bouts) < min_bouts]

    logger.info(f"\nIncluded: {len(included)} individuals (≥{min_bouts} bouts)")
    if excluded:
        logger.info(f"Excluded: {len(excluded)} individuals (<{min_bouts} bouts)")
        for ind, count in sorted(excluded):
            logger.info(f"  {ind}: {count} bouts")

    # Create concatenated files per split
    concat_dir = output_dir / "hyrax_id_concatenated"
    concat_dir.mkdir(parents=True, exist_ok=True)

    manifest_splits = {'train': [], 'val': [], 'test': []}

    for individual in sorted(included):
        bouts = bouts_per_individual[individual]
        n_bouts = len(bouts)

        logger.info(f"\nProcessing {individual}: {n_bouts} bouts")

        indices = np.random.permutation(n_bouts)
        n_train = int(0.8 * n_bouts)
        n_val = max(1, int(0.1 * n_bouts))
        n_test = n_bouts - n_train - n_val
        if n_test < 1:
            n_train, n_val, n_test = n_bouts - 2, 1, 1

        splits_indices = {
            'train': indices[:n_train],
            'val': indices[n_train:n_train + n_val],
            'test': indices[n_train + n_val:]
        }

        for split_name, split_idx in splits_indices.items():
            bout_list = [bouts[i] for i in split_idx]
            concat_file = concat_dir / f"{individual}_{split_name}.wav"

            logger.info(f"  {split_name}: Concatenating {len(bout_list)} bouts...")
            # Concatenate and save
            audio = concatenate_bouts(bout_list)
            if len(audio) > 0:
                save_audio(str(concat_file), audio, sr=16000)
                logger.info(f"  {split_name}: Saved {concat_file.name} ({len(audio)/16000:.2f}s)")

            manifest_splits[split_name].append({
                'file': str(concat_file),
                'individual': individual,
                'num_bouts': len(bout_list),
                'duration': len(audio) / 16000
            })

    manifest = {
        'task': 'hyrax_id',
        'description': f'18-class hyrax individual recognition - all individuals ≥{min_bouts} bouts, per-individual 80/10/10 bout splits',
        'num_classes': len(included),
        'individuals': sorted(included),
        'excluded': [{'individual': ind, 'bout_count': cnt} for ind, cnt in sorted(excluded)],
        'class_to_idx': {ind: idx for idx, ind in enumerate(sorted(included))},
        'class_weights': {ind: 1.0 for ind in included},
        'splits': manifest_splits,
        'split_counts': {k: len(v) for k, v in manifest_splits.items()},
        'seed': seed
    }

    manifest_file = output_dir / "hyrax_id.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\n✓ Hyrax ID manifest: {manifest_file}")
    logger.info(f"  Classes: {len(included)} | Train: {len(manifest_splits['train'])} | Val: {len(manifest_splits['val'])} | Test: {len(manifest_splits['test'])}")

    return manifest


def create_session_holdout_manifest(bouts_per_individual, session_profile, output_dir, logger, seed=42):
    """Session holdout diagnostic: Session-stratified split for 8 individuals with ≥4 sessions and ≥100 bouts."""
    logger.info("\n" + "=" * 80)
    logger.info("HYRAX ID - SESSION HOLDOUT DIAGNOSTIC")
    logger.info("Session-stratified splits for leakage sensitivity test")
    logger.info("Inclusion criteria: ≥4 sessions, ≥100 bouts, clean date labels")
    logger.info("=" * 80)

    np.random.seed(seed)

    # 8 individuals meeting criteria (≥4 sessions, ≥100 bouts with clean date labels)
    target = ['R3', 'Q7', 'P1', 'P8', 'O7', 'M9', 'U7', 'Kashtan']

    # Junk sessions to exclude (non-date labels)
    junk_sessions = {
        'P8': ['1301'],          # Location code
        'Kashtan': ['7893', 'maybeKashtan']  # Location code + ambiguous label
    }

    # Select held-out session per individual (largest valid session)
    held_out = {}
    for ind in target:
        if ind not in session_profile:
            continue

        # Filter out junk sessions
        valid_sessions = {s: c for s, c in session_profile[ind].items()
                          if s not in junk_sessions.get(ind, [])}

        # Sort by bout count, pick largest
        sessions = sorted(valid_sessions.items(), key=lambda x: x[1], reverse=True)
        held_out[ind] = sessions[0][0] if sessions else None

        logger.info(f"{ind}: {len(valid_sessions)} valid sessions (excluded {len(session_profile[ind]) - len(valid_sessions)} junk)")

    concat_dir = output_dir / "session_holdout_concatenated"
    concat_dir.mkdir(parents=True, exist_ok=True)

    manifest_splits = {'train': [], 'test': []}

    for ind in target:
        if ind not in bouts_per_individual or held_out[ind] is None:
            continue

        # Filter bouts: exclude junk sessions AND separate held-out
        valid_bouts = [b for b in bouts_per_individual[ind]
                       if b['session'] not in junk_sessions.get(ind, [])]

        train_bouts = [b for b in valid_bouts if b['session'] != held_out[ind]]
        test_bouts = [b for b in valid_bouts if b['session'] == held_out[ind]]

        logger.info(f"\n{ind}: Held-out session={held_out[ind]} | Train={len(train_bouts)} | Test={len(test_bouts)}")

        for split_name, bout_list in [('train', train_bouts), ('test', test_bouts)]:
            if not bout_list:
                continue

            concat_file = concat_dir / f"{ind}_{split_name}.wav"
            audio = concatenate_bouts(bout_list)
            if len(audio) > 0:
                save_audio(str(concat_file), audio, sr=16000)

            manifest_splits[split_name].append({
                'file': str(concat_file),
                'individual': ind,
                'num_bouts': len(bout_list),
                'duration': len(audio) / 16000,
                'held_out_session': held_out[ind] if split_name == 'test' else None
            })

    # Class weights (inverse frequency based on train split)
    train_class_counts = defaultdict(int)
    for item in manifest_splits['train']:
        train_class_counts[item['individual']] += 1

    total_train = len(manifest_splits['train'])
    class_weights = {ind: total_train / (len(target) * train_class_counts[ind])
                     for ind in target}

    manifest = {
        'task': 'hyrax_id_session_holdout',
        'description': 'Session holdout diagnostic - 8 individuals (≥4 sessions, ≥100 bouts, clean date labels)',
        'inclusion_criteria': '≥4 sessions AND ≥100 bouts with clean date labels',
        'num_classes': len(target),
        'individuals': sorted(target),
        'class_to_idx': {ind: idx for idx, ind in enumerate(sorted(target))},
        'class_weights': class_weights,
        'held_out_sessions': held_out,
        'excluded_sessions': junk_sessions,
        'splits': manifest_splits,
        'split_counts': {k: len(v) for k, v in manifest_splits.items()},
        'seed': seed
    }

    manifest_file = output_dir / "hyrax_id_session_holdout.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\n✓ Session holdout manifest: {manifest_file}")
    logger.info(f"  Classes: {len(target)} | Train: {len(manifest_splits['train'])} | Test: {len(manifest_splits['test'])}")

    return manifest


def main():
    """Main pipeline."""

    log_dir = Path("outputs/phase3/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("Phase3_Manifests", log_file=str(log_dir / "manifest_creation.log"))

    logger.info("=" * 80)
    logger.info("PHASE 3 - STEP 2: MANIFEST CREATION")
    logger.info("=" * 80)

    # Paths
    data_dir = Path("Data/YearLocation")
    hyrax_dir = Path("outputs/phase3/hyrax_data")
    phase2_manifests_dir = Path("outputs/phase2/manifests")
    output_dir = Path("outputs/phase3/manifests")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse Hyrax-ID bout labels
    bouts_per_individual, session_profile = parse_hyrax_id_labels(data_dir, logger)

    # Save session profile
    profile = {'individuals': {ind: {'total_bouts': sum(sessions.values()), 'sessions': dict(sessions)}
                                for ind, sessions in session_profile.items()}}
    with open(output_dir / "hyrax_session_profile.json", 'w') as f:
        json.dump(profile, f, indent=2)

    # Create Hyrax ID manifests
    hyrax_id_manifest = create_hyrax_id_manifest(bouts_per_individual, session_profile, output_dir, logger)
    session_holdout_manifest = create_session_holdout_manifest(bouts_per_individual, session_profile, output_dir, logger)

    # Load old concatenated hyrax data for species_id
    hyrax_data = load_hyrax_data(hyrax_dir)
    train_ids, val_ids, test_ids = split_individuals(list(hyrax_data.keys()))

    # Create Species ID manifest
    species_id_manifest = create_species_id_manifest(
        hyrax_data, train_ids, val_ids, test_ids,
        phase2_manifests_dir, output_dir, logger
    )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("MANIFEST CREATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutput: {output_dir}")
    logger.info(f"  - hyrax_session_profile.json")
    logger.info(f"  - hyrax_id.json ({hyrax_id_manifest['num_classes']} classes - MAIN TASK)")
    logger.info(f"  - hyrax_id_session_holdout.json ({session_holdout_manifest['num_classes']} classes - DIAGNOSTIC)")
    logger.info(f"  - species_id.json (8 classes)")
    logger.info("\n✓ Ready for experiments!")


if __name__ == "__main__":
    main()
