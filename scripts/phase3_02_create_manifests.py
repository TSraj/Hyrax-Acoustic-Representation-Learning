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
from functools import lru_cache

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


def parse_hyrax_id_labels(data_dir, logger, audio_subdir="BIODA/denoised"):
    """Parse GTLabels to extract bout info per individual.

    Args:
        audio_subdir: Audio version folder relative to the location dir.
                      "Audio" (original), "BIODA/denoised" (default), or "ACA".
    """
    logger.info(f"\nParsing GTLabels bout annotations (audio source: {audio_subdir})...")

    bouts_per_individual = defaultdict(list)
    session_profile = defaultdict(lambda: defaultdict(int))

    label_files = list(data_dir.glob("*/GTLabels/*.txt"))
    logger.info(f"Found {len(label_files)} label files")

    for label_file in sorted(label_files):
        # Find corresponding audio for the selected version
        bioda_dir = label_file.parent.parent / audio_subdir

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


@lru_cache(maxsize=4)
def _load_audio_cached(path, target_sr=16000):
    """Load+resample a source wav once; bouts from the same file reuse it.

    Small cache: bouts are concatenated in file-sorted order, so a handful of
    slots is enough to avoid re-decoding a multi-minute wav per bout.
    """
    audio, sr = load_audio(path, target_sr=target_sr, mono=True)
    return audio, sr


def concatenate_bouts(bout_list, target_sr=16000):
    """Concatenate bout segments without silence.

    Bouts are ordered by (source file, start time) so the concatenation is
    deterministic and independent of how the split was drawn. For the
    session-holdout task this is a no-op (bouts already arrive in that order).
    """
    ordered = sorted(bout_list, key=lambda b: (str(b['audio_file']), b['start']))

    segments = []
    for bout in ordered:
        try:
            audio, sr = _load_audio_cached(str(bout['audio_file']), target_sr)
            start_sample = max(0, int(bout['start'] * sr))
            end_sample = min(len(audio), int(bout['end'] * sr))
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


# --- Shared cohort definition for the 8-individual session tasks -------------
# 8 individuals meeting criteria (>=4 sessions, >=100 bouts with clean date labels)
SESSION_TASK_INDIVIDUALS = ['R3', 'Q7', 'P1', 'P8', 'O7', 'M9', 'U7', 'Kashtan']

# Junk sessions to exclude (non-date labels)
SESSION_TASK_JUNK = {
    'P8': ['1301'],                      # Location code
    'Kashtan': ['7893', 'maybeKashtan'],  # Location code + ambiguous label
}


def valid_bouts_for(individual, bouts_per_individual):
    """Bouts for an individual with junk sessions removed (shared by both session tasks)."""
    return [b for b in bouts_per_individual.get(individual, [])
            if b['session'] not in SESSION_TASK_JUNK.get(individual, [])]


def create_within_session_manifest(bouts_per_individual, session_profile, output_dir, logger, seed=42):
    """Within-session control: SAME 8 individuals and SAME valid bouts as the session
    holdout task, but bouts are split 80/20 at RANDOM so train and test share sessions.

    This is the leaky counterpart to create_session_holdout_manifest(): the only thing
    that differs between the two manifests is the split rule.
    """
    logger.info("\n" + "=" * 80)
    logger.info("HYRAX ID - WITHIN-SESSION CONTROL")
    logger.info("Random 80/20 bout split (train and test share sessions)")
    logger.info("=" * 80)

    np.random.seed(seed)

    target = SESSION_TASK_INDIVIDUALS

    concat_dir = output_dir / "within_session_concatenated"
    concat_dir.mkdir(parents=True, exist_ok=True)

    manifest_splits = {'train': [], 'test': []}
    bout_inventory = {}

    for ind in target:
        valid = valid_bouts_for(ind, bouts_per_individual)
        if not valid:
            logger.warning(f"{ind}: no valid bouts, skipping")
            continue

        n = len(valid)
        idx = np.random.permutation(n)
        n_train = int(0.8 * n)
        if n_train < 1 or n - n_train < 1:
            n_train = max(1, n - 1)

        splits_idx = {'train': idx[:n_train], 'test': idx[n_train:]}
        bout_inventory[ind] = {
            'total_valid_bouts': n,
            'sessions': sorted({b['session'] for b in valid}),
        }

        logger.info(f"\n{ind}: {n} valid bouts | Train={len(splits_idx['train'])} | Test={len(splits_idx['test'])}")

        for split_name, sel in splits_idx.items():
            bout_list = [valid[i] for i in sel]
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
                'duration': len(audio) / 16000
            })

    # Class weights (inverse frequency based on train split)
    train_class_counts = defaultdict(int)
    for item in manifest_splits['train']:
        train_class_counts[item['individual']] += 1

    total_train = len(manifest_splits['train'])
    class_weights = {ind: total_train / (len(target) * train_class_counts[ind])
                     for ind in target if train_class_counts[ind] > 0}

    manifest = {
        'task': 'hyrax_id_within_session',
        'description': 'Within-session control - same 8 individuals and same valid bouts as '
                       'the session holdout task, random 80/20 bout split (sessions shared)',
        'num_classes': len(target),
        'individuals': sorted(target),
        'class_to_idx': {ind: idx for idx, ind in enumerate(sorted(target))},
        'class_weights': class_weights,
        'excluded_sessions': SESSION_TASK_JUNK,
        'bout_inventory': bout_inventory,
        'splits': manifest_splits,
        'split_counts': {k: len(v) for k, v in manifest_splits.items()},
        'seed': seed
    }

    manifest_file = output_dir / "hyrax_id_within_session.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\n✓ Within-session manifest: {manifest_file}")
    logger.info(f"  Classes: {len(target)} | Train: {len(manifest_splits['train'])} | Test: {len(manifest_splits['test'])}")

    return manifest


def create_session_holdout_ft_manifest(bouts_per_individual, session_profile, output_dir,
                                       logger, seed=42):
    """Session holdout WITH a session-disjoint validation split (for fine-tuning).

    Identical to create_session_holdout_manifest() except that a second session per
    individual is carved out as validation:

        test  = largest valid session   (SAME session as the zero-shot session-holdout
                                         manifest, so results stay comparable to that
                                         baseline)
        val   = second-largest valid session
        train = all remaining valid sessions

    train is therefore slightly smaller than in the zero-shot manifest, since the
    val session is removed from it.
    """
    logger.info("\n" + "=" * 80)
    logger.info("HYRAX ID - SESSION HOLDOUT + SESSION-DISJOINT VAL (fine-tuning)")
    logger.info("test = largest session (unchanged) | val = 2nd largest | train = rest")
    logger.info("=" * 80)

    np.random.seed(seed)

    target = SESSION_TASK_INDIVIDUALS
    held_out, val_sessions = {}, {}

    for ind in target:
        if ind not in session_profile:
            continue
        valid = {s: c for s, c in session_profile[ind].items()
                 if s not in SESSION_TASK_JUNK.get(ind, [])}
        ordered = sorted(valid.items(), key=lambda x: (-x[1], x[0]))
        held_out[ind] = ordered[0][0] if ordered else None
        val_sessions[ind] = ordered[1][0] if len(ordered) > 1 else None

    concat_dir = output_dir / "session_holdout_ft_concatenated"
    concat_dir.mkdir(parents=True, exist_ok=True)

    manifest_splits = {'train': [], 'val': [], 'test': []}
    bout_inventory = {}

    for ind in target:
        if ind not in bouts_per_individual or held_out.get(ind) is None:
            continue

        valid_bouts = valid_bouts_for(ind, bouts_per_individual)
        bout_inventory[ind] = {
            'total_valid_bouts': len(valid_bouts),
            'sessions': sorted({b['session'] for b in valid_bouts}),
        }

        by_split = {
            'test': [b for b in valid_bouts if b['session'] == held_out[ind]],
            'val': [b for b in valid_bouts if b['session'] == val_sessions[ind]],
            'train': [b for b in valid_bouts
                      if b['session'] not in (held_out[ind], val_sessions[ind])],
        }

        logger.info(f"\n{ind}: test={held_out[ind]} ({len(by_split['test'])}) | "
                    f"val={val_sessions[ind]} ({len(by_split['val'])}) | "
                    f"train={len(by_split['train'])} bouts")
        if not by_split['train']:
            logger.warning(f"  {ind}: NO training bouts left after removing test+val sessions")

        for split_name, bout_list in by_split.items():
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
                'session': (held_out[ind] if split_name == 'test'
                            else val_sessions[ind] if split_name == 'val' else None)
            })

    train_class_counts = defaultdict(int)
    for item in manifest_splits['train']:
        train_class_counts[item['individual']] += 1
    total_train = len(manifest_splits['train'])
    class_weights = {ind: total_train / (len(target) * train_class_counts[ind])
                     for ind in target if train_class_counts[ind] > 0}

    manifest = {
        'task': 'hyrax_id_session_holdout_ft',
        'description': 'Session holdout with session-disjoint val split, for fine-tuning. '
                       'Test session matches the zero-shot session-holdout manifest.',
        'num_classes': len(target),
        'individuals': sorted(target),
        'class_to_idx': {ind: idx for idx, ind in enumerate(sorted(target))},
        'class_weights': class_weights,
        'held_out_sessions': held_out,
        'val_sessions': val_sessions,
        'excluded_sessions': SESSION_TASK_JUNK,
        'bout_inventory': bout_inventory,
        'splits': manifest_splits,
        'split_counts': {k: len(v) for k, v in manifest_splits.items()},
        'seed': seed
    }

    manifest_file = output_dir / "hyrax_id_session_holdout_ft.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\n✓ Session holdout (fine-tuning) manifest: {manifest_file}")
    logger.info(f"  Classes: {len(target)} | " +
                " | ".join(f"{k.capitalize()}: {len(v)}" for k, v in manifest_splits.items()))

    return manifest


def create_session_holdout_manifest(bouts_per_individual, session_profile, output_dir, logger, seed=42):
    """Session holdout diagnostic: Session-stratified split for 8 individuals with ≥4 sessions and ≥100 bouts."""
    logger.info("\n" + "=" * 80)
    logger.info("HYRAX ID - SESSION HOLDOUT DIAGNOSTIC")
    logger.info("Session-stratified splits for leakage sensitivity test")
    logger.info("Inclusion criteria: ≥4 sessions, ≥100 bouts, clean date labels")
    logger.info("=" * 80)

    np.random.seed(seed)

    target = SESSION_TASK_INDIVIDUALS
    junk_sessions = SESSION_TASK_JUNK

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
    bout_inventory = {}

    for ind in target:
        if ind not in bouts_per_individual or held_out[ind] is None:
            continue

        # Filter bouts: exclude junk sessions AND separate held-out
        valid_bouts = valid_bouts_for(ind, bouts_per_individual)
        bout_inventory[ind] = {
            'total_valid_bouts': len(valid_bouts),
            'sessions': sorted({b['session'] for b in valid_bouts}),
        }

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
        'bout_inventory': bout_inventory,
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


AUDIO_SOURCES = {
    'original': 'Audio',
    'bioda': 'BIODA/denoised',
    'aca': 'ACA',
}


def main():
    """Main pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3 - Step 2: Manifest creation")
    parser.add_argument("--audio-source", default="bioda", choices=sorted(AUDIO_SOURCES),
                        help="Which audio version the bouts are cut from (default: bioda)")
    parser.add_argument("--output-dir", default="outputs/phase3/manifests",
                        help="Where manifests + concatenated wavs are written")
    parser.add_argument("--tasks", default="all",
                        choices=["all", "session_screen", "session_ft"],
                        help="'all' = full Phase 3 set; 'session_screen' = the two "
                             "8-individual session tasks (denoiser screen); "
                             "'session_ft' = session holdout with a session-disjoint "
                             "val split (fine-tuning)")
    parser.add_argument("--log-tag", default=None,
                        help="Suffix for the log file name (default: derived from audio source)")
    args = parser.parse_args()

    audio_subdir = AUDIO_SOURCES[args.audio_source]
    tag = args.log_tag or args.audio_source

    log_dir = Path("outputs/phase3/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("Phase3_Manifests",
                          log_file=str(log_dir / f"manifest_creation_{tag}.log"))

    logger.info("=" * 80)
    logger.info("PHASE 3 - STEP 2: MANIFEST CREATION")
    logger.info(f"Audio source: {args.audio_source} -> */{audio_subdir}/")
    logger.info(f"Tasks: {args.tasks}")
    logger.info("=" * 80)

    # Paths
    data_dir = Path("Data/YearLocation")
    hyrax_dir = Path("outputs/phase3/hyrax_data")
    phase2_manifests_dir = Path("outputs/phase2/manifests")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse Hyrax-ID bout labels
    bouts_per_individual, session_profile = parse_hyrax_id_labels(
        data_dir, logger, audio_subdir=audio_subdir
    )

    # Save session profile
    profile = {'audio_source': args.audio_source,
               'audio_subdir': audio_subdir,
               'individuals': {ind: {'total_bouts': sum(sessions.values()), 'sessions': dict(sessions)}
                               for ind, sessions in session_profile.items()}}
    with open(output_dir / "hyrax_session_profile.json", 'w') as f:
        json.dump(profile, f, indent=2)

    within_session_manifest = session_holdout_manifest = session_ft_manifest = None

    if args.tasks in ("all", "session_screen"):
        within_session_manifest = create_within_session_manifest(
            bouts_per_individual, session_profile, output_dir, logger
        )
        session_holdout_manifest = create_session_holdout_manifest(
            bouts_per_individual, session_profile, output_dir, logger
        )

    if args.tasks in ("all", "session_ft"):
        session_ft_manifest = create_session_holdout_ft_manifest(
            bouts_per_individual, session_profile, output_dir, logger
        )

    hyrax_id_manifest = None
    if args.tasks == "all":
        hyrax_id_manifest = create_hyrax_id_manifest(
            bouts_per_individual, session_profile, output_dir, logger
        )

        # Load old concatenated hyrax data for species_id
        hyrax_data = load_hyrax_data(hyrax_dir)
        train_ids, val_ids, test_ids = split_individuals(list(hyrax_data.keys()))

        create_species_id_manifest(
            hyrax_data, train_ids, val_ids, test_ids,
            phase2_manifests_dir, output_dir, logger
        )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("MANIFEST CREATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutput: {output_dir}")
    logger.info(f"  - hyrax_session_profile.json")
    if within_session_manifest is not None:
        logger.info(f"  - hyrax_id_within_session.json ({within_session_manifest['num_classes']} classes - CONTROL)")
        logger.info(f"  - hyrax_id_session_holdout.json ({session_holdout_manifest['num_classes']} classes - DIAGNOSTIC)")
    if session_ft_manifest is not None:
        logger.info(f"  - hyrax_id_session_holdout_ft.json ({session_ft_manifest['num_classes']} classes - FINE-TUNING)")
    if hyrax_id_manifest is not None:
        logger.info(f"  - hyrax_id.json ({hyrax_id_manifest['num_classes']} classes - MAIN TASK)")
        logger.info(f"  - species_id.json (8 classes)")
    logger.info("\n✓ Ready for experiments!")


if __name__ == "__main__":
    main()
