#!/usr/bin/env python3
"""
Phase 3 - Step 1: Extract and Concatenate Hyrax BOUTs
Extracts BOUT segments from BIODA denoised files and concatenates per individual.
"""

import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
from src.utils.audio_utils import load_audio, save_audio


def parse_gtlabels(label_file):
    """
    Parse GTLabels annotation file.

    Format: start_time  end_time  label
    Returns list of (start, end, label) tuples for BOUT segments only.
    """
    bouts = []

    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 3:
                continue

            start, end, label = parts

            # Only keep BOUT labels (bout_0, bout_1, etc.)
            if label.lower().startswith('bout_'):
                try:
                    bouts.append((float(start), float(end), label))
                except ValueError:
                    continue

    return bouts


def extract_bout_segments(audio_path, bouts, sr=16000):
    """
    Extract BOUT segments from audio file.

    Args:
        audio_path: Path to audio file
        bouts: List of (start, end, label) tuples
        sr: Target sample rate

    Returns:
        List of audio segments (numpy arrays)
    """
    # Load full audio
    audio, actual_sr = load_audio(str(audio_path), target_sr=sr, mono=True)

    segments = []
    for start, end, label in bouts:
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        # Extract segment
        segment = audio[start_sample:end_sample]

        if len(segment) > 0:
            segments.append(segment)

    return segments


def concatenate_segments(segments):
    """Concatenate audio segments (removes silences between BOUTs)."""
    if not segments:
        return np.array([])

    return np.concatenate(segments)


def extract_individual_id(filename):
    """
    Extract individual ID from filename.
    Format: IndividualID_Take###_...
    """
    # Remove extension
    name = Path(filename).stem

    # Take first part before underscore
    parts = name.split('_')
    if parts:
        return parts[0]

    return None


def main():
    """Main extraction pipeline."""

    # Setup logging
    log_dir = Path("outputs/phase3/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("Phase3_BOUTExtraction", log_file=str(log_dir / "bout_extraction.log"))

    logger.info("=" * 80)
    logger.info("PHASE 3 - STEP 1: HYRAX BOUT EXTRACTION")
    logger.info("=" * 80)

    # Paths
    data_root = Path("data/YearLocation")
    output_dir = Path("outputs/phase3/hyrax_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all location folders
    location_folders = [d for d in data_root.iterdir() if d.is_dir()]

    logger.info(f"Found {len(location_folders)} location folders")

    # Collect all recordings per individual
    individual_recordings = defaultdict(list)  # {individual_id: [(audio_path, label_path), ...]}

    logger.info("\nScanning for BIODA and GTLabels files...")

    for location in location_folders:
        logger.info(f"  Processing: {location.name}")

        bioda_dir = location / "BIODA" / "denoised"
        gtlabels_dir = location / "GTLabels"

        if not bioda_dir.exists() or not gtlabels_dir.exists():
            logger.warning(f"    Missing BIODA or GTLabels in {location.name}, skipping")
            continue

        # Match audio files with label files
        audio_files = sorted(bioda_dir.glob("*.wav"))

        for audio_file in audio_files:
            # Find corresponding label file
            label_file = gtlabels_dir / (audio_file.stem + ".txt")

            if not label_file.exists():
                logger.warning(f"    No labels for {audio_file.name}, skipping")
                continue

            # Extract individual ID
            individual_id = extract_individual_id(audio_file.name)

            if individual_id:
                individual_recordings[individual_id].append((audio_file, label_file))

    logger.info(f"\n✓ Found recordings for {len(individual_recordings)} individuals")

    # Sort individuals for consistent ordering
    individuals = sorted(individual_recordings.keys())

    logger.info(f"Individuals: {', '.join(individuals)}")

    # Statistics
    stats = {
        'individuals': {},
        'total_recordings': 0,
        'total_bouts': 0,
        'total_duration_sec': 0
    }

    # Extract and concatenate BOUTs per individual
    logger.info("\n" + "=" * 80)
    logger.info("EXTRACTING AND CONCATENATING BOUTS")
    logger.info("=" * 80)

    for individual_id in tqdm(individuals, desc="Processing individuals"):
        recordings = individual_recordings[individual_id]

        logger.info(f"\n{individual_id}: {len(recordings)} recordings")

        all_segments = []
        bout_count = 0

        for audio_path, label_path in recordings:
            # Parse annotations
            bouts = parse_gtlabels(label_path)

            if not bouts:
                continue

            # Extract BOUT segments
            segments = extract_bout_segments(audio_path, bouts)
            all_segments.extend(segments)
            bout_count += len(bouts)

        if not all_segments:
            logger.warning(f"  No BOUTs found for {individual_id}, skipping")
            continue

        # Concatenate all BOUTs
        concatenated = concatenate_segments(all_segments)

        # Save concatenated audio
        output_path = output_dir / f"{individual_id}_concatenated.wav"
        save_audio(str(output_path), concatenated, sr=16000)

        duration_sec = len(concatenated) / 16000

        logger.info(f"  BOUTs: {bout_count}")
        logger.info(f"  Duration: {duration_sec:.2f}s")
        logger.info(f"  Saved: {output_path.name}")

        # Update stats
        stats['individuals'][individual_id] = {
            'num_recordings': len(recordings),
            'num_bouts': bout_count,
            'duration_sec': duration_sec,
            'audio_file': output_path.name
        }
        stats['total_recordings'] += len(recordings)
        stats['total_bouts'] += bout_count
        stats['total_duration_sec'] += duration_sec

    # Save statistics
    stats_file = output_dir / "extraction_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Individuals processed: {len(stats['individuals'])}")
    logger.info(f"Total recordings: {stats['total_recordings']}")
    logger.info(f"Total BOUTs: {stats['total_bouts']}")
    logger.info(f"Total duration: {stats['total_duration_sec']/60:.2f} minutes")
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"Statistics saved: {stats_file}")

    # Create summary table
    summary_rows = []
    for ind_id, ind_stats in sorted(stats['individuals'].items()):
        summary_rows.append({
            'Individual': ind_id,
            'Recordings': ind_stats['num_recordings'],
            'BOUTs': ind_stats['num_bouts'],
            'Duration (min)': f"{ind_stats['duration_sec']/60:.2f}",
            'File': ind_stats['audio_file']
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "extraction_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    logger.info(f"Summary table saved: {summary_csv}")
    logger.info("\n✓ BOUT extraction complete!")


if __name__ == "__main__":
    main()
