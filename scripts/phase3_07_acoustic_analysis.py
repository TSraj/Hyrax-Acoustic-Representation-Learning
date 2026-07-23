#!/usr/bin/env python3
"""
Phase 3 - Step 7: Acoustic Characteristics Analysis
Analyzes which acoustic characteristics predict successful transfer from speech models.
Research Question 4: frequency range, noise, duration, taxonomic group.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.logging_utils import setup_logger


def extract_acoustic_features(audio_path, sr=16000):
    """Extract acoustic characteristics from audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=sr)

        # Duration
        duration = len(y) / sr

        # Frequency range (spectral analysis)
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)

        # Frequency range: find range containing 90% of energy
        energy_per_freq = np.sum(S, axis=1)
        cumsum_energy = np.cumsum(energy_per_freq)
        total_energy = cumsum_energy[-1]

        idx_5 = np.argmax(cumsum_energy >= 0.05 * total_energy)
        idx_95 = np.argmax(cumsum_energy >= 0.95 * total_energy)

        freq_min = freqs[idx_5]
        freq_max = freqs[idx_95]
        freq_range = freq_max - freq_min

        # Spectral centroid (dominant frequency)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        # Signal-to-noise ratio estimate (using spectral flatness)
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        snr_estimate = -10 * np.log10(spectral_flatness + 1e-10)  # Higher = less noisy

        # RMS energy
        rms = np.mean(librosa.feature.rms(y=y))

        return {
            'duration': duration,
            'freq_min': freq_min,
            'freq_max': freq_max,
            'freq_range': freq_range,
            'spectral_centroid': spectral_centroid,
            'snr_estimate': snr_estimate,
            'rms_energy': rms,
            'valid': True
        }
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def load_predictions(model_name, task, condition='zero_shot'):
    """Load model predictions and ground truth."""
    if condition == 'zero_shot':
        result_file = Path(f"outputs/phase3/zero_shot/{task}/{model_name}/results.json")
    else:  # fine_tuned
        result_file = Path(f"outputs/phase3/fine_tuning/{task}/{model_name}/fine_tuning_results.json")

    if not result_file.exists():
        return None

    with open(result_file, 'r') as f:
        return json.load(f)


def analyze_per_sample_performance(manifest_path, predictions, model_name, task, logger):
    """Analyze performance per sample with acoustic features."""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    test_items = manifest['splits']['test']

    rows = []
    logger.info(f"Extracting acoustic features from {len(test_items)} test samples...")

    for i, item in enumerate(test_items):
        if i % 50 == 0:
            logger.info(f"  Processing {i}/{len(test_items)}...")

        # Get file path
        file_path = item['file']
        if not Path(file_path).exists() and not file_path.startswith('outputs/'):
            file_path = f"data/{file_path}"

        # Extract features
        features = extract_acoustic_features(file_path)

        if not features['valid']:
            continue

        # Get species/individual and taxonomic info
        if task == 'species_id':
            label = item['species']
            taxonomic_group = item.get('taxonomic_group', 'unknown')
        else:  # hyrax_id
            label = item['individual']
            taxonomic_group = 'hyrax'

        rows.append({
            'file': file_path,
            'label': label,
            'taxonomic_group': taxonomic_group,
            'duration': features['duration'],
            'freq_min': features['freq_min'],
            'freq_max': features['freq_max'],
            'freq_range': features['freq_range'],
            'spectral_centroid': features['spectral_centroid'],
            'snr_estimate': features['snr_estimate'],
            'rms_energy': features['rms_energy']
        })

    df = pd.DataFrame(rows)
    logger.info(f"✓ Extracted features for {len(df)} samples")

    return df


def compute_correlations(df_features, df_performance, output_dir, model_name, task, logger):
    """Compute correlations between acoustic features and performance."""
    # Merge by label (aggregate per class)
    class_features = df_features.groupby('label').agg({
        'duration': 'mean',
        'freq_min': 'mean',
        'freq_max': 'mean',
        'freq_range': 'mean',
        'spectral_centroid': 'mean',
        'snr_estimate': 'mean',
        'rms_energy': 'mean'
    }).reset_index()

    # Get per-class accuracy from performance
    # This needs to be extracted from confusion matrix or classification report

    logger.info("Computing feature correlations...")

    # For now, compute summary statistics
    summary = df_features.describe()
    summary_file = output_dir / f"acoustic_features_summary_{task}_{model_name}.csv"
    summary.to_csv(summary_file)
    logger.info(f"✓ Feature summary: {summary_file}")

    return class_features


def plot_acoustic_distributions(df, task, output_dir):
    """Plot distributions of acoustic characteristics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    features = ['duration', 'freq_range', 'spectral_centroid',
                'snr_estimate', 'freq_min', 'freq_max']

    for ax, feature in zip(axes.flat, features):
        if feature in df.columns:
            ax.hist(df[feature], bins=30, color='#0173B2', alpha=0.7, edgecolor='black')
            ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.grid(alpha=0.3)

    plt.suptitle(f'Acoustic Characteristics Distribution\n{task.replace("_", " ").title()}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / f"acoustic_distributions_{task}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def plot_taxonomic_comparison(df, task, output_dir):
    """Compare acoustic features across taxonomic groups."""
    if task != 'species_id' or 'taxonomic_group' not in df.columns:
        return None

    # Box plots for key features by taxonomic group
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    features = ['duration', 'freq_range', 'snr_estimate']

    for ax, feature in zip(axes, features):
        groups = df.groupby('taxonomic_group')[feature].apply(list)
        data = [groups[g] for g in groups.index]

        bp = ax.boxplot(data, labels=groups.index, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#0173B2')
            patch.set_alpha(0.7)

        ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=11)
        ax.set_xlabel('Taxonomic Group', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Acoustic Features by Taxonomic Group',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / f"taxonomic_comparison_{task}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file


def create_acoustic_report(df, task, output_dir):
    """Create report on acoustic characteristics."""
    report_file = output_dir / f"acoustic_characteristics_report_{task}.txt"

    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ACOUSTIC CHARACTERISTICS ANALYSIS\n")
        f.write("Research Question 4: Predicting Transfer Success\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Task: {task}\n")
        f.write(f"Total samples analyzed: {len(df)}\n\n")

        f.write("=" * 80 + "\n")
        f.write("DURATION STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"  Mean: {df['duration'].mean():.2f} seconds\n")
        f.write(f"  Std:  {df['duration'].std():.2f} seconds\n")
        f.write(f"  Min:  {df['duration'].min():.2f} seconds\n")
        f.write(f"  Max:  {df['duration'].max():.2f} seconds\n\n")

        f.write("=" * 80 + "\n")
        f.write("FREQUENCY RANGE STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"  Mean range: {df['freq_range'].mean():.0f} Hz\n")
        f.write(f"  Mean min freq: {df['freq_min'].mean():.0f} Hz\n")
        f.write(f"  Mean max freq: {df['freq_max'].mean():.0f} Hz\n")
        f.write(f"  Mean spectral centroid: {df['spectral_centroid'].mean():.0f} Hz\n\n")

        f.write("=" * 80 + "\n")
        f.write("SIGNAL QUALITY STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"  Mean SNR estimate: {df['snr_estimate'].mean():.2f} dB\n")
        f.write(f"  Mean RMS energy: {df['rms_energy'].mean():.4f}\n\n")

        if 'taxonomic_group' in df.columns:
            f.write("=" * 80 + "\n")
            f.write("BY TAXONOMIC GROUP\n")
            f.write("=" * 80 + "\n\n")

            for group in df['taxonomic_group'].unique():
                group_df = df[df['taxonomic_group'] == group]
                f.write(f"{group}:\n")
                f.write(f"  Samples: {len(group_df)}\n")
                f.write(f"  Duration: {group_df['duration'].mean():.2f}s\n")
                f.write(f"  Freq range: {group_df['freq_range'].mean():.0f} Hz\n")
                f.write(f"  SNR: {group_df['snr_estimate'].mean():.2f} dB\n\n")

    return report_file


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Acoustic Characteristics Analysis")
    parser.add_argument("--task", required=True, choices=["species_id", "hyrax_id"])
    parser.add_argument("--model", required=True, help="Model to analyze")
    parser.add_argument("--debug", action="store_true", help="Limit samples for testing")
    args = parser.parse_args()

    logger = setup_logger(f"Phase3_AcousticAnalysis_{args.task}_{args.model}")

    logger.info("=" * 80)
    logger.info("ACOUSTIC CHARACTERISTICS ANALYSIS")
    logger.info("=" * 80)

    # Load manifest
    manifest_path = Path(f"outputs/phase3/manifests/{args.task}_manifest.json")

    # Analyze acoustic features
    logger.info(f"\nAnalyzing task: {args.task}")
    logger.info(f"Model: {args.model}")

    df_features = analyze_per_sample_performance(
        manifest_path, None, args.model, args.task, logger
    )

    if args.debug:
        df_features = df_features.head(100)
        logger.info(f"[DEBUG] Limited to {len(df_features)} samples")

    # Output directory
    output_dir = Path("outputs/phase3/acoustic_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    features_file = output_dir / f"acoustic_features_{args.task}.csv"
    df_features.to_csv(features_file, index=False)
    logger.info(f"\n✓ Features saved: {features_file}")

    # Generate visualizations
    logger.info("\nGenerating distributions plot...")
    dist_file = plot_acoustic_distributions(df_features, args.task, output_dir)
    logger.info(f"✓ {dist_file}")

    logger.info("\nGenerating taxonomic comparison...")
    tax_file = plot_taxonomic_comparison(df_features, args.task, output_dir)
    if tax_file:
        logger.info(f"✓ {tax_file}")

    # Create report
    logger.info("\nCreating acoustic report...")
    report_file = create_acoustic_report(df_features, args.task, output_dir)
    logger.info(f"✓ {report_file}")

    logger.info("\n" + "=" * 80)
    logger.info("ACOUSTIC ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
