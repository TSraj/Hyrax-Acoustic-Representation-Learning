"""
Audio preprocessor for standardizing audio files across datasets.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from src.utils.audio_utils import (
    load_audio,
    normalize_audio,
    trim_silence,
    validate_audio_duration
)
from src.utils.logging_utils import setup_logger


class AudioPreprocessor:
    """Preprocesses audio files to standardized format for wav2vec models."""

    def __init__(self, config: dict, log_level: str = "INFO"):
        """
        Initialize audio preprocessor.

        Args:
            config: Configuration dictionary
            log_level: Logging level
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__, log_level)

        # Preprocessing parameters
        self.target_sr = config['preprocessing']['target_sample_rate']
        self.mono = config['preprocessing']['channels'] == 1
        self.normalize = config['preprocessing']['normalize']
        self.trim_silence_enabled = config['preprocessing']['trim_silence']
        self.min_duration = config['preprocessing'].get('min_duration')
        self.max_duration = config['preprocessing'].get('max_duration')

    def preprocess_audio(self, file_path: str) -> Tuple[np.ndarray, int, bool]:
        """
        Preprocess a single audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (preprocessed audio, sample rate, success flag)
        """
        try:
            # Load audio
            audio, sr = load_audio(file_path, target_sr=self.target_sr, mono=self.mono)

            # Validate duration
            if not validate_audio_duration(audio, sr, self.min_duration, self.max_duration):
                duration = len(audio) / sr
                self.logger.warning(f"Audio duration {duration:.2f}s out of range for {file_path}")
                return None, sr, False

            # Trim silence if enabled
            if self.trim_silence_enabled:
                audio = trim_silence(audio, sr)

            # Normalize if enabled
            if self.normalize:
                audio = normalize_audio(audio, method='peak')

            return audio, sr, True

        except Exception as e:
            self.logger.error(f"Error preprocessing {file_path}: {e}")
            return None, None, False

    def preprocess_dataset(
        self,
        input_dir: str,
        output_dir: str,
        file_pattern: str = "**/*.wav"
    ) -> Dict:
        """
        Preprocess all audio files in a dataset directory.

        Args:
            input_dir: Input directory containing audio files
            output_dir: Output directory for preprocessed files
            file_pattern: Glob pattern for audio files

        Returns:
            Dictionary containing preprocessing statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all audio files
        audio_files = list(input_path.glob(file_pattern))
        self.logger.info(f"Found {len(audio_files)} audio files in {input_dir}")

        stats = {
            'total_files': len(audio_files),
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'failed_files': []
        }

        # Process each file
        for file_path in tqdm(audio_files, desc="Preprocessing audio"):
            # Maintain directory structure
            relative_path = file_path.relative_to(input_path)
            output_file = output_path / relative_path

            # Create output directory
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Preprocess audio
            audio, sr, success = self.preprocess_audio(str(file_path))

            if success and audio is not None:
                # Save preprocessed audio
                sf.write(str(output_file), audio, sr, subtype='PCM_16')
                stats['successful'] += 1
            else:
                stats['failed'] += 1
                stats['failed_files'].append(str(file_path))

        self.logger.info(
            f"Preprocessing complete: {stats['successful']}/{stats['total_files']} successful"
        )

        return stats

    def preprocess_macaque_subset(self, subset_dir: str, output_dir: str) -> Dict:
        """
        Preprocess Macaque subset.

        Args:
            subset_dir: Directory containing subset
            output_dir: Output directory for preprocessed files

        Returns:
            Preprocessing statistics
        """
        self.logger.info("Preprocessing Macaque subset...")
        input_path = Path(subset_dir) / "macaque"
        output_path = Path(output_dir) / "macaque"

        return self.preprocess_dataset(str(input_path), str(output_path))

    def preprocess_zebra_finch_subset(self, subset_dir: str, output_dir: str) -> Dict:
        """
        Preprocess Zebra Finch subset.

        Args:
            subset_dir: Directory containing subset
            output_dir: Output directory for preprocessed files

        Returns:
            Preprocessing statistics
        """
        self.logger.info("Preprocessing Zebra Finch subset...")
        input_path = Path(subset_dir) / "zebra_finch"
        output_path = Path(output_dir) / "zebra_finch"

        return self.preprocess_dataset(str(input_path), str(output_path))

    def preprocess_all_subsets(
        self,
        subset_dir: str = None,
        output_dir: str = None
    ) -> Dict:
        """
        Preprocess all dataset subsets.

        Args:
            subset_dir: Directory containing subsets
            output_dir: Output directory for preprocessed files

        Returns:
            Combined preprocessing statistics
        """
        if subset_dir is None:
            subset_dir = self.config['paths']['processed_dir'] + "/subsets"
        if output_dir is None:
            output_dir = self.config['paths']['processed_dir'] + "/preprocessed_subsets"

        all_stats = {
            'macaque': self.preprocess_macaque_subset(subset_dir, output_dir),
            'zebra_finch': self.preprocess_zebra_finch_subset(subset_dir, output_dir)
        }

        # Compute total statistics
        total_successful = sum(s['successful'] for s in all_stats.values())
        total_files = sum(s['total_files'] for s in all_stats.values())

        self.logger.info(f"All subsets preprocessed: {total_successful}/{total_files} files successful")

        return all_stats
