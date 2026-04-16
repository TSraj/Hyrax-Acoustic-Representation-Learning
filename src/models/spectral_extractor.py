"""
Spectral Feature Extractor

Extracts traditional spectral and time-domain features from audio files.
These handcrafted features serve as baselines for comparison with learned representations.

Features extracted:
- Spectral centroid (brightness)
- Spectral bandwidth (spread)
- Spectral rolloff (frequency below which X% of energy is contained)
- Spectral contrast (peak-valley differences in spectrum)
- Zero-crossing rate (time-domain feature)
- RMS energy (amplitude)
- Chroma features (pitch class profile)

Author: Raj
Date: 2026-04-11
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SpectralExtractor:
    """
    Extract spectral and time-domain features from audio files.

    These features capture different aspects of the audio signal:
    - Spectral: Frequency-domain characteristics
    - Temporal: Time-domain characteristics
    - Tonal: Pitch-related characteristics
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,  # Reduced from 2048 for short audio
        hop_length: int = 256,  # Reduced from 512
        n_chroma: int = 12,
        n_contrast_bands: int = 6,
        window: str = 'hamming',
        include_chroma: bool = True,
        include_contrast: bool = True,
    ):
        """
        Initialize spectral feature extractor.

        Args:
            sample_rate: Target sampling rate
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_chroma: Number of chroma bins
            n_contrast_bands: Number of spectral contrast bands
            window: Window function type
            include_chroma: Whether to include chroma features
            include_contrast: Whether to include spectral contrast
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_chroma = n_chroma
        self.n_contrast_bands = n_contrast_bands
        self.window = window
        self.include_chroma = include_chroma
        self.include_contrast = include_contrast

        # Calculate expected feature dimension
        self.feature_dim = self._calculate_feature_dim()

        logger.info(f"Initialized spectral extractor: "
                   f"chroma={include_chroma}, contrast={include_contrast}, "
                   f"total dim={self.feature_dim}")

    def _calculate_feature_dim(self) -> int:
        """Calculate total feature dimension."""
        dim = 0
        dim += 1  # spectral centroid
        dim += 1  # spectral bandwidth
        dim += 1  # spectral rolloff
        dim += 1  # zero crossing rate
        dim += 1  # RMS energy

        if self.include_contrast:
            dim += self.n_contrast_bands + 1  # contrast bands + valley

        if self.include_chroma:
            dim += self.n_chroma

        return dim

    def extract_from_audio(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract spectral features from audio array.

        Args:
            audio: Audio time series (1D numpy array)
            sr: Sample rate of audio (if different from self.sample_rate)

        Returns:
            Spectral features of shape (n_frames, feature_dim)
        """
        if sr is None:
            sr = self.sample_rate

        # Resample if necessary
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        features = []

        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window
        )
        features.append(centroid.T)

        # Spectral bandwidth (spread around centroid)
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window
        )
        features.append(bandwidth.T)

        # Spectral rolloff (frequency below which X% of energy)
        rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            roll_percent=0.85
        )
        features.append(rolloff.T)

        # Zero crossing rate (time-domain)
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        features.append(zcr.T)

        # RMS energy
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        features.append(rms.T)

        # Spectral contrast (peak-valley differences)
        if self.include_contrast:
            contrast = librosa.feature.spectral_contrast(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_bands=self.n_contrast_bands
            )
            features.append(contrast.T)

        # Chroma features (pitch class profile)
        if self.include_chroma:
            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_chroma=self.n_chroma
            )
            features.append(chroma.T)

        # Concatenate all features
        all_features = np.concatenate(features, axis=1)

        return all_features

    def extract_from_file(
        self,
        audio_path: Union[str, Path]
    ) -> np.ndarray:
        """
        Extract spectral features from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Spectral features of shape (n_frames, feature_dim)
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Extract features
        features = self.extract_from_audio(audio, sr)

        return features

    def pool_features(
        self,
        features: np.ndarray,
        method: str = 'mean'
    ) -> np.ndarray:
        """
        Pool frame-level features into a single vector.

        Args:
            features: Frame-level features of shape (n_frames, feature_dim)
            method: Pooling method ('mean', 'max', 'mean_std', 'first', 'last')

        Returns:
            Pooled feature vector
        """
        if method == 'mean':
            return np.mean(features, axis=0)

        elif method == 'max':
            return np.max(features, axis=0)

        elif method == 'mean_std':
            # Concatenate mean and std
            mean_feat = np.mean(features, axis=0)
            std_feat = np.std(features, axis=0)
            return np.concatenate([mean_feat, std_feat])

        elif method == 'first':
            return features[0]

        elif method == 'last':
            return features[-1]

        else:
            raise ValueError(f"Unknown pooling method: {method}")

    def extract_and_pool(
        self,
        audio_path: Union[str, Path],
        pooling_method: str = 'mean'
    ) -> np.ndarray:
        """
        Extract and pool spectral features in one step.

        Args:
            audio_path: Path to audio file
            pooling_method: Pooling method to use

        Returns:
            Pooled feature vector
        """
        features = self.extract_from_file(audio_path)
        pooled = self.pool_features(features, pooling_method)
        return pooled

    def extract_batch(
        self,
        audio_paths: List[Union[str, Path]],
        pooling_method: str = 'mean',
        verbose: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from multiple audio files.

        Args:
            audio_paths: List of paths to audio files
            pooling_method: Pooling method to use
            verbose: Whether to show progress

        Returns:
            features: Array of shape (n_samples, feature_dim)
            successful_paths: List of successfully processed file paths
        """
        features_list = []
        successful_paths = []
        failed_count = 0

        for i, path in enumerate(audio_paths):
            try:
                feat = self.extract_and_pool(path, pooling_method)
                features_list.append(feat)
                successful_paths.append(str(path))

                if verbose and (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(audio_paths)} files")

            except Exception as e:
                logger.warning(f"Failed to extract features from {path}: {e}")
                failed_count += 1
                continue

        if failed_count > 0:
            logger.warning(f"Failed to process {failed_count}/{len(audio_paths)} files")

        features_array = np.array(features_list)

        if verbose:
            logger.info(f"Extracted spectral features: {features_array.shape}")

        return features_array, successful_paths

    def get_feature_names(self) -> List[str]:
        """
        Get feature names for interpretability.

        Returns:
            List of feature names
        """
        names = [
            'spectral_centroid',
            'spectral_bandwidth',
            'spectral_rolloff',
            'zero_crossing_rate',
            'rms_energy'
        ]

        # Spectral contrast bands
        if self.include_contrast:
            for i in range(self.n_contrast_bands + 1):
                names.append(f'spectral_contrast_{i}')

        # Chroma features
        if self.include_chroma:
            chroma_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            for i in range(self.n_chroma):
                names.append(f'chroma_{chroma_names[i]}')

        return names

    def __repr__(self) -> str:
        return (f"SpectralExtractor(chroma={self.include_chroma}, "
                f"contrast={self.include_contrast}, "
                f"feature_dim={self.feature_dim})")


def create_default_spectral_extractor(sample_rate: int = 16000) -> SpectralExtractor:
    """
    Create spectral extractor with standard configuration.

    Args:
        sample_rate: Target sampling rate

    Returns:
        Configured SpectralExtractor instance
    """
    return SpectralExtractor(
        sample_rate=sample_rate,
        n_fft=512,  # Reduced for short audio
        hop_length=256,  # Reduced for short audio
        n_chroma=12,
        n_contrast_bands=6,
        include_chroma=True,
        include_contrast=True
    )
