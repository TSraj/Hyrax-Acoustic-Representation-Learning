"""
MFCC Feature Extractor

Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio files.
This serves as a traditional handcrafted acoustic baseline for comparison with wav2vec features.

Features extracted:
- 13 MFCC coefficients
- 13 delta (Δ) coefficients
- 13 delta-delta (ΔΔ) coefficients
Total: 39 features per frame, then pooled over time

Author: Raj
Date: 2026-04-11
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MFCCExtractor:
    """
    Extract MFCC features from audio files.

    MFCCs are widely used in speech and audio processing as a compact
    representation of the spectral envelope. They capture timbral characteristics
    and are standard baselines in audio classification tasks.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 512,  # Reduced from 2048 for short audio
        hop_length: int = 256,  # Reduced from 512
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        window: str = 'hamming',
        include_deltas: bool = True,
        include_delta_deltas: bool = True,
    ):
        """
        Initialize MFCC extractor.

        Args:
            sample_rate: Target sampling rate
            n_mfcc: Number of MFCC coefficients to extract
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of Mel bands
            fmin: Minimum frequency
            fmax: Maximum frequency (None = sample_rate/2)
            window: Window function type
            include_deltas: Whether to include delta coefficients
            include_delta_deltas: Whether to include delta-delta coefficients
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2.0
        self.window = window
        self.include_deltas = include_deltas
        self.include_delta_deltas = include_delta_deltas

        # Calculate expected feature dimension
        self.feature_dim = n_mfcc
        if include_deltas:
            self.feature_dim += n_mfcc
        if include_delta_deltas:
            self.feature_dim += n_mfcc

        logger.info(f"Initialized MFCC extractor: {n_mfcc} MFCCs, "
                   f"deltas={include_deltas}, delta-deltas={include_delta_deltas}, "
                   f"total dim={self.feature_dim}")

    def extract_from_audio(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract MFCC features from audio array.

        Args:
            audio: Audio time series (1D numpy array)
            sr: Sample rate of audio (if different from self.sample_rate)

        Returns:
            MFCC features of shape (n_frames, feature_dim)
        """
        if sr is None:
            sr = self.sample_rate

        # Resample if necessary
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            window=self.window
        )

        # MFCCs are returned as (n_mfcc, n_frames), transpose to (n_frames, n_mfcc)
        features = [mfccs.T]

        # Add delta features
        if self.include_deltas:
            delta = librosa.feature.delta(mfccs)
            features.append(delta.T)

        # Add delta-delta features
        if self.include_delta_deltas:
            delta_delta = librosa.feature.delta(mfccs, order=2)
            features.append(delta_delta.T)

        # Concatenate all features
        all_features = np.concatenate(features, axis=1)

        return all_features

    def extract_from_file(
        self,
        audio_path: Union[str, Path]
    ) -> np.ndarray:
        """
        Extract MFCC features from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            MFCC features of shape (n_frames, feature_dim)
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
        Extract and pool MFCC features in one step.

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
            logger.info(f"Extracted MFCC features: {features_array.shape}")

        return features_array, successful_paths

    def get_feature_names(self) -> List[str]:
        """
        Get feature names for interpretability.

        Returns:
            List of feature names
        """
        names = []

        # MFCC coefficients
        for i in range(self.n_mfcc):
            names.append(f"mfcc_{i}")

        # Delta coefficients
        if self.include_deltas:
            for i in range(self.n_mfcc):
                names.append(f"delta_mfcc_{i}")

        # Delta-delta coefficients
        if self.include_delta_deltas:
            for i in range(self.n_mfcc):
                names.append(f"delta2_mfcc_{i}")

        return names

    def __repr__(self) -> str:
        return (f"MFCCExtractor(n_mfcc={self.n_mfcc}, "
                f"deltas={self.include_deltas}, "
                f"delta_deltas={self.include_delta_deltas}, "
                f"feature_dim={self.feature_dim})")


def create_default_mfcc_extractor(sample_rate: int = 16000) -> MFCCExtractor:
    """
    Create MFCC extractor with standard configuration.

    Args:
        sample_rate: Target sampling rate

    Returns:
        Configured MFCCExtractor instance
    """
    return MFCCExtractor(
        sample_rate=sample_rate,
        n_mfcc=13,
        n_fft=512,  # Reduced for short audio
        hop_length=256,  # Reduced for short audio
        n_mels=128,
        include_deltas=True,
        include_delta_deltas=False  # Disabled for very short audio
    )
