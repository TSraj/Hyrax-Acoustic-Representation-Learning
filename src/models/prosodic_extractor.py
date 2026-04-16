"""
Prosodic Feature Extractor

Extracts prosodic features (pitch, energy, duration) from audio signals.
These features capture temporal and tonal characteristics important for
individual identification in animal vocalizations.

Prosodic features include:
- Pitch (F0): Fundamental frequency and its statistics
- Energy: Signal amplitude/loudness patterns
- Duration: Temporal characteristics

Author: Raj
Date: 2026-04-16
"""

import numpy as np
import librosa
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ProsodicFeatureExtractor:
    """
    Extract prosodic features from audio signals.

    Captures pitch, energy, and temporal characteristics that are
    important for individual identification.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 512,
        frame_length: int = 2048,
        fmin: float = 50.0,
        fmax: float = 8000.0,
        n_statistics: int = 10
    ):
        """
        Initialize prosodic feature extractor.

        Args:
            sample_rate: Audio sample rate
            hop_length: Number of samples between frames
            frame_length: Frame size for analysis
            fmin: Minimum frequency for pitch detection (Hz)
            fmax: Maximum frequency for pitch detection (Hz)
            n_statistics: Number of statistical features to compute
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.fmin = fmin
        self.fmax = fmax
        self.n_statistics = n_statistics

        logger.info(f"Initialized ProsodicFeatureExtractor (sr={sample_rate})")

    def extract_pitch_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract pitch-related features using pyin algorithm.

        Args:
            audio: Audio signal array

        Returns:
            Pitch feature vector with statistics
        """
        try:
            # Extract pitch using pyin (probabilistic YIN)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                frame_length=self.frame_length
            )

            # Remove NaN values for statistics
            f0_voiced = f0[~np.isnan(f0)]

            if len(f0_voiced) == 0:
                # No voiced segments detected, return zeros
                return np.zeros(self.n_statistics)

            # Compute statistics
            features = [
                np.mean(f0_voiced),              # Mean F0
                np.std(f0_voiced),               # F0 std dev
                np.min(f0_voiced),               # Min F0
                np.max(f0_voiced),               # Max F0
                np.ptp(f0_voiced),               # F0 range (peak-to-peak)
                np.median(f0_voiced),            # Median F0
                np.percentile(f0_voiced, 25),    # 25th percentile
                np.percentile(f0_voiced, 75),    # 75th percentile
                np.mean(voiced_flag),            # Voicing ratio
                np.mean(voiced_probs[~np.isnan(voiced_probs)])  # Mean voicing confidence
            ]

            return np.array(features)

        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
            return np.zeros(self.n_statistics)

    def extract_energy_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract energy-related features (RMS energy).

        Args:
            audio: Audio signal array

        Returns:
            Energy feature vector with statistics
        """
        # Compute RMS energy
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]

        # Compute statistics
        features = [
            np.mean(rms),           # Mean energy
            np.std(rms),            # Energy std dev
            np.min(rms),            # Min energy
            np.max(rms),            # Max energy
            np.ptp(rms),            # Energy range
            np.median(rms),         # Median energy
            np.percentile(rms, 25), # 25th percentile
            np.percentile(rms, 75), # 75th percentile
        ]

        return np.array(features)

    def extract_duration_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract duration and temporal features.

        Args:
            audio: Audio signal array

        Returns:
            Duration feature vector
        """
        # Total duration
        duration_sec = len(audio) / self.sample_rate

        # Zero crossing rate (relates to frequency content and noisiness)
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]

        features = [
            duration_sec,           # Total duration in seconds
            np.mean(zcr),          # Mean zero crossing rate
            np.std(zcr),           # ZCR std dev
            np.max(zcr),           # Max ZCR
        ]

        return np.array(features)

    def extract_spectral_temporal_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract additional spectral-temporal features.

        Args:
            audio: Audio signal array

        Returns:
            Spectral-temporal feature vector
        """
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.frame_length
        )[0]

        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.frame_length
        )[0]

        features = [
            np.mean(spectral_centroids),   # Mean spectral centroid
            np.std(spectral_centroids),    # Centroid variation
            np.mean(spectral_rolloff),     # Mean spectral rolloff
            np.std(spectral_rolloff),      # Rolloff variation
        ]

        return np.array(features)

    def extract(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all prosodic features from audio.

        Args:
            audio: Audio signal array (mono, sampled at self.sample_rate)

        Returns:
            Dictionary containing:
                - 'features': Combined feature vector
                - 'pitch': Pitch features
                - 'energy': Energy features
                - 'duration': Duration features
                - 'spectral_temporal': Spectral-temporal features
        """
        # Extract individual feature groups
        pitch_features = self.extract_pitch_features(audio)
        energy_features = self.extract_energy_features(audio)
        duration_features = self.extract_duration_features(audio)
        spectral_temporal_features = self.extract_spectral_temporal_features(audio)

        # Concatenate all features
        combined_features = np.concatenate([
            pitch_features,
            energy_features,
            duration_features,
            spectral_temporal_features
        ])

        return {
            'features': combined_features,
            'pitch': pitch_features,
            'energy': energy_features,
            'duration': duration_features,
            'spectral_temporal': spectral_temporal_features
        }

    def get_feature_names(self) -> list:
        """
        Get names of all features in order.

        Returns:
            List of feature names
        """
        pitch_names = [
            'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 'pitch_range',
            'pitch_median', 'pitch_q25', 'pitch_q75', 'voicing_ratio', 'voicing_confidence'
        ]

        energy_names = [
            'energy_mean', 'energy_std', 'energy_min', 'energy_max',
            'energy_range', 'energy_median', 'energy_q25', 'energy_q75'
        ]

        duration_names = [
            'duration_sec', 'zcr_mean', 'zcr_std', 'zcr_max'
        ]

        spectral_temporal_names = [
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std'
        ]

        return pitch_names + energy_names + duration_names + spectral_temporal_names

    def get_num_features(self) -> int:
        """Get total number of features."""
        return len(self.get_feature_names())

    def extract_batch(
        self,
        audio_paths: list,
        pooling_method: str = 'mean',
        verbose: bool = True
    ) -> Tuple[np.ndarray, list]:
        """
        Extract prosodic features from a batch of audio files.

        Args:
            audio_paths: List of paths to audio files
            pooling_method: Pooling method (prosodic features are already aggregated)
            verbose: Whether to show progress bar

        Returns:
            features: Array of shape (n_samples, n_features)
            successful_paths: List of paths that were successfully processed
        """
        import soundfile as sf
        from tqdm import tqdm

        features_list = []
        successful_paths = []

        iterator = tqdm(audio_paths, desc="Extracting prosodic features") if verbose else audio_paths

        for audio_path in iterator:
            try:
                # Load audio
                audio, sr = sf.read(str(audio_path))

                # Convert to mono if stereo
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)

                # Resample if needed (prosodic extractor expects self.sample_rate)
                if sr != self.sample_rate:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

                # Extract features
                result = self.extract(audio)
                features_list.append(result['features'])
                successful_paths.append(str(audio_path))

            except Exception as e:
                logger.warning(f"Failed to extract prosodic features from {audio_path}: {e}")
                continue

        if len(features_list) == 0:
            logger.error("No prosodic features were successfully extracted!")
            return np.array([]), []

        features = np.array(features_list)
        logger.info(f"Extracted prosodic features: {features.shape}")

        return features, successful_paths

    def __repr__(self) -> str:
        return (f"ProsodicFeatureExtractor(sr={self.sample_rate}, "
                f"n_features={self.get_num_features()})")


def create_default_prosodic_extractor(sample_rate: int = 16000) -> ProsodicFeatureExtractor:
    """
    Create prosodic feature extractor with default settings.

    Args:
        sample_rate: Audio sample rate

    Returns:
        Configured ProsodicFeatureExtractor instance
    """
    return ProsodicFeatureExtractor(
        sample_rate=sample_rate,
        hop_length=512,
        frame_length=2048,
        fmin=50.0,
        fmax=8000.0,
        n_statistics=10
    )
