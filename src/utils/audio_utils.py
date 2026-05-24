"""
Audio utility functions for loading, processing, and analyzing audio files.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional


def load_audio(
    file_path: str,
    target_sr: int = 16000,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and optionally resample and convert to mono.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default: 16000 for wav2vec)
        mono: Convert to mono if True

    Returns:
        Tuple of (audio array, sample rate)
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=mono)
        return audio, sr
    except Exception as e:
        raise ValueError(f"Error loading audio file {file_path}: {str(e)}")


def get_audio_info(file_path: str) -> dict:
    """
    Get metadata information about an audio file.

    Args:
        file_path: Path to audio file

    Returns:
        Dictionary containing audio metadata
    """
    try:
        info = sf.info(file_path)
        return {
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'duration': info.duration,
            'frames': info.frames,
            'format': info.format,
            'subtype': info.subtype
        }
    except Exception as e:
        raise ValueError(f"Error getting audio info for {file_path}: {str(e)}")


def normalize_audio(audio: np.ndarray, method: str = 'peak') -> np.ndarray:
    """
    Normalize audio signal.

    Args:
        audio: Audio signal array
        method: Normalization method ('peak' or 'rms')

    Returns:
        Normalized audio array
    """
    if method == 'peak':
        # Peak normalization
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio
    elif method == 'rms':
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            return audio / rms
        return audio
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def trim_silence(
    audio: np.ndarray,
    sr: int,
    threshold: float = 0.01,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.

    Args:
        audio: Audio signal array
        sr: Sample rate
        threshold: Silence threshold (amplitude)
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis

    Returns:
        Trimmed audio array
    """
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=20 * np.log10(threshold) if threshold > 0 else 60,
        frame_length=frame_length,
        hop_length=hop_length
    )
    return trimmed


def validate_audio_duration(
    audio: np.ndarray,
    sr: int,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None
) -> bool:
    """
    Check if audio duration is within specified range.

    Args:
        audio: Audio signal array
        sr: Sample rate
        min_duration: Minimum duration in seconds (optional)
        max_duration: Maximum duration in seconds (optional)

    Returns:
        True if duration is valid, False otherwise
    """
    duration = len(audio) / sr

    if min_duration is not None and duration < min_duration:
        return False
    if max_duration is not None and duration > max_duration:
        return False

    return True


def audio_to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono by averaging channels.

    Args:
        audio: Audio signal array (can be mono or stereo)

    Returns:
        Mono audio array
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        return np.mean(audio, axis=0)
    else:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")


def chunk_audio(
    audio: np.ndarray,
    sr: int,
    chunk_size: float,
    overlap: float = 0.0
) -> list:
    """
    Split audio into overlapping chunks.

    Args:
        audio: Audio signal array
        sr: Sample rate
        chunk_size: Size of each chunk in seconds
        overlap: Overlap between chunks in seconds

    Returns:
        List of audio chunks
    """
    chunk_samples = int(chunk_size * sr)
    hop_samples = int((chunk_size - overlap) * sr)

    chunks = []
    start = 0

    while start < len(audio):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]

        # Only add chunk if it's long enough
        if len(chunk) >= hop_samples:
            chunks.append(chunk)

        start += hop_samples

        # Break if we've reached the end
        if end >= len(audio):
            break

    return chunks


def find_audio_files(directory: Path, recursive: bool = True) -> list:
    """
    Find all audio files (WAV, MP3, FLAC) in a directory.

    Args:
        directory: Directory to search
        recursive: Search recursively if True

    Returns:
        List of audio file paths
    """
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.WAV', '*.MP3', '*.FLAC']
    audio_files = []

    for ext in audio_extensions:
        if recursive:
            audio_files.extend(directory.glob(f"**/{ext}"))
        else:
            audio_files.extend(directory.glob(ext))

    # Filter out macOS metadata files
    audio_files = [f for f in audio_files if not f.name.startswith('._')]

    return sorted(audio_files)
