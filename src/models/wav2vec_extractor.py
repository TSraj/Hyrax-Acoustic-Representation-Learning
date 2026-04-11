"""
Wav2Vec feature extractor for extracting layer-wise representations.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

from src.utils.audio_utils import load_audio
from src.utils.logging_utils import setup_logger


class Wav2VecFeatureExtractor:
    """Extracts features from audio using pretrained Wav2Vec models."""

    def __init__(
        self,
        model_name: str,
        config: dict,
        device: str = None,
        log_level: str = "INFO"
    ):
        """
        Initialize Wav2Vec feature extractor.

        Args:
            model_name: Name of the model ('wav2vec2_base' or 'wav2vec2_xlsr')
            config: Configuration dictionary
            device: Device to run model on ('cpu', 'cuda', or 'mps')
            log_level: Logging level
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__, log_level)

        # Get model configuration
        self.model_name = model_name
        self.model_config = config['models'][model_name]
        self.model_id = self.model_config['model_id']
        self.num_layers = self.model_config['num_layers']

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.logger.info(f"Using device: {self.device}")

        # Load model and feature extractor
        self.logger.info(f"Loading model: {self.model_id}")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_id)
        self.model = Wav2Vec2Model.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

        self.logger.info(f"Model loaded successfully: {self.model_name}")

    def extract_features_from_audio(
        self,
        audio: np.ndarray,
        sr: int,
        extract_all_layers: bool = True,
        layers: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        """
        Extract features from a single audio array.

        Args:
            audio: Audio array
            sr: Sample rate (should be 16000 for wav2vec)
            extract_all_layers: Whether to extract from all layers
            layers: Specific layers to extract (if extract_all_layers is False)

        Returns:
            Dictionary mapping layer index to feature array
        """
        # Process audio
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Get hidden states from all layers
        hidden_states = outputs.hidden_states  # Tuple of (batch_size, sequence_length, hidden_size)

        # Determine which layers to extract
        if extract_all_layers:
            layer_indices = range(len(hidden_states))
        elif layers is not None:
            layer_indices = layers
        else:
            layer_indices = [len(hidden_states) - 1]  # Last layer only

        # Extract features from specified layers
        features = {}
        for layer_idx in layer_indices:
            if layer_idx < len(hidden_states):
                layer_output = hidden_states[layer_idx]  # (batch_size, seq_len, hidden_dim)
                features[layer_idx] = layer_output.cpu().numpy()[0]  # Remove batch dimension

        return features

    def extract_features_from_file(
        self,
        file_path: str,
        extract_all_layers: bool = True,
        layers: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        """
        Extract features from an audio file.

        Args:
            file_path: Path to audio file
            extract_all_layers: Whether to extract from all layers
            layers: Specific layers to extract

        Returns:
            Dictionary mapping layer index to feature array
        """
        # Load audio
        audio, sr = load_audio(file_path, target_sr=16000, mono=True)

        # Extract features
        return self.extract_features_from_audio(audio, sr, extract_all_layers, layers)

    def extract_features_from_dataset(
        self,
        data_dir: str,
        file_pattern: str = "**/*.wav",
        extract_all_layers: bool = True,
        layers: Optional[List[int]] = None
    ) -> Dict:
        """
        Extract features from all files in a dataset directory.

        Args:
            data_dir: Directory containing audio files
            file_pattern: Glob pattern for audio files
            extract_all_layers: Whether to extract from all layers
            layers: Specific layers to extract

        Returns:
            Dictionary containing features and metadata
        """
        data_path = Path(data_dir)
        audio_files = sorted(list(data_path.glob(file_pattern)))

        self.logger.info(f"Extracting features from {len(audio_files)} files...")

        all_features = {}
        metadata = {
            'model_name': self.model_name,
            'model_id': self.model_id,
            'num_files': len(audio_files),
            'layers': [],
            'file_paths': [],
            'labels': []
        }

        for file_path in tqdm(audio_files, desc=f"Extracting features ({self.model_name})"):
            try:
                # Extract features
                features = self.extract_features_from_file(
                    str(file_path),
                    extract_all_layers,
                    layers
                )

                # Store features
                file_key = str(file_path.relative_to(data_path))
                all_features[file_key] = features

                # Extract label from directory structure or filename
                label = self._extract_label(file_path, data_path)
                metadata['file_paths'].append(file_key)
                metadata['labels'].append(label)

            except Exception as e:
                self.logger.error(f"Error extracting features from {file_path}: {e}")

        # Record which layers were extracted
        if all_features:
            first_file = next(iter(all_features.values()))
            metadata['layers'] = sorted(list(first_file.keys()))

        self.logger.info(f"Feature extraction complete: {len(all_features)} files processed")

        return {
            'features': all_features,
            'metadata': metadata
        }

    def _extract_label(self, file_path: Path, data_path: Path) -> str:
        """
        Extract label from file path.

        Args:
            file_path: Path to audio file
            data_path: Base data directory

        Returns:
            Label string
        """
        # For Macaque: parent directory name is the individual ID
        # For Zebra Finch: first part of filename is individual ID
        relative_path = file_path.relative_to(data_path)

        if len(relative_path.parts) > 1:
            # Label is parent directory (e.g., macaque/AL/file.wav -> AL)
            return relative_path.parts[-2]
        else:
            # Label is from filename (e.g., zebra_finch/HPiHPi4748_*.wav -> HPiHPi4748)
            return file_path.stem.split('_')[0]

    def save_features(self, features_dict: Dict, output_path: str):
        """
        Save extracted features to disk.

        Args:
            features_dict: Dictionary containing features and metadata
            output_path: Path to save features (will be saved as .npz)
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert nested dict structure to saveable format
        save_dict = {
            'metadata': features_dict['metadata']
        }

        # Flatten features for saving
        for file_key, layer_features in features_dict['features'].items():
            for layer_idx, features in layer_features.items():
                key = f"{file_key}__layer_{layer_idx}"
                save_dict[key] = features

        np.savez_compressed(output_file, **save_dict)
        self.logger.info(f"Features saved to {output_path}")

    @staticmethod
    def load_features(feature_path: str) -> Dict:
        """
        Load features from disk.

        Args:
            feature_path: Path to saved features (.npz file)

        Returns:
            Dictionary containing features and metadata
        """
        data = np.load(feature_path, allow_pickle=True)

        # Reconstruct original structure
        metadata = data['metadata'].item()
        features = {}

        for key in data.keys():
            if key != 'metadata' and '__layer_' in key:
                file_key, layer_part = key.rsplit('__layer_', 1)
                layer_idx = int(layer_part)

                if file_key not in features:
                    features[file_key] = {}

                features[file_key][layer_idx] = data[key]

        return {
            'features': features,
            'metadata': metadata
        }
