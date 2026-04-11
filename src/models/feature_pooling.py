"""
Feature pooling strategies for converting sequence-level features to sample-level.
"""

import numpy as np
from typing import Dict, List
from src.utils.logging_utils import setup_logger


class FeaturePooler:
    """Pools sequence-level features into fixed-size vectors."""

    def __init__(self, config: dict, log_level: str = "INFO"):
        """
        Initialize feature pooler.

        Args:
            config: Configuration dictionary
            log_level: Logging level
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__, log_level)

        # Get pooling methods from config
        self.pooling_methods = config['pooling']['methods']
        self.default_method = config['pooling']['default']

    def mean_pool(self, features: np.ndarray) -> np.ndarray:
        """
        Mean pooling over time dimension.

        Args:
            features: Feature array of shape (time_steps, hidden_dim)

        Returns:
            Pooled features of shape (hidden_dim,)
        """
        return np.mean(features, axis=0)

    def max_pool(self, features: np.ndarray) -> np.ndarray:
        """
        Max pooling over time dimension.

        Args:
            features: Feature array of shape (time_steps, hidden_dim)

        Returns:
            Pooled features of shape (hidden_dim,)
        """
        return np.max(features, axis=0)

    def first_token_pool(self, features: np.ndarray) -> np.ndarray:
        """
        Use first token as representation.

        Args:
            features: Feature array of shape (time_steps, hidden_dim)

        Returns:
            Pooled features of shape (hidden_dim,)
        """
        return features[0]

    def last_token_pool(self, features: np.ndarray) -> np.ndarray:
        """
        Use last token as representation.

        Args:
            features: Feature array of shape (time_steps, hidden_dim)

        Returns:
            Pooled features of shape (hidden_dim,)
        """
        return features[-1]

    def weighted_pool(self, features: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
        """
        Weighted pooling over time dimension.

        Args:
            features: Feature array of shape (time_steps, hidden_dim)
            weights: Weight array of shape (time_steps,). If None, uses softmax of mean.

        Returns:
            Pooled features of shape (hidden_dim,)
        """
        if weights is None:
            # Simple attention: compute weights from feature magnitudes
            magnitudes = np.linalg.norm(features, axis=1)  # (time_steps,)
            weights = np.exp(magnitudes) / np.sum(np.exp(magnitudes))

        # Apply weights
        weighted_features = features * weights[:, np.newaxis]
        return np.sum(weighted_features, axis=0)

    def pool_features(
        self,
        features: np.ndarray,
        method: str = None
    ) -> np.ndarray:
        """
        Pool features using specified method.

        Args:
            features: Feature array of shape (time_steps, hidden_dim)
            method: Pooling method ('mean', 'max', 'first', 'last', 'weighted')

        Returns:
            Pooled features of shape (hidden_dim,)
        """
        if method is None:
            method = self.default_method

        if method == 'mean':
            return self.mean_pool(features)
        elif method == 'max':
            return self.max_pool(features)
        elif method == 'first':
            return self.first_token_pool(features)
        elif method == 'last':
            return self.last_token_pool(features)
        elif method == 'weighted':
            return self.weighted_pool(features)
        else:
            raise ValueError(f"Unknown pooling method: {method}")

    def pool_layer_features(
        self,
        layer_features: Dict[int, np.ndarray],
        method: str = None
    ) -> Dict[int, np.ndarray]:
        """
        Pool features for all layers.

        Args:
            layer_features: Dictionary mapping layer index to features
            method: Pooling method

        Returns:
            Dictionary mapping layer index to pooled features
        """
        pooled = {}
        for layer_idx, features in layer_features.items():
            pooled[layer_idx] = self.pool_features(features, method)
        return pooled

    def pool_dataset_features(
        self,
        features_dict: Dict,
        method: str = None,
        methods: List[str] = None
    ) -> Dict:
        """
        Pool features for entire dataset.

        Args:
            features_dict: Dictionary containing features and metadata
            method: Single pooling method (if None, use all methods from config)
            methods: List of pooling methods (overrides method parameter)

        Returns:
            Dictionary containing pooled features for each method
        """
        if methods is None:
            if method is not None:
                methods = [method]
            else:
                methods = self.pooling_methods

        self.logger.info(f"Pooling features using methods: {methods}")

        pooled_features = {}

        for pool_method in methods:
            self.logger.info(f"  Pooling with {pool_method}...")

            method_features = {}
            for file_key, layer_features in features_dict['features'].items():
                method_features[file_key] = self.pool_layer_features(
                    layer_features,
                    pool_method
                )

            pooled_features[pool_method] = {
                'features': method_features,
                'metadata': features_dict['metadata'].copy()
            }

        return pooled_features

    def get_pooled_arrays(
        self,
        pooled_features: Dict,
        layer_idx: int,
        pooling_method: str = 'mean'
    ) -> tuple:
        """
        Convert pooled features to numpy arrays for analysis.

        Args:
            pooled_features: Dictionary of pooled features
            layer_idx: Layer index to extract
            pooling_method: Pooling method to use

        Returns:
            Tuple of (feature_array, labels_array)
        """
        method_data = pooled_features[pooling_method]
        features_dict = method_data['features']
        metadata = method_data['metadata']

        # Extract features and labels in consistent order
        feature_list = []
        labels = []

        for file_path, label in zip(metadata['file_paths'], metadata['labels']):
            if file_path in features_dict:
                layer_features = features_dict[file_path]
                if layer_idx in layer_features:
                    feature_list.append(layer_features[layer_idx])
                    labels.append(label)

        return np.array(feature_list), np.array(labels)

    def save_pooled_features(self, pooled_features: Dict, output_path: str):
        """
        Save pooled features to disk.

        Args:
            pooled_features: Dictionary of pooled features
            output_path: Path to save features
        """
        save_dict = {}

        for method, method_data in pooled_features.items():
            # Save metadata
            save_dict[f'{method}__metadata'] = method_data['metadata']

            # Save pooled features
            for file_key, layer_features in method_data['features'].items():
                for layer_idx, features in layer_features.items():
                    key = f"{method}__{file_key}__layer_{layer_idx}"
                    save_dict[key] = features

        np.savez_compressed(output_path, **save_dict)
        self.logger.info(f"Pooled features saved to {output_path}")

    @staticmethod
    def load_pooled_features(feature_path: str) -> Dict:
        """
        Load pooled features from disk.

        Args:
            feature_path: Path to saved features

        Returns:
            Dictionary of pooled features
        """
        data = np.load(feature_path, allow_pickle=True)

        pooled_features = {}

        for key in data.keys():
            if '__metadata' in key:
                method = key.split('__metadata')[0]
                if method not in pooled_features:
                    pooled_features[method] = {'metadata': data[key].item(), 'features': {}}
                else:
                    pooled_features[method]['metadata'] = data[key].item()
            elif '__layer_' in key:
                parts = key.split('__')
                method = parts[0]
                layer_idx = int(parts[-1].replace('layer_', ''))
                file_key = '__'.join(parts[1:-1])

                if method not in pooled_features:
                    pooled_features[method] = {'features': {}, 'metadata': {}}

                if file_key not in pooled_features[method]['features']:
                    pooled_features[method]['features'][file_key] = {}

                pooled_features[method]['features'][file_key][layer_idx] = data[key]

        return pooled_features
