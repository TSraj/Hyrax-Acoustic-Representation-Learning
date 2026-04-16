"""
Layer Combiner Module

Combines features from multiple wav2vec layers to create enriched representations.

Modern deep learning often benefits from combining features from different depths:
- Early layers: Low-level acoustic features
- Middle layers: Mid-level phonetic/prosodic features
- Late layers: High-level semantic features

This module implements strategies for combining multi-layer wav2vec representations.

Research question:
"Can we get better representations by combining information from multiple layers?"

Author: Raj
Date: 2026-04-11
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LayerCombiner:
    """
    Combine features from multiple wav2vec layers.

    Supports various combination strategies to investigate whether
    multi-layer fusion improves representation quality.
    """

    def __init__(
        self,
        combination_strategy: str = 'concatenation',
        layer_selection: str = 'specified'
    ):
        """
        Initialize layer combiner.

        Args:
            combination_strategy: How to combine layers
                - 'concatenation': Simple concatenation
                - 'mean': Average across layers
                - 'weighted_mean': Weighted average (learned or specified)
                - 'max': Max pooling across layers
            layer_selection: How to select layers
                - 'specified': Use explicitly specified layers
                - 'early': First 3 layers
                - 'middle': Middle 3 layers
                - 'late': Last 3 layers
                - 'early_middle_late': One from each group
                - 'all': All layers
        """
        self.combination_strategy = combination_strategy
        self.layer_selection = layer_selection
        self.layer_weights = None

        logger.info(f"Initialized LayerCombiner: {combination_strategy}, {layer_selection}")

    def select_layers(
        self,
        total_layers: int,
        specified_layers: Optional[List[int]] = None
    ) -> List[int]:
        """
        Select which layers to combine based on selection strategy.

        Args:
            total_layers: Total number of layers in the model
            specified_layers: Explicitly specified layer indices (if layer_selection='specified')

        Returns:
            List of layer indices to use
        """
        if self.layer_selection == 'specified':
            if specified_layers is None:
                raise ValueError("specified_layers must be provided for 'specified' selection")
            return specified_layers

        elif self.layer_selection == 'early':
            # First 3 layers
            return list(range(min(3, total_layers)))

        elif self.layer_selection == 'middle':
            # Middle 3 layers
            mid = total_layers // 2
            start = max(0, mid - 1)
            end = min(total_layers, mid + 2)
            return list(range(start, end))

        elif self.layer_selection == 'late':
            # Last 3 layers
            start = max(0, total_layers - 3)
            return list(range(start, total_layers))

        elif self.layer_selection == 'early_middle_late':
            # One from each group
            early = 0
            middle = total_layers // 2
            late = total_layers - 1
            return [early, middle, late]

        elif self.layer_selection == 'all':
            # All layers
            return list(range(total_layers))

        else:
            raise ValueError(f"Unknown layer selection: {self.layer_selection}")

    def set_layer_weights(
        self,
        weights: List[float]
    ) -> None:
        """
        Set explicit weights for weighted_mean strategy.

        Args:
            weights: List of weights (one per layer to combine)
        """
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        self.layer_weights = weights
        logger.info(f"Set layer weights: {weights}")

    def combine_layers(
        self,
        layer_features: Dict[int, np.ndarray],
        layers_to_combine: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Combine features from specified layers.

        Args:
            layer_features: Dictionary mapping layer_idx -> features
                           Each features array has shape (n_samples, feature_dim)
            layers_to_combine: Which layers to combine (if None, use all available)

        Returns:
            Combined features of shape (n_samples, combined_dim)
        """
        # Determine which layers to use
        if layers_to_combine is None:
            layers_to_combine = sorted(layer_features.keys())

        # Extract features for selected layers
        selected_features = []
        for layer_idx in layers_to_combine:
            if layer_idx not in layer_features:
                raise ValueError(f"Layer {layer_idx} not found in layer_features")
            selected_features.append(layer_features[layer_idx])

        # Check all features have same number of samples
        n_samples = selected_features[0].shape[0]
        for feat in selected_features:
            if feat.shape[0] != n_samples:
                raise ValueError("All layer features must have same number of samples")

        # Combine based on strategy
        if self.combination_strategy == 'concatenation':
            # Simple concatenation along feature dimension
            combined = np.concatenate(selected_features, axis=1)
            logger.info(f"Concatenated {len(selected_features)} layers: "
                       f"{[f.shape[1] for f in selected_features]} → {combined.shape[1]} dims")

        elif self.combination_strategy == 'mean':
            # Average across layers
            # Stack: (n_layers, n_samples, feature_dim)
            stacked = np.stack(selected_features, axis=0)
            combined = np.mean(stacked, axis=0)
            logger.info(f"Averaged {len(selected_features)} layers: → {combined.shape[1]} dims")

        elif self.combination_strategy == 'weighted_mean':
            # Weighted average
            if self.layer_weights is None:
                # Default: weight inversely proportional to layer depth
                # (early layers get higher weight)
                n_layers = len(selected_features)
                weights = np.array([1.0 / (i + 1) for i in range(n_layers)])
                weights = weights / weights.sum()
                self.layer_weights = weights
                logger.info(f"Using default layer weights: {weights}")

            if len(self.layer_weights) != len(selected_features):
                raise ValueError(f"Number of weights ({len(self.layer_weights)}) "
                               f"!= number of layers ({len(selected_features)})")

            # Weighted average
            stacked = np.stack(selected_features, axis=0)  # (n_layers, n_samples, feature_dim)
            combined = np.average(stacked, axis=0, weights=self.layer_weights)
            logger.info(f"Weighted average of {len(selected_features)} layers: → {combined.shape[1]} dims")

        elif self.combination_strategy == 'max':
            # Max pooling across layers
            stacked = np.stack(selected_features, axis=0)
            combined = np.max(stacked, axis=0)
            logger.info(f"Max pooled {len(selected_features)} layers: → {combined.shape[1]} dims")

        else:
            raise ValueError(f"Unknown combination strategy: {self.combination_strategy}")

        return combined

    def combine_from_dict(
        self,
        embeddings_dict: Dict[str, Dict[int, np.ndarray]],
        layers_to_combine: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Combine layers for multiple pooling methods.

        Args:
            embeddings_dict: Nested dictionary with structure:
                            {pooling_method: {layer_idx: features}}
            layers_to_combine: Which layers to combine

        Returns:
            Dictionary mapping pooling_method -> combined_features
        """
        combined_dict = {}

        for pooling_method, layer_features in embeddings_dict.items():
            combined = self.combine_layers(layer_features, layers_to_combine)
            combined_dict[pooling_method] = combined
            logger.info(f"Combined layers for {pooling_method}: {combined.shape}")

        return combined_dict

    def get_layer_contribution_scores(
        self,
        layer_features: Dict[int, np.ndarray],
        labels: np.ndarray,
        metric: str = 'silhouette'
    ) -> Dict[int, float]:
        """
        Compute contribution score for each layer.

        This can help understand which layers are most informative
        and guide weight selection for weighted_mean strategy.

        Args:
            layer_features: Dictionary mapping layer_idx -> features
            labels: Ground truth labels
            metric: Metric to use ('silhouette', 'calinski', 'davies_bouldin')

        Returns:
            Dictionary mapping layer_idx -> contribution_score
        """
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

        scores = {}

        for layer_idx, features in layer_features.items():
            try:
                if metric == 'silhouette':
                    score = silhouette_score(features, labels, metric='cosine')
                elif metric == 'calinski':
                    score = calinski_harabasz_score(features, labels)
                elif metric == 'davies_bouldin':
                    score = davies_bouldin_score(features, labels)
                    # Lower is better for Davies-Bouldin, so invert
                    score = 1.0 / (score + 1e-6)
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                scores[layer_idx] = score
                logger.info(f"Layer {layer_idx} {metric} score: {score:.4f}")

            except Exception as e:
                logger.warning(f"Failed to compute score for layer {layer_idx}: {e}")
                scores[layer_idx] = 0.0

        return scores

    def compute_optimal_weights_from_scores(
        self,
        contribution_scores: Dict[int, float]
    ) -> np.ndarray:
        """
        Compute optimal layer weights based on contribution scores.

        Args:
            contribution_scores: Dictionary mapping layer_idx -> score

        Returns:
            Array of normalized weights
        """
        layers = sorted(contribution_scores.keys())
        scores = np.array([contribution_scores[l] for l in layers])

        # Normalize scores to sum to 1
        if scores.sum() > 0:
            weights = scores / scores.sum()
        else:
            # Fallback to uniform weights
            weights = np.ones(len(scores)) / len(scores)

        logger.info(f"Computed optimal weights: {weights}")

        return weights

    def __repr__(self) -> str:
        return (f"LayerCombiner(strategy={self.combination_strategy}, "
                f"selection={self.layer_selection})")


# Convenience functions for common configurations

def create_concatenation_combiner(layers: Optional[List[int]] = None) -> LayerCombiner:
    """Create combiner that concatenates specified layers."""
    return LayerCombiner(
        combination_strategy='concatenation',
        layer_selection='specified' if layers else 'early_middle_late'
    )


def create_mean_combiner(layers: Optional[List[int]] = None) -> LayerCombiner:
    """Create combiner that averages specified layers."""
    return LayerCombiner(
        combination_strategy='mean',
        layer_selection='specified' if layers else 'early_middle_late'
    )


def create_weighted_combiner(
    layers: Optional[List[int]] = None,
    weights: Optional[List[float]] = None
) -> LayerCombiner:
    """Create combiner with weighted averaging."""
    combiner = LayerCombiner(
        combination_strategy='weighted_mean',
        layer_selection='specified' if layers else 'early_middle_late'
    )

    if weights is not None:
        combiner.set_layer_weights(weights)

    return combiner


def create_early_middle_late_combiner(
    strategy: str = 'concatenation'
) -> LayerCombiner:
    """
    Create combiner that uses one layer from early, middle, and late groups.

    This is a common configuration that captures multi-scale features.
    """
    return LayerCombiner(
        combination_strategy=strategy,
        layer_selection='early_middle_late'
    )
