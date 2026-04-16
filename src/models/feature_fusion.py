"""
Feature Fusion Module

Combines multiple feature representations to create enriched feature vectors.

This module implements different strategies for fusing:
- Handcrafted features (MFCC, spectral)
- Learned features (wav2vec embeddings)
- Multiple layers of wav2vec

Fusion strategies:
1. Simple concatenation
2. Weighted concatenation
3. Dimensionality-matched fusion (PCA before fusion)

Research question:
"Do handcrafted and learned features provide complementary information?"

Author: Raj
Date: 2026-04-11
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class FeatureFusion:
    """
    Feature fusion module for combining multiple feature representations.

    Supports multiple fusion strategies to investigate whether
    different feature types provide complementary information.
    """

    def __init__(
        self,
        fusion_strategy: str = 'concatenation',
        normalize: bool = True,
        pca_dims: Optional[int] = None
    ):
        """
        Initialize feature fusion module.

        Args:
            fusion_strategy: Fusion method ('concatenation', 'weighted', 'pca_fusion')
            normalize: Whether to normalize features before fusion
            pca_dims: Target dimensions for PCA (if using pca_fusion)
        """
        self.fusion_strategy = fusion_strategy
        self.normalize = normalize
        self.pca_dims = pca_dims

        # Will be set during fit
        self.scalers = {}
        self.pcas = {}
        self.weights = {}
        self.is_fitted = False

        logger.info(f"Initialized FeatureFusion with strategy: {fusion_strategy}")

    def _normalize_features(
        self,
        features: np.ndarray,
        feature_name: str,
        fit: bool = False
    ) -> np.ndarray:
        """
        Normalize features using StandardScaler.

        Args:
            features: Feature matrix
            feature_name: Name for this feature set (for caching scaler)
            fit: Whether to fit scaler (True) or use existing (False)

        Returns:
            Normalized features
        """
        if fit:
            scaler = StandardScaler()
            normalized = scaler.fit_transform(features)
            self.scalers[feature_name] = scaler
        else:
            if feature_name not in self.scalers:
                raise ValueError(f"Scaler for {feature_name} not fitted")
            normalized = self.scalers[feature_name].transform(features)

        return normalized

    def _apply_pca(
        self,
        features: np.ndarray,
        feature_name: str,
        n_components: int,
        fit: bool = False
    ) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction.

        Args:
            features: Feature matrix
            feature_name: Name for this feature set
            n_components: Number of PCA components
            fit: Whether to fit PCA (True) or use existing (False)

        Returns:
            PCA-transformed features
        """
        if fit:
            pca = PCA(n_components=n_components, random_state=42)
            reduced = pca.fit_transform(features)
            self.pcas[feature_name] = pca
            logger.info(f"  PCA for {feature_name}: {features.shape} → {reduced.shape}")
            logger.info(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        else:
            if feature_name not in self.pcas:
                raise ValueError(f"PCA for {feature_name} not fitted")
            reduced = self.pcas[feature_name].transform(features)

        return reduced

    def fit(
        self,
        feature_dict: Dict[str, np.ndarray]
    ) -> None:
        """
        Fit fusion parameters (scalers, PCA, weights).

        Args:
            feature_dict: Dictionary mapping feature_name -> features
                         e.g., {'mfcc': mfcc_features, 'wav2vec': w2v_features}
        """
        logger.info(f"Fitting fusion with {len(feature_dict)} feature types...")

        # Clear existing state
        self.scalers = {}
        self.pcas = {}
        self.weights = {}

        # Fit scalers if normalization enabled
        if self.normalize:
            for name, features in feature_dict.items():
                self._normalize_features(features, name, fit=True)
                logger.info(f"  Fitted scaler for {name}: {features.shape}")

        # Fit PCA if using pca_fusion strategy
        if self.fusion_strategy == 'pca_fusion':
            if self.pca_dims is None:
                raise ValueError("pca_dims must be specified for pca_fusion strategy")

            for name, features in feature_dict.items():
                # Normalize first if enabled
                if self.normalize:
                    features = self._normalize_features(features, name, fit=False)

                # Apply PCA
                self._apply_pca(features, name, self.pca_dims, fit=True)

        # Compute weights for weighted fusion
        if self.fusion_strategy == 'weighted':
            # Simple strategy: weight inversely proportional to dimensionality
            total_dims = sum(f.shape[1] for f in feature_dict.values())
            for name, features in feature_dict.items():
                # Higher weight for lower-dimensional features
                self.weights[name] = total_dims / (features.shape[1] * len(feature_dict))
                logger.info(f"  Weight for {name}: {self.weights[name]:.3f}")

        self.is_fitted = True
        logger.info("Fusion fitting complete")

    def transform(
        self,
        feature_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Fuse multiple feature representations into a single representation.

        Args:
            feature_dict: Dictionary mapping feature_name -> features

        Returns:
            Fused features
        """
        if not self.is_fitted:
            raise ValueError("Fusion not fitted. Call fit() first.")

        fused_features_list = []

        for name, features in feature_dict.items():
            if self.fusion_strategy == 'concatenation':
                # Simple concatenation (with optional normalization)
                if self.normalize:
                    features = self._normalize_features(features, name, fit=False)
                fused_features_list.append(features)

            elif self.fusion_strategy == 'weighted':
                # Weighted concatenation
                if self.normalize:
                    features = self._normalize_features(features, name, fit=False)
                weighted = features * self.weights[name]
                fused_features_list.append(weighted)

            elif self.fusion_strategy == 'pca_fusion':
                # PCA to same dimension, then concatenate
                if self.normalize:
                    features = self._normalize_features(features, name, fit=False)
                reduced = self._apply_pca(features, name, self.pca_dims, fit=False)
                fused_features_list.append(reduced)

            else:
                raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # Concatenate all features
        fused = np.concatenate(fused_features_list, axis=1)

        logger.info(f"Fused {len(feature_dict)} feature types → shape: {fused.shape}")

        return fused

    def fit_transform(
        self,
        feature_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            feature_dict: Dictionary mapping feature_name -> features

        Returns:
            Fused features
        """
        self.fit(feature_dict)
        return self.transform(feature_dict)

    def get_feature_info(self) -> Dict:
        """
        Get information about fitted fusion.

        Returns:
            Dictionary with fusion information
        """
        if not self.is_fitted:
            return {'fitted': False}

        info = {
            'fitted': True,
            'strategy': self.fusion_strategy,
            'normalize': self.normalize,
            'n_feature_types': len(self.scalers),
            'feature_names': list(self.scalers.keys())
        }

        if self.fusion_strategy == 'weighted':
            info['weights'] = self.weights

        if self.fusion_strategy == 'pca_fusion':
            info['pca_dims'] = self.pca_dims
            info['pca_variance_explained'] = {
                name: pca.explained_variance_ratio_.sum()
                for name, pca in self.pcas.items()
            }

        return info

    def __repr__(self) -> str:
        return (f"FeatureFusion(strategy={self.fusion_strategy}, "
                f"normalize={self.normalize}, "
                f"fitted={self.is_fitted})")


class MultiLayerFusion:
    """
    Fuse features from multiple wav2vec layers.

    Combines representations from different layers to create
    a more comprehensive representation.
    """

    def __init__(
        self,
        fusion_strategy: str = 'concatenation',
        normalize: bool = True,
        pca_dims: Optional[int] = None
    ):
        """
        Initialize multi-layer fusion.

        Args:
            fusion_strategy: Fusion method ('concatenation', 'mean', 'weighted', 'pca_fusion')
            normalize: Whether to normalize before fusion
            pca_dims: Target dimensions for PCA (if using pca_fusion)
        """
        self.fusion_strategy = fusion_strategy
        self.normalize = normalize
        self.pca_dims = pca_dims

        self.scaler = None
        self.pca = None
        self.layer_weights = None
        self.is_fitted = False

        logger.info(f"Initialized MultiLayerFusion with strategy: {fusion_strategy}")

    def fit(
        self,
        layer_features_list: List[np.ndarray],
        layer_names: Optional[List[str]] = None
    ) -> None:
        """
        Fit fusion parameters.

        Args:
            layer_features_list: List of feature matrices, one per layer
            layer_names: Optional names for layers (for logging)
        """
        logger.info(f"Fitting multi-layer fusion for {len(layer_features_list)} layers...")

        if layer_names is None:
            layer_names = [f"layer_{i}" for i in range(len(layer_features_list))]

        # Concatenate or average first
        if self.fusion_strategy in ['concatenation', 'pca_fusion']:
            combined = np.concatenate(layer_features_list, axis=1)
        elif self.fusion_strategy == 'mean':
            combined = np.mean(layer_features_list, axis=0)
        elif self.fusion_strategy == 'weighted':
            # Learn weights (inverse of layer index, favoring early layers)
            n_layers = len(layer_features_list)
            weights = np.array([1.0 / (i + 1) for i in range(n_layers)])
            weights = weights / weights.sum()
            self.layer_weights = weights
            logger.info(f"  Layer weights: {weights}")

            # Weighted average
            combined = np.average(layer_features_list, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # Fit scaler if normalization enabled
        if self.normalize:
            self.scaler = StandardScaler()
            combined = self.scaler.fit_transform(combined)

        # Fit PCA if using pca_fusion
        if self.fusion_strategy == 'pca_fusion':
            if self.pca_dims is None:
                raise ValueError("pca_dims must be specified for pca_fusion")

            self.pca = PCA(n_components=self.pca_dims, random_state=42)
            self.pca.fit(combined)
            logger.info(f"  PCA: {combined.shape[1]} → {self.pca_dims} dims")
            logger.info(f"  Explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")

        self.is_fitted = True
        logger.info("Multi-layer fusion fitting complete")

    def transform(
        self,
        layer_features_list: List[np.ndarray]
    ) -> np.ndarray:
        """
        Transform features using fitted fusion.

        Args:
            layer_features_list: List of feature matrices, one per layer

        Returns:
            Fused features
        """
        if not self.is_fitted:
            raise ValueError("Fusion not fitted. Call fit() first.")

        # Combine layers
        if self.fusion_strategy in ['concatenation', 'pca_fusion']:
            combined = np.concatenate(layer_features_list, axis=1)
        elif self.fusion_strategy == 'mean':
            combined = np.mean(layer_features_list, axis=0)
        elif self.fusion_strategy == 'weighted':
            combined = np.average(layer_features_list, axis=0, weights=self.layer_weights)

        # Normalize if enabled
        if self.normalize and self.scaler is not None:
            combined = self.scaler.transform(combined)

        # Apply PCA if using pca_fusion
        if self.fusion_strategy == 'pca_fusion' and self.pca is not None:
            combined = self.pca.transform(combined)

        logger.info(f"Fused {len(layer_features_list)} layers → shape: {combined.shape}")

        return combined

    def fit_transform(
        self,
        layer_features_list: List[np.ndarray],
        layer_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(layer_features_list, layer_names)
        return self.transform(layer_features_list)

    def __repr__(self) -> str:
        return (f"MultiLayerFusion(strategy={self.fusion_strategy}, "
                f"normalize={self.normalize}, "
                f"fitted={self.is_fitted})")


def create_simple_fusion(normalize: bool = True) -> FeatureFusion:
    """Create feature fusion with simple concatenation."""
    return FeatureFusion(
        fusion_strategy='concatenation',
        normalize=normalize
    )


def create_weighted_fusion(normalize: bool = True) -> FeatureFusion:
    """Create feature fusion with weighted concatenation."""
    return FeatureFusion(
        fusion_strategy='weighted',
        normalize=normalize
    )


def create_pca_fusion(pca_dims: int = 128, normalize: bool = True) -> FeatureFusion:
    """Create feature fusion with PCA dimensionality reduction."""
    return FeatureFusion(
        fusion_strategy='pca_fusion',
        normalize=normalize,
        pca_dims=pca_dims
    )


def create_layer_fusion(strategy: str = 'concatenation', normalize: bool = True) -> MultiLayerFusion:
    """Create multi-layer fusion."""
    return MultiLayerFusion(
        fusion_strategy=strategy,
        normalize=normalize
    )
