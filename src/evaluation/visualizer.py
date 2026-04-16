"""
Visualization module for dimensionality reduction and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from umap import UMAP

from src.utils.logging_utils import setup_logger


class EmbeddingVisualizer:
    """Visualizes high-dimensional embeddings using dimensionality reduction."""

    def __init__(self, config: dict, log_level: str = "INFO"):
        """
        Initialize embedding visualizer.

        Args:
            config: Configuration dictionary
            log_level: Logging level
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__, log_level)

        # Visualization settings
        self.viz_config = config['visualization']
        self.figure_size = tuple(self.viz_config['figure_size'])
        self.dpi = self.viz_config['dpi']
        self.format = self.viz_config['format']

        # Set style
        try:
            plt.style.use(self.viz_config.get('style', 'default'))
        except:
            plt.style.use('default')
        sns.set_palette(self.viz_config['color_palette'])

    def reduce_pca(
        self,
        features: np.ndarray,
        n_components: int = 2,
        **kwargs
    ) -> Tuple[np.ndarray, object]:
        """
        Reduce dimensionality using PCA.

        Args:
            features: Feature array of shape (n_samples, n_features)
            n_components: Number of components
            **kwargs: Additional PCA parameters

        Returns:
            Tuple of (reduced features, fitted model)
        """
        pca_config = self.viz_config['methods']['pca'].copy()
        pca_config.pop('n_components', None)  # Remove to avoid duplicate
        pca_config.pop('enabled', None)  # Remove non-PCA parameter
        pca = PCA(n_components=n_components, **pca_config, **kwargs)
        reduced = pca.fit_transform(features)

        self.logger.info(f"PCA: Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        return reduced, pca

    def reduce_lda(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_components: int = 2,
        **kwargs
    ) -> Tuple[np.ndarray, object]:
        """
        Reduce dimensionality using LDA.

        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            n_components: Number of components (max = n_classes - 1)
            **kwargs: Additional LDA parameters

        Returns:
            Tuple of (reduced features, fitted model)
        """
        # LDA requires n_components <= n_classes - 1
        n_classes = len(np.unique(labels))
        n_components = min(n_components, n_classes - 1)

        if n_components < 1:
            raise ValueError(f"Need at least 2 classes for LDA, got {n_classes}")

        lda = LDA(n_components=n_components, **kwargs)
        reduced = lda.fit_transform(features, labels)

        self.logger.info(f"LDA: Using {n_components} components for {n_classes} classes")
        return reduced, lda

    def reduce_tsne(
        self,
        features: np.ndarray,
        n_components: int = 2,
        **kwargs
    ) -> Tuple[np.ndarray, object]:
        """
        Reduce dimensionality using t-SNE.

        Args:
            features: Feature array of shape (n_samples, n_features)
            n_components: Number of components
            **kwargs: Additional t-SNE parameters

        Returns:
            Tuple of (reduced features, fitted model)
        """
        tsne_config = self.viz_config['methods']['tsne'].copy()
        tsne_config.pop('n_components', None)  # Remove to avoid duplicate
        tsne_config.pop('enabled', None)  # Remove non-TSNE parameter
        tsne = TSNE(n_components=n_components, **{**tsne_config, **kwargs})
        reduced = tsne.fit_transform(features)

        self.logger.info(f"t-SNE: Reduced to {n_components} dimensions")
        return reduced, tsne

    def reduce_umap(
        self,
        features: np.ndarray,
        n_components: int = 2,
        **kwargs
    ) -> Tuple[np.ndarray, object]:
        """
        Reduce dimensionality using UMAP.

        Args:
            features: Feature array of shape (n_samples, n_features)
            n_components: Number of components
            **kwargs: Additional UMAP parameters

        Returns:
            Tuple of (reduced features, fitted model)
        """
        umap_config = self.viz_config['methods']['umap'].copy()
        umap_config.pop('n_components', None)  # Remove to avoid duplicate
        umap_config.pop('enabled', None)  # Remove non-UMAP parameter
        reducer = UMAP(n_components=n_components, **{**umap_config, **kwargs})
        reduced = reducer.fit_transform(features)

        self.logger.info(f"UMAP: Reduced to {n_components} dimensions")
        return reduced, reducer

    def plot_2d_embedding(
        self,
        embedding: np.ndarray,
        labels: np.ndarray,
        title: str,
        output_path: Optional[str] = None,
        legend_title: str = "Individual"
    ) -> plt.Figure:
        """
        Plot 2D embedding with colored labels.

        Args:
            embedding: 2D embedding of shape (n_samples, 2)
            labels: Label array of shape (n_samples,)
            title: Plot title
            output_path: Path to save figure (optional)
            legend_title: Title for legend

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Get unique labels for coloring
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)

        # Create color map
        colors = sns.color_palette(self.viz_config['color_palette'], n_labels)
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

        # Plot each class
        for label in unique_labels:
            mask = labels == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                label=label,
                alpha=self.viz_config['alpha'],
                s=self.viz_config['marker_size'],
                c=[label_to_color[label]],
                edgecolors='black',
                linewidths=0.5
            )

        ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)

        # Configure legend
        if n_labels <= 20:
            ax.legend(
                title=legend_title,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                fontsize=9,
                framealpha=0.9
            )
        else:
            # Too many labels, skip legend
            self.logger.info(f"Skipping legend ({n_labels} labels)")

        plt.tight_layout()

        # Save figure if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight', format=self.format)
            self.logger.info(f"Figure saved to {output_path}")

        return fig

    def visualize_layer(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        layer_idx: int,
        model_name: str,
        dataset_name: str,
        output_dir: str,
        methods: List[str] = None
    ) -> Dict[str, plt.Figure]:
        """
        Visualize a single layer using multiple dimensionality reduction methods.

        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            layer_idx: Layer index
            model_name: Name of the model
            dataset_name: Name of the dataset
            output_dir: Directory to save figures
            methods: List of methods to use (default: all enabled)

        Returns:
            Dictionary of method name to figure
        """
        if methods is None:
            methods = [m for m in ['pca', 'lda', 'tsne', 'umap']
                      if self.viz_config['methods'][m]['enabled']]

        figures = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Visualizing layer {layer_idx} with methods: {methods}")

        for method in methods:
            try:
                # Reduce dimensionality
                if method == 'pca':
                    reduced, _ = self.reduce_pca(features)
                    title = f"{model_name} - Layer {layer_idx} - PCA\n{dataset_name}"
                elif method == 'lda':
                    reduced, _ = self.reduce_lda(features, labels)
                    title = f"{model_name} - Layer {layer_idx} - LDA\n{dataset_name}"
                elif method == 'tsne':
                    reduced, _ = self.reduce_tsne(features)
                    title = f"{model_name} - Layer {layer_idx} - t-SNE\n{dataset_name}"
                elif method == 'umap':
                    reduced, _ = self.reduce_umap(features)
                    title = f"{model_name} - Layer {layer_idx} - UMAP\n{dataset_name}"
                else:
                    continue

                # Plot
                fig_path = output_path / f"{dataset_name}_{model_name}_layer{layer_idx}_{method}.{self.format}"
                fig = self.plot_2d_embedding(
                    reduced,
                    labels,
                    title,
                    str(fig_path)
                )
                figures[method] = fig

                plt.close(fig)

            except Exception as e:
                self.logger.error(f"Error visualizing with {method}: {e}")

        return figures

    def create_layer_comparison_grid(
        self,
        features_per_layer: Dict[int, np.ndarray],
        labels: np.ndarray,
        model_name: str,
        dataset_name: str,
        output_path: str,
        method: str = 'umap',
        layers: List[int] = None
    ):
        """
        Create a grid of visualizations comparing multiple layers.

        Args:
            features_per_layer: Dictionary mapping layer index to features
            labels: Label array
            model_name: Name of the model
            dataset_name: Name of the dataset
            output_path: Path to save figure
            method: Dimensionality reduction method
            layers: List of layers to visualize (default: all)
        """
        if layers is None:
            layers = sorted(features_per_layer.keys())

        n_layers = len(layers)
        n_cols = 4
        n_rows = (n_layers + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows), dpi=100)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        colors = sns.color_palette(self.viz_config['color_palette'], n_labels)
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

        for idx, layer_idx in enumerate(layers):
            ax = axes[idx]
            features = features_per_layer[layer_idx]

            # Reduce dimensionality
            if method == 'pca':
                reduced, _ = self.reduce_pca(features)
            elif method == 'lda':
                reduced, _ = self.reduce_lda(features, labels)
            elif method == 'tsne':
                reduced, _ = self.reduce_tsne(features)
            elif method == 'umap':
                reduced, _ = self.reduce_umap(features)
            else:
                reduced, _ = self.reduce_pca(features)

            # Plot
            for label in unique_labels:
                mask = labels == label
                ax.scatter(
                    reduced[mask, 0],
                    reduced[mask, 1],
                    label=label if idx == 0 else None,
                    alpha=0.6,
                    s=30,
                    c=[label_to_color[label]],
                    edgecolors='black',
                    linewidths=0.3
                )

            ax.set_title(f"Layer {layer_idx}", fontsize=11)
            ax.set_xlabel('Comp 1', fontsize=9)
            ax.set_ylabel('Comp 2', fontsize=9)

        # Hide unused subplots
        for idx in range(n_layers, len(axes)):
            axes[idx].axis('off')

        # Add suptitle first - reserve more space for two-line title
        fig.suptitle(f"{model_name} - Layer Comparison ({method.upper()})\n{dataset_name}",
                     fontsize=14, fontweight='bold', y=0.99)

        # Call tight_layout FIRST to position subplots, then add legend outside
        plt.tight_layout(rect=[0, 0, 0.95, 0.94])  # Leave 5% on right for legend, 6% on top for title

        # Add legend AFTER tight_layout - position outside plot area on the right
        if n_labels <= 20:
            handles, labels_legend = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels_legend, loc='center left',
                      bbox_to_anchor=(0.96, 0.5), fontsize=8, framealpha=0.95,
                      title='Individual', title_fontsize=9)

        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight', format=self.format)
        self.logger.info(f"Layer comparison grid saved to {output_path}")

        plt.close(fig)
