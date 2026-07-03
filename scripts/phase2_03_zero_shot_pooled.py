#!/usr/bin/env python3
"""
Phase 2 - Stage 3: Zero-Shot Pooled Evaluation
Evaluates models on the pooled manifest (all 7 datasets combined).
Tests whether models identify animals or dataset artifacts.
"""

import json
import yaml
import sys
import torch
torch.backends.cudnn.enabled = False  # V100 (CC 7.0) incompatible with bundled cuDNN
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
import umap

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
from src.utils.audio_utils import load_audio


class PooledZeroShotEvaluator:
    """Zero-shot evaluation on pooled multi-dataset manifest."""

    def __init__(self, config, model_name, manifest_path, output_dir, logger):
        """Initialize pooled evaluator."""
        self.config = config
        self.model_name = model_name
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        self.num_classes = len(self.manifest['individuals'])
        self.class_to_idx = {ind: idx for idx, ind in enumerate(self.manifest['individuals'])}
        self.idx_to_class = {idx: ind for ind, idx in self.class_to_idx.items()}

        # Map individuals to source datasets
        self.individual_to_dataset = {}
        for individual in self.manifest['individuals']:
            # Individual format: "dataset_individualname"
            dataset = individual.split('_')[0]
            self.individual_to_dataset[individual] = dataset

        # Get unique datasets
        self.source_datasets = sorted(set(self.individual_to_dataset.values()))

        # Get class weights
        class_weights = [self.manifest['class_weights'][ind] for ind in self.manifest['individuals']]
        self.class_weights = torch.FloatTensor(class_weights)

        # Set device
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Pooled dataset: {len(self.source_datasets)} source datasets")
        self.logger.info(f"Total individuals: {self.num_classes}")
        self.logger.info(f"Model: {self.model_name}")

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the appropriate pretrained model."""
        from transformers import (
            Wav2Vec2Model,
            Wav2Vec2FeatureExtractor,
            WavLMModel
        )

        self.logger.info(f"Loading model: {self.model_name}")

        if self.model_name == "wav2vec2_base":
            model_id = "facebook/wav2vec2-base"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.model = Wav2Vec2Model.from_pretrained(model_id, use_safetensors=True)
            self.model_type = "transformer"

        elif self.model_name == "wav2vec2_base_960h":
            model_id = "facebook/wav2vec2-base-960h"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.model = Wav2Vec2Model.from_pretrained(model_id, use_safetensors=True)
            self.model_type = "transformer"

        elif self.model_name == "xls_r":
            model_id = "facebook/wav2vec2-xls-r-300m"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.model = Wav2Vec2Model.from_pretrained(model_id, use_safetensors=True)
            self.model_type = "transformer"

        elif self.model_name == "wavlm":
            model_id = "microsoft/wavlm-base-plus"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.model = WavLMModel.from_pretrained(model_id, use_safetensors=True)
            self.model_type = "transformer"

        elif self.model_name == "ecapa_tdnn":
            from speechbrain.inference.speaker import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/ecapa_tdnn"
            )
            self.feature_extractor = None
            self.model_type = "ecapa"

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        if self.model_type == "transformer":
            self.model.to(self.device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.logger.info(f"✓ Model loaded: {self.model_name}")

        # Initialize cache
        self.layer_cache = {}  # {layer_idx: {split: (embeddings, labels)}}
        self.cache_dir = self.output_dir / "embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_embedding(self, audio_path, layer_idx=None):
        """Extract embedding from audio file."""
        audio, sr = load_audio(audio_path, target_sr=16000, mono=True)

        # Truncate long files
        max_duration = self.config.get('feature_extraction', {}).get('max_audio_duration', 30)
        max_samples = int(max_duration * 16000)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Pad very short files (ECAPA needs minimum length)
        min_samples = int(0.5 * 16000)  # Minimum 0.5 seconds
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')

        if self.model_type == "ecapa":
            with torch.no_grad():
                embedding = self.model.encode_batch(torch.FloatTensor(audio).unsqueeze(0))
            return embedding.squeeze().cpu()
        else:
            inputs = self.feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states

            if layer_idx is None:
                layer_idx = len(hidden_states) - 1

            layer_output = hidden_states[layer_idx]
            embedding = layer_output.mean(dim=1).squeeze().cpu()

            return embedding

    def extract_all_layers_cached(self, split):
        """
        Extract embeddings from ALL layers for a given split and cache them.
        This is called ONCE before evaluating any layer.

        Args:
            split: 'train', 'val', or 'test'
        """
        import pickle

        cache_file = self.cache_dir / f"{split}_all_layers.pkl"

        # Check if cache exists
        if cache_file.exists():
            self.logger.info(f"  → Loading cached embeddings from {cache_file.name}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            return cache_data['layer_embeddings'], cache_data['labels'], cache_data['num_layers']

        # Extract all layers for all samples
        self.logger.info(f"  → Extracting ALL layers for {split} split (will be cached)...")

        items = self.manifest[split]
        data_dir = Path(self.config['paths']['data_dir'])

        all_samples_all_layers = []  # [sample_idx][layer_idx] -> embedding
        all_labels = []

        for item in tqdm(items, desc=f"Extracting {split}"):
            audio_path = str(data_dir / item['file'])
            individual = item['individual']
            label = self.class_to_idx[individual]

            # Extract from ALL layers at once
            audio, sr = load_audio(audio_path, target_sr=16000, mono=True)

            # Truncate long files
            max_duration = self.config.get('feature_extraction', {}).get('max_audio_duration', 30)
            max_samples = int(max_duration * 16000)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # Pad very short files
            min_samples = int(0.5 * 16000)
            if len(audio) < min_samples:
                audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')

            # Process audio
            inputs = self.feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states

            # Extract embeddings from ALL layers
            sample_layer_embeddings = []
            for layer_output in hidden_states:
                embedding = layer_output.mean(dim=1).squeeze().cpu().numpy()
                sample_layer_embeddings.append(embedding)

            all_samples_all_layers.append(sample_layer_embeddings)
            all_labels.append(label)

        num_layers = len(all_samples_all_layers[0])

        # Reorganize: layer_embeddings[layer_idx] = array of shape (num_samples, embedding_dim)
        layer_embeddings = {}
        for layer_idx in range(num_layers):
            embeddings = np.array([sample[layer_idx] for sample in all_samples_all_layers])
            layer_embeddings[layer_idx] = embeddings

        labels = np.array(all_labels)

        # Cache to disk
        cache_data = {
            'layer_embeddings': layer_embeddings,
            'labels': labels,
            'num_layers': num_layers
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
        self.logger.info(f"  ✓ Cached {num_layers} layers for {len(items)} samples ({cache_size_mb:.1f} MB)")

        return layer_embeddings, labels, num_layers

    def extract_embeddings_for_visualization(self, split='test', layer_idx=None, max_samples_per_individual=50):
        """
        Extract embeddings for visualization (t-SNE/UMAP).

        Args:
            split: Which split to use
            layer_idx: Layer to extract from
            max_samples_per_individual: Limit samples per individual (for speed)

        Returns:
            embeddings (array), labels (list), datasets (list)
        """
        self.logger.info(f"\nExtracting embeddings for visualization...")
        self.logger.info(f"  Split: {split}")
        self.logger.info(f"  Layer: {layer_idx}")
        self.logger.info(f"  Max samples per individual: {max_samples_per_individual}")

        # Sample from manifest
        from collections import defaultdict
        samples_by_individual = defaultdict(list)

        for item in self.manifest[split]:
            samples_by_individual[item['individual']].append(item)

        # Limit samples per individual
        selected_items = []
        for individual, items in samples_by_individual.items():
            selected = items[:max_samples_per_individual]
            selected_items.extend(selected)

        self.logger.info(f"  Total samples to extract: {len(selected_items)}")

        # Extract embeddings
        embeddings = []
        labels = []
        datasets = []

        data_dir = Path(self.config['paths']['data_dir'])

        for item in tqdm(selected_items, desc="Extracting embeddings"):
            audio_path = data_dir / item['file']
            individual = item['individual']
            dataset = self.individual_to_dataset[individual]

            try:
                embedding = self.extract_embedding(str(audio_path), layer_idx)
                embeddings.append(embedding.numpy())
                labels.append(individual)
                datasets.append(dataset)
            except Exception as e:
                self.logger.warning(f"Failed to extract {audio_path}: {e}")

        embeddings = np.array(embeddings)
        self.logger.info(f"✓ Extracted {len(embeddings)} embeddings, shape: {embeddings.shape}")

        return embeddings, labels, datasets

    def visualize_embeddings_tsne(self, embeddings, labels, datasets, output_path):
        """Create t-SNE visualization colored by dataset."""
        self.logger.info("Running t-SNE...")

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        embeddings_2d = tsne.fit_transform(embeddings)

        # Create color map for datasets
        dataset_colors = {ds: plt.cm.tab10(i) for i, ds in enumerate(self.source_datasets)}

        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot by dataset
        for dataset in self.source_datasets:
            mask = np.array(datasets) == dataset
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[dataset_colors[dataset]],
                label=dataset,
                alpha=0.6,
                s=30,
                edgecolors='black',
                linewidths=0.5
            )

        ax.set_title(f't-SNE Visualization (Colored by Dataset)\n{self.model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ t-SNE visualization saved: {output_path}")

    def visualize_embeddings_umap(self, embeddings, labels, datasets, output_path):
        """Create UMAP visualization colored by dataset."""
        self.logger.info("Running UMAP...")

        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings) - 1))
        embeddings_2d = reducer.fit_transform(embeddings)

        dataset_colors = {ds: plt.cm.tab10(i) for i, ds in enumerate(self.source_datasets)}

        fig, ax = plt.subplots(figsize=(14, 10))

        for dataset in self.source_datasets:
            mask = np.array(datasets) == dataset
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[dataset_colors[dataset]],
                label=dataset,
                alpha=0.6,
                s=30,
                edgecolors='black',
                linewidths=0.5
            )

        ax.set_title(f'UMAP Visualization (Colored by Dataset)\n{self.model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ UMAP visualization saved: {output_path}")

    def analyze_bird_clustering(self, embeddings, labels, datasets, output_dir):
        """
        Analyze whether bird embeddings cluster by individual or by dataset.
        Birds span 4 datasets: bengalese_finch, picidae, wetlands_bird, zebra_finch.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("BIRD CLUSTERING ANALYSIS")
        self.logger.info("="*60)

        bird_datasets = ['bengalese_finch', 'picidae', 'wetlands_bird', 'zebra_finch']

        # Filter to bird samples only
        bird_mask = np.array([ds in bird_datasets for ds in datasets])
        bird_embeddings = embeddings[bird_mask]
        bird_labels = [labels[i] for i in range(len(labels)) if bird_mask[i]]
        bird_datasets_filtered = [datasets[i] for i in range(len(datasets)) if bird_mask[i]]

        if len(bird_embeddings) == 0:
            self.logger.warning("No bird samples found!")
            return None

        if len(bird_embeddings) < 2:
            self.logger.warning(f"Insufficient bird samples for analysis ({len(bird_embeddings)} < 2). Skipping bird clustering.")
            return None

        self.logger.info(f"Total bird samples: {len(bird_embeddings)}")
        self.logger.info(f"Bird datasets present: {set(bird_datasets_filtered)}")

        # t-SNE for birds only
        self.logger.info("\nRunning t-SNE on bird samples only...")
        perplexity = min(30, max(1, len(bird_embeddings) - 1))
        if perplexity < 5:
            self.logger.warning(f"Very few bird samples ({len(bird_embeddings)}), t-SNE may not be meaningful.")

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        bird_2d = tsne.fit_transform(bird_embeddings)

        # Plot colored by dataset
        dataset_colors = {ds: plt.cm.tab10(i) for i, ds in enumerate(bird_datasets)}

        fig, ax = plt.subplots(figsize=(14, 10))

        for dataset in bird_datasets:
            mask = np.array(bird_datasets_filtered) == dataset
            if mask.sum() == 0:
                continue
            ax.scatter(
                bird_2d[mask, 0],
                bird_2d[mask, 1],
                c=[dataset_colors[dataset]],
                label=dataset,
                alpha=0.7,
                s=50,
                edgecolors='black',
                linewidths=1
            )

        ax.set_title(f'Bird Clustering Analysis (46 individuals, 4 datasets)\n{self.model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)

        # Add interpretation note
        note = "If clusters form BY DATASET → artifact (bad)\nIf clusters mix across datasets → identifying birds (good)"
        ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_dir / "bird_clustering_by_dataset.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ Bird clustering plot saved: {output_dir / 'bird_clustering_by_dataset.png'}")

        # Calculate clustering metric (silhouette score)
        from sklearn.metrics import silhouette_score

        # Create dataset labels (numeric)
        dataset_label_map = {ds: i for i, ds in enumerate(bird_datasets)}
        dataset_labels_numeric = [dataset_label_map[ds] for ds in bird_datasets_filtered]

        if len(set(dataset_labels_numeric)) > 1:
            silhouette_by_dataset = silhouette_score(bird_embeddings, dataset_labels_numeric)
            self.logger.info(f"\nSilhouette score (by dataset): {silhouette_by_dataset:.4f}")
            self.logger.info("  Interpretation: Higher score → embeddings cluster by dataset (BAD)")
            self.logger.info("                  Lower score → embeddings mix across datasets (GOOD)")

            # Save metric
            metric = {
                'silhouette_by_dataset': silhouette_by_dataset,
                'num_bird_samples': len(bird_embeddings),
                'bird_datasets': list(set(bird_datasets_filtered)),
                'interpretation': 'high=clustered_by_dataset(bad), low=mixed_across_datasets(good)'
            }

            with open(output_dir / "bird_clustering_metric.json", 'w') as f:
                json.dump(metric, f, indent=2)

            return metric
        else:
            self.logger.warning("Only one bird dataset present, cannot compute silhouette score")
            return None

    def train_and_evaluate(self, layer_idx=None):
        """Train FC head on pooled data and evaluate."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Training on Pooled Dataset")
        self.logger.info(f"{'='*60}")

        # Debug mode: reduce training
        if hasattr(self, 'debug_mode') and self.debug_mode:
            max_epochs = 3
            patience = 2
            self.logger.info(f"  - Debug mode: max_epochs={max_epochs}, patience={patience}")
        else:
            max_epochs = 100
            patience = 10

        # Get embedding dimension
        sample_embedding = self.extract_embedding(
            str(Path(self.config['paths']['data_dir']) / self.manifest['train'][0]['file']),
            layer_idx
        )
        embedding_dim = sample_embedding.shape[0]
        self.logger.info(f"Embedding dimension: {embedding_dim}")

        # Create dataloaders with caching
        from torch.utils.data import Dataset, DataLoader, TensorDataset

        # For transformer models: use cached embeddings
        if self.model_type == "transformer":
            # Resolve layer_idx=None to actual last layer index
            if layer_idx is None:
                num_layers = len(self.model.encoder.layers)
                actual_layer_idx = num_layers - 1
            else:
                actual_layer_idx = layer_idx

            # Check if this layer is already in memory cache
            if actual_layer_idx not in self.layer_cache:
                self.layer_cache[actual_layer_idx] = {}

            # Load or extract embeddings for each split
            for split_name in ['train', 'val', 'test']:
                if split_name not in self.layer_cache[actual_layer_idx]:
                    # Load all layers for this split (from disk cache or extract)
                    layer_embeddings, labels, num_layers = self.extract_all_layers_cached(split_name)
                    # Store in memory cache
                    self.layer_cache[actual_layer_idx][split_name] = (layer_embeddings[actual_layer_idx], labels)

            # Get cached embeddings for each split
            train_emb, train_labels = self.layer_cache[actual_layer_idx]['train']
            val_emb, val_labels = self.layer_cache[actual_layer_idx]['val']
            test_emb, test_labels = self.layer_cache[actual_layer_idx]['test']

            # Create TensorDatasets
            train_dataset = TensorDataset(torch.FloatTensor(train_emb), torch.LongTensor(train_labels))
            val_dataset = TensorDataset(torch.FloatTensor(val_emb), torch.LongTensor(val_labels))
            test_dataset = TensorDataset(torch.FloatTensor(test_emb), torch.LongTensor(test_labels))

        else:
            # ECAPA: extract on-the-fly (no caching)
            class AudioDataset(Dataset):
                def __init__(self, manifest_items, class_to_idx, data_dir, extractor_fn):
                    self.items = manifest_items
                    self.class_to_idx = class_to_idx
                    self.data_dir = Path(data_dir)
                    self.extractor_fn = extractor_fn

                def __len__(self):
                    return len(self.items)

                def __getitem__(self, idx):
                    item = self.items[idx]
                    audio_path = self.data_dir / item['file']
                    individual = item['individual']
                    label = self.class_to_idx[individual]
                    embedding = self.extractor_fn(str(audio_path))
                    return embedding, label

            train_dataset = AudioDataset(
                self.manifest['train'],
                self.class_to_idx,
                Path(self.config['paths']['data_dir']),
                lambda path: self.extract_embedding(path, layer_idx)
            )

            val_dataset = AudioDataset(
                self.manifest['val'],
                self.class_to_idx,
                Path(self.config['paths']['data_dir']),
                lambda path: self.extract_embedding(path, layer_idx)
            )

            test_dataset = AudioDataset(
                self.manifest['test'],
                self.class_to_idx,
                Path(self.config['paths']['data_dir']),
                lambda path: self.extract_embedding(path, layer_idx)
            )

        # Use num_workers=0 to avoid CUDA fork errors in multiprocessing
        num_workers = 0
        pin_memory = self.device == 'cuda'

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        # Train FC head
        fc_head = nn.Linear(embedding_dim, self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        optimizer = optim.Adam(fc_head.parameters(), lr=1e-3)

        # Training loop (simplified - full version in Stage 2)
        best_val_acc = 0.0
        best_model_state = None
        epochs_without_improvement = 0

        self.logger.info("Training FC head...")

        for epoch in range(max_epochs):
            fc_head.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for embeddings, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]", leave=False):
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = fc_head(embeddings)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            # Validate
            fc_head.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for embeddings, labels in val_loader:
                    embeddings = embeddings.to(self.device)
                    labels = labels.to(self.device)
                    outputs = fc_head(embeddings)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_acc = val_correct / val_total

            self.logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = fc_head.state_dict().copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model (if any improvement happened)
        if best_model_state is not None:
            fc_head.load_state_dict(best_model_state)
        else:
            self.logger.warning(f"⚠️  No validation improvement. Using final epoch model.")
            best_val_acc = val_acc

        # Test
        fc_head.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for embeddings, labels in tqdm(test_loader, desc="Testing"):
                embeddings = embeddings.to(self.device)
                outputs = fc_head(embeddings)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        test_acc = accuracy_score(all_labels, all_preds)
        self.logger.info(f"\n✓ Test accuracy: {test_acc:.4f}")

        # Per-dataset breakdown
        dataset_accuracy = self._compute_per_dataset_accuracy(all_preds, all_labels)

        results = {
            'model': self.model_name,
            'test_accuracy': test_acc,
            'best_val_accuracy': best_val_acc,
            'per_dataset_accuracy': dataset_accuracy,
            'num_classes': self.num_classes,
            'embedding_dim': embedding_dim
        }

        # Save model
        torch.save(fc_head.state_dict(), self.output_dir / "fc_head_pooled.pth")

        # Save results
        with open(self.output_dir / "pooled_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _compute_per_dataset_accuracy(self, predictions, labels):
        """Compute accuracy breakdown by source dataset."""
        dataset_correct = {ds: 0 for ds in self.source_datasets}
        dataset_total = {ds: 0 for ds in self.source_datasets}

        for pred, label in zip(predictions, labels):
            individual = self.idx_to_class[label]
            dataset = self.individual_to_dataset[individual]

            dataset_total[dataset] += 1
            if pred == label:
                dataset_correct[dataset] += 1

        dataset_accuracy = {}
        for ds in self.source_datasets:
            if dataset_total[ds] > 0:
                dataset_accuracy[ds] = dataset_correct[ds] / dataset_total[ds]
            else:
                dataset_accuracy[ds] = 0.0

        self.logger.info("\nPer-dataset accuracy breakdown:")
        for ds, acc in dataset_accuracy.items():
            self.logger.info(f"  {ds}: {acc:.4f} ({dataset_correct[ds]}/{dataset_total[ds]})")

        return dataset_accuracy

    def run_full_evaluation(self, layer_idx=None):
        """Run complete pooled evaluation."""
        # 1. Train and evaluate
        results = self.train_and_evaluate(layer_idx)

        # 2. Extract embeddings for visualization
        embeddings, labels, datasets = self.extract_embeddings_for_visualization(
            split='test',
            layer_idx=layer_idx,
            max_samples_per_individual=50
        )

        # 3. Visualize embeddings
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        self.visualize_embeddings_tsne(embeddings, labels, datasets, viz_dir / "tsne_by_dataset.png")
        self.visualize_embeddings_umap(embeddings, labels, datasets, viz_dir / "umap_by_dataset.png")

        # 4. Bird clustering analysis
        bird_dir = self.output_dir / "bird_analysis"
        bird_dir.mkdir(exist_ok=True)

        bird_metric = self.analyze_bird_clustering(embeddings, labels, datasets, bird_dir)

        # 5. Save final summary
        summary = {
            **results,
            'bird_clustering_metric': bird_metric
        }

        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"\n{'='*60}")
        self.logger.info("POOLED EVALUATION COMPLETE")
        self.logger.info(f"{'='*60}")

        return summary


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 - Zero-Shot Pooled Evaluation")
    parser.add_argument("--model", required=True, choices=["wav2vec2_base", "wav2vec2_base_960h", "xls_r", "wavlm", "ecapa_tdnn"])
    parser.add_argument("--layer", type=int, default=None, help="Layer to use (default: last layer)")
    parser.add_argument("--debug", action="store_true", help="Debug mode: use small subset and reduce training")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    logger = setup_logger("Phase2_PooledZeroShot", config['experiment']['log_level'])

    logger.info("="*80)
    logger.info("PHASE 2 - STAGE 3: ZERO-SHOT POOLED EVALUATION")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    if args.debug:
        logger.info("⚠️  DEBUG MODE ENABLED: Using small subset")

    # Get pooled manifest
    manifest_path = Path(config['paths']['output_dir']) / "phase2" / "manifests" / "pooled_manifest.json"

    if not manifest_path.exists():
        logger.error(f"Pooled manifest not found: {manifest_path}")
        logger.error("Run phase2_01_create_manifests.py first")
        return

    # Create output directory
    output_dir = Path(config['paths']['output_dir']) / "phase2" / "zero_shot" / "pooled" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    evaluator = PooledZeroShotEvaluator(config, args.model, manifest_path, output_dir, logger)

    # Apply debug mode
    if args.debug:
        logger.info("\n🔧 Applying debug mode modifications...")
        evaluator.manifest['train'] = evaluator.manifest['train'][:50]
        evaluator.manifest['val'] = evaluator.manifest['val'][:20]
        evaluator.manifest['test'] = evaluator.manifest['test'][:20]
        logger.info(f"  - Train samples: 50")
        logger.info(f"  - Val samples: 20")
        logger.info(f"  - Test samples: 20")
        evaluator.debug_mode = True
    else:
        evaluator.debug_mode = False

    summary = evaluator.run_full_evaluation(layer_idx=args.layer)

    logger.info(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
