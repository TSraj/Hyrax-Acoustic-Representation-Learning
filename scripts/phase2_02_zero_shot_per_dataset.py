#!/usr/bin/env python3
"""
Phase 2 - Stage 2: Zero-Shot Per-Dataset Evaluation
Evaluates 5 models on each dataset separately using frozen backbone + trained FC head.
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
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
from src.utils.audio_utils import load_audio


class ZeroShotEvaluator:
    """Zero-shot evaluation with frozen backbone + trained FC head."""

    def __init__(self, config, model_name, manifest_path, output_dir, logger):
        """
        Initialize evaluator.

        Args:
            config: Configuration dictionary
            model_name: One of: wav2vec2_base, wav2vec2_base_960h, xls_r, wavlm, ecapa_tdnn
            manifest_path: Path to dataset manifest JSON
            output_dir: Output directory for results
            logger: Logger instance
        """
        self.config = config
        self.model_name = model_name
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        self.dataset_name = self.manifest['dataset']
        self.num_classes = len(self.manifest['individuals'])
        self.class_to_idx = {ind: idx for idx, ind in enumerate(self.manifest['individuals'])}
        self.idx_to_class = {idx: ind for ind, idx in self.class_to_idx.items()}

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
        self.logger.info(f"Dataset: {self.dataset_name}")
        self.logger.info(f"Num classes: {self.num_classes}")
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
            # Pretrained-only, no ASR fine-tuning
            model_id = "facebook/wav2vec2-base"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.model = Wav2Vec2Model.from_pretrained(model_id)
            self.model_type = "transformer"

        elif self.model_name == "wav2vec2_base_960h":
            # ASR fine-tuned on 960h English
            model_id = "facebook/wav2vec2-base-960h"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.model = Wav2Vec2Model.from_pretrained(model_id)
            self.model_type = "transformer"

        elif self.model_name == "xls_r":
            # XLS-R 300M multilingual
            model_id = "facebook/wav2vec2-xls-r-300m"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.model = Wav2Vec2Model.from_pretrained(model_id)
            self.model_type = "transformer"

        elif self.model_name == "wavlm":
            # WavLM Base+ (uses same feature extractor as Wav2Vec2)
            model_id = "microsoft/wavlm-base-plus"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.model = WavLMModel.from_pretrained(model_id)
            self.model_type = "transformer"

        elif self.model_name == "ecapa_tdnn":
            # ECAPA-TDNN from SpeechBrain
            from speechbrain.inference.speaker import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/ecapa_tdnn"
            )
            self.feature_extractor = None  # ECAPA has internal preprocessing
            self.model_type = "ecapa"

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        # Move model to device and freeze
        if self.model_type == "transformer":
            self.model.to(self.device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.logger.info(f"✓ Model loaded and frozen: {self.model_name}")

    def extract_embedding(self, audio_path, layer_idx=None):
        """
        Extract embedding from audio file.

        Args:
            audio_path: Path to audio file
            layer_idx: Transformer layer index (ignored for ECAPA)

        Returns:
            Embedding tensor (1D)
        """
        audio, sr = load_audio(audio_path, target_sr=16000, mono=True)

        # Truncate long files
        max_duration = self.config.get('feature_extraction', {}).get('max_audio_duration', 30)
        max_samples = int(max_duration * 16000)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        if self.model_type == "ecapa":
            # ECAPA-TDNN: single pooled embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(torch.FloatTensor(audio).unsqueeze(0))
            return embedding.squeeze().cpu()

        else:
            # Transformer models: layer-wise extraction
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

            # Extract from specified layer (or last layer if None)
            if layer_idx is None:
                layer_idx = len(hidden_states) - 1

            layer_output = hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)

            # Mean pooling over sequence length
            embedding = layer_output.mean(dim=1).squeeze().cpu()

            return embedding

    def create_dataloader(self, split, layer_idx=None, batch_size=32):
        """
        Create PyTorch dataloader for a split.

        Args:
            split: 'train', 'val', or 'test'
            layer_idx: Layer to extract from (for transformer models)
            batch_size: Batch size

        Returns:
            DataLoader
        """
        from torch.utils.data import Dataset, DataLoader

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

                # Extract embedding
                embedding = self.extractor_fn(str(audio_path))

                return embedding, label

        dataset = AudioDataset(
            self.manifest[split],
            self.class_to_idx,
            Path(self.config['paths']['data_dir']),
            lambda path: self.extract_embedding(path, layer_idx)
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=4,  # HPC optimization: parallel data loading
            pin_memory=True  # Faster GPU transfer
        )

    def train_fc_head(self, train_loader, val_loader, embedding_dim, max_epochs=100, patience=10):
        """
        Train FC head on frozen embeddings.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            embedding_dim: Embedding dimension
            max_epochs: Maximum training epochs
            patience: Early stopping patience

        Returns:
            Trained model, training history
        """
        # Define FC head
        fc_head = nn.Linear(embedding_dim, self.num_classes).to(self.device)

        # Loss function with class weighting
        criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))

        # Optimizer
        optimizer = optim.Adam(fc_head.parameters(), lr=1e-3)

        # Training loop
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0
        best_model_state = None
        epochs_without_improvement = 0

        self.logger.info(f"Training FC head (embedding_dim={embedding_dim}, classes={self.num_classes})")
        self.logger.info(f"Max epochs: {max_epochs}, Early stopping patience: {patience}")

        for epoch in range(max_epochs):
            # Train
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
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for embeddings, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]", leave=False):
                    embeddings = embeddings.to(self.device)
                    labels = labels.to(self.device)

                    outputs = fc_head(embeddings)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            self.logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = fc_head.state_dict().copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Restore best model
        fc_head.load_state_dict(best_model_state)

        self.logger.info(f"✓ Training complete. Best val acc: {best_val_acc:.4f}")

        return fc_head, history

    def evaluate_on_test(self, fc_head, test_loader):
        """
        Evaluate FC head on test set.

        Args:
            fc_head: Trained FC head
            test_loader: Test dataloader

        Returns:
            Dictionary with evaluation metrics
        """
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

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.manifest['individuals'],
            output_dict=True,
            zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)

        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds,
            'labels': all_labels
        }

        self.logger.info(f"Test accuracy: {accuracy:.4f}")

        return results

    def plot_confusion_matrix(self, cm, layer_idx, save_path):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(max(10, self.num_classes * 0.5), max(8, self.num_classes * 0.4)))
        sns.heatmap(
            cm,
            annot=True if self.num_classes <= 20 else False,
            fmt='d',
            cmap='Blues',
            xticklabels=self.manifest['individuals'],
            yticklabels=self.manifest['individuals']
        )
        plt.title(f'Confusion Matrix - {self.model_name} Layer {layer_idx}\n{self.dataset_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ Confusion matrix saved: {save_path}")

    def evaluate_layer(self, layer_idx):
        """
        Evaluate a single layer.

        Args:
            layer_idx: Layer index to evaluate

        Returns:
            Dictionary with results for this layer
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Evaluating Layer {layer_idx}")
        self.logger.info(f"{'='*60}")

        # Extract embeddings and get embedding dimension
        sample_embedding = self.extract_embedding(
            str(Path(self.config['paths']['data_dir']) / self.manifest['train'][0]['file']),
            layer_idx
        )
        embedding_dim = sample_embedding.shape[0]
        self.logger.info(f"Embedding dimension: {embedding_dim}")

        # Create dataloaders
        self.logger.info("Creating dataloaders...")
        train_loader = self.create_dataloader('train', layer_idx, batch_size=32)
        val_loader = self.create_dataloader('val', layer_idx, batch_size=32)
        test_loader = self.create_dataloader('test', layer_idx, batch_size=32)

        # Train FC head
        fc_head, history = self.train_fc_head(train_loader, val_loader, embedding_dim)

        # Evaluate on test set
        test_results = self.evaluate_on_test(fc_head, test_loader)

        # Save results
        layer_output_dir = self.output_dir / f"layer_{layer_idx}"
        layer_output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(fc_head.state_dict(), layer_output_dir / "fc_head.pth")

        # Save metrics
        results = {
            'layer': layer_idx,
            'embedding_dim': embedding_dim,
            'test_accuracy': test_results['accuracy'],
            'training_history': history,
            'classification_report': test_results['classification_report']
        }

        with open(layer_output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Plot confusion matrix
        self.plot_confusion_matrix(
            np.array(test_results['confusion_matrix']),
            layer_idx,
            layer_output_dir / "confusion_matrix.png"
        )

        return results

    def evaluate_all_layers(self):
        """Evaluate all layers (for transformer models) or single embedding (for ECAPA)."""
        if self.model_type == "ecapa":
            # ECAPA: single embedding only
            self.logger.info("ECAPA-TDNN: Evaluating single pooled embedding")
            results = self.evaluate_layer(layer_idx=0)  # Use 0 as placeholder

            # Save summary
            summary = {
                'model': self.model_name,
                'dataset': self.dataset_name,
                'model_type': 'ecapa',
                'best_accuracy': results['test_accuracy'],
                'embedding_dim': results['embedding_dim']
            }

        else:
            # Transformer: evaluate all layers
            num_layers = len(self.model.encoder.layers)
            self.logger.info(f"Transformer model: Evaluating {num_layers} layers")

            layer_results = []
            for layer_idx in range(num_layers):
                results = self.evaluate_layer(layer_idx)
                layer_results.append(results)

            # Find best layer
            best_layer = max(layer_results, key=lambda x: x['test_accuracy'])

            # Save layer-wise summary
            summary = {
                'model': self.model_name,
                'dataset': self.dataset_name,
                'model_type': 'transformer',
                'num_layers': num_layers,
                'best_layer': best_layer['layer'],
                'best_accuracy': best_layer['test_accuracy'],
                'layer_wise_accuracy': [r['test_accuracy'] for r in layer_results]
            }

            # Save layer-wise CSV
            import pandas as pd
            df = pd.DataFrame([
                {
                    'layer': r['layer'],
                    'test_accuracy': r['test_accuracy'],
                    'embedding_dim': r['embedding_dim']
                }
                for r in layer_results
            ])
            df.to_csv(self.output_dir / "layer_wise_results.csv", index=False)

        # Save summary
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"EVALUATION COMPLETE")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Best accuracy: {summary['best_accuracy']:.4f}")
        if summary['model_type'] == 'transformer':
            self.logger.info(f"Best layer: {summary['best_layer']}")

        return summary


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 - Zero-Shot Per-Dataset Evaluation")
    parser.add_argument("--model", required=True, choices=["wav2vec2_base", "wav2vec2_base_960h", "xls_r", "wavlm", "ecapa_tdnn"])
    parser.add_argument("--dataset", required=True, help="Dataset key (e.g., macaque, anuraset)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64 for HPC)")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    logger = setup_logger("Phase2_ZeroShot", config['experiment']['log_level'])

    logger.info("="*80)
    logger.info("PHASE 2 - STAGE 2: ZERO-SHOT PER-DATASET EVALUATION")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Batch size: {args.batch_size}")

    # Get manifest path
    manifest_path = Path(config['paths']['output_dir']) / "phase2" / "manifests" / f"{args.dataset}_manifest.json"

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        logger.error("Run phase2_01_create_manifests.py first")
        return

    # Create output directory
    output_dir = Path(config['paths']['output_dir']) / "phase2" / "zero_shot" / "per_dataset" / args.dataset / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    evaluator = ZeroShotEvaluator(config, args.model, manifest_path, output_dir, logger)
    summary = evaluator.evaluate_all_layers()

    logger.info(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
