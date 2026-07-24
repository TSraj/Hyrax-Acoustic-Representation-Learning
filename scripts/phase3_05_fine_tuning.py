#!/usr/bin/env python3
"""
Phase 3 - Step 5: Fine-Tuning with Data Efficiency
Fine-tunes selected models (best monolingual + multilingual) with 10%/25%/50%/100% data splits.
Evaluates data efficiency and animal adaptation.
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

# Metrics
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_auc_score
)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
from src.utils.audio_utils import load_audio


class FineTuner:
    """Fine-tune selected models with data efficiency experiments."""

    def __init__(self, config, model_name, task, manifest_path, output_dir,
                 zero_shot_results, logger, debug=False):
        """
        Initialize fine-tuner.

        Args:
            config: Configuration dict
            model_name: Model name (hubert_base or xls_r)
            task: Task name (species_id or hyrax_id)
            manifest_path: Path to manifest JSON
            output_dir: Output directory
            zero_shot_results: Zero-shot baseline results for comparison
            logger: Logger instance
            debug: If True, limit samples for quick testing
        """
        self.config = config
        self.model_name = model_name
        self.task = task
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug
        self.zero_shot_acc = zero_shot_results.get('test_metrics', {}).get('accuracy', 0.0)

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        self.num_classes = self.manifest['num_classes']

        # Create class mappings
        if task == 'hyrax_id':
            self.class_to_idx = self.manifest['class_to_idx']
            self.class_names = self.manifest['individuals']
        else:  # species_id
            self.class_to_idx = self.manifest['species_to_idx']
            self.class_names = self.manifest['species']

        # Class weights
        class_weights = [self.manifest['class_weights'][cls] for cls in self.class_names]
        self.class_weights = torch.FloatTensor(class_weights)

        # Device
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Task: {task}")
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Num classes: {self.num_classes}")
        self.logger.info(f"Zero-shot baseline: {self.zero_shot_acc:.4f}")

        # Load model
        self._load_model()

    def _load_model(self):
        """Load pretrained model for fine-tuning."""
        from transformers import (
            Wav2Vec2Model, Wav2Vec2FeatureExtractor,
            HubertModel, WavLMModel
        )

        self.logger.info(f"Loading model: {self.model_name}")

        if self.model_name == "wav2vec2_base":
            model_id = "facebook/wav2vec2-base"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.model = Wav2Vec2Model.from_pretrained(model_id, use_safetensors=True)
            self.model_type = "transformer"
            self.embedding_dim = 768

        elif self.model_name == "wav2vec2_base_960h":
            model_id = "facebook/wav2vec2-base-960h"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.model = Wav2Vec2Model.from_pretrained(model_id, use_safetensors=True)
            self.model_type = "transformer"
            self.embedding_dim = 768

        elif self.model_name == "hubert_base":
            model_id = "facebook/hubert-base-ls960"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.model = HubertModel.from_pretrained(model_id, use_safetensors=True)
            self.model_type = "transformer"
            self.embedding_dim = 768

        elif self.model_name == "xls_r":
            model_id = "facebook/wav2vec2-xls-r-300m"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.model = Wav2Vec2Model.from_pretrained(model_id, use_safetensors=True)
            self.model_type = "transformer"
            self.embedding_dim = 1024

        elif self.model_name == "wavlm":
            model_id = "microsoft/wavlm-base-plus"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.model = WavLMModel.from_pretrained(model_id, use_safetensors=True)
            self.model_type = "transformer"
            self.embedding_dim = 768

        elif self.model_name == "ecapa_tdnn":
            from speechbrain.inference.speaker import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb"
            )
            self.model_type = "ecapa"
            self.embedding_dim = 192

        # Move to device
        if self.model_type == "transformer":
            self.model = self.model.to(self.device)

        # Add classifier head
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes).to(self.device)

        self.logger.info(f"✓ Model loaded with {self.embedding_dim}-dim embeddings")

    def load_data_subset(self, split_name, data_fraction=1.0):
        """
        Load a fraction of the training data.

        Args:
            split_name: 'train', 'val', or 'test'
            data_fraction: Fraction of data to use (0.1, 0.25, 0.5, 1.0)

        Returns:
            List of data items
        """
        items = self.manifest['splits'][split_name]

        # Debug mode: limit samples
        if self.debug:
            items = items[:50]

        # Apply data fraction for training split
        if split_name == 'train' and data_fraction < 1.0:
            n_samples = max(self.num_classes, int(len(items) * data_fraction))

            # Stratified sampling: ensure all classes represented
            items_by_class = defaultdict(list)
            for item in items:
                if self.task == 'hyrax_id':
                    cls = item['individual']
                else:
                    cls = item['species']
                items_by_class[cls].append(item)

            # Sample proportionally from each class
            sampled_items = []
            for cls in self.class_names:
                cls_items = items_by_class[cls]
                n_cls = max(1, int(len(cls_items) * data_fraction))
                sampled_items.extend(cls_items[:n_cls])

            items = sampled_items

        return items

    def extract_embedding(self, audio_path):
        """Extract embedding from audio file."""
        audio, sr = load_audio(audio_path, target_sr=16000, mono=True)

        if self.model_type == "transformer":
            inputs = self.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                embedding = hidden_states.mean(dim=1).squeeze(0)

            return embedding

        else:  # ECAPA
            embedding = self.model.encode_batch(torch.FloatTensor(audio).unsqueeze(0))
            return embedding.squeeze(0).to(self.device)

    def fine_tune(self, train_items, val_items, data_fraction):
        """
        Fine-tune model on training data.

        Args:
            train_items: Training data items
            val_items: Validation data items
            data_fraction: Fraction of training data used

        Returns:
            Training history (losses, accuracies per epoch)
        """
        self.logger.info(f"\nFine-tuning with {len(train_items)} train / {len(val_items)} val samples...")

        # Unfreeze model for fine-tuning
        if self.model_type == "transformer":
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True

        self.classifier.train()

        # Optimizer with different LRs for encoder and classifier
        optimizer = optim.AdamW([
            {'params': self.model.parameters() if self.model_type == "transformer" else [], 'lr': 1e-5},
            {'params': self.classifier.parameters(), 'lr': 1e-3}
        ])

        # Loss with class weights
        criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))

        # Training parameters (reduced for debug mode)
        max_epochs = 5 if self.debug else 50
        patience = 2 if self.debug else 10
        best_val_acc = 0
        patience_counter = 0

        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        for epoch in range(max_epochs):
            # Training
            train_losses = []
            train_correct = 0
            train_total = 0

            for item in tqdm(train_items, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False):
                try:
                    file_path = item['file']
                    if not Path(file_path).exists() and not file_path.startswith('outputs/'):
                        file_path = f"Data/{file_path}"

                    # Extract embedding
                    embedding = self.extract_embedding(file_path)

                    # Get label
                    if self.task == 'hyrax_id':
                        label = self.class_to_idx[item['individual']]
                    else:
                        label = self.class_to_idx[item['species']]

                    label = torch.LongTensor([label]).to(self.device)

                    # Forward
                    outputs = self.classifier(embedding.unsqueeze(0))
                    loss = criterion(outputs, label)

                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Track metrics
                    train_losses.append(loss.item())
                    pred = outputs.argmax(dim=1).item()
                    train_correct += (pred == label.item())
                    train_total += 1

                except Exception as e:
                    continue

            # Validation
            val_acc, val_loss = self._evaluate(val_items, criterion)

            # History
            history['train_loss'].append(np.mean(train_losses))
            history['train_acc'].append(train_correct / max(train_total, 1))
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Logging
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.logger.info(
                    f"Epoch {epoch+1}: Train Loss={history['train_loss'][-1]:.4f}, "
                    f"Train Acc={history['train_acc'][-1]:.4f}, "
                    f"Val Acc={val_acc:.4f}"
                )

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict() if self.model_type == "transformer" else None
                best_classifier_state = self.classifier.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if best_model_state and self.model_type == "transformer":
            self.model.load_state_dict(best_model_state)
        self.classifier.load_state_dict(best_classifier_state)

        self.logger.info(f"✓ Best validation accuracy: {best_val_acc:.4f}")

        return history, best_val_acc

    def _evaluate(self, items, criterion):
        """Evaluate model on data items."""
        self.model.eval() if self.model_type == "transformer" else None
        self.classifier.eval()

        all_preds = []
        all_labels = []
        all_losses = []

        with torch.no_grad():
            for item in items:
                try:
                    file_path = item['file']
                    if not Path(file_path).exists() and not file_path.startswith('outputs/'):
                        file_path = f"Data/{file_path}"

                    embedding = self.extract_embedding(file_path)

                    if self.task == 'hyrax_id':
                        label = self.class_to_idx[item['individual']]
                    else:
                        label = self.class_to_idx[item['species']]

                    label_tensor = torch.LongTensor([label]).to(self.device)

                    outputs = self.classifier(embedding.unsqueeze(0))
                    loss = criterion(outputs, label_tensor)

                    all_losses.append(loss.item())
                    all_preds.append(outputs.argmax(dim=1).item())
                    all_labels.append(label)

                except Exception as e:
                    continue

        accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        avg_loss = np.mean(all_losses) if all_losses else 0.0

        self.model.train() if self.model_type == "transformer" else None
        self.classifier.train()

        return accuracy, avg_loss

    def evaluate_full(self, items):
        """Full evaluation with all metrics."""
        self.model.eval() if self.model_type == "transformer" else None
        self.classifier.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for item in tqdm(items, desc="Evaluating"):
                try:
                    file_path = item['file']
                    if not Path(file_path).exists() and not file_path.startswith('outputs/'):
                        file_path = f"Data/{file_path}"

                    embedding = self.extract_embedding(file_path)

                    if self.task == 'hyrax_id':
                        label = self.class_to_idx[item['individual']]
                    else:
                        label = self.class_to_idx[item['species']]

                    outputs = self.classifier(embedding.unsqueeze(0))
                    all_preds.append(outputs.argmax(dim=1).item())
                    all_labels.append(label)

                except Exception as e:
                    continue

        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0)
        }

        return metrics, all_preds, all_labels

    def run(self):
        """Run full fine-tuning pipeline with data efficiency experiments."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("FINE-TUNING WITH DATA EFFICIENCY")
        self.logger.info("=" * 80)

        # Data fractions to test
        data_fractions = [0.1, 0.25, 0.5, 1.0]

        results = {}

        # Load val and test sets (always full)
        val_items = self.load_data_subset('val', data_fraction=1.0)
        test_items = self.load_data_subset('test', data_fraction=1.0)

        for fraction in data_fractions:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"DATA FRACTION: {int(fraction*100)}%")
            self.logger.info(f"{'='*80}")

            # Reload model for each fraction
            self._load_model()

            # Load training subset
            train_items = self.load_data_subset('train', data_fraction=fraction)

            # Fine-tune
            history, best_val_acc = self.fine_tune(train_items, val_items, fraction)

            # Evaluate on test set
            test_metrics, test_preds, test_labels = self.evaluate_full(test_items)

            self.logger.info(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
            self.logger.info(f"Improvement over zero-shot: {test_metrics['accuracy'] - self.zero_shot_acc:+.4f}")

            results[fraction] = {
                'n_train_samples': len(train_items),
                'history': history,
                'best_val_acc': best_val_acc,
                'test_metrics': test_metrics,
                'test_preds': test_preds,
                'test_labels': test_labels
            }

        return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3 - Fine-Tuning")
    parser.add_argument("--model", required=True,
                       choices=["wav2vec2_base", "wav2vec2_base_960h", "hubert_base",
                               "xls_r", "wavlm", "ecapa_tdnn"])
    parser.add_argument("--task", required=True,
                       choices=["species_id", "hyrax_id"])
    parser.add_argument("--debug", action="store_true", help="Debug mode: small subset")
    args = parser.parse_args()

    # Setup logging
    log_dir = Path("outputs/phase3/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        f"Phase3_FineTune_{args.task}_{args.model}",
        log_file=str(log_dir / f"fine_tune_{args.task}_{args.model}.log")
    )

    # Paths
    manifest_path = Path(f"outputs/phase3/manifests/{args.task}.json")
    output_dir = Path(f"outputs/phase3/fine_tuning/{args.task}/{args.model}")

    # Load zero-shot baseline
    zero_shot_file = Path(f"outputs/phase3/zero_shot/{args.task}/{args.model}/results.json")
    if zero_shot_file.exists():
        with open(zero_shot_file, 'r') as f:
            zero_shot_results = json.load(f)
    else:
        logger.warning(f"Zero-shot baseline not found: {zero_shot_file}")
        zero_shot_results = {}

    # Load config
    config = {}

    # Run fine-tuning
    fine_tuner = FineTuner(
        config, args.model, args.task, manifest_path, output_dir,
        zero_shot_results, logger, debug=args.debug
    )

    results = fine_tuner.run()

    # Save results
    results_file = output_dir / "fine_tuning_results.json"
    # Convert numpy arrays to lists for JSON
    results_json = {}
    for fraction, data in results.items():
        results_json[str(fraction)] = {
            'n_train_samples': data['n_train_samples'],
            'history': data['history'],
            'best_val_acc': data['best_val_acc'],
            'test_metrics': data['test_metrics']
        }

    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    logger.info(f"\n✓ Results saved: {results_file}")
    logger.info("\n✓ Fine-tuning complete!")


if __name__ == "__main__":
    main()
