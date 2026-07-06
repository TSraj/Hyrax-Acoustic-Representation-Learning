#!/usr/bin/env python3
"""
Phase 2 - Stage 5: Fine-Tuning Pipeline
Fine-tunes the best model from Stage 4 on multi-dataset pooled data.
Fine-tunes first 4 layers only, freezes the rest.
"""

import json
import yaml
import sys
import torch
torch.backends.cudnn.enabled = False  # V100 (CC 7.0) incompatible with bundled cuDNN
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
from src.utils.audio_utils import load_audio


class AudioDataset(Dataset):
    """Dataset for audio files with on-the-fly loading."""

    def __init__(self, manifest_items, class_to_idx, data_dir, max_duration=30):
        self.items = manifest_items
        self.class_to_idx = class_to_idx
        self.data_dir = Path(data_dir)
        self.max_duration = max_duration

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_path = self.data_dir / item['file']
        individual = item['individual']
        label = self.class_to_idx[individual]

        # Load audio
        audio, sr = load_audio(str(audio_path), target_sr=16000, mono=True)

        # Truncate if too long
        max_samples = int(self.max_duration * 16000)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Pad very short files (minimum 0.5 seconds)
        min_samples = int(0.5 * 16000)
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')

        return torch.FloatTensor(audio), label


def collate_fn(batch):
    """Custom collate function to pad variable-length audio to same length."""
    audios, labels = zip(*batch)

    # Find max length in batch
    max_len = max(audio.shape[0] for audio in audios)

    # Pad all audios to max length
    padded_audios = []
    for audio in audios:
        if audio.shape[0] < max_len:
            padding = torch.zeros(max_len - audio.shape[0])
            audio = torch.cat([audio, padding])
        padded_audios.append(audio)

    # Stack into batch
    audio_batch = torch.stack(padded_audios)
    label_batch = torch.LongTensor(labels)

    return audio_batch, label_batch


class FineTuner:
    """Fine-tunes the best model on multi-dataset pooled data."""

    def __init__(self, config, model_name, manifest_path, output_dir, logger):
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
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Num classes: {self.num_classes}")

        # Load model
        self._load_and_prepare_model()

    def _load_and_prepare_model(self):
        """Load model and prepare for fine-tuning (first 4 layers only)."""
        from transformers import (
            Wav2Vec2Model,
            Wav2Vec2FeatureExtractor,
            WavLMModel
        )

        self.logger.info(f"Loading model: {self.model_name}")

        # Determine model ID
        if self.model_name == "wav2vec2_base":
            model_id = "facebook/wav2vec2-base"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.backbone = Wav2Vec2Model.from_pretrained(model_id, use_safetensors=True)

        elif self.model_name == "wav2vec2_base_960h":
            model_id = "facebook/wav2vec2-base-960h"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.backbone = Wav2Vec2Model.from_pretrained(model_id, use_safetensors=True)

        elif self.model_name == "xls_r":
            model_id = "facebook/wav2vec2-xls-r-300m"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.backbone = Wav2Vec2Model.from_pretrained(model_id, use_safetensors=True)

        elif self.model_name == "wavlm":
            model_id = "microsoft/wavlm-base-plus"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.backbone = WavLMModel.from_pretrained(model_id, use_safetensors=True)

        elif self.model_name == "ecapa_tdnn":
            # ECAPA-TDNN from SpeechBrain
            from speechbrain.inference.speaker import EncoderClassifier

            self.logger.info("Loading ECAPA-TDNN for fine-tuning...")
            self.backbone = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/ecapa_tdnn"
            )
            self.feature_extractor = None  # ECAPA has internal preprocessing
            self.model_type = "ecapa"

            # For ECAPA, we'll freeze the feature extraction layers and fine-tune the embedding layer
            # ECAPA structure: feature extraction -> TDNN blocks -> embedding layer -> classifier
            # We'll freeze everything except the last few layers

            # Move to device (ECAPA modules)
            if hasattr(self.backbone, 'mods'):
                for module in self.backbone.mods.values():
                    if hasattr(module, 'to'):
                        module.to(self.device)

            # Freeze all parameters first
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

            # Unfreeze only the final embedding layers (last 2 TDNN blocks)
            for name, param in self.backbone.named_parameters():
                if 'tdnn' in name.lower() or 'embedding' in name.lower():
                    # Check if it's in the last layers (heuristic: contains "5", "6", or "embedding")
                    if any(x in name for x in ['layer5', 'layer6', 'tdnn.5', 'tdnn.6', 'embedding']):
                        param.requires_grad = True
                        self.logger.info(f"  Unfreezing: {name}")

            # Create classification head for ECAPA (embedding size is 192)
            hidden_size = 192
            self.classifier = nn.Linear(hidden_size, self.num_classes).to(self.device)
            self.logger.info(f"ECAPA embedding size: {hidden_size}")

            # Mark as ECAPA model type
            self.is_ecapa = True

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        # For transformer models only
        if self.model_name != "ecapa_tdnn":
            self.is_ecapa = False
            self.backbone.to(self.device)

            # Freeze all parameters first
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze first 4 transformer layers only
            num_layers = len(self.backbone.encoder.layers)
            self.logger.info(f"Total transformer layers: {num_layers}")
            self.logger.info(f"Fine-tuning first 4 layers, freezing remaining {num_layers - 4} layers")

            for i in range(min(4, num_layers)):
                for param in self.backbone.encoder.layers[i].parameters():
                    param.requires_grad = True

            # Create classification head
            # Get embedding dimension from last hidden state
            hidden_size = self.backbone.config.hidden_size
            self.classifier = nn.Linear(hidden_size, self.num_classes).to(self.device)

        self.logger.info(f"Classification head: {hidden_size} -> {self.num_classes}")
        self.logger.info("✓ Model prepared for fine-tuning")

        # Count trainable parameters
        if self.is_ecapa:
            # For ECAPA, count all named parameters
            trainable_params = sum(p.numel() for n, p in self.backbone.named_parameters() if p.requires_grad)
            total_params = sum(p.numel() for n, p in self.backbone.named_parameters())
        else:
            trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.backbone.parameters())

        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        self.logger.info(f"Backbone trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
        self.logger.info(f"Classifier params: {classifier_params:,}")

    def forward(self, audio):
        """Forward pass through backbone + classifier."""
        if self.is_ecapa:
            # ECAPA-TDNN forward pass
            # audio should be a list of tensors or a batched tensor
            if isinstance(audio, list):
                # Batch of audio samples
                embeddings = []
                for aud in audio:
                    emb = self.backbone.encode_batch(aud.unsqueeze(0).to(self.device))
                    embeddings.append(emb.squeeze())
                embeddings = torch.stack(embeddings)  # (batch, 192)
            else:
                # Single batched tensor
                embeddings = self.backbone.encode_batch(audio.to(self.device))  # (batch, 192)
                if embeddings.dim() == 3:
                    embeddings = embeddings.squeeze(1)  # Remove extra dimension if present

            # Classifier
            logits = self.classifier(embeddings)
            return logits

        else:
            # Transformer models forward pass
            # Preprocess audio
            inputs = self.feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward through backbone
            outputs = self.backbone(**inputs)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

            # Mean pooling
            embeddings = hidden_states.mean(dim=1)  # (batch, hidden_dim)

            # Classifier
            logits = self.classifier(embeddings)

            return logits

    def create_dataloaders(self, batch_size=16):
        """Create train/val/test dataloaders."""
        self.logger.info("\nCreating dataloaders...")

        data_dir = Path(self.config['paths']['data_dir'])

        train_dataset = AudioDataset(
            self.manifest['train'],
            self.class_to_idx,
            data_dir,
            max_duration=30
        )

        val_dataset = AudioDataset(
            self.manifest['val'],
            self.class_to_idx,
            data_dir,
            max_duration=30
        )

        test_dataset = AudioDataset(
            self.manifest['test'],
            self.class_to_idx,
            data_dir,
            max_duration=30
        )

        # Debug mode: use small subset
        if hasattr(self, 'debug_mode') and self.debug_mode:
            self.logger.info("  🐛 DEBUG MODE: Using small subset")
            train_dataset.items = train_dataset.items[:50]
            val_dataset.items = val_dataset.items[:20]
            test_dataset.items = test_dataset.items[:20]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

        self.logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        self.logger.info(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
        self.logger.info(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")

        return train_loader, val_loader, test_loader

    def train(self, train_loader, val_loader, max_epochs=50, patience=10, lr=1e-4):
        """
        Train the model with fine-tuning.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            lr: Learning rate
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING")
        self.logger.info("="*80)

        # Loss function
        criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))

        # Optimizer (both backbone layers and classifier)
        if self.is_ecapa:
            # For ECAPA, collect trainable parameters via named_parameters
            backbone_params = [p for n, p in self.backbone.named_parameters() if p.requires_grad]
        else:
            backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': lr},
            {'params': self.classifier.parameters(), 'lr': lr * 10}  # Higher LR for classifier
        ])

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

        # Training state
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

        best_val_acc = 0.0
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(max_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{max_epochs}")
            self.logger.info("-" * 60)

            # Train mode
            if not self.is_ecapa:
                self.backbone.train()
            else:
                # ECAPA modules need to be set to train mode individually
                if hasattr(self.backbone, 'mods'):
                    for module in self.backbone.mods.values():
                        if hasattr(module, 'train'):
                            module.train()
            self.classifier.train()

            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for audio_batch, labels in tqdm(train_loader, desc="Training"):
                labels = labels.to(self.device)

                optimizer.zero_grad()

                # Forward
                logits = self.forward(audio_batch.numpy())
                loss = criterion(logits, labels)

                # Backward
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            # Eval mode
            if not self.is_ecapa:
                self.backbone.eval()
            else:
                # ECAPA modules need to be set to eval mode individually
                if hasattr(self.backbone, 'mods'):
                    for module in self.backbone.mods.values():
                        if hasattr(module, 'eval'):
                            module.eval()
            self.classifier.eval()

            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for audio_batch, labels in tqdm(val_loader, desc="Validation"):
                    labels = labels.to(self.device)

                    logits = self.forward(audio_batch.numpy())
                    loss = criterion(logits, labels)

                    val_loss += loss.item()
                    _, predicted = logits.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            # Update scheduler
            scheduler.step(val_acc)

            # Record history
            current_lr = optimizer.param_groups[0]['lr']
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)

            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            self.logger.info(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
            self.logger.info(f"LR: {current_lr:.6f}")

            # Save checkpoint if best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                # Save best checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'backbone_state_dict': self.backbone.state_dict(),
                    'classifier_state_dict': self.classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }
                torch.save(checkpoint, self.output_dir / "checkpoints" / "best_model.pth")
                self.logger.info(f"✓ Saved best model (val_acc: {val_acc:.4f})")
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                self.logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        self.logger.info(f"\n✓ Training complete")
        self.logger.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

        return history, best_val_acc

    def evaluate(self, test_loader):
        """Evaluate on test set."""
        self.logger.info("\n" + "="*80)
        self.logger.info("TESTING")
        self.logger.info("="*80)

        # Load best checkpoint
        checkpoint_path = self.output_dir / "checkpoints" / "best_model.pth"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if self.is_ecapa:
            # For ECAPA, load state dict with strict=False to handle SpeechBrain structure
            try:
                self.backbone.load_state_dict(checkpoint['backbone_state_dict'], strict=False)
            except:
                self.logger.warning("Could not load ECAPA backbone state dict - using current state")
        else:
            self.backbone.load_state_dict(checkpoint['backbone_state_dict'])

        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])

        self.logger.info(f"✓ Loaded best checkpoint from epoch {checkpoint['epoch']}")

        # Evaluate
        if not self.is_ecapa:
            self.backbone.eval()
        else:
            # ECAPA modules need to be set to eval mode individually
            if hasattr(self.backbone, 'mods'):
                for module in self.backbone.mods.values():
                    if hasattr(module, 'eval'):
                        module.eval()
        self.classifier.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for audio_batch, labels in tqdm(test_loader, desc="Testing"):
                logits = self.forward(audio_batch.numpy())
                _, predicted = logits.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Calculate metrics
        test_acc = accuracy_score(all_labels, all_preds)

        # Get only classes present in test set (important for debug mode with small samples)
        unique_labels = sorted(set(all_labels))
        present_class_names = [self.manifest['individuals'][i] for i in unique_labels]

        report = classification_report(
            all_labels,
            all_preds,
            labels=unique_labels,
            target_names=present_class_names,
            output_dict=True,
            zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)

        self.logger.info(f"\nTest Accuracy: {test_acc:.4f}")

        results = {
            'test_accuracy': test_acc,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds,
            'labels': all_labels
        }

        return results

    def plot_training_curves(self, history, save_dir):
        """Plot training curves as separate individual figures."""
        epochs = range(1, len(history['train_loss']) + 1)
        best_val_acc = max(history['val_acc'])
        best_epoch = history['val_acc'].index(best_val_acc) + 1

        # Figure 1: Loss curves
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
        ax.set_ylabel('Loss', fontweight='bold', fontsize=12)
        ax.set_title('Training and Validation Loss', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "1_loss_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  ✓ Loss curves saved")

        # Figure 2: Accuracy curves
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        ax.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, linewidth=2, label=f'Best Epoch ({best_epoch})')
        ax.plot(best_epoch, best_val_acc, 'g*', markersize=20, label=f'Best Val Acc: {best_val_acc:.4f}')
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
        ax.set_title('Training and Validation Accuracy', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "2_accuracy_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  ✓ Accuracy curves saved")

        # Figure 3: Learning rate schedule
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, history['lr'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
        ax.set_ylabel('Learning Rate', fontweight='bold', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontweight='bold', fontsize=14)
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "3_learning_rate_schedule.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  ✓ Learning rate schedule saved")

        # Figure 4: Training summary table
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')

        table_data = [
            ['Metric', 'Value'],
            ['Model', self.model_name],
            ['Fine-tuned Layers', 'First 4 layers'],
            ['Num Classes', str(self.num_classes)],
            ['Total Epochs', str(len(epochs))],
            ['Best Epoch', str(best_epoch)],
            ['Best Val Accuracy', f'{best_val_acc:.4f}'],
            ['Final Train Loss', f'{history["train_loss"][-1]:.4f}'],
            ['Final Val Loss', f'{history["val_loss"][-1]:.4f}'],
        ]

        table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                        colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.scale(1, 3)

        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=14)

        # Highlight best epoch row
        table[(5, 0)].set_facecolor('#E8F5E9')
        table[(5, 1)].set_facecolor('#E8F5E9')
        table[(6, 0)].set_facecolor('#E8F5E9')
        table[(6, 1)].set_facecolor('#E8F5E9')

        ax.set_title('Training Summary', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(save_dir / "4_training_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"  ✓ Training summary saved")

    def plot_confusion_matrix(self, cm, save_path):
        """Plot confusion matrix with high quality and readable labels."""
        # Dynamic figure size based on number of classes
        fig_width = max(16, self.num_classes * 0.4)
        fig_height = max(14, self.num_classes * 0.35)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Show labels only if <= 30 classes, otherwise too crowded
        if self.num_classes <= 30:
            xticklabels = self.manifest['individuals']
            yticklabels = self.manifest['individuals']
            fontsize = max(8, min(12, 300 // self.num_classes))  # Scale font with class count
        else:
            xticklabels = False
            yticklabels = False
            fontsize = 10

        sns.heatmap(
            cm,
            annot=False,  # No numbers in cells (too crowded)
            cmap='Blues',
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            cbar_kws={'label': 'Count', 'shrink': 0.8},
            linewidths=0,  # No grid lines for cleaner look
            square=True,  # Square cells
            ax=ax
        )

        # Title with clear info
        ax.set_title(f'Confusion Matrix - Fine-Tuned {self.model_name}\n{self.num_classes} Classes',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')

        # Rotate labels if shown
        if self.num_classes <= 30:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsize)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        self.logger.info(f"✓ Confusion matrix saved: {save_path}")

    def run(self, batch_size=16, max_epochs=50, patience=10, lr=1e-4, debug=False):
        """Run complete fine-tuning pipeline."""
        self.debug_mode = debug

        if debug:
            self.logger.info("\n" + "="*80)
            self.logger.info("🐛 DEBUG MODE: PHASE 2 - STAGE 5: FINE-TUNING")
            self.logger.info("="*80)
            max_epochs = 3
            patience = 2
            self.logger.info(f"  Using reduced epochs: {max_epochs}, patience: {patience}")
        else:
            self.logger.info("\n" + "="*80)
            self.logger.info("PHASE 2 - STAGE 5: FINE-TUNING")
            self.logger.info("="*80)

        # Create dataloaders
        train_loader, val_loader, test_loader = self.create_dataloaders(batch_size)

        # Train
        history, best_val_acc = self.train(train_loader, val_loader, max_epochs, patience, lr)

        # Evaluate
        test_results = self.evaluate(test_loader)

        # Save results
        results_dir = self.output_dir / "results"
        results_dir.mkdir(exist_ok=True)

        summary = {
            'model': self.model_name,
            'num_classes': self.num_classes,
            'fine_tuned_layers': 4,
            'best_val_accuracy': best_val_acc,
            'test_accuracy': test_results['test_accuracy'],
            'training_epochs': len(history['train_loss']),
            'batch_size': batch_size,
            'learning_rate': lr
        }

        with open(results_dir / "fine_tuning_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        with open(results_dir / "test_results.json", 'w') as f:
            # Remove numpy arrays for JSON serialization
            test_results_json = {k: v for k, v in test_results.items() if k not in ['predictions', 'labels']}
            json.dump(test_results_json, f, indent=2)

        # Plot training curves (now saves multiple individual figures)
        curves_dir = self.output_dir / "training_curves"
        curves_dir.mkdir(exist_ok=True)
        self.plot_training_curves(history, curves_dir)

        # Plot confusion matrix
        self.plot_confusion_matrix(np.array(test_results['confusion_matrix']),
                                   results_dir / "confusion_matrix.png")

        self.logger.info(f"\n{'='*80}")
        self.logger.info("FINE-TUNING COMPLETE")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Best Val Accuracy: {best_val_acc:.4f}")
        self.logger.info(f"Test Accuracy: {test_results['test_accuracy']:.4f}")
        self.logger.info(f"\nAll results saved to: {self.output_dir}")

        return summary


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 - Fine-Tuning")
    parser.add_argument("--model", help="Model to fine-tune (default: load from Stage 4 selection)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--max-epochs", type=int, default=16, help="Max epochs (default: 16)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--debug", action="store_true", help="Debug mode: use small subset and 3 epochs")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    logger = setup_logger("Phase2_FineTuning", config['experiment']['log_level'])

    logger.info("="*80)
    logger.info("PHASE 2 - STAGE 5: FINE-TUNING")
    logger.info("="*80)

    # Determine which model to fine-tune
    if args.model:
        model_name = args.model
        logger.info(f"Using specified model: {model_name}")
    else:
        # Load from Stage 4 selection
        selection_path = Path(config['paths']['output_dir']) / "phase2" / "model_selection" / "best_model_selection.json"

        if not selection_path.exists():
            logger.error(f"Best model selection not found: {selection_path}")
            logger.error("Run phase2_04_model_selection.py first, or specify --model")
            return

        with open(selection_path, 'r') as f:
            selection = json.load(f)

        model_name = selection['best_model']
        logger.info(f"Using best model from Stage 4: {model_name}")

    # Get pooled manifest
    manifest_path = Path(config['paths']['output_dir']) / "phase2" / "manifests" / "pooled_manifest.json"

    if not manifest_path.exists():
        logger.error(f"Pooled manifest not found: {manifest_path}")
        logger.error("Run phase2_01_create_manifests.py first")
        return

    # Create output directory
    output_dir = Path(config['paths']['output_dir']) / "phase2" / "fine_tuning" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "training_curves").mkdir(exist_ok=True)

    # Run fine-tuning
    finetuner = FineTuner(config, model_name, manifest_path, output_dir, logger)
    summary = finetuner.run(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        debug=args.debug
    )

    logger.info("\n✓ Fine-tuning complete - Ready for Hyrax evaluation")


if __name__ == "__main__":
    main()
