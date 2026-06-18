#!/usr/bin/env python3
"""
Phase 2 - Stage 6: Sampling Rate Experiment
Tests whether information is lost by resampling to 16kHz.
Uses ResNet-18 on mel spectrograms: original rate vs 16kHz.
"""

import json
import yaml
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torchvision import models

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
from src.utils.audio_utils import load_audio


class MelSpectrogramDataset(Dataset):
    """Dataset for mel spectrogram extraction."""

    def __init__(self, manifest_items, class_to_idx, data_dir, target_sr, n_mels=128, max_duration=10):
        self.items = manifest_items
        self.class_to_idx = class_to_idx
        self.data_dir = Path(data_dir)
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.max_duration = max_duration

        # Mel spectrogram parameters (proportional to sampling rate)
        self.n_fft = int(2048 * (target_sr / 22050))
        self.hop_length = int(512 * (target_sr / 22050))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_path = self.data_dir / item['file']
        individual = item['individual']
        label = self.class_to_idx[individual]

        # Load audio at target sampling rate
        audio, sr = load_audio(str(audio_path), target_sr=self.target_sr, mono=True)

        # Truncate if too long
        max_samples = int(self.max_duration * self.target_sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

        # Pad or crop to fixed width (time dimension)
        target_width = int(self.max_duration * self.target_sr / self.hop_length)
        if mel_spec_norm.shape[1] < target_width:
            pad_width = target_width - mel_spec_norm.shape[1]
            mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_norm = mel_spec_norm[:, :target_width]

        # Convert to 3-channel (RGB) for ResNet
        mel_spec_rgb = np.stack([mel_spec_norm] * 3, axis=0)

        return torch.FloatTensor(mel_spec_rgb), label


class ResNet18Classifier(nn.Module):
    """ResNet-18 classifier for mel spectrograms."""

    def __init__(self, num_classes):
        super(ResNet18Classifier, self).__init__()

        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=True)

        # Replace final layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class SamplingRateExperiment:
    """Compares model performance at original vs 16kHz sampling rates."""

    def __init__(self, config, dataset_name, manifest_path, output_dir, logger):
        self.config = config
        self.dataset_name = dataset_name
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
        self.logger.info(f"Dataset: {self.dataset_name}")
        self.logger.info(f"Num classes: {self.num_classes}")

    def get_original_sampling_rate(self):
        """Detect original sampling rate from first audio file."""
        first_file = self.manifest['train'][0]['file']
        audio_path = Path(self.config['paths']['data_dir']) / first_file

        import soundfile as sf
        info = sf.info(str(audio_path))

        self.logger.info(f"Original sampling rate detected: {info.samplerate} Hz")
        return info.samplerate

    def create_dataloaders(self, target_sr, batch_size=32):
        """Create dataloaders for a specific sampling rate."""
        data_dir = Path(self.config['paths']['data_dir'])

        train_dataset = MelSpectrogramDataset(
            self.manifest['train'],
            self.class_to_idx,
            data_dir,
            target_sr=target_sr,
            n_mels=128,
            max_duration=10
        )

        val_dataset = MelSpectrogramDataset(
            self.manifest['val'],
            self.class_to_idx,
            data_dir,
            target_sr=target_sr,
            n_mels=128,
            max_duration=10
        )

        test_dataset = MelSpectrogramDataset(
            self.manifest['test'],
            self.class_to_idx,
            data_dir,
            target_sr=target_sr,
            n_mels=128,
            max_duration=10
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_loader, val_loader, test_loader

    def train_model(self, train_loader, val_loader, sampling_rate, max_epochs=50, patience=10, lr=1e-3):
        """Train ResNet-18 model."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"TRAINING: {sampling_rate} Hz")
        self.logger.info(f"{'='*80}")

        # Create model
        model = ResNet18Classifier(self.num_classes).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        # Training state
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        best_val_acc = 0.0
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(max_epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for mel_specs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]", leave=False):
                mel_specs = mel_specs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(mel_specs)
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
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for mel_specs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]", leave=False):
                    mel_specs = mel_specs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(mel_specs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            # Update scheduler
            scheduler.step(val_acc)

            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            self.logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        model.load_state_dict(best_model_state)

        self.logger.info(f"✓ Training complete. Best val acc: {best_val_acc:.4f}")

        return model, history, best_val_acc

    def evaluate_model(self, model, test_loader):
        """Evaluate model on test set."""
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for mel_specs, labels in tqdm(test_loader, desc="Testing"):
                mel_specs = mel_specs.to(self.device)
                outputs = model(mel_specs)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        test_acc = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        self.logger.info(f"Test accuracy: {test_acc:.4f}")

        return {
            'accuracy': test_acc,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels
        }

    def run_experiment(self, original_sr, batch_size=32, max_epochs=50):
        """Run complete experiment: original rate vs 16kHz."""
        self.logger.info("\n" + "="*80)
        self.logger.info("PHASE 2 - STAGE 6: SAMPLING RATE EXPERIMENT")
        self.logger.info("="*80)
        self.logger.info(f"Dataset: {self.dataset_name}")
        self.logger.info(f"Original sampling rate: {original_sr} Hz")
        self.logger.info(f"Comparison sampling rate: 16000 Hz")

        results = {}

        # Experiment 1: Original sampling rate
        self.logger.info("\n" + "="*80)
        self.logger.info("EXPERIMENT 1: ORIGINAL SAMPLING RATE")
        self.logger.info("="*80)

        train_loader_orig, val_loader_orig, test_loader_orig = self.create_dataloaders(
            target_sr=original_sr,
            batch_size=batch_size
        )

        model_orig, history_orig, best_val_acc_orig = self.train_model(
            train_loader_orig,
            val_loader_orig,
            sampling_rate=original_sr,
            max_epochs=max_epochs
        )

        test_results_orig = self.evaluate_model(model_orig, test_loader_orig)

        results['original'] = {
            'sampling_rate': original_sr,
            'best_val_accuracy': best_val_acc_orig,
            'test_accuracy': test_results_orig['accuracy'],
            'training_history': history_orig
        }

        # Save original rate model
        torch.save(model_orig.state_dict(), self.output_dir / "original_rate" / "model.pth")

        # Experiment 2: 16kHz
        self.logger.info("\n" + "="*80)
        self.logger.info("EXPERIMENT 2: 16kHz RESAMPLED")
        self.logger.info("="*80)

        train_loader_16k, val_loader_16k, test_loader_16k = self.create_dataloaders(
            target_sr=16000,
            batch_size=batch_size
        )

        model_16k, history_16k, best_val_acc_16k = self.train_model(
            train_loader_16k,
            val_loader_16k,
            sampling_rate=16000,
            max_epochs=max_epochs
        )

        test_results_16k = self.evaluate_model(model_16k, test_loader_16k)

        results['16khz'] = {
            'sampling_rate': 16000,
            'best_val_accuracy': best_val_acc_16k,
            'test_accuracy': test_results_16k['accuracy'],
            'training_history': history_16k
        }

        # Save 16kHz model
        torch.save(model_16k.state_dict(), self.output_dir / "16khz" / "model.pth")

        # Comparison
        accuracy_diff = test_results_orig['accuracy'] - test_results_16k['accuracy']
        percent_loss = (accuracy_diff / test_results_orig['accuracy']) * 100 if test_results_orig['accuracy'] > 0 else 0

        results['comparison'] = {
            'accuracy_difference': accuracy_diff,
            'percent_information_loss': percent_loss,
            'original_better': test_results_orig['accuracy'] > test_results_16k['accuracy']
        }

        self.logger.info("\n" + "="*80)
        self.logger.info("COMPARISON RESULTS")
        self.logger.info("="*80)
        self.logger.info(f"Original ({original_sr} Hz): {test_results_orig['accuracy']*100:.2f}%")
        self.logger.info(f"16kHz resampled:           {test_results_16k['accuracy']*100:.2f}%")
        self.logger.info(f"Accuracy difference:       {accuracy_diff*100:.2f}%")
        self.logger.info(f"Information loss:          {percent_loss:.2f}%")

        if test_results_orig['accuracy'] > test_results_16k['accuracy']:
            self.logger.info("✓ Original sampling rate performs better")
        else:
            self.logger.info("⚠ 16kHz performs equal or better (no information loss detected)")

        # Save results
        with open(self.output_dir / "comparison" / "results.json", 'w') as f:
            # Remove large arrays for JSON
            results_json = {
                'original': {k: v for k, v in results['original'].items() if k != 'training_history'},
                '16khz': {k: v for k, v in results['16khz'].items() if k != 'training_history'},
                'comparison': results['comparison']
            }
            json.dump(results_json, f, indent=2)

        # Create visualizations
        self.create_comparison_plots(results, test_results_orig, test_results_16k, original_sr)

        return results

    def create_comparison_plots(self, results, test_results_orig, test_results_16k, original_sr):
        """Create comparison visualizations."""
        comparison_dir = self.output_dir / "comparison"
        comparison_dir.mkdir(exist_ok=True)

        # 1. Training curves comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        history_orig = results['original']['training_history']
        history_16k = results['16khz']['training_history']

        epochs_orig = range(1, len(history_orig['train_acc']) + 1)
        epochs_16k = range(1, len(history_16k['train_acc']) + 1)

        # Accuracy
        axes[0].plot(epochs_orig, history_orig['val_acc'], 'b-', linewidth=2, label=f'Original ({original_sr} Hz)')
        axes[0].plot(epochs_16k, history_16k['val_acc'], 'r-', linewidth=2, label='16kHz')
        axes[0].set_xlabel('Epoch', fontweight='bold')
        axes[0].set_ylabel('Validation Accuracy', fontweight='bold')
        axes[0].set_title('Validation Accuracy Comparison', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Loss
        axes[1].plot(epochs_orig, history_orig['val_loss'], 'b-', linewidth=2, label=f'Original ({original_sr} Hz)')
        axes[1].plot(epochs_16k, history_16k['val_loss'], 'r-', linewidth=2, label='16kHz')
        axes[1].set_xlabel('Epoch', fontweight='bold')
        axes[1].set_ylabel('Validation Loss', fontweight='bold')
        axes[1].set_title('Validation Loss Comparison', fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(comparison_dir / "training_curves_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Test accuracy comparison bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        models = [f'Original\n({original_sr} Hz)', '16kHz\nResampled']
        accuracies = [test_results_orig['accuracy'] * 100, test_results_16k['accuracy'] * 100]

        colors = ['blue' if accuracies[0] > accuracies[1] else 'red', 'red' if accuracies[0] > accuracies[1] else 'blue']

        bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=2, alpha=0.7)

        ax.set_ylabel('Test Accuracy (%)', fontweight='bold', fontsize=12)
        ax.set_title(f'Sampling Rate Experiment: {self.dataset_name}\nTest Accuracy Comparison', fontweight='bold', fontsize=14)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, acc + 1, f'{acc:.2f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)

        # Add difference annotation
        diff = accuracies[0] - accuracies[1]
        ax.text(0.5, max(accuracies) - 5, f'Δ = {diff:.2f}%', ha='center',
               fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        plt.tight_layout()
        plt.savefig(comparison_dir / "test_accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ Comparison plots saved to: {comparison_dir}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 - Sampling Rate Experiment")
    parser.add_argument("--dataset", required=True, choices=["picidae", "wetlands_bird"],
                       help="Dataset to use (Picidae or Wetlands Bird)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--max-epochs", type=int, default=50, help="Max epochs (default: 50)")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    logger = setup_logger("Phase2_SamplingRateExperiment", config['experiment']['log_level'])

    logger.info("="*80)
    logger.info("PHASE 2 - STAGE 6: SAMPLING RATE EXPERIMENT")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset}")

    # Get manifest path
    manifest_path = Path(config['paths']['output_dir']) / "phase2" / "manifests" / f"{args.dataset}_manifest.json"

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        logger.error("Run phase2_01_create_manifests.py first")
        return

    # Create output directory
    output_dir = Path(config['paths']['output_dir']) / "phase2" / "sampling_rate_experiment" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "original_rate").mkdir(exist_ok=True)
    (output_dir / "16khz").mkdir(exist_ok=True)
    (output_dir / "comparison").mkdir(exist_ok=True)

    # Run experiment
    experiment = SamplingRateExperiment(config, args.dataset, manifest_path, output_dir, logger)
    original_sr = experiment.get_original_sampling_rate()

    results = experiment.run_experiment(
        original_sr=original_sr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs
    )

    logger.info(f"\n✓ Experiment complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
