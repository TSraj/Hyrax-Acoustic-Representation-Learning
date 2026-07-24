#!/usr/bin/env python3
"""
Phase 3 - Step 3: Zero-Shot Evaluation
Evaluates 6 models on Species ID (8-class) and Hyrax ID (18-class) tasks.
Frozen encoder + trained classifier head.
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict

# Metrics
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_auc_score, top_k_accuracy_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
from src.utils.audio_utils import load_audio


class ZeroShotEvaluator:
    """Zero-shot evaluator with frozen backbone + trained FC head."""

    def __init__(self, config, model_name, task, manifest_path, output_dir, logger, debug=False):
        """
        Initialize evaluator.

        Args:
            config: Configuration dict
            model_name: Model name (wav2vec2_base, hubert_base, etc.)
            task: Task name (species_id or hyrax_id)
            manifest_path: Path to manifest JSON
            output_dir: Output directory
            logger: Logger instance
            debug: If True, limit to small subset for quick testing
        """
        self.config = config
        self.model_name = model_name
        self.task = task
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        self.num_classes = self.manifest['num_classes']

        # Create class mappings based on task
        if task == 'hyrax_id':
            self.class_to_idx = self.manifest['class_to_idx']
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
            self.class_names = self.manifest['individuals']
        else:  # species_id
            self.class_to_idx = self.manifest['species_to_idx']
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
            self.class_names = self.manifest['species']

        # Get class weights
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

        # Load model
        self._load_model()

    def _load_model(self):
        """Load pretrained model."""
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

        elif self.model_name == "wav2vec2_base_960h":
            model_id = "facebook/wav2vec2-base-960h"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.model = Wav2Vec2Model.from_pretrained(model_id, use_safetensors=True)
            self.model_type = "transformer"

        elif self.model_name == "hubert_base":
            model_id = "facebook/hubert-base-ls960"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            self.model = HubertModel.from_pretrained(model_id, use_safetensors=True)
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

        # Move to device and freeze
        if self.model_type == "transformer":
            self.model.to(self.device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.logger.info(f"✓ Model loaded and frozen")

    def extract_embedding(self, audio_path):
        """Extract embedding from audio file (use last layer)."""
        # Load audio
        audio, sr = load_audio(str(audio_path), target_sr=16000, mono=True)

        # Truncate if too long (30 seconds max)
        max_samples = 30 * 16000
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Extract embedding
        if self.model_type == "transformer":
            inputs = self.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )

            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)

                # Use last hidden state, mean pool over time
                hidden_states = outputs.last_hidden_state  # [1, T, D]
                embedding = hidden_states.mean(dim=1).squeeze(0)  # [D]

            return embedding.cpu().numpy()

        else:  # ECAPA
            with torch.no_grad():
                # encode_batch returns [batch, embedding_dim]
                embedding = self.model.encode_batch(torch.FloatTensor(audio).unsqueeze(0))
                # Squeeze batch dimension to get [embedding_dim]
                embedding = embedding.squeeze(0)
                # If still 2D (has time dimension), take mean
                if embedding.dim() > 1:
                    embedding = embedding.mean(dim=0)
                return embedding.cpu().numpy()

    def extract_embeddings_for_split(self, split_name):
        """Extract embeddings for all items in a split."""
        items = self.manifest['splits'][split_name]

        # Debug mode: limit to small subset
        if self.debug:
            max_samples = 50  # 50 samples per split for quick smoke test
            items = items[:max_samples]
            self.logger.info(f"\n[DEBUG MODE] Limiting to {len(items)} samples")

        embeddings = []
        labels = []

        self.logger.info(f"\nExtracting embeddings for {split_name} split ({len(items)} files)...")

        for item in tqdm(items, desc=f"{split_name}"):
            try:
                file_path = item['file']

                # Handle Phase 2 paths (relative to data/ folder)
                if not Path(file_path).exists() and not file_path.startswith('outputs/'):
                    # Phase 2 files are in data/ folder
                    file_path = f"Data/{file_path}"

                embedding = self.extract_embedding(file_path)

                # Get label
                if self.task == 'hyrax_id':
                    label = self.class_to_idx[item['individual']]
                else:  # species_id
                    label = self.class_to_idx[item['species']]

                embeddings.append(embedding)
                labels.append(label)

            except Exception as e:
                self.logger.warning(f"Failed to extract {item['file']}: {e}")
                continue

        return np.array(embeddings), np.array(labels)

    def train_classifier(self, train_embeddings, train_labels, val_embeddings, val_labels):
        """Train linear classifier on frozen embeddings."""
        self.logger.info("\nTraining linear classifier...")

        embedding_dim = train_embeddings.shape[1]

        # Simple linear classifier
        classifier = nn.Linear(embedding_dim, self.num_classes).to(self.device)

        # Loss with class weights
        criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

        # Convert to tensors
        train_X = torch.FloatTensor(train_embeddings).to(self.device)
        train_y = torch.LongTensor(train_labels).to(self.device)
        val_X = torch.FloatTensor(val_embeddings).to(self.device)
        val_y = torch.LongTensor(val_labels).to(self.device)

        # Training loop
        best_val_acc = 0
        best_state = classifier.state_dict()  # Initialize with initial state
        patience = 10
        patience_counter = 0
        max_epochs = 100

        for epoch in range(max_epochs):
            classifier.train()

            # Forward
            outputs = classifier(train_X)
            loss = criterion(outputs, train_y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            classifier.eval()
            with torch.no_grad():
                val_outputs = classifier(val_X)
                val_preds = val_outputs.argmax(dim=1)
                val_acc = (val_preds == val_y).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = classifier.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Val Acc={val_acc:.4f}")

            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        classifier.load_state_dict(best_state)
        self.logger.info(f"✓ Best validation accuracy: {best_val_acc:.4f}")

        return classifier

    def evaluate_classifier(self, classifier, embeddings, labels):
        """Evaluate classifier and compute all metrics."""
        classifier.eval()

        X = torch.FloatTensor(embeddings).to(self.device)
        y_true = labels

        with torch.no_grad():
            logits = classifier(X)
            y_pred = logits.argmax(dim=1).cpu().numpy()
            y_proba = torch.softmax(logits, dim=1).cpu().numpy()

        # Compute metrics
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

        # F1, Precision, Recall (macro & weighted)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        # ROC-AUC (one-vs-rest)
        try:
            metrics['roc_auc_macro'] = roc_auc_score(y_true, y_proba, average='macro', multi_class='ovr')
            metrics['roc_auc_weighted'] = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr')
        except:
            metrics['roc_auc_macro'] = 0.0
            metrics['roc_auc_weighted'] = 0.0

        # Top-k accuracy (for multi-class)
        if self.num_classes > 5:
            k = min(3, self.num_classes)
            try:
                metrics[f'top_{k}_accuracy'] = top_k_accuracy_score(
                    y_true, y_proba, k=k,
                    labels=list(range(self.num_classes))
                )
            except:
                metrics[f'top_{k}_accuracy'] = 0.0

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Per-class metrics
        per_class_report = classification_report(
            y_true, y_pred,
            labels=list(range(self.num_classes)),
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )

        return metrics, cm, per_class_report, y_pred, y_proba

    def save_results(self, train_metrics, val_metrics, test_metrics,
                    test_cm, test_per_class, test_preds, test_labels, test_proba):
        """Save all results to files."""

        # Summary results JSON
        results = {
            'model': self.model_name,
            'task': self.task,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'test_per_class': test_per_class
        }

        results_file = self.output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"✓ Results saved: {results_file}")

        # Save confusion matrix as PNG
        self._plot_confusion_matrix(test_cm, test_labels)

        # Save per-class metrics CSV
        self._save_per_class_csv(test_per_class)

        # Save per-class metrics chart
        self._plot_per_class_metrics(test_per_class)

        # Save ROC curves (if sufficient samples)
        self._plot_roc_curves(test_labels, test_proba)

        # Save summary report
        self._save_summary_report(train_metrics, val_metrics, test_metrics)

    def _plot_confusion_matrix(self, cm, test_labels):
        """Plot and save confusion matrix (300 DPI PNG)."""
        plt.figure(figsize=(12, 10))

        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Normalized Count'})

        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title(f'Confusion Matrix: {self.model_name} on {self.task}', fontsize=14)
        plt.tight_layout()

        cm_file = self.output_dir / "confusion_matrix.png"
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ Confusion matrix saved: {cm_file}")

    def _save_per_class_csv(self, per_class_report):
        """Save per-class metrics to CSV."""
        import pandas as pd

        rows = []
        for class_name in self.class_names:
            if class_name in per_class_report:
                rows.append({
                    'class': class_name,
                    'precision': per_class_report[class_name]['precision'],
                    'recall': per_class_report[class_name]['recall'],
                    'f1-score': per_class_report[class_name]['f1-score'],
                    'support': per_class_report[class_name]['support']
                })

        df = pd.DataFrame(rows)
        csv_file = self.output_dir / "per_class_metrics.csv"
        df.to_csv(csv_file, index=False)

        self.logger.info(f"✓ Per-class metrics saved: {csv_file}")

    def _save_summary_report(self, train_metrics, val_metrics, test_metrics):
        """Save human-readable summary report."""
        report_file = self.output_dir / "summary_report.txt"

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"ZERO-SHOT EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Task: {self.task}\n")
            f.write(f"Num Classes: {self.num_classes}\n\n")

            f.write("=" * 80 + "\n")
            f.write("TEST METRICS\n")
            f.write("=" * 80 + "\n\n")

            for metric, value in test_metrics.items():
                f.write(f"{metric:30s}: {value:.4f}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("VALIDATION METRICS\n")
            f.write("=" * 80 + "\n\n")

            for metric, value in val_metrics.items():
                f.write(f"{metric:30s}: {value:.4f}\n")

        self.logger.info(f"✓ Summary report saved: {report_file}")

    def _plot_per_class_metrics(self, per_class_report):
        """
        Plot per-class precision, recall, F1 as grouped bar chart.
        IEEE publication ready: 300 DPI PNG, colorblind-safe palette.
        """
        # Extract metrics
        classes = []
        precision = []
        recall = []
        f1 = []

        for class_name in self.class_names:
            if class_name in per_class_report and per_class_report[class_name]['support'] > 0:
                classes.append(class_name)
                precision.append(per_class_report[class_name]['precision'])
                recall.append(per_class_report[class_name]['recall'])
                f1.append(per_class_report[class_name]['f1-score'])

        if not classes:
            self.logger.warning("No per-class metrics to plot")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))

        x = np.arange(len(classes))
        width = 0.25

        # Colorblind-safe palette
        colors = ['#0173B2', '#DE8F05', '#029E73']  # Blue, Orange, Green

        bars1 = ax.bar(x - width, precision, width, label='Precision', color=colors[0])
        bars2 = ax.bar(x, recall, width, label='Recall', color=colors[1])
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color=colors[2])

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Per-Class Metrics: {self.model_name} on {self.task}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        metrics_file = self.output_dir / "per_class_metrics.png"
        plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ Per-class metrics chart saved: {metrics_file}")

    def _plot_roc_curves(self, test_labels, test_proba):
        """
        Plot ROC curves (one-vs-rest) for multi-class classification.
        Only plotted for tasks with sufficient samples (skip tiny hyrax_id).
        IEEE publication ready: 300 DPI PNG, colorblind-safe palette.
        """
        n_samples = len(test_labels)

        # Skip if too few samples
        if n_samples < 20:
            self.logger.info("✓ Skipping ROC curves (insufficient test samples)")
            return

        # Binarize labels
        y_test_bin = label_binarize(test_labels, classes=list(range(self.num_classes)))

        # Compute ROC curve for each class
        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(self.num_classes):
            # Skip classes with no positive samples
            if y_test_bin[:, i].sum() == 0:
                continue

            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], test_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        if not roc_auc:
            self.logger.warning("No ROC curves to plot (no positive samples)")
            return

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Colorblind-safe palette (extended)
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))

        for i in roc_auc.keys():
            ax.plot(fpr[i], tpr[i], color=colors[i],
                   label=f'{self.class_names[i]} (AUC={roc_auc[i]:.2f})',
                   linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Chance')

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves: {self.model_name} on {self.task}', fontsize=14)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        roc_file = self.output_dir / "roc_curves.png"
        plt.savefig(roc_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ ROC curves saved: {roc_file}")

    def run(self):
        """Run full zero-shot evaluation pipeline."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STARTING ZERO-SHOT EVALUATION")
        self.logger.info("=" * 80)

        # Extract embeddings
        train_emb, train_labels = self.extract_embeddings_for_split('train')
        val_emb, val_labels = self.extract_embeddings_for_split('val')
        test_emb, test_labels = self.extract_embeddings_for_split('test')

        self.logger.info(f"\nEmbedding shapes:")
        self.logger.info(f"  Train: {train_emb.shape}")
        self.logger.info(f"  Val: {val_emb.shape}")
        self.logger.info(f"  Test: {test_emb.shape}")

        # Train classifier
        classifier = self.train_classifier(train_emb, train_labels, val_emb, val_labels)

        # Evaluate on all splits
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("=" * 80)

        train_metrics, _, _, _, _ = self.evaluate_classifier(classifier, train_emb, train_labels)
        val_metrics, _, _, _, _ = self.evaluate_classifier(classifier, val_emb, val_labels)
        test_metrics, test_cm, test_per_class, test_preds, test_proba = self.evaluate_classifier(
            classifier, test_emb, test_labels
        )

        self.logger.info(f"\nTrain Accuracy: {train_metrics['accuracy']:.4f}")
        self.logger.info(f"Val Accuracy:   {val_metrics['accuracy']:.4f}")
        self.logger.info(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")

        # Save all results
        self.save_results(train_metrics, val_metrics, test_metrics,
                         test_cm, test_per_class, test_preds, test_labels, test_proba)

        self.logger.info("\n✓ Zero-shot evaluation complete!")

        return test_metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Phase 3 - Zero-Shot Evaluation")
    parser.add_argument("--model", required=True,
                       choices=["wav2vec2_base", "wav2vec2_base_960h", "hubert_base",
                               "xls_r", "wavlm", "ecapa_tdnn"])
    parser.add_argument("--task", required=True,
                       choices=["species_id", "hyrax_id", "hyrax_id_session_holdout"])
    parser.add_argument("--debug", action="store_true", help="Debug mode: small subset")
    args = parser.parse_args()

    # Setup logging
    log_dir = Path("outputs/phase3/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        f"Phase3_ZeroShot_{args.task}_{args.model}",
        log_file=str(log_dir / f"zero_shot_{args.task}_{args.model}.log")
    )

    # Paths - use new manifest naming
    manifest_path = Path(f"outputs/phase3/manifests/{args.task}.json")

    # Map task to output subfolder
    if args.task == "hyrax_id_session_holdout":
        output_dir = Path(f"outputs/phase3/zero_shot/hyrax_id/session_holdout/{args.model}")
    else:
        output_dir = Path(f"outputs/phase3/zero_shot/{args.task}/{args.model}")

    # Load config (minimal)
    config = {}

    # Determine actual task type for evaluator (hyrax_id or species_id)
    eval_task = "hyrax_id" if "hyrax_id" in args.task else args.task

    # Run evaluation
    evaluator = ZeroShotEvaluator(
        config, args.model, eval_task, manifest_path, output_dir, logger, debug=args.debug
    )

    evaluator.run()


if __name__ == "__main__":
    main()
