"""
Scaling Evaluator Module

Evaluates how classifier performance scales with task difficulty
by varying the number of individuals (classes) in the classification problem.

This module answers the question:
"How robust is the representation when the recognition task becomes harder?"

Typical scaling analysis:
- 2 individuals (binary classification)
- 3 individuals
- 5 individuals
- 8 individuals
- All individuals

Author: Raj
Date: 2026-04-11
"""

import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Callable
import logging
from itertools import combinations
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ScalingEvaluator:
    """
    Evaluate classifier performance across different numbers of individuals.

    This class:
    1. Creates subsets with varying numbers of individuals
    2. Evaluates classifiers on each subset
    3. Generates scaling curves showing performance vs. task difficulty
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        n_trials: int = 5
    ):
        """
        Initialize scaling evaluator.

        Args:
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            n_trials: Number of random trials per configuration (for robustness)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.n_trials = n_trials

        logger.info(f"Initialized scaling evaluator with {n_trials} trials per config")

    def _create_subset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        individuals: List[str],
        selected_individuals: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create subset with only selected individuals.

        Args:
            features: All features
            labels: All labels
            individuals: All individual IDs
            selected_individuals: Individuals to include in subset

        Returns:
            subset_features: Features for selected individuals
            subset_labels: Labels for selected individuals
        """
        # Find indices for selected individuals
        mask = np.isin(labels, selected_individuals)

        subset_features = features[mask]
        subset_labels = labels[mask]

        return subset_features, subset_labels

    def evaluate_scaling(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        classifier_fn: Callable,
        num_individuals_list: List[int],
        individual_selection: str = 'random'
    ) -> Dict:
        """
        Evaluate classifier performance across different numbers of individuals.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            classifier_fn: Function that creates and returns a classifier instance
                          Should have train_and_evaluate(X_train, y_train, X_test, y_test)
            num_individuals_list: List of numbers of individuals to test (e.g., [2, 3, 5, 8])
            individual_selection: How to select individuals ('random', 'most_samples', 'least_samples')

        Returns:
            Dictionary with scaling results
        """
        logger.info(f"Starting scaling evaluation for {num_individuals_list}")

        # Get unique individuals
        unique_individuals = np.unique(labels)
        n_total_individuals = len(unique_individuals)

        # Sort individuals by number of samples
        individual_counts = {ind: np.sum(labels == ind) for ind in unique_individuals}
        sorted_individuals = sorted(individual_counts.keys(), key=lambda x: individual_counts[x], reverse=True)

        results = {
            'num_individuals': [],
            'mean_accuracy': [],
            'std_accuracy': [],
            'mean_balanced_accuracy': [],
            'std_balanced_accuracy': [],
            'mean_macro_f1': [],
            'std_macro_f1': [],
            'trial_details': []
        }

        for n_individuals in num_individuals_list:
            if n_individuals > n_total_individuals:
                logger.warning(f"Skipping n={n_individuals} (only {n_total_individuals} available)")
                continue

            logger.info(f"\nEvaluating with {n_individuals} individuals...")

            trial_accuracies = []
            trial_balanced_accs = []
            trial_macro_f1s = []
            trial_configs = []

            for trial in range(self.n_trials):
                # Select individuals for this trial
                if individual_selection == 'random':
                    # Random selection
                    rng = np.random.RandomState(self.random_state + trial)
                    selected = rng.choice(unique_individuals, size=n_individuals, replace=False)

                elif individual_selection == 'most_samples':
                    # Select individuals with most samples (deterministic)
                    selected = sorted_individuals[:n_individuals]

                elif individual_selection == 'least_samples':
                    # Select individuals with least samples (deterministic)
                    selected = sorted_individuals[-n_individuals:]

                else:
                    raise ValueError(f"Unknown selection method: {individual_selection}")

                # Create subset
                subset_features, subset_labels = self._create_subset(
                    features, labels, unique_individuals, selected
                )

                # Split into train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    subset_features,
                    subset_labels,
                    test_size=self.test_size,
                    stratify=subset_labels,
                    random_state=self.random_state + trial
                )

                # Create classifier instance
                classifier = classifier_fn()

                # Train and evaluate
                try:
                    eval_results = classifier.train_and_evaluate(
                        X_train, y_train, X_test, y_test,
                        perform_grid_search=False  # Faster for scaling experiments
                    )

                    trial_accuracies.append(eval_results['accuracy'])
                    trial_balanced_accs.append(eval_results['balanced_accuracy'])
                    trial_macro_f1s.append(eval_results['macro_f1'])

                    trial_configs.append({
                        'trial': trial,
                        'selected_individuals': list(selected),
                        'n_train_samples': len(y_train),
                        'n_test_samples': len(y_test),
                        'accuracy': eval_results['accuracy'],
                        'balanced_accuracy': eval_results['balanced_accuracy'],
                        'macro_f1': eval_results['macro_f1']
                    })

                    logger.info(f"  Trial {trial+1}/{self.n_trials}: "
                               f"Acc={eval_results['accuracy']:.4f}, "
                               f"Bal_Acc={eval_results['balanced_accuracy']:.4f}, "
                               f"F1={eval_results['macro_f1']:.4f}")

                except Exception as e:
                    logger.warning(f"  Trial {trial+1} failed: {e}")
                    continue

            # Aggregate results
            if len(trial_accuracies) > 0:
                results['num_individuals'].append(n_individuals)
                results['mean_accuracy'].append(np.mean(trial_accuracies))
                results['std_accuracy'].append(np.std(trial_accuracies))
                results['mean_balanced_accuracy'].append(np.mean(trial_balanced_accs))
                results['std_balanced_accuracy'].append(np.std(trial_balanced_accs))
                results['mean_macro_f1'].append(np.mean(trial_macro_f1s))
                results['std_macro_f1'].append(np.std(trial_macro_f1s))
                results['trial_details'].append(trial_configs)

                logger.info(f"  → Mean Accuracy: {np.mean(trial_accuracies):.4f} ± {np.std(trial_accuracies):.4f}")
                logger.info(f"  → Mean Balanced Acc: {np.mean(trial_balanced_accs):.4f} ± {np.std(trial_balanced_accs):.4f}")
                logger.info(f"  → Mean Macro F1: {np.mean(trial_macro_f1s):.4f} ± {np.std(trial_macro_f1s):.4f}")

        return results

    def plot_scaling_curves(
        self,
        scaling_results: Dict,
        title: str = "Performance vs. Number of Individuals",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4)
    ) -> None:
        """
        Plot scaling curves showing performance vs. number of individuals.

        Args:
            scaling_results: Results from evaluate_scaling()
            title: Plot title
            save_path: Path to save figure (if None, show only)
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        num_individuals = scaling_results['num_individuals']

        # Accuracy
        axes[0].errorbar(
            num_individuals,
            scaling_results['mean_accuracy'],
            yerr=scaling_results['std_accuracy'],
            marker='o',
            capsize=5,
            label='Accuracy'
        )
        axes[0].set_xlabel('Number of Individuals')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy vs. Task Difficulty')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.05])

        # Balanced Accuracy
        axes[1].errorbar(
            num_individuals,
            scaling_results['mean_balanced_accuracy'],
            yerr=scaling_results['std_balanced_accuracy'],
            marker='s',
            capsize=5,
            color='orange',
            label='Balanced Accuracy'
        )
        axes[1].set_xlabel('Number of Individuals')
        axes[1].set_ylabel('Balanced Accuracy')
        axes[1].set_title('Balanced Accuracy vs. Task Difficulty')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])

        # Macro F1
        axes[2].errorbar(
            num_individuals,
            scaling_results['mean_macro_f1'],
            yerr=scaling_results['std_macro_f1'],
            marker='^',
            capsize=5,
            color='green',
            label='Macro F1'
        )
        axes[2].set_xlabel('Number of Individuals')
        axes[2].set_ylabel('Macro F1')
        axes[2].set_title('Macro F1 vs. Task Difficulty')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1.05])

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved scaling plot to: {save_path}")

        plt.close()

    def compare_multiple_scalings(
        self,
        features_dict: Dict[str, np.ndarray],
        labels: np.ndarray,
        classifier_fn_dict: Dict[str, Callable],
        num_individuals_list: List[int],
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Compare scaling behavior across multiple feature types and classifiers.

        Args:
            features_dict: Dictionary mapping feature_name -> features
            labels: Labels
            classifier_fn_dict: Dictionary mapping classifier_name -> classifier_fn
            num_individuals_list: List of numbers of individuals to test
            save_dir: Directory to save plots

        Returns:
            Dictionary with all scaling results
        """
        all_results = {}

        for feat_name, features in features_dict.items():
            for clf_name, clf_fn in classifier_fn_dict.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"Scaling: {feat_name} + {clf_name}")
                logger.info(f"{'='*60}")

                key = f"{feat_name}_{clf_name}"

                results = self.evaluate_scaling(
                    features=features,
                    labels=labels,
                    classifier_fn=clf_fn,
                    num_individuals_list=num_individuals_list
                )

                all_results[key] = results

                # Plot
                if save_dir:
                    from pathlib import Path
                    save_path = Path(save_dir) / f"scaling_{key}.png"
                    self.plot_scaling_curves(
                        results,
                        title=f"Scaling: {feat_name} + {clf_name}",
                        save_path=str(save_path)
                    )

        return all_results

    def __repr__(self) -> str:
        return f"ScalingEvaluator(n_trials={self.n_trials}, test_size={self.test_size})"


def create_default_scaling_evaluator(
    n_trials: int = 5,
    random_state: int = 42
) -> ScalingEvaluator:
    """Create scaling evaluator with default configuration."""
    return ScalingEvaluator(
        test_size=0.2,
        random_state=random_state,
        n_trials=n_trials
    )
