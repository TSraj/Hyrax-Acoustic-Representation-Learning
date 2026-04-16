"""
SVM Classifier Module

Support Vector Machine classifiers for evaluating acoustic representations.
Implements both Linear and RBF (Radial Basis Function) kernels with hyperparameter tuning.

SVMs are standard baselines in machine learning and provide:
- Linear SVM: Fast, interpretable, works well with high-dimensional data
- RBF SVM: Captures non-linear patterns, more powerful but slower

Author: Raj
Date: 2026-04-11
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, classification_report,
    confusion_matrix
)
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class SVMClassifier:
    """
    Support Vector Machine classifier with hyperparameter tuning.

    Supports both Linear and RBF kernels with automatic scaling
    and grid search for optimal hyperparameters.
    """

    def __init__(
        self,
        kernel: str = 'rbf',
        C_values: Optional[List[float]] = None,
        gamma_values: Optional[Union[List[float], List[str]]] = None,
        cv_folds: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 0
    ):
        """
        Initialize SVM classifier.

        Args:
            kernel: SVM kernel type ('linear' or 'rbf')
            C_values: Regularization parameter values to search
            gamma_values: Kernel coefficient values to search (for RBF)
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 = all cores)
            verbose: Verbosity level for sklearn
        """
        self.kernel = kernel
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Default C values if not provided
        if C_values is None:
            self.C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        else:
            self.C_values = C_values

        # Default gamma values for RBF kernel
        if kernel == 'rbf':
            if gamma_values is None:
                self.gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
            else:
                self.gamma_values = gamma_values
        else:
            self.gamma_values = None

        # Will be set during training
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = None
        self.best_score = None

        logger.info(f"Initialized SVM classifier with {kernel} kernel")

    def _create_param_grid(self) -> Dict:
        """Create parameter grid for grid search."""
        if self.kernel == 'linear':
            param_grid = {
                'C': self.C_values,
                'kernel': ['linear']
            }
        elif self.kernel == 'rbf':
            param_grid = {
                'C': self.C_values,
                'gamma': self.gamma_values,
                'kernel': ['rbf']
            }
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

        return param_grid

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        perform_grid_search: bool = True
    ) -> Dict:
        """
        Train SVM classifier with optional grid search.

        Args:
            X_train: Training features of shape (n_samples, n_features)
            y_train: Training labels of shape (n_samples,)
            perform_grid_search: Whether to perform grid search for hyperparameters

        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {self.kernel} SVM on {X_train.shape[0]} samples...")

        # Scale features (important for SVM)
        X_train_scaled = self.scaler.fit_transform(X_train)

        if perform_grid_search:
            # Grid search for best hyperparameters
            param_grid = self._create_param_grid()

            cv = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )

            grid_search = GridSearchCV(
                estimator=SVC(random_state=self.random_state, class_weight='balanced'),
                param_grid=param_grid,
                cv=cv,
                scoring='balanced_accuracy',
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                refit=True
            )

            logger.info(f"Performing grid search over {len(param_grid['C'])} C values...")
            if self.kernel == 'rbf':
                logger.info(f"  and {len(param_grid['gamma'])} gamma values...")

            grid_search.fit(X_train_scaled, y_train)

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_

            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV score: {self.best_score:.4f}")

        else:
            # Train with default parameters
            if self.kernel == 'linear':
                self.model = SVC(
                    kernel='linear',
                    C=1.0,
                    random_state=self.random_state,
                    class_weight='balanced'
                )
            else:  # rbf
                self.model = SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    random_state=self.random_state,
                    class_weight='balanced'
                )

            self.model.fit(X_train_scaled, y_train)
            self.best_params = self.model.get_params()

        # Return training info
        return {
            'best_params': self.best_params,
            'best_cv_score': self.best_score if perform_grid_search else None
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features of shape (n_samples, n_features)

        Returns:
            Predicted labels of shape (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Scale features using fitted scaler
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled)

        return predictions

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate model on test data.

        Args:
            X_test: Test features of shape (n_samples, n_features)
            y_test: Test labels of shape (n_samples,)

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {self.kernel} SVM on {X_test.shape[0]} samples...")

        # Make predictions
        y_pred = self.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

        # Per-class F1 scores
        per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Classification report
        class_report = classification_report(y_test, y_pred, zero_division=0)

        results = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'per_class_f1': per_class_f1,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'best_params': self.best_params
        }

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
        logger.info(f"Macro F1: {macro_f1:.4f}")

        return results

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        perform_grid_search: bool = True
    ) -> Dict:
        """
        Train and evaluate in one step.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            perform_grid_search: Whether to perform grid search

        Returns:
            Dictionary with training and evaluation results
        """
        # Train
        train_results = self.train(X_train, y_train, perform_grid_search)

        # Evaluate
        eval_results = self.evaluate(X_test, y_test)

        # Combine results
        results = {**train_results, **eval_results}

        return results

    def __repr__(self) -> str:
        return f"SVMClassifier(kernel={self.kernel}, cv_folds={self.cv_folds})"


def create_linear_svm(
    C_values: Optional[List[float]] = None,
    cv_folds: int = 5,
    random_state: int = 42
) -> SVMClassifier:
    """
    Create Linear SVM classifier with default configuration.

    Args:
        C_values: Regularization parameter values to search
        cv_folds: Number of cross-validation folds
        random_state: Random seed

    Returns:
        Configured SVMClassifier instance with linear kernel
    """
    return SVMClassifier(
        kernel='linear',
        C_values=C_values,
        cv_folds=cv_folds,
        random_state=random_state
    )


def create_rbf_svm(
    C_values: Optional[List[float]] = None,
    gamma_values: Optional[List] = None,
    cv_folds: int = 5,
    random_state: int = 42
) -> SVMClassifier:
    """
    Create RBF SVM classifier with default configuration.

    Args:
        C_values: Regularization parameter values to search
        gamma_values: Kernel coefficient values to search
        cv_folds: Number of cross-validation folds
        random_state: Random seed

    Returns:
        Configured SVMClassifier instance with RBF kernel
    """
    return SVMClassifier(
        kernel='rbf',
        C_values=C_values,
        gamma_values=gamma_values,
        cv_folds=cv_folds,
        random_state=random_state
    )
