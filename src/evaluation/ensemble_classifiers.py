"""
Ensemble Classifier Module

Tree-based ensemble classifiers for evaluating acoustic representations.
Implements Random Forest and XGBoost with hyperparameter tuning.

These classifiers complement linear and SVM baselines by:
- Capturing non-linear feature interactions
- Providing feature importance analysis
- Handling complex decision boundaries
- Being robust to feature scaling

Author: Raj
Date: 2026-04-11
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, classification_report,
    confusion_matrix
)
from typing import Dict, List, Tuple, Optional
import logging

# XGBoost is optional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not installed. Install with: pip install xgboost")

logger = logging.getLogger(__name__)


class RandomForestEvaluator:
    """
    Random Forest classifier with hyperparameter tuning.

    Random Forest is an ensemble of decision trees that:
    - Reduces overfitting through bagging
    - Provides feature importance scores
    - Works well with high-dimensional data
    - Requires no feature scaling
    """

    def __init__(
        self,
        n_estimators_values: Optional[List[int]] = None,
        max_depth_values: Optional[List[Optional[int]]] = None,
        min_samples_split_values: Optional[List[int]] = None,
        cv_folds: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 0
    ):
        """
        Initialize Random Forest classifier.

        Args:
            n_estimators_values: Number of trees to test
            max_depth_values: Maximum tree depth values to test
            min_samples_split_values: Minimum samples to split values to test
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 = all cores)
            verbose: Verbosity level for sklearn
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Default hyperparameter values
        if n_estimators_values is None:
            self.n_estimators_values = [50, 100, 200]
        else:
            self.n_estimators_values = n_estimators_values

        if max_depth_values is None:
            self.max_depth_values = [10, 20, None]  # None = unlimited
        else:
            self.max_depth_values = max_depth_values

        if min_samples_split_values is None:
            self.min_samples_split_values = [2, 5, 10]
        else:
            self.min_samples_split_values = min_samples_split_values

        # Will be set during training
        self.model = None
        self.best_params = None
        self.best_score = None
        self.feature_importances = None

        logger.info("Initialized Random Forest classifier")

    def _create_param_grid(self) -> Dict:
        """Create parameter grid for grid search."""
        param_grid = {
            'n_estimators': self.n_estimators_values,
            'max_depth': self.max_depth_values,
            'min_samples_split': self.min_samples_split_values
        }
        return param_grid

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        perform_grid_search: bool = True
    ) -> Dict:
        """
        Train Random Forest classifier.

        Args:
            X_train: Training features of shape (n_samples, n_features)
            y_train: Training labels of shape (n_samples,)
            perform_grid_search: Whether to perform grid search

        Returns:
            Dictionary with training results
        """
        logger.info(f"Training Random Forest on {X_train.shape[0]} samples...")

        if perform_grid_search:
            # Grid search for best hyperparameters
            param_grid = self._create_param_grid()

            cv = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )

            grid_search = GridSearchCV(
                estimator=RandomForestClassifier(
                    random_state=self.random_state,
                    class_weight='balanced',
                    n_jobs=self.n_jobs
                ),
                param_grid=param_grid,
                cv=cv,
                scoring='balanced_accuracy',
                n_jobs=1,  # RF already uses n_jobs
                verbose=self.verbose,
                refit=True
            )

            logger.info(f"Performing grid search over parameter combinations...")

            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_

            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV score: {self.best_score:.4f}")

        else:
            # Train with default parameters
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=self.n_jobs
            )

            self.model.fit(X_train, y_train)
            self.best_params = self.model.get_params()

        # Store feature importances
        self.feature_importances = self.model.feature_importances_

        return {
            'best_params': self.best_params,
            'best_cv_score': self.best_score if perform_grid_search else None,
            'feature_importances': self.feature_importances
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """Evaluate model on test data."""
        logger.info(f"Evaluating Random Forest on {X_test.shape[0]} samples...")

        y_pred = self.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
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
            'best_params': self.best_params,
            'feature_importances': self.feature_importances
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
        """Train and evaluate in one step."""
        train_results = self.train(X_train, y_train, perform_grid_search)
        eval_results = self.evaluate(X_test, y_test)
        return {**train_results, **eval_results}

    def __repr__(self) -> str:
        return f"RandomForestEvaluator(cv_folds={self.cv_folds})"


class XGBoostEvaluator:
    """
    XGBoost classifier with hyperparameter tuning.

    XGBoost is a gradient boosting framework that:
    - Often achieves state-of-the-art performance
    - Handles imbalanced data well
    - Provides feature importance
    - Supports GPU acceleration
    """

    def __init__(
        self,
        n_estimators_values: Optional[List[int]] = None,
        max_depth_values: Optional[List[int]] = None,
        learning_rate_values: Optional[List[float]] = None,
        cv_folds: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 0
    ):
        """Initialize XGBoost classifier."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Default hyperparameter values
        if n_estimators_values is None:
            self.n_estimators_values = [50, 100, 200]
        else:
            self.n_estimators_values = n_estimators_values

        if max_depth_values is None:
            self.max_depth_values = [3, 5, 7]
        else:
            self.max_depth_values = max_depth_values

        if learning_rate_values is None:
            self.learning_rate_values = [0.01, 0.1, 0.3]
        else:
            self.learning_rate_values = learning_rate_values

        # Will be set during training
        self.model = None
        self.best_params = None
        self.best_score = None
        self.feature_importances = None

        logger.info("Initialized XGBoost classifier")

    def _create_param_grid(self) -> Dict:
        """Create parameter grid for grid search."""
        param_grid = {
            'n_estimators': self.n_estimators_values,
            'max_depth': self.max_depth_values,
            'learning_rate': self.learning_rate_values
        }
        return param_grid

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        perform_grid_search: bool = True
    ) -> Dict:
        """Train XGBoost classifier."""
        logger.info(f"Training XGBoost on {X_train.shape[0]} samples...")

        # Encode labels if they're strings
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        self.label_encoder = le

        if perform_grid_search:
            param_grid = self._create_param_grid()

            cv = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )

            grid_search = GridSearchCV(
                estimator=xgb.XGBClassifier(
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    eval_metric='logloss'
                ),
                param_grid=param_grid,
                cv=cv,
                scoring='balanced_accuracy',
                n_jobs=1,  # XGB already uses n_jobs
                verbose=self.verbose,
                refit=True
            )

            logger.info(f"Performing grid search over parameter combinations...")

            grid_search.fit(X_train, y_train_encoded)

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_

            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV score: {self.best_score:.4f}")

        else:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                eval_metric='logloss'
            )

            self.model.fit(X_train, y_train_encoded)
            self.best_params = self.model.get_params()

        # Store feature importances
        self.feature_importances = self.model.feature_importances_

        return {
            'best_params': self.best_params,
            'best_cv_score': self.best_score if perform_grid_search else None,
            'feature_importances': self.feature_importances
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """Evaluate model on test data."""
        logger.info(f"Evaluating XGBoost on {X_test.shape[0]} samples...")

        y_pred = self.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
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
            'best_params': self.best_params,
            'feature_importances': self.feature_importances
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
        """Train and evaluate in one step."""
        train_results = self.train(X_train, y_train, perform_grid_search)
        eval_results = self.evaluate(X_test, y_test)
        return {**train_results, **eval_results}

    def __repr__(self) -> str:
        return f"XGBoostEvaluator(cv_folds={self.cv_folds})"


def create_random_forest(
    n_estimators_values: Optional[List[int]] = None,
    cv_folds: int = 5,
    random_state: int = 42
) -> RandomForestEvaluator:
    """Create Random Forest classifier with default configuration."""
    return RandomForestEvaluator(
        n_estimators_values=n_estimators_values,
        cv_folds=cv_folds,
        random_state=random_state
    )


def create_xgboost(
    n_estimators_values: Optional[List[int]] = None,
    cv_folds: int = 5,
    random_state: int = 42
) -> XGBoostEvaluator:
    """Create XGBoost classifier with default configuration."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")

    return XGBoostEvaluator(
        n_estimators_values=n_estimators_values,
        cv_folds=cv_folds,
        random_state=random_state
    )
