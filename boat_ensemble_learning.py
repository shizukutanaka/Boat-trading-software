#!/usr/bin/env python3
"""
Ensemble Learning and Model Stacking for Trading
==================================================

Advanced ensemble methods for combining multiple models:
  - Model stacking with meta-learners
  - Weighted ensemble averaging
  - Boosting and bagging frameworks
  - Cross-validation strategy selection
  - Ensemble diversity metrics
  - Out-of-fold predictions

Based on 2025 research on ensemble methods and meta-learning in finance.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Ensemble prediction result"""
    predictions: np.ndarray
    confidence: float
    constituent_predictions: Dict[str, np.ndarray]
    weights: Dict[str, float]
    diversity_score: float


class ModelStacking:
    """Stack multiple models with meta-learner"""

    def __init__(self, base_models: Dict[str, Callable], meta_model_type: str = 'linear'):
        """
        Initialize stacking ensemble

        Args:
            base_models: Dictionary of base model prediction functions
            meta_model_type: Type of meta-model ('linear', 'tree', 'neural')
        """
        self.base_models = base_models
        self.meta_model_type = meta_model_type
        self.meta_weights = {}
        self.scaler_mean = {}
        self.scaler_std = {}

    def fit_stacking(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Fit stacking ensemble using cross-validation

        Args:
            X_train: Training features (N, D)
            y_train: Training targets (N,)
            cv_folds: Number of cross-validation folds

        Returns:
            Meta-model weights
        """
        N = len(X_train)
        fold_size = N // cv_folds

        # Generate out-of-fold predictions
        meta_features = np.zeros((N, len(self.base_models)))

        for fold in range(cv_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < cv_folds - 1 else N

            train_idx = np.concatenate([
                np.arange(0, start_idx),
                np.arange(end_idx, N)
            ])
            val_idx = np.arange(start_idx, end_idx)

            X_train_fold = X_train[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]

            # Collect base model predictions
            for i, (model_name, model_func) in enumerate(self.base_models.items()):
                meta_features[val_idx, i] = model_func(X_val_fold)

        # Normalize meta-features
        self.scaler_mean = np.mean(meta_features, axis=0)
        self.scaler_std = np.std(meta_features, axis=0) + 1e-8
        meta_features_norm = (meta_features - self.scaler_mean) / self.scaler_std

        # Fit meta-model via linear regression
        X_meta = np.column_stack([np.ones(N), meta_features_norm])
        beta = np.linalg.lstsq(X_meta, y_train, rcond=None)[0]

        self.meta_weights['intercept'] = float(beta[0])
        for i, model_name in enumerate(self.base_models.keys()):
            self.meta_weights[model_name] = float(beta[i + 1])

        return self.meta_weights

    def predict_stacking(self, X_test: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict using stacking ensemble

        Args:
            X_test: Test features

        Returns:
            (predictions, confidence)
        """
        n_models = len(self.base_models)
        meta_features = np.zeros((len(X_test), n_models))

        # Collect base model predictions
        for i, (model_name, model_func) in enumerate(self.base_models.items()):
            meta_features[:, i] = model_func(X_test)

        # Normalize using training statistics
        meta_features_norm = (meta_features - self.scaler_mean) / self.scaler_std

        # Apply meta-model
        predictions = self.meta_weights['intercept'] + np.sum(
            np.column_stack([
                meta_features_norm * self.meta_weights[name]
                for name in self.base_models.keys()
            ]),
            axis=1
        )

        # Confidence: variance of base predictions
        confidence = 1.0 / (1.0 + np.mean(np.std(meta_features, axis=1)))

        return predictions, confidence


class WeightedEnsemble:
    """Weighted averaging of multiple models"""

    def __init__(self, base_models: Dict[str, Callable]):
        """Initialize weighted ensemble"""
        self.base_models = base_models
        self.weights = {name: 1.0 / len(base_models) for name in base_models}

    def fit_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        method: str = 'mse'
    ) -> Dict[str, float]:
        """
        Learn ensemble weights via validation performance

        Args:
            X_val: Validation features
            y_val: Validation targets
            method: Optimization criterion ('mse', 'correlation', 'sharpe')

        Returns:
            Optimized weights
        """
        n_models = len(self.base_models)
        predictions = {}

        # Get predictions from each model
        for model_name, model_func in self.base_models.items():
            predictions[model_name] = model_func(X_val)

        # Optimize weights
        best_score = np.inf
        best_weights = np.ones(n_models) / n_models

        for _ in range(100):
            # Random weight search
            weights = np.random.dirichlet(np.ones(n_models))

            # Ensemble prediction
            ensemble_pred = np.zeros_like(y_val)
            for i, (model_name, weight) in enumerate(zip(self.base_models.keys(), weights)):
                ensemble_pred += weight * predictions[model_name]

            # Score
            if method == 'mse':
                score = np.mean((ensemble_pred - y_val) ** 2)
            elif method == 'correlation':
                score = -np.corrcoef(ensemble_pred, y_val)[0, 1]
            else:  # sharpe
                score = -np.mean(ensemble_pred - y_val) / (np.std(ensemble_pred - y_val) + 1e-8)

            if score < best_score:
                best_score = score
                best_weights = weights

        # Update weights
        for model_name, weight in zip(self.base_models.keys(), best_weights):
            self.weights[model_name] = float(weight)

        return self.weights

    def predict_weighted(self, X_test: np.ndarray) -> np.ndarray:
        """Predict using weighted ensemble"""
        ensemble_pred = np.zeros(len(X_test))

        for model_name, model_func in self.base_models.items():
            weight = self.weights[model_name]
            ensemble_pred += weight * model_func(X_test)

        return ensemble_pred


class BaggingEnsemble:
    """Bootstrap aggregating ensemble"""

    def __init__(self, base_model_func: Callable, n_estimators: int = 10):
        """
        Initialize bagging ensemble

        Args:
            base_model_func: Base model training function
            n_estimators: Number of bootstrap samples
        """
        self.base_model_func = base_model_func
        self.n_estimators = n_estimators
        self.models = []

    def fit_bagging(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """
        Fit bagging ensemble

        Args:
            X_train: Training features
            y_train: Training targets
        """
        N = len(X_train)

        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(N, size=N, replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]

            # Train model
            model = self.base_model_func(X_boot, y_boot)
            self.models.append(model)

    def predict_bagging(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using bagging ensemble

        Args:
            X_test: Test features

        Returns:
            (mean_predictions, uncertainty)
        """
        predictions = np.zeros((len(X_test), self.n_estimators))

        for i, model in enumerate(self.models):
            predictions[:, i] = model(X_test)

        mean_pred = np.mean(predictions, axis=1)
        uncertainty = np.std(predictions, axis=1)

        return mean_pred, uncertainty


class AdaBoostEnsemble:
    """Adaptive boosting ensemble"""

    def __init__(self, base_model_func: Callable, n_estimators: int = 10, learning_rate: float = 0.1):
        """Initialize AdaBoost"""
        self.base_model_func = base_model_func
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.weights = []

    def fit_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """
        Fit AdaBoost ensemble

        Args:
            X_train: Training features
            y_train: Training targets
        """
        N = len(X_train)
        sample_weights = np.ones(N) / N

        for _ in range(self.n_estimators):
            # Weighted training
            # Adjust training probability by sample weights
            indices = np.random.choice(N, size=N, replace=True, p=sample_weights)
            X_weighted = X_train[indices]
            y_weighted = y_train[indices]

            # Train model
            model = self.base_model_func(X_weighted, y_weighted)
            predictions = model(X_train)

            # Calculate weighted error
            errors = (predictions - y_train) ** 2
            weighted_error = np.sum(sample_weights * errors)

            # Model weight
            model_weight = self.learning_rate / (weighted_error + 1e-8)
            self.models.append(model)
            self.weights.append(model_weight)

            # Update sample weights
            sample_weights *= np.exp(-model_weight * errors)
            sample_weights /= np.sum(sample_weights)

    def predict_boosting(self, X_test: np.ndarray) -> np.ndarray:
        """Predict using AdaBoost ensemble"""
        ensemble_pred = np.zeros(len(X_test))

        total_weight = sum(self.weights)

        for model, weight in zip(self.models, self.weights):
            ensemble_pred += (weight / total_weight) * model(X_test)

        return ensemble_pred


class EnsembleDiversityMetrics:
    """Calculate diversity metrics for ensemble"""

    @staticmethod
    def calculate_diversity(
        predictions_list: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate ensemble diversity

        Args:
            predictions_list: List of model predictions

        Returns:
            Diversity metrics
        """
        predictions_array = np.array(predictions_list)
        n_models, n_samples = predictions_array.shape

        # Pairwise correlation
        correlations = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                corr = np.corrcoef(predictions_array[i], predictions_array[j])[0, 1]
                correlations.append(corr)

        # Disagreement
        mean_pred = np.mean(predictions_array, axis=0)
        disagreement = np.mean(np.std(predictions_array, axis=0))

        # Entropy of predictions
        entropy = np.zeros(n_samples)
        for s in range(n_samples):
            preds = predictions_array[:, s]
            normalized = (preds - np.min(preds)) / (np.max(preds) - np.min(preds) + 1e-8)
            entropy[s] = -np.sum(normalized * np.log(normalized + 1e-8))

        return {
            'mean_correlation': float(np.mean(correlations)),
            'std_correlation': float(np.std(correlations)),
            'disagreement': float(disagreement),
            'mean_entropy': float(np.mean(entropy)),
            'diversity_score': float(1.0 - np.abs(np.mean(correlations)))
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate sample data
    X_train = np.random.randn(200, 10)
    y_train = X_train[:, 0] + 0.5 * X_train[:, 1] + np.random.randn(200) * 0.1

    X_test = np.random.randn(50, 10)
    y_test = X_test[:, 0] + 0.5 * X_test[:, 1] + np.random.randn(50) * 0.1

    # Base models
    def model1(X):
        return X[:, 0]

    def model2(X):
        return 0.5 * X[:, 1]

    def model3(X):
        return X[:, 0] + X[:, 1]

    base_models = {
        'model1': model1,
        'model2': model2,
        'model3': model3
    }

    # Stacking
    stacker = ModelStacking(base_models)
    stacker.fit_stacking(X_train, y_train, cv_folds=5)
    stack_pred, stack_conf = stacker.predict_stacking(X_test)

    logger.info("Stacking Results:")
    logger.info(f"Meta-weights: {stacker.meta_weights}")
    logger.info(f"Confidence: {stack_conf:.4f}")
    logger.info(f"MSE: {np.mean((stack_pred - y_test) ** 2):.6f}")

    # Weighted ensemble
    weighted = WeightedEnsemble(base_models)
    weighted.fit_weights(X_train, y_train, method='mse')
    weighted_pred = weighted.predict_weighted(X_test)

    logger.info("\nWeighted Ensemble Results:")
    logger.info(f"Weights: {weighted.weights}")
    logger.info(f"MSE: {np.mean((weighted_pred - y_test) ** 2):.6f}")

    # Diversity metrics
    predictions = [model1(X_test), model2(X_test), model3(X_test)]
    diversity = EnsembleDiversityMetrics.calculate_diversity(predictions)

    logger.info("\nDiversity Metrics:")
    for metric, value in diversity.items():
        logger.info(f"  {metric}: {value:.4f}")
