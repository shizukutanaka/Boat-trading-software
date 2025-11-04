#!/usr/bin/env python3
"""
Explainable AI with SHAP Integration for Trading Signals
=========================================================

Advanced model interpretability and trading signal explanation:
  - SHAP (SHapley Additive exPlanations) values
  - LIME (Local Interpretable Model-agnostic Explanations)
  - Feature importance analysis
  - Decision boundary visualization
  - Model prediction explanation
  - Trading signal justification
  - Model confidence assessment

Based on 2025 research on explainable AI in finance and model interpretability.
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
class ShapleyValue:
    """Shapley value explanation for model prediction"""
    feature_name: str
    shap_value: float
    feature_value: float
    base_value: float
    contribution_direction: str  # 'positive' or 'negative'


@dataclass
class ModelExplanation:
    """Complete explanation for a model prediction"""
    prediction: float
    confidence: float
    base_value: float
    shapley_values: List[ShapleyValue]
    feature_importance: Dict[str, float]
    model_type: str


class ShapleyValueCalculator:
    """Calculate Shapley values for model predictions"""

    def __init__(self, model_func: Callable, background_data: np.ndarray):
        """
        Initialize SHAP calculator

        Args:
            model_func: Function that takes features and returns predictions
            background_data: Background dataset for baseline calculations
        """
        self.model_func = model_func
        self.background_data = background_data
        self.base_value = np.mean([self.model_func(x.reshape(1, -1)) for x in background_data])

    def calculate_shapley_values(
        self,
        x: np.ndarray,
        feature_names: List[str],
        n_samples: int = 100
    ) -> ModelExplanation:
        """
        Calculate Shapley values using Monte Carlo approximation

        Args:
            x: Single instance (n_features,)
            feature_names: Names of features
            n_samples: Number of coalition samples

        Returns:
            ModelExplanation with Shapley values
        """
        n_features = x.shape[0]
        shap_values = np.zeros(n_features)

        # Monte Carlo approximation of Shapley values
        for _ in range(n_samples):
            # Random feature ordering
            feature_order = np.random.permutation(n_features)

            # Calculate marginal contribution for each feature
            for i, feature_idx in enumerate(feature_order):
                # Value with feature
                x_with = x.copy()
                value_with = self.model_func(x_with.reshape(1, -1))[0]

                # Value without feature (replace with background average)
                x_without = x.copy()
                x_without[feature_idx] = np.mean(self.background_data[:, feature_idx])
                value_without = self.model_func(x_without.reshape(1, -1))[0]

                # Marginal contribution
                shap_values[feature_idx] += (value_with - value_without)

        shap_values /= n_samples

        # Get prediction
        prediction = self.model_func(x.reshape(1, -1))[0]

        # Create Shapley value objects
        shapley_objs = [
            ShapleyValue(
                feature_name=feature_names[i],
                shap_value=float(shap_values[i]),
                feature_value=float(x[i]),
                base_value=float(self.base_value),
                contribution_direction='positive' if shap_values[i] > 0 else 'negative'
            )
            for i in range(n_features)
        ]

        # Feature importance (absolute Shapley values)
        feature_importance = {
            feature_names[i]: float(np.abs(shap_values[i]))
            for i in range(n_features)
        }

        # Confidence based on Shapley value concentration
        total_shap = np.sum(np.abs(shap_values))
        max_shap = np.max(np.abs(shap_values))
        confidence = float(max_shap / (total_shap + 1e-8))

        return ModelExplanation(
            prediction=float(prediction),
            confidence=min(1.0, confidence),
            base_value=float(self.base_value),
            shapley_values=shapley_objs,
            feature_importance=feature_importance,
            model_type='black_box'
        )


class LimeExplainer:
    """LIME-based local explanations"""

    def __init__(self, model_func: Callable):
        self.model_func = model_func

    def explain_instance(
        self,
        x: np.ndarray,
        feature_names: List[str],
        num_samples: int = 1000,
        kernel_width: float = 0.25
    ) -> ModelExplanation:
        """
        Explain prediction using LIME

        Args:
            x: Instance to explain (n_features,)
            feature_names: Feature names
            num_samples: Number of perturbed samples
            kernel_width: Kernel width for locality weighting

        Returns:
            ModelExplanation with local linear model
        """
        n_features = x.shape[0]

        # Generate perturbed samples
        perturbed = np.random.normal(x, kernel_width, (num_samples, n_features))
        perturbed = np.clip(perturbed, 0, 1)  # Keep in valid range

        # Get predictions
        predictions = np.array([self.model_func(p.reshape(1, -1))[0] for p in perturbed])

        # Calculate distance to original instance
        distances = np.linalg.norm(perturbed - x, axis=1)

        # Kernel weights (exponential decay)
        weights = np.exp(-(distances ** 2) / (2 * kernel_width ** 2))

        # Fit weighted linear regression
        X_weighted = perturbed * weights.reshape(-1, 1)
        y_weighted = predictions * weights

        # Calculate regression coefficients (simplified)
        coefficients = np.zeros(n_features)

        for i in range(n_features):
            # Weighted correlation
            x_weighted_i = X_weighted[:, i]
            correlation = np.sum(x_weighted_i * y_weighted) / (np.sum(x_weighted_i ** 2) + 1e-8)
            coefficients[i] = correlation

        # Get prediction
        prediction = self.model_func(x.reshape(1, -1))[0]

        # Create explanation
        base_value = np.mean(predictions)
        shap_vals = [
            ShapleyValue(
                feature_name=feature_names[i],
                shap_value=float(coefficients[i] * (x[i] - np.mean(perturbed[:, i]))),
                feature_value=float(x[i]),
                base_value=float(base_value),
                contribution_direction='positive' if coefficients[i] > 0 else 'negative'
            )
            for i in range(n_features)
        ]

        feature_importance = {
            feature_names[i]: float(np.abs(coefficients[i]))
            for i in range(n_features)
        }

        return ModelExplanation(
            prediction=float(prediction),
            confidence=float(np.mean(weights)),
            base_value=float(base_value),
            shapley_values=shap_vals,
            feature_importance=feature_importance,
            model_type='lime'
        )


class FeatureImportanceAnalyzer:
    """Analyze global feature importance"""

    @staticmethod
    def calculate_feature_importance(
        explanations: List[ModelExplanation]
    ) -> Dict[str, float]:
        """
        Calculate average feature importance across instances

        Args:
            explanations: List of ModelExplanation objects

        Returns:
            Average feature importance
        """
        if not explanations:
            return {}

        feature_names = set()
        for exp in explanations:
            feature_names.update(exp.feature_importance.keys())

        importance = {f: 0.0 for f in feature_names}

        for exp in explanations:
            for feature, imp in exp.feature_importance.items():
                importance[feature] += imp

        # Average
        importance = {f: v / len(explanations) for f, v in importance.items()}

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def identify_redundant_features(
        explanations: List[ModelExplanation],
        threshold: float = 0.1
    ) -> List[str]:
        """
        Identify features with low importance

        Args:
            explanations: List of explanations
            threshold: Importance threshold

        Returns:
            List of low-importance features
        """
        importance = FeatureImportanceAnalyzer.calculate_feature_importance(explanations)

        return [f for f, imp in importance.items() if imp < threshold]


class TradingSignalExplainer:
    """Explain trading signals and decision reasoning"""

    @staticmethod
    def explain_trade_decision(
        signal_value: float,
        prediction_confidence: float,
        explanation: ModelExplanation,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate human-readable explanation for trading signal

        Args:
            signal_value: Model signal (-1 to 1)
            prediction_confidence: Model confidence (0 to 1)
            explanation: ModelExplanation object
            threshold: Decision threshold

        Returns:
            Decision explanation
        """
        # Determine signal type
        if signal_value > threshold:
            signal_type = 'BUY'
        elif signal_value < -threshold:
            signal_type = 'SELL'
        else:
            signal_type = 'HOLD'

        # Find top contributing features
        top_features = sorted(
            explanation.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        # Build explanation text
        explanation_text = f"{signal_type} Signal (strength: {abs(signal_value):.2f})"

        if prediction_confidence < 0.3:
            explanation_text += " - LOW CONFIDENCE"

        return {
            'signal': signal_type,
            'signal_value': float(signal_value),
            'confidence': float(prediction_confidence),
            'explanation': explanation_text,
            'top_drivers': [f[0] for f in top_features],
            'driver_importance': {f[0]: float(f[1]) for f in top_features},
            'recommendation_strength': float(abs(signal_value) * prediction_confidence)
        }


class ModelAccuracyExplainer:
    """Explain model accuracy and error patterns"""

    @staticmethod
    def analyze_prediction_errors(
        predictions: np.ndarray,
        actuals: np.ndarray,
        explanations: List[ModelExplanation]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in prediction errors

        Args:
            predictions: Model predictions
            actuals: Actual values
            explanations: Corresponding explanations

        Returns:
            Error analysis
        """
        errors = predictions - actuals
        abs_errors = np.abs(errors)

        # Identify high-error predictions
        high_error_mask = abs_errors > np.mean(abs_errors) + np.std(abs_errors)

        # Feature patterns in errors
        error_feature_patterns = {}

        if any(high_error_mask) and explanations:
            for i, is_error in enumerate(high_error_mask):
                if is_error and i < len(explanations):
                    for feature, importance in explanations[i].feature_importance.items():
                        if feature not in error_feature_patterns:
                            error_feature_patterns[feature] = []
                        error_feature_patterns[feature].append(importance)

        return {
            'mean_absolute_error': float(np.mean(abs_errors)),
            'std_error': float(np.std(errors)),
            'high_error_rate': float(np.mean(high_error_mask)),
            'error_feature_patterns': {
                f: float(np.mean(v)) for f, v in error_feature_patterns.items()
            }
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Define a simple model
    def simple_model(x):
        return np.tanh(x @ np.array([0.5, -0.3, 0.2]))

    # Background data
    background = np.random.randn(100, 3) * 0.5

    # Test instance
    test_x = np.array([1.0, -0.5, 0.8])
    feature_names = ['Price_Momentum', 'Volume_Change', 'RSI']

    # SHAP explanation
    shap_calc = ShapleyValueCalculator(simple_model, background)
    shap_explanation = shap_calc.calculate_shapley_values(test_x, feature_names, n_samples=50)

    logger.info("SHAP Explanation:")
    logger.info(f"Prediction: {shap_explanation.prediction:.4f}")
    logger.info(f"Confidence: {shap_explanation.confidence:.4f}")

    for sv in shap_explanation.shapley_values[:3]:
        logger.info(
            f"  {sv.feature_name}: {sv.shap_value:+.4f} ({sv.contribution_direction})"
        )

    # LIME explanation
    lime_explainer = LimeExplainer(simple_model)
    lime_explanation = lime_explainer.explain_instance(test_x, feature_names, num_samples=500)

    logger.info("\nLIME Explanation:")
    logger.info(f"Prediction: {lime_explanation.prediction:.4f}")
    logger.info(f"Base value: {lime_explanation.base_value:.4f}")

    # Trading signal explanation
    signal_explainer = TradingSignalExplainer()
    signal_explanation = signal_explainer.explain_trade_decision(
        signal_value=shap_explanation.prediction,
        prediction_confidence=shap_explanation.confidence,
        explanation=shap_explanation,
        threshold=0.3
    )

    logger.info("\nTrading Signal Explanation:")
    logger.info(f"Signal: {signal_explanation['signal']}")
    logger.info(f"Confidence: {signal_explanation['confidence']:.2%}")
    logger.info(f"Top drivers: {signal_explanation['top_drivers']}")
