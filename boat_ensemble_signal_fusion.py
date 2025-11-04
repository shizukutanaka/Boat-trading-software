#!/usr/bin/env python3
"""
Ensemble Deep Learning Signal Fusion
=====================================

Multi-model blending for optimal trading signal combination:
  - Multiple deep learning models (CNN, LSTM, Transformer)
  - Stacking and blending ensemble methods
  - Signal correlation analysis
  - Weighted voting for consensus signals
  - Meta-learner optimization

Based on 2025 research (ensemble methods, signal blending, stacking).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnsembleSignal:
    """Ensemble trading signal"""
    timestamp: int
    signal: float  # -1: sell, 0: hold, 1: buy
    confidence: float  # 0-1
    component_signals: Dict[str, float]  # Individual model signals
    weights: Dict[str, float]  # Model weights in ensemble


class ModelBase:
    """Base model for ensemble"""

    def __init__(self, name: str):
        """Initialize model"""
        self.name = name
        self.weights = np.random.randn(16) * 0.01
        self.bias = 0.0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        return (x @ self.weights) + self.bias

    def predict(self, x: np.ndarray) -> float:
        """Generate signal"""
        output = self.forward(x)
        signal = np.tanh(output)  # Normalize to [-1, 1]
        return float(signal)


class CNNModel(ModelBase):
    """CNN-based signal model"""

    def __init__(self):
        """Initialize CNN model"""
        super().__init__("CNN")
        self.conv_weights = np.random.randn(5, 3) * 0.01
        self.conv_bias = np.zeros(3)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """CNN forward pass (simplified)"""
        # Simulate convolution
        n = len(x)
        output = np.zeros(max(1, n - 4))
        for i in range(len(output)):
            window = x[i:i + 5]
            output[i] = np.sum(window * self.conv_weights[:, 0]) + self.conv_bias[0]
        return output.mean() if len(output) > 0 else 0.0

    def predict(self, x: np.ndarray) -> float:
        """CNN prediction"""
        output = self.forward(x)
        return float(np.tanh(output))


class LSTMModel(ModelBase):
    """LSTM-based signal model"""

    def __init__(self):
        """Initialize LSTM model"""
        super().__init__("LSTM")
        self.hidden_state = np.zeros(8)

    def forward(self, x: np.ndarray) -> float:
        """LSTM forward pass (simplified)"""
        for val in x:
            self.hidden_state = np.tanh(self.hidden_state * 0.9 + val * self.weights[:8] * 0.1)
        return float(np.mean(self.hidden_state))

    def predict(self, x: np.ndarray) -> float:
        """LSTM prediction"""
        self.hidden_state = np.zeros(8)
        output = self.forward(x)
        return float(np.tanh(output))


class TransformerModel(ModelBase):
    """Transformer-based signal model"""

    def __init__(self):
        """Initialize Transformer model"""
        super().__init__("Transformer")
        self.attention_scores = np.zeros(10)

    def forward(self, x: np.ndarray) -> float:
        """Transformer forward pass (simplified)"""
        n = len(x)
        # Self-attention on input
        self.attention_scores = np.exp(x) / (np.sum(np.exp(x)) + 1e-8)
        weighted_sum = np.sum(self.attention_scores[:n] * x)
        return weighted_sum / max(1, n)

    def predict(self, x: np.ndarray) -> float:
        """Transformer prediction"""
        output = self.forward(x)
        return float(np.tanh(output))


class SignalBlender:
    """Signal blending ensemble"""

    def __init__(self, models: List[ModelBase]):
        """Initialize blender"""
        self.models = models
        self.model_weights = np.ones(len(models)) / len(models)
        self.meta_learner_w = np.random.randn(len(models)) * 0.01

    def compute_signal_correlations(self, signals: List[float]) -> np.ndarray:
        """Compute correlation between model signals"""
        n = len(signals)
        correlation = np.ones((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Simplified: correlation based on signal similarity
                corr = 1.0 - abs(signals[i] - signals[j]) / 2.0
                correlation[i, j] = corr
                correlation[j, i] = corr

        return correlation

    def blend_signals(self, component_signals: Dict[str, float]) -> Tuple[float, float]:
        """
        Blend multiple signals into consensus signal

        Args:
            component_signals: Individual model signals

        Returns:
            (blended_signal, confidence)
        """
        signals = list(component_signals.values())

        # Weighted average (stacking)
        blended = np.sum(self.model_weights * signals)

        # Confidence based on agreement
        signal_std = np.std(signals)
        confidence = 1.0 / (1.0 + signal_std)  # Higher confidence when signals agree

        return float(blended), float(confidence)

    def update_weights(self, validation_signals: List[Dict[str, float]], validation_returns: np.ndarray):
        """
        Update model weights based on validation performance

        Args:
            validation_signals: List of component signal dicts
            validation_returns: Actual returns
        """
        n_samples = len(validation_signals)
        model_names = list(validation_signals[0].keys())
        n_models = len(model_names)

        # Compute performance per model
        model_returns = np.zeros(n_models)
        for t in range(n_samples):
            signals = [validation_signals[t][name] for name in model_names]
            returns = validation_returns[t:t + 1]

            for m, signal in enumerate(signals):
                model_returns[m] += signal * returns[0]

        # Normalize to weights
        model_returns = np.clip(model_returns, 0, 1)
        self.model_weights = model_returns / (np.sum(model_returns) + 1e-8)

        logger.info(f"Updated weights: {dict(zip(model_names, self.model_weights))}")


class EnsembleSignalGenerator:
    """Ensemble-based signal generation"""

    def __init__(self):
        """Initialize ensemble"""
        self.models = [
            CNNModel(),
            LSTMModel(),
            TransformerModel()
        ]
        self.blender = SignalBlender(self.models)

    def generate_signals(self, price_features: np.ndarray, n_predictions: int = 5) -> List[EnsembleSignal]:
        """
        Generate ensemble signals

        Args:
            price_features: (n_periods, feature_dim) features
            n_predictions: Number of predictions

        Returns:
            List of ensemble signals
        """
        signals = []

        for t in range(max(0, len(price_features) - n_predictions), len(price_features)):
            features = price_features[t]

            # Get component signals
            component_signals = {}
            for model in self.models:
                component_signals[model.name] = model.predict(features)

            # Blend signals
            blended, confidence = self.blender.blend_signals(component_signals)

            # Create ensemble signal
            signal = EnsembleSignal(
                timestamp=t,
                signal=float(blended),
                confidence=confidence,
                component_signals=component_signals,
                weights=dict(zip([m.name for m in self.models], self.blender.model_weights))
            )
            signals.append(signal)

        return signals

    def backtest(self, signals: List[EnsembleSignal], returns: np.ndarray) -> Dict[str, float]:
        """
        Backtest ensemble strategy

        Args:
            signals: List of ensemble signals
            returns: (n_periods,) realized returns

        Returns:
            Performance metrics
        """
        n = min(len(signals), len(returns))

        # Strategy returns
        signal_values = np.array([s.signal for s in signals[:n]])
        strategy_returns = signal_values * returns[:n]

        # Metrics
        total_return = np.sum(strategy_returns)
        sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
        hit_rate = np.sum((signal_values * returns[:n]) > 0) / n

        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'hit_rate': float(hit_rate),
            'avg_return': float(np.mean(strategy_returns)),
            'max_loss': float(np.min(strategy_returns))
        }


if __name__ == "__main__":
    logger.info("Ensemble Deep Learning Signal Fusion")
    logger.info("=" * 50)

    np.random.seed(42)

    # Generate synthetic price features
    logger.info("\nGenerating price features")
    n_periods = 252
    feature_dim = 16

    price_features = np.random.randn(n_periods, feature_dim)
    actual_returns = np.random.randn(n_periods) * 0.01

    logger.info(f"  Periods: {n_periods}, Feature dim: {feature_dim}")

    # Initialize ensemble
    ensemble = EnsembleSignalGenerator()

    logger.info("\nGenerating ensemble signals")
    signals = ensemble.generate_signals(price_features, n_predictions=5)

    logger.info(f"  Generated {len(signals)} signals")

    # Display sample signals
    logger.info("\nSample Signals (last 5):")
    for signal in signals[-5:]:
        logger.info(f"  T={signal.timestamp}:")
        logger.info(f"    Blended: {signal.signal:.4f}, Confidence: {signal.confidence:.4f}")
        logger.info(f"    Components: {', '.join([f'{name}={val:.4f}' for name, val in signal.component_signals.items()])}")

    # Backtest
    logger.info("\nBacktesting ensemble strategy")
    metrics = ensemble.backtest(signals, actual_returns)

    logger.info(f"  Total Return: {metrics['total_return']:.4f}")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    logger.info(f"  Hit Rate: {metrics['hit_rate']:.2%}")
    logger.info(f"  Avg Return: {metrics['avg_return']:.4f}")
    logger.info(f"  Max Loss: {metrics['max_loss']:.4f}")

    # Model weights
    logger.info("\nEnsemble Model Weights:")
    for name, weight in zip([m.name for m in ensemble.models], ensemble.blender.model_weights):
        logger.info(f"  {name}: {weight:.4f}")

    logger.info("\nSignal Fusion Complete")
