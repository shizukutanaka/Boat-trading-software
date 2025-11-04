#!/usr/bin/env python3
"""
Mixture of Experts for Financial Prediction
=============================================

Conditional computing with expert gating for specialized predictions:
  - Multiple expert networks for different market regimes
  - Gating network for expert selection
  - Load balancing across experts
  - Sparsity for efficient computation
  - Portfolio ensemble with expert weighting

Based on 2025 research (MoE in finance, expert aggregation, conditional compute).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExpertPrediction:
    """Expert network prediction"""
    expert_id: int
    prediction: float
    confidence: float
    load: float


@dataclass
class MoEOutput:
    """Mixture of Experts output"""
    ensemble_prediction: float
    expert_predictions: List[ExpertPrediction]
    gating_weights: np.ndarray  # (n_experts,)
    active_experts: List[int]


class Expert:
    """Individual expert network"""

    def __init__(self, expert_id: int, input_dim: int = 16, hidden_dim: int = 32):
        """Initialize expert"""
        self.expert_id = expert_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Network weights (2-layer MLP)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros(1)

    def forward(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Forward pass

        Args:
            x: Input features

        Returns:
            (prediction, confidence)
        """
        # Ensure correct shape
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # First layer
        h = np.maximum(x @ self.W1 + self.b1, 0)  # ReLU

        # Output layer
        output = h @ self.W2 + self.b2
        prediction = float(output[0, 0])

        # Confidence based on magnitude
        confidence = float(1.0 / (1.0 + np.abs(prediction)))

        return prediction, confidence

    def predict(self, x: np.ndarray) -> ExpertPrediction:
        """Generate prediction"""
        prediction, confidence = self.forward(x)
        return ExpertPrediction(
            expert_id=self.expert_id,
            prediction=prediction,
            confidence=confidence,
            load=0.0
        )


class GatingNetwork:
    """Gating network for expert selection"""

    def __init__(self, n_experts: int, input_dim: int = 16, sparsity_k: int = 2):
        """Initialize gating network"""
        self.n_experts = n_experts
        self.input_dim = input_dim
        self.sparsity_k = min(sparsity_k, n_experts)  # Top-k experts

        # Gating weights
        self.W_gate = np.random.randn(input_dim, n_experts) * 0.01
        self.b_gate = np.zeros(n_experts)

    def forward(self, x: np.ndarray, temperature: float = 1.0) -> Tuple[np.ndarray, List[int]]:
        """
        Compute gating weights via softmax

        Args:
            x: Input features
            temperature: Softmax temperature for sparsity

        Returns:
            (gating_weights, active_expert_indices)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Compute logits
        logits = x @ self.W_gate + self.b_gate  # (1, n_experts)
        logits = logits[0]  # Remove batch dimension

        # Temperature-scaled softmax
        scaled_logits = logits / temperature
        e_logits = np.exp(scaled_logits - np.max(scaled_logits))
        weights = e_logits / np.sum(e_logits)

        # Top-k sparsity
        top_k_indices = np.argsort(weights)[-self.sparsity_k:][::-1]

        # Zero out non-selected experts
        sparse_weights = np.zeros_like(weights)
        sparse_weights[top_k_indices] = weights[top_k_indices]
        sparse_weights = sparse_weights / (np.sum(sparse_weights) + 1e-8)

        return sparse_weights, top_k_indices.tolist()


class MixtureOfExpertsEnsemble:
    """Mixture of Experts ensemble"""

    def __init__(self, n_experts: int = 4, input_dim: int = 16, sparsity_k: int = 2):
        """Initialize MoE ensemble"""
        self.n_experts = n_experts
        self.input_dim = input_dim

        # Create experts
        self.experts = [Expert(i, input_dim) for i in range(n_experts)]

        # Gating network
        self.gating = GatingNetwork(n_experts, input_dim, sparsity_k)

        # Load balancing
        self.expert_loads = np.zeros(n_experts)

    def forward(self, x: np.ndarray, temperature: float = 1.0) -> MoEOutput:
        """
        Forward pass through MoE

        Args:
            x: Input features
            temperature: Temperature for gating softmax

        Returns:
            MoEOutput with ensemble prediction
        """
        # Get gating weights
        gating_weights, active_indices = self.gating.forward(x, temperature)

        # Get expert predictions
        expert_preds = []
        for i, expert in enumerate(self.experts):
            pred = expert.predict(x)
            pred.load = gating_weights[i]
            expert_preds.append(pred)

        # Update expert loads
        self.expert_loads += gating_weights

        # Ensemble prediction (weighted average of active experts only)
        ensemble_pred = 0.0
        active_weight_sum = 0.0

        for idx in active_indices:
            weight = gating_weights[idx]
            pred = expert_preds[idx].prediction
            ensemble_pred += weight * pred
            active_weight_sum += weight

        if active_weight_sum > 0:
            ensemble_pred = ensemble_pred / active_weight_sum

        return MoEOutput(
            ensemble_prediction=float(ensemble_pred),
            expert_predictions=expert_preds,
            gating_weights=gating_weights,
            active_experts=active_indices
        )

    def get_load_imbalance(self) -> float:
        """Compute load imbalance penalty"""
        if np.sum(self.expert_loads) == 0:
            return 0.0

        normalized_loads = self.expert_loads / np.sum(self.expert_loads)
        ideal_load = 1.0 / self.n_experts

        imbalance = np.mean((normalized_loads - ideal_load) ** 2)
        return float(imbalance)

    def rebalance_experts(self):
        """Reset load counters for next iteration"""
        self.expert_loads = np.zeros(self.n_experts)


class MarketRegimeExpertMoE:
    """MoE specialized for different market regimes"""

    def __init__(self, regimes: List[str] = ["bull", "bear", "sideways"]):
        """Initialize regime-specific MoE"""
        self.regimes = regimes
        self.regime_moes = {
            regime: MixtureOfExpertsEnsemble(n_experts=4, sparsity_k=2)
            for regime in regimes
        }

    def detect_regime(self, price_data: np.ndarray) -> str:
        """
        Detect current market regime

        Args:
            price_data: Recent price data

        Returns:
            Regime label
        """
        returns = np.diff(np.log(price_data))
        avg_return = np.mean(returns)
        volatility = np.std(returns)

        if avg_return > 0.001 and volatility < 0.02:
            return "bull"
        elif avg_return < -0.001 and volatility < 0.02:
            return "bear"
        else:
            return "sideways"

    def predict(self, x: np.ndarray, price_data: np.ndarray) -> Tuple[float, str]:
        """
        Predict using regime-specific MoE

        Args:
            x: Input features
            price_data: Price data for regime detection

        Returns:
            (prediction, regime)
        """
        regime = self.detect_regime(price_data)
        moe = self.regime_moes[regime]

        output = moe.forward(x, temperature=1.0)

        return output.ensemble_prediction, regime


if __name__ == "__main__":
    logger.info("Mixture of Experts for Financial Prediction")
    logger.info("=" * 50)

    np.random.seed(42)

    # Generate synthetic data
    logger.info("\nGenerating synthetic market data")
    n_samples = 100
    input_dim = 16

    features = np.random.randn(n_samples, input_dim)
    price_data = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)

    logger.info(f"  Samples: {n_samples}, Features: {input_dim}")

    # Initialize MoE
    logger.info("\nInitializing Mixture of Experts")
    moe = MixtureOfExpertsEnsemble(n_experts=4, input_dim=input_dim, sparsity_k=2)

    logger.info(f"  Experts: 4, Top-k: 2")

    # Make predictions
    logger.info("\nMaking predictions")
    predictions = []
    for i in range(5):
        output = moe.forward(features[i])
        predictions.append(output.ensemble_prediction)

        logger.info(f"  Sample {i}:")
        logger.info(f"    Ensemble: {output.ensemble_prediction:.4f}")
        logger.info(f"    Active experts: {output.active_experts}")
        logger.info(f"    Weights: {output.gating_weights}")

    # Load balancing
    logger.info("\nLoad Balancing Analysis")
    load_imbalance = moe.get_load_imbalance()
    logger.info(f"  Load imbalance penalty: {load_imbalance:.6f}")

    logger.info(f"  Expert loads: {moe.expert_loads}")
    normalized_loads = moe.expert_loads / np.sum(moe.expert_loads)
    logger.info(f"  Normalized loads: {normalized_loads}")

    # Regime-specific MoE
    logger.info("\nRegime-Specific MoE")
    regime_moe = MarketRegimeExpertMoE(regimes=["bull", "bear", "sideways"])

    # Test on different market regimes
    test_regimes = [
        price_data[-50:],  # Recent bull
        price_data[-50:] * 0.95,  # Bear
        price_data[-50:]  # Sideways
    ]

    for test_prices in test_regimes:
        pred, regime = regime_moe.predict(features[0], test_prices)
        logger.info(f"  Regime: {regime}, Prediction: {pred:.4f}")

    logger.info("\nMixture of Experts Complete")
