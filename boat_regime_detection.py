#!/usr/bin/env python3
"""
Market Regime Detection and Classification
===========================================

Identify and classify market conditions using machine learning:
  - Hidden Markov Models for regime detection
  - Gaussian Mixture Models for regime clustering
  - Regime-specific factor importance
  - Market condition forecasting
  - Regime transition probabilities
  - Strategy adaptation by regime

Based on 2025 research on market regime classification and HMM approaches.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RegimeState:
    """Market regime classification"""
    regime_id: int
    regime_type: str  # 'bull', 'bear', 'sideways', 'high_vol', 'low_vol'
    probability: float
    volatility: float
    trend: float
    correlation: float
    transition_prob: float


class HiddenMarkovRegimeDetector:
    """Detect market regimes using Hidden Markov Models"""

    def __init__(self, n_states: int = 3, n_iterations: int = 100):
        self.n_states = n_states
        self.n_iterations = n_iterations

        # HMM parameters
        self.transition_matrix = np.ones((n_states, n_states)) / n_states
        self.emission_means = np.random.randn(n_states)
        self.emission_stds = np.ones(n_states) * 0.5
        self.initial_probs = np.ones(n_states) / n_states

    def fit(self, returns: np.ndarray) -> None:
        """
        Fit HMM to return data using Baum-Welch algorithm

        Args:
            returns: Daily returns (T,)
        """
        T = len(returns)

        # Simplified Baum-Welch: iterate EM steps
        for iteration in range(self.n_iterations):
            # E-step: calculate forward and backward probabilities
            forward = self._forward_algorithm(returns)
            backward = self._backward_algorithm(returns)

            # Calculate posterior probabilities
            posteriors = forward * backward
            posteriors /= np.sum(posteriors, axis=1, keepdims=True)

            # M-step: update parameters
            # Update transition matrix
            for i in range(self.n_states):
                for j in range(self.n_states):
                    numerator = 0
                    denominator = 0

                    for t in range(T - 1):
                        numerator += posteriors[t, i] * self._emission_prob(returns[t+1], j)
                        denominator += posteriors[t, i]

                    self.transition_matrix[i, j] = numerator / (denominator + 1e-8)

            # Update emission parameters
            for j in range(self.n_states):
                weight = posteriors[:, j]
                self.emission_means[j] = np.average(returns, weights=weight + 1e-8)
                variance = np.average((returns - self.emission_means[j]) ** 2, weights=weight + 1e-8)
                self.emission_stds[j] = np.sqrt(variance + 1e-8)

            # Update initial probabilities
            self.initial_probs = posteriors[0]

    def _forward_algorithm(self, returns: np.ndarray) -> np.ndarray:
        """Compute forward probabilities"""
        T = len(returns)
        forward = np.zeros((T, self.n_states))

        # Initialization
        for j in range(self.n_states):
            forward[0, j] = self.initial_probs[j] * self._emission_prob(returns[0], j)

        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                forward[t, j] = (
                    np.sum(forward[t-1] * self.transition_matrix[:, j]) *
                    self._emission_prob(returns[t], j)
                )

        return forward

    def _backward_algorithm(self, returns: np.ndarray) -> np.ndarray:
        """Compute backward probabilities"""
        T = len(returns)
        backward = np.zeros((T, self.n_states))

        # Initialization
        backward[T-1] = 1.0

        # Recursion
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                backward[t, i] = np.sum(
                    self.transition_matrix[i] *
                    np.array([self._emission_prob(returns[t+1], j) for j in range(self.n_states)]) *
                    backward[t+1]
                )

        return backward

    def _emission_prob(self, observation: float, state: int) -> float:
        """Gaussian emission probability"""
        mean = self.emission_means[state]
        std = self.emission_stds[state]

        exponent = -0.5 * ((observation - mean) / (std + 1e-8)) ** 2
        return (1.0 / (np.sqrt(2 * np.pi) * (std + 1e-8))) * np.exp(exponent)

    def predict(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regime sequence using Viterbi algorithm

        Args:
            returns: Daily returns (T,)

        Returns:
            (regime_sequence, regime_probabilities)
        """
        T = len(returns)

        # Forward pass
        viterbi = np.zeros((T, self.n_states))
        backpointer = np.zeros((T, self.n_states), dtype=int)

        # Initialization
        for j in range(self.n_states):
            viterbi[0, j] = self.initial_probs[j] * self._emission_prob(returns[0], j)

        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                temp = viterbi[t-1] * self.transition_matrix[:, j]
                backpointer[t, j] = np.argmax(temp)
                viterbi[t, j] = np.max(temp) * self._emission_prob(returns[t], j)

        # Backtrack
        regime_sequence = np.zeros(T, dtype=int)
        regime_sequence[-1] = np.argmax(viterbi[-1])

        for t in range(T-2, -1, -1):
            regime_sequence[t] = backpointer[t+1, regime_sequence[t+1]]

        # Calculate probabilities
        regime_probs = np.zeros((T, self.n_states))
        for t in range(T):
            regime_probs[t, regime_sequence[t]] = np.max(viterbi[t])

        return regime_sequence, regime_probs


class GaussianMixtureRegimeDetector:
    """Detect regimes using Gaussian Mixture Models"""

    def __init__(self, n_regimes: int = 3, n_iterations: int = 50):
        self.n_regimes = n_regimes
        self.n_iterations = n_iterations

        self.means = np.random.randn(n_regimes)
        self.variances = np.ones(n_regimes)
        self.weights = np.ones(n_regimes) / n_regimes

    def fit(self, returns: np.ndarray) -> None:
        """Fit GMM using EM algorithm"""
        for iteration in range(self.n_iterations):
            # E-step: calculate responsibilities
            responsibilities = self._calculate_responsibilities(returns)

            # M-step: update parameters
            N_k = np.sum(responsibilities, axis=0)

            # Update weights
            self.weights = N_k / len(returns)

            # Update means
            for k in range(self.n_regimes):
                self.means[k] = np.sum(responsibilities[:, k] * returns) / (N_k[k] + 1e-8)

            # Update variances
            for k in range(self.n_regimes):
                diff_sq = (returns - self.means[k]) ** 2
                self.variances[k] = np.sum(responsibilities[:, k] * diff_sq) / (N_k[k] + 1e-8)

    def _calculate_responsibilities(self, returns: np.ndarray) -> np.ndarray:
        """Calculate soft assignments to regimes"""
        numerator = np.zeros((len(returns), self.n_regimes))

        for k in range(self.n_regimes):
            numerator[:, k] = self.weights[k] * self._gaussian_pdf(returns, self.means[k], self.variances[k])

        denominator = np.sum(numerator, axis=1, keepdims=True)
        return numerator / (denominator + 1e-8)

    @staticmethod
    def _gaussian_pdf(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
        """Gaussian probability density"""
        numerator = np.exp(-0.5 * ((x - mean) ** 2) / (variance + 1e-8))
        denominator = np.sqrt(2 * np.pi * (variance + 1e-8))
        return numerator / denominator

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Assign regimes to observations"""
        responsibilities = self._calculate_responsibilities(returns)
        return np.argmax(responsibilities, axis=1)


class RegimeCharacterization:
    """Characterize and describe market regimes"""

    @staticmethod
    def characterize_regimes(
        returns: np.ndarray,
        regime_labels: np.ndarray,
        window: int = 20
    ) -> Dict[int, Dict[str, float]]:
        """
        Characterize each regime

        Args:
            returns: Daily returns
            regime_labels: Assigned regime for each day
            window: Rolling window for metrics

        Returns:
            Dictionary of regime characteristics
        """
        characteristics = {}

        for regime_id in np.unique(regime_labels):
            mask = regime_labels == regime_id
            regime_returns = returns[mask]

            characteristics[regime_id] = {
                'mean_return': float(np.mean(regime_returns)),
                'volatility': float(np.std(regime_returns)),
                'skewness': float(np.mean(((regime_returns - np.mean(regime_returns)) /
                                           (np.std(regime_returns) + 1e-8)) ** 3)),
                'kurtosis': float(np.mean(((regime_returns - np.mean(regime_returns)) /
                                           (np.std(regime_returns) + 1e-8)) ** 4)) - 3,
                'max_drawdown': float(np.min(np.cumsum(regime_returns))),
                'sharpe_ratio': float(np.mean(regime_returns) / (np.std(regime_returns) + 1e-8)),
                'frequency': float(np.sum(mask) / len(regime_labels))
            }

        return characteristics

    @staticmethod
    def classify_regime_type(characteristics: Dict[str, float]) -> str:
        """
        Classify regime type based on characteristics

        Args:
            characteristics: Regime characteristics

        Returns:
            Regime type name
        """
        vol = characteristics['volatility']
        return_mean = characteristics['mean_return']

        if vol > np.percentile([v['volatility'] for k, v in characteristics.items()], 75):
            return 'high_volatility'
        elif vol < np.percentile([v['volatility'] for k, v in characteristics.items()], 25):
            return 'low_volatility'
        elif return_mean > 0:
            return 'bull_market'
        elif return_mean < 0:
            return 'bear_market'
        else:
            return 'sideways_market'


class RegimeAdaptiveStrategy:
    """Adapt trading strategy based on regime"""

    @staticmethod
    def get_regime_specific_parameters(
        regime_type: str
    ) -> Dict[str, float]:
        """
        Get strategy parameters for specific regime

        Args:
            regime_type: Type of market regime

        Returns:
            Recommended parameters
        """
        params = {
            'high_volatility': {
                'position_size': 0.5,  # Reduce position size
                'stop_loss': 0.02,     # Tighter stops
                'profit_target': 0.03,
                'holding_period': 5,   # Shorter holds
                'leverage': 1.0
            },
            'low_volatility': {
                'position_size': 1.0,
                'stop_loss': 0.05,
                'profit_target': 0.10,
                'holding_period': 20,
                'leverage': 1.5
            },
            'bull_market': {
                'position_size': 1.0,
                'stop_loss': 0.10,     # Wider stops
                'profit_target': 0.20,
                'holding_period': 30,
                'leverage': 2.0,
                'bias': 'long'
            },
            'bear_market': {
                'position_size': 0.5,
                'stop_loss': 0.05,
                'profit_target': 0.05,
                'holding_period': 5,
                'leverage': 1.0,
                'bias': 'short'
            },
            'sideways_market': {
                'position_size': 0.75,
                'stop_loss': 0.03,
                'profit_target': 0.05,
                'holding_period': 10,
                'leverage': 1.0,
                'strategy': 'range_trading'
            }
        }

        return params.get(regime_type, params['sideways_market'])


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate sample returns with regime changes
    T = 500

    # Different regimes
    bull_returns = np.random.randn(150) * 0.01 + 0.002
    bear_returns = np.random.randn(200) * 0.015 - 0.001
    sideways_returns = np.random.randn(150) * 0.005

    returns = np.concatenate([bull_returns, bear_returns, sideways_returns])

    # HMM Regime Detection
    hmm = HiddenMarkovRegimeDetector(n_states=3, n_iterations=20)
    hmm.fit(returns)
    hmm_regimes, hmm_probs = hmm.predict(returns)

    logger.info("HMM Regime Detection:")
    logger.info(f"Unique regimes: {np.unique(hmm_regimes)}")
    logger.info(f"Regime distribution: {np.bincount(hmm_regimes)}")

    # GMM Regime Detection
    gmm = GaussianMixtureRegimeDetector(n_regimes=3, n_iterations=50)
    gmm.fit(returns)
    gmm_regimes = gmm.predict(returns)

    logger.info("\nGMM Regime Detection:")
    logger.info(f"Unique regimes: {np.unique(gmm_regimes)}")
    logger.info(f"Regime distribution: {np.bincount(gmm_regimes)}")

    # Characterize regimes
    characteristics = RegimeCharacterization.characterize_regimes(returns, hmm_regimes)

    logger.info("\nRegime Characteristics:")
    for regime_id, chars in characteristics.items():
        logger.info(f"\nRegime {regime_id}:")
        for metric, value in chars.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Classify regime type
        regime_type = RegimeCharacterization.classify_regime_type(chars)
        logger.info(f"  Classification: {regime_type}")

        # Get strategy parameters
        params = RegimeAdaptiveStrategy.get_regime_specific_parameters(regime_type)
        logger.info(f"  Recommended position size: {params['position_size']}")
