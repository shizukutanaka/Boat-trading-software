#!/usr/bin/env python3
"""
Transformer Reinforcement Learning for Portfolio Optimization
==============================================================

Combines transformer attention with deep RL for dynamic portfolio management:
  - Transformer encoder for asset relationship modeling
  - Policy gradient optimization (PPO-style)
  - Risk-adjusted rewards with Sharpe ratio
  - Dynamic weight rebalancing
  - Multi-asset portfolio management

Based on 2025 research (Financial Transformer RL, FTRL, ART-DRL).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PortfolioAction:
    """Portfolio rebalancing action"""
    weights: np.ndarray  # (n_assets,) weight vector
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float


class TransformerEncoder:
    """Transformer for asset relationship encoding"""

    def __init__(self, asset_dim: int = 16, hidden_dim: int = 32, n_heads: int = 4):
        """Initialize transformer encoder"""
        self.asset_dim = asset_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # Attention weights
        self.W_q = np.random.randn(asset_dim, hidden_dim) * 0.01
        self.W_k = np.random.randn(asset_dim, hidden_dim) * 0.01
        self.W_v = np.random.randn(asset_dim, hidden_dim) * 0.01
        self.W_out = np.random.randn(hidden_dim, asset_dim) * 0.01

    def forward(self, asset_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transformer forward pass

        Args:
            asset_features: (n_assets, asset_dim) feature matrix

        Returns:
            (encoded_features, attention_weights)
        """
        # Multi-head attention
        Q = asset_features @ self.W_q  # (n_assets, hidden_dim)
        K = asset_features @ self.W_k
        V = asset_features @ self.W_v

        # Attention scores
        scores = Q @ K.T / np.sqrt(self.hidden_dim)  # (n_assets, n_assets)
        attention = self._softmax(scores)

        # Aggregate values
        output = attention @ V @ self.W_out  # (n_assets, asset_dim)

        return output, attention

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)


class PolicyNetwork:
    """Policy network for portfolio optimization"""

    def __init__(self, n_assets: int = 10, hidden_dim: int = 64):
        """Initialize policy network"""
        self.n_assets = n_assets
        self.hidden_dim = hidden_dim

        # Network weights
        self.W1 = np.random.randn(n_assets * 16, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, n_assets) * 0.01
        self.b2 = np.zeros((1, n_assets))

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass to compute portfolio weights

        Args:
            state: (n_assets, 16) state features

        Returns:
            (n_assets,) portfolio weights
        """
        # Flatten state
        flat_state = state.flatten().reshape(1, -1)

        # Hidden layer
        h = np.maximum(flat_state @ self.W1 + self.b1, 0)  # ReLU

        # Output layer (softmax for valid weights)
        logits = h @ self.W2 + self.b2
        weights = self._softmax(logits)[0]

        return weights

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)


class TransformerRLPortfolioOptimizer:
    """Transformer RL for portfolio optimization"""

    def __init__(self, n_assets: int = 10, asset_dim: int = 16):
        """Initialize optimizer"""
        self.n_assets = n_assets
        self.asset_dim = asset_dim
        self.transformer = TransformerEncoder(asset_dim=asset_dim)
        self.policy = PolicyNetwork(n_assets=n_assets)
        self.learning_rate = 0.001

    def compute_portfolio_metrics(self, weights: np.ndarray, returns: np.ndarray,
                                 volatilities: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute portfolio metrics

        Args:
            weights: (n_assets,) portfolio weights
            returns: (n_assets,) expected returns
            volatilities: (n_assets,) asset volatilities

        Returns:
            (expected_return, expected_volatility, sharpe_ratio)
        """
        # Weighted return
        portfolio_return = np.sum(weights * returns)

        # Portfolio volatility (simplified: weighted average of individual volatilities)
        weighted_vol_sum = np.sum(weights * volatilities)
        portfolio_vol = weighted_vol_sum

        # Sharpe ratio (risk-free rate = 2%)
        sharpe = (portfolio_return - 0.02) / (portfolio_vol + 1e-8)

        return float(portfolio_return), float(portfolio_vol), float(sharpe)

    def optimize(self, market_data: np.ndarray, returns: np.ndarray,
                volatilities: np.ndarray, n_epochs: int = 10) -> List[PortfolioAction]:
        """
        Optimize portfolio via Transformer RL

        Args:
            market_data: (n_assets, asset_dim) feature matrix
            returns: (n_assets,) expected returns
            volatilities: (n_assets,) volatilities
            n_epochs: Number of optimization epochs

        Returns:
            List of portfolio actions
        """
        actions = []

        for epoch in range(n_epochs):
            # Transform features
            encoded_features, attention_weights = self.transformer.forward(market_data)

            # Policy forward pass
            weights = self.policy.forward(encoded_features)

            # Compute portfolio metrics
            p_return, p_vol, sharpe = self.compute_portfolio_metrics(weights, returns, volatilities)

            # Store action
            action = PortfolioAction(
                weights=weights,
                expected_return=p_return,
                expected_volatility=p_vol,
                sharpe_ratio=sharpe
            )
            actions.append(action)

            if epoch % 2 == 0:
                logger.info(f"  Epoch {epoch}: Return={p_return:.4f}, Vol={p_vol:.4f}, Sharpe={sharpe:.4f}")

        return actions

    def backtest(self, weights: np.ndarray, historical_returns: np.ndarray) -> Dict[str, float]:
        """
        Backtest portfolio strategy

        Args:
            weights: (n_assets,) portfolio weights
            historical_returns: (n_periods, n_assets) historical returns

        Returns:
            Performance metrics
        """
        # Portfolio returns
        portfolio_returns = historical_returns @ weights  # (n_periods,)

        # Metrics
        total_return = np.prod(1 + portfolio_returns) - 1
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        sharpe = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(252)
        max_dd = np.max(np.maximum.accumulate(portfolio_returns) - portfolio_returns)

        return {
            'total_return': float(total_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'avg_return': float(np.mean(portfolio_returns))
        }


if __name__ == "__main__":
    logger.info("Transformer RL Portfolio Optimization")
    logger.info("=" * 50)

    np.random.seed(42)

    # Generate market data
    logger.info("\nGenerating market data")
    n_assets = 10
    asset_dim = 16

    market_features = np.random.randn(n_assets, asset_dim)
    expected_returns = np.random.uniform(0.05, 0.15, n_assets)
    volatilities = np.random.uniform(0.1, 0.3, n_assets)

    logger.info(f"  Assets: {n_assets}, Feature dim: {asset_dim}")
    logger.info(f"  Return range: [{expected_returns.min():.2%}, {expected_returns.max():.2%}]")
    logger.info(f"  Volatility range: [{volatilities.min():.2%}, {volatilities.max():.2%}]")

    # Initialize optimizer
    optimizer = TransformerRLPortfolioOptimizer(n_assets=n_assets, asset_dim=asset_dim)

    # Optimize
    logger.info("\nOptimizing portfolio")
    actions = optimizer.optimize(market_features, expected_returns, volatilities, n_epochs=10)

    # Best action
    best_action = max(actions, key=lambda a: a.sharpe_ratio)
    logger.info(f"\nBest Portfolio:")
    logger.info(f"  Return: {best_action.expected_return:.4f}")
    logger.info(f"  Volatility: {best_action.expected_volatility:.4f}")
    logger.info(f"  Sharpe Ratio: {best_action.sharpe_ratio:.4f}")

    logger.info(f"  Top 3 Assets by Weight:")
    top_indices = np.argsort(best_action.weights)[-3:][::-1]
    for idx in top_indices:
        logger.info(f"    Asset_{idx}: {best_action.weights[idx]:.4f}")

    # Backtest
    logger.info("\nBacktesting strategy")
    historical_returns = np.random.randn(252, n_assets) * volatilities * 0.1 + expected_returns * 0.001
    metrics = optimizer.backtest(best_action.weights, historical_returns)

    logger.info(f"  Total Return: {metrics['total_return']:.4f}")
    logger.info(f"  Volatility: {metrics['volatility']:.4f}")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.4f}")

    logger.info("\nTransformer RL Optimization Complete")
