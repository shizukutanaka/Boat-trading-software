#!/usr/bin/env python3
"""
Graph Attention Networks for Asset Correlation & Prediction
============================================================

Spatio-temporal attention mechanisms for financial networks:
  - Graph Attention Layer with multi-head attention
  - Dynamic asset correlation graphs
  - Temporal graph convolutions
  - Asset price forecasting with graph context
  - Portfolio risk propagation via attention
  - Time-varying network structure

Based on 2025 research on attention-based GNNs for finance.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GraphAttentionOutput:
    """Output from graph attention layer"""
    node_features: np.ndarray
    attention_weights: np.ndarray
    aggregated_features: np.ndarray


class CorrelationGraphBuilder:
    """Build asset correlation graphs"""

    @staticmethod
    def build_correlation_graph(returns: np.ndarray, threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build correlation-based asset graph

        Args:
            returns: Asset returns (N_samples, N_assets)
            threshold: Correlation threshold for edges

        Returns:
            (adjacency_matrix, correlation_matrix)
        """
        corr_matrix = np.corrcoef(returns.T)

        # Threshold to create sparse graph
        adjacency = np.abs(corr_matrix) > threshold
        adjacency = adjacency.astype(float)
        np.fill_diagonal(adjacency, 0)  # Remove self-loops

        return adjacency, corr_matrix

    @staticmethod
    def build_rolling_graph(returns: np.ndarray, window: int = 60, step: int = 10) -> List[np.ndarray]:
        """
        Build rolling correlation graphs

        Args:
            returns: Asset returns
            window: Rolling window size
            step: Step size for rolling

        Returns:
            List of adjacency matrices over time
        """
        graphs = []

        for start in range(0, len(returns) - window, step):
            end = start + window
            window_returns = returns[start:end]
            adj, _ = CorrelationGraphBuilder.build_correlation_graph(window_returns, threshold=0.3)
            graphs.append(adj)

        return graphs


class GraphAttentionLayer:
    """Single graph attention layer"""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 8):
        """
        Initialize GAT layer

        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            n_heads: Number of attention heads
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads

        # Linear transformation for all heads
        self.W = np.random.randn(n_heads, in_dim, out_dim) * np.sqrt(2.0 / (in_dim + out_dim))

        # Attention parameters for all heads
        self.a = np.random.randn(n_heads, 2 * out_dim) * 0.01

    def forward(
        self,
        features: np.ndarray,
        adjacency: np.ndarray
    ) -> GraphAttentionOutput:
        """
        Forward pass through GAT layer

        Args:
            features: Node features (N_nodes, in_dim)
            adjacency: Adjacency matrix (N_nodes, N_nodes)

        Returns:
            GraphAttentionOutput
        """
        N = features.shape[0]
        all_head_outputs = []
        all_attention_weights = []

        for head in range(self.n_heads):
            # Linear transformation: (N, in_dim) @ (in_dim, out_dim) -> (N, out_dim)
            h = np.dot(features, self.W[head])

            # Attention logits: e_ij = LeakyReLU(a^T [h_i || h_j])
            e = np.zeros((N, N))

            for i in range(N):
                for j in range(N):
                    # Concatenate features
                    concat = np.concatenate([h[i], h[j]])
                    # Attention score
                    e[i, j] = np.dot(self.a[head], concat)

            # Apply LeakyReLU
            e = np.where(e > 0, e, 0.01 * e)

            # Mask: only attend to neighbors (adjacency matrix)
            e = np.where(adjacency > 0, e, -1e9)

            # Softmax
            attention_weights = self._softmax(e, axis=1)
            all_attention_weights.append(attention_weights)

            # Aggregate: (N, N) @ (N, out_dim) -> (N, out_dim)
            aggregated = np.dot(attention_weights, h)
            all_head_outputs.append(aggregated)

        # Concatenate heads
        aggregated_features = np.concatenate(all_head_outputs, axis=1)
        attention_weights = np.mean(all_attention_weights, axis=0)  # Average attention across heads

        return GraphAttentionOutput(
            node_features=features,
            attention_weights=attention_weights,
            aggregated_features=aggregated_features
        )

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
        """Softmax with numerical stability"""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)


class GraphAttentionNetwork:
    """Multi-layer GAT for financial forecasting"""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32, n_heads: int = 8, n_layers: int = 2):
        """
        Initialize GAT network

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden feature dimension
            output_dim: Output feature dimension
            n_heads: Attention heads per layer
            n_layers: Number of attention layers
        """
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.layers = []

        # Build layers
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]

        for i in range(n_layers):
            # Each head outputs hidden_dim // n_heads
            out_per_head = dims[i + 1] // n_heads
            layer = GraphAttentionLayer(
                in_dim=dims[i],
                out_dim=out_per_head,
                n_heads=n_heads
            )
            self.layers.append(layer)

    def forward(
        self,
        features: np.ndarray,
        adjacency: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through GAT

        Args:
            features: Node features
            adjacency: Adjacency matrix

        Returns:
            (output_features, list_of_attention_weights)
        """
        x = features
        attention_weights = []

        for layer in self.layers:
            output = layer.forward(x, adjacency)
            x = output.aggregated_features
            attention_weights.append(output.attention_weights)

        return x, attention_weights

    def predict_prices(
        self,
        features: np.ndarray,
        adjacency: np.ndarray
    ) -> np.ndarray:
        """
        Predict asset prices using graph context

        Args:
            features: Node features (N_assets, features)
            adjacency: Asset correlation graph

        Returns:
            Price predictions (N_assets,)
        """
        # Forward pass through GAT
        gat_features, _ = self.forward(features, adjacency)

        # Simple linear predictor on top of GAT features
        # (in practice, this would be learned)
        predictions = np.sum(gat_features, axis=1)

        return predictions


class TemporalGraphAttention:
    """Temporal graph attention for multi-step forecasting"""

    def __init__(self, n_assets: int, look_back: int = 20):
        """
        Initialize temporal GAT

        Args:
            n_assets: Number of assets
            look_back: Number of historical time steps
        """
        self.n_assets = n_assets
        self.look_back = look_back
        self.gat = GraphAttentionNetwork(
            input_dim=look_back,
            hidden_dim=64,
            output_dim=32,
            n_heads=8,
            n_layers=2
        )

    def forecast(
        self,
        returns_history: np.ndarray,
        correlation_graphs: Optional[List[np.ndarray]] = None,
        steps_ahead: int = 5
    ) -> np.ndarray:
        """
        Forecast multiple steps ahead using temporal attention

        Args:
            returns_history: Historical returns (N_samples, N_assets, look_back) or (N_assets, look_back)
            correlation_graphs: List of correlation matrices over time
            steps_ahead: Number of steps to forecast

        Returns:
            Forecasted returns (steps_ahead, N_assets)
        """
        # Handle 3D input by flattening to 2D
        if returns_history.ndim == 3:
            # (N_samples, N_assets, look_back) -> flatten last dimension
            hist_2d = returns_history[-1]  # Last sample (N_assets, look_back)
        else:
            # Assume (look_back, N_assets)
            hist_2d = returns_history.T  # Transpose to (N_assets, look_back)

        # Build graph if not provided
        if correlation_graphs is None:
            # Convert to correlation: transpose to (look_back, N_assets) for correlation
            adj, _ = CorrelationGraphBuilder.build_correlation_graph(hist_2d.T, threshold=0.3)
        else:
            adj = correlation_graphs[-1] if correlation_graphs else np.eye(self.n_assets)

        forecasts = []

        # Get GAT features
        gat_out, attention_weights = self.gat.forward(hist_2d, adj)

        # Forecast using attention-weighted combinations
        for step in range(steps_ahead):
            # Simple autoregressive step
            forecast = np.mean(gat_out, axis=0)
            forecasts.append(forecast)

            # Update for next step (in practice would use recurrent mechanism)
            if step < steps_ahead - 1:
                gat_out = gat_out[1:] + forecast.reshape(1, -1)

        return np.array(forecasts)


class PortfolioRiskViaAttention:
    """Calculate portfolio risk using graph attention mechanism"""

    @staticmethod
    def attention_adjusted_volatility(
        weights: np.ndarray,
        volatilities: np.ndarray,
        attention_matrix: np.ndarray
    ) -> float:
        """
        Calculate portfolio volatility with attention-based correlation weighting

        Args:
            weights: Portfolio weights
            volatilities: Asset volatilities
            attention_matrix: Graph attention weights (N_assets, N_assets)

        Returns:
            Portfolio volatility
        """
        # Attention-weighted correlation matrix
        N = len(weights)
        corr_matrix = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                # Base correlation from attention
                corr_matrix[i, j] = attention_matrix[i, j]

        # Diagonal elements are 1
        np.fill_diagonal(corr_matrix, 1.0)

        # Cov = diag(σ) @ ρ @ diag(σ)
        vol_diag = np.diag(volatilities)
        cov_matrix = vol_diag @ corr_matrix @ vol_diag

        # Portfolio variance
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))

        return float(np.sqrt(portfolio_variance))

    @staticmethod
    def risk_contribution(
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        attention_weights: np.ndarray
    ) -> np.ndarray:
        """
        Calculate risk contribution of each asset with attention adjustment

        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix
            attention_weights: Attention scores

        Returns:
            Risk contribution (N_assets,)
        """
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

        # Marginal contribution to risk
        mcr = np.dot(cov_matrix, weights) / portfolio_vol

        # Attention-adjusted contribution
        risk_contrib = weights * mcr

        # Weight by attention
        attention_scores = np.sum(attention_weights, axis=1)
        attention_normalized = attention_scores / np.sum(attention_scores)

        risk_contrib_adjusted = risk_contrib * attention_normalized

        return risk_contrib_adjusted


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Synthetic multi-asset data
    n_assets = 10
    n_samples = 100
    n_features = 5

    returns = np.random.randn(n_samples, n_assets) * 0.02 + 0.001

    # Build correlation graph
    adj, corr_mat = CorrelationGraphBuilder.build_correlation_graph(returns, threshold=0.3)

    logger.info(f"Graph sparsity: {np.sum(adj) / (n_assets ** 2 - n_assets) * 100:.1f}% edges")

    # Initialize GAT
    features = np.random.randn(n_assets, n_features)
    gat = GraphAttentionNetwork(
        input_dim=n_features,
        hidden_dim=64,
        output_dim=32,
        n_heads=8,
        n_layers=2
    )

    # Forward pass
    output, attention_weights = gat.forward(features, adj)

    logger.info(f"GAT output shape: {output.shape}")
    logger.info(f"Attention matrix shape: {attention_weights[0].shape}")

    # Price prediction
    prices = gat.predict_prices(features, adj)
    logger.info(f"Predicted prices: {prices[:3]}")

    # Temporal forecasting
    returns_3d = np.random.randn(50, n_assets, 20) * 0.02 + 0.001
    temporal_gat = TemporalGraphAttention(n_assets=n_assets, look_back=20)
    forecast = temporal_gat.forecast(returns_3d, steps_ahead=5)

    logger.info(f"Forecast shape: {forecast.shape}")
    logger.info(f"Mean forecast: {np.mean(forecast):.6f}")

    # Risk calculation
    weights = np.ones(n_assets) / n_assets
    volatilities = np.std(returns, axis=0)
    attention_matrix = attention_weights[0]

    portfolio_vol = PortfolioRiskViaAttention.attention_adjusted_volatility(
        weights, volatilities, attention_matrix
    )

    logger.info(f"Portfolio volatility: {portfolio_vol:.6f}")

    logger.info("Graph Attention Network Complete")
