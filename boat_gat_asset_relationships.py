#!/usr/bin/env python3
"""
Graph Attention Networks for Asset Relationships
=================================================

GAT for capturing inter-asset correlations and spatio-temporal dynamics:
  - Graph attention mechanisms for asset networks
  - Spatio-temporal feature fusion
  - Dynamic correlation estimation
  - Asset clustering and network analysis
  - Cross-asset price prediction

Based on 2025 research (STGAT, FSTGAT, GAT for finance).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AttentionOutput:
    """Graph attention output"""
    node_features: np.ndarray  # (n_nodes, feature_dim)
    attention_weights: np.ndarray  # (n_nodes, n_nodes)
    graph_embedding: np.ndarray  # Global graph representation


class GraphAttentionLayer:
    """Graph attention network layer"""

    def __init__(self, in_features: int, out_features: int, n_heads: int = 4):
        """Initialize GAT layer"""
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads

        # Attention parameters per head
        self.W_per_head = [np.random.randn(in_features, out_features) * 0.01
                           for _ in range(n_heads)]
        self.a_per_head = [np.random.randn(2 * out_features) * 0.01
                           for _ in range(n_heads)]

    def _attention_coefficient(self, features: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Compute attention coefficients"""
        n = len(features)
        attention = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                # Concatenate features
                feat_ij = np.concatenate([features[i], features[j]])
                # Attention score
                score = np.dot(a, feat_ij)
                attention[i, j] = score

        # Apply softmax
        attention = self._softmax(attention)
        return attention

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax normalization"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / (np.sum(e_x, axis=1, keepdims=True) + 1e-8)

    def forward(self, node_features: np.ndarray, adjacency: Optional[np.ndarray] = None) -> AttentionOutput:
        """
        GAT forward pass

        Args:
            node_features: (n_nodes, in_features)
            adjacency: (n_nodes, n_nodes) adjacency matrix (optional)

        Returns:
            AttentionOutput with aggregated features and attention
        """
        n_nodes = len(node_features)
        output_features_list = []
        attention_list = []

        # Multi-head attention
        for head in range(self.n_heads):
            # Transform features
            transformed = node_features @ self.W_per_head[head]  # (n_nodes, out_features)

            # Compute attention
            attention = self._attention_coefficient(transformed, self.a_per_head[head])

            # Mask with adjacency if provided
            if adjacency is not None:
                attention = attention * adjacency

            # Apply attention
            output = attention @ transformed  # (n_nodes, out_features)
            output_features_list.append(output)
            attention_list.append(attention)

        # Concatenate or average heads
        output_features = np.mean(output_features_list, axis=0)  # (n_nodes, out_features)
        attention_weights = np.mean(attention_list, axis=0)  # (n_nodes, n_nodes)

        # Global embedding
        graph_embedding = np.mean(output_features, axis=0)

        return AttentionOutput(
            node_features=output_features,
            attention_weights=attention_weights,
            graph_embedding=graph_embedding
        )


class AssetNetwork:
    """Asset correlation network"""

    def __init__(self, asset_names: List[str]):
        """Initialize asset network"""
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.adjacency = np.ones((self.n_assets, self.n_assets))  # Fully connected
        np.fill_diagonal(self.adjacency, 1)

    def compute_correlation_matrix(self, price_data: np.ndarray) -> np.ndarray:
        """
        Compute asset correlation matrix

        Args:
            price_data: (n_periods, n_assets) price data

        Returns:
            (n_assets, n_assets) correlation matrix
        """
        returns = np.diff(np.log(price_data), axis=0)
        correlation = np.corrcoef(returns.T)
        return correlation

    def update_adjacency_from_correlation(self, correlation: np.ndarray, threshold: float = 0.3):
        """Update adjacency based on correlation"""
        self.adjacency = (np.abs(correlation) > threshold).astype(float)
        np.fill_diagonal(self.adjacency, 1)


class SpatioTemporalGAT:
    """Spatio-temporal Graph Attention Network"""

    def __init__(self, asset_names: List[str], feature_dim: int = 32, n_heads: int = 4):
        """Initialize ST-GAT"""
        self.asset_network = AssetNetwork(asset_names)
        self.feature_dim = feature_dim
        self.n_heads = n_heads

        # Temporal CNN layers (simplified)
        self.temporal_conv_w = np.random.randn(3, 1, 16) * 0.01  # (kernel, 1, out_channels)

        # GAT layers
        self.gat_layer = GraphAttentionLayer(16, feature_dim, n_heads=n_heads)

    def temporal_convolution(self, time_series: np.ndarray) -> np.ndarray:
        """
        Apply temporal convolution (simplified)

        Args:
            time_series: (n_periods, n_assets)

        Returns:
            (n_assets, feature_dim) temporal features
        """
        n_periods, n_assets = time_series.shape
        features = np.zeros((n_assets, 16))

        for asset_idx in range(n_assets):
            ts = time_series[:, asset_idx]
            # Compute rolling statistics
            for i in range(len(ts) - 2):
                window = ts[i:i + 3]
                features[asset_idx, 0] = np.mean(ts[-10:])
                features[asset_idx, 1] = np.std(ts[-10:])
                features[asset_idx, 2] = np.max(window)
                features[asset_idx, 3] = np.min(window)

            # Add momentum and volatility
            features[asset_idx, 4:8] = np.gradient(ts[-4:])
            features[asset_idx, 8:16] = np.random.randn(8)  # Placeholder

        return features

    def forward(self, price_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through ST-GAT

        Args:
            price_data: (n_periods, n_assets) price time series

        Returns:
            (asset_embeddings, attention_weights, correlation_matrix)
        """
        # Compute temporal features
        temporal_features = self.temporal_convolution(price_data)

        # Compute correlation
        correlation = self.asset_network.compute_correlation_matrix(price_data)
        self.asset_network.update_adjacency_from_correlation(correlation)

        # Apply GAT
        gat_output = self.gat_layer.forward(temporal_features, self.asset_network.adjacency)

        return gat_output.node_features, gat_output.attention_weights, correlation

    def predict_asset_prices(self, current_prices: np.ndarray, embeddings: np.ndarray,
                            horizon: int = 1) -> np.ndarray:
        """
        Predict future asset prices using embeddings

        Args:
            current_prices: (n_assets,) current prices
            embeddings: (n_assets, feature_dim) asset embeddings
            horizon: prediction horizon

        Returns:
            (horizon, n_assets) predicted prices
        """
        predictions = np.zeros((horizon, len(current_prices)))

        for t in range(horizon):
            # Simple prediction: weighted average of embeddings
            price_change = embeddings @ np.random.randn(self.feature_dim) * 0.01
            current_prices = current_prices * (1 + price_change)
            predictions[t] = current_prices

        return predictions


class AssetClusteringGAT:
    """Asset clustering using GAT embeddings"""

    @staticmethod
    def cluster_assets(embeddings: np.ndarray, n_clusters: int = 3) -> Dict[int, List[int]]:
        """
        Cluster assets using embeddings (k-means style)

        Args:
            embeddings: (n_assets, embedding_dim)
            n_clusters: Number of clusters

        Returns:
            Cluster assignment dictionary
        """
        n_assets = len(embeddings)

        # Initialize cluster centers randomly
        center_indices = np.random.choice(n_assets, n_clusters, replace=False)
        centers = embeddings[center_indices]

        # K-means iterations
        for iteration in range(10):
            # Assign to nearest cluster
            distances = np.zeros((n_assets, n_clusters))
            for i in range(n_assets):
                for k in range(n_clusters):
                    distances[i, k] = np.linalg.norm(embeddings[i] - centers[k])

            assignments = np.argmin(distances, axis=1)

            # Update centers
            for k in range(n_clusters):
                cluster_members = embeddings[assignments == k]
                if len(cluster_members) > 0:
                    centers[k] = np.mean(cluster_members, axis=0)

        # Create cluster dictionary
        clusters = {}
        for k in range(n_clusters):
            clusters[k] = np.where(assignments == k)[0].tolist()

        return clusters


if __name__ == "__main__":
    logger.info("Graph Attention Networks for Asset Relationships")
    logger.info("=" * 50)

    np.random.seed(42)

    # Create asset network
    asset_names = [f"ASSET_{i:02d}" for i in range(10)]
    logger.info(f"\nAssets: {', '.join(asset_names[:5])}...")

    # Generate synthetic price data
    logger.info("\nGenerating price data")
    n_periods = 252
    n_assets = len(asset_names)

    # Correlated price data
    price_data = 100 + np.cumsum(
        np.random.randn(n_periods, n_assets) * 0.5 +
        np.random.randn(n_assets).reshape(1, -1) * 2,
        axis=0
    )

    # Initialize ST-GAT
    stgat = SpatioTemporalGAT(asset_names, feature_dim=32, n_heads=4)

    # Forward pass
    logger.info("\nRunning ST-GAT")
    embeddings, attention_weights, correlation = stgat.forward(price_data)

    logger.info(f"  Asset embeddings shape: {embeddings.shape}")
    logger.info(f"  Attention weights shape: {attention_weights.shape}")

    # Compute statistics
    avg_correlation = np.mean(np.abs(correlation[np.triu_indices_from(correlation, k=1)]))
    logger.info(f"  Average asset correlation: {avg_correlation:.4f}")

    # Top correlations
    logger.info("\nTop Asset Correlations:")
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            if abs(correlation[i, j]) > 0.5:
                logger.info(f"  {asset_names[i]} - {asset_names[j]}: {correlation[i, j]:.4f}")

    # Predict prices
    logger.info("\nPredicting future prices")
    current_prices = price_data[-1]
    predictions = stgat.predict_asset_prices(current_prices.copy(), embeddings, horizon=5)

    logger.info(f"  Predictions shape: {predictions.shape}")
    logger.info(f"  Asset_00 predicted returns (5 days): {(predictions[:, 0] / current_prices[0] - 1)[:5]}")

    # Clustering
    logger.info("\nClustering assets")
    clusters = AssetClusteringGAT.cluster_assets(embeddings, n_clusters=3)

    for cluster_id, asset_indices in clusters.items():
        asset_list = [asset_names[i] for i in asset_indices]
        logger.info(f"  Cluster {cluster_id}: {', '.join(asset_list)}")

    logger.info("\nGraph Attention Analysis Complete")
