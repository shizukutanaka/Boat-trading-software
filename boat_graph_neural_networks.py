#!/usr/bin/env python3
"""
Graph Neural Networks for Market Topology and Asset Interdependency
=====================================================================

Advanced market structure modeling using graph neural networks:
  - Dynamic correlation-based asset graphs
  - Temporal graph convolutions
  - Node embedding for asset representation
  - Market leadership detection
  - Volatility clustering analysis
  - Portfolio risk propagation modeling
  - Spillover and contagion analysis

Based on 2025 research on GNNs in financial markets and network analysis.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AssetGraph:
    """Build and maintain dynamic asset correlation graphs"""

    def __init__(self, symbols: List[str], correlation_threshold: float = 0.3):
        self.symbols = symbols
        self.correlation_threshold = correlation_threshold
        self.nodes = {s: {'id': i, 'symbol': s} for i, s in enumerate(symbols)}
        self.edges: Dict[Tuple[str, str], float] = {}
        self.correlation_matrix = None
        self.timestamp = None

    def build_from_returns(self, returns_df: pd.DataFrame) -> None:
        """
        Build graph from asset returns correlation

        Args:
            returns_df: DataFrame with asset returns (T, n_assets)
        """
        # Calculate correlation matrix
        self.correlation_matrix = returns_df.corr()
        self.timestamp = datetime.utcnow()

        # Create edges based on correlation threshold
        self.edges = {}

        for i, sym1 in enumerate(self.symbols):
            for j, sym2 in enumerate(self.symbols):
                if i < j:  # Avoid duplicates
                    corr = abs(self.correlation_matrix.loc[sym1, sym2])

                    if corr >= self.correlation_threshold:
                        edge_key = (sym1, sym2)
                        self.edges[edge_key] = corr

    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get adjacency matrix representation

        Returns:
            n x n adjacency matrix
        """
        n = len(self.symbols)
        adj_matrix = np.zeros((n, n))

        for (sym1, sym2), weight in self.edges.items():
            i, j = self.nodes[sym1]['id'], self.nodes[sym2]['id']
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight  # Symmetric

        return adj_matrix

    def get_node_degree(self) -> Dict[str, int]:
        """
        Calculate node degree (number of connections)

        Returns:
            Dictionary of symbol -> degree
        """
        degree = {s: 0 for s in self.symbols}

        for (sym1, sym2) in self.edges.keys():
            degree[sym1] += 1
            degree[sym2] += 1

        return degree


@dataclass
class GraphEmbedding:
    """Node embedding in latent space"""
    symbol: str
    embedding: np.ndarray  # d-dimensional
    centrality: float
    betweenness: float
    clustering_coefficient: float
    influence_score: float


class GraphConvolutionalNetwork:
    """Simple GCN for asset embedding learning"""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 16, embedding_dim: int = 8):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Weight matrices (in practice, would be learned via backprop)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W2 = np.random.randn(hidden_dim, embedding_dim) * 0.01
        self.embeddings = {}

    def forward(
        self,
        node_features: np.ndarray,
        adjacency_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass through GCN layers

        Args:
            node_features: (n_nodes, input_dim) feature matrix
            adjacency_matrix: (n_nodes, n_nodes) adjacency matrix

        Returns:
            (n_nodes, embedding_dim) embeddings
        """
        # Normalize adjacency matrix
        deg = np.sum(adjacency_matrix, axis=1)
        deg_inv = np.power(deg, -0.5)
        deg_inv[np.isinf(deg_inv)] = 0
        D_inv_sqrt = np.diag(deg_inv)

        # Normalized adjacency: D^{-1/2} A D^{-1/2}
        A_norm = D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt

        # First GCN layer: A * X * W1
        h1 = (A_norm @ node_features) @ self.W1
        h1 = np.maximum(h1, 0)  # ReLU activation

        # Second GCN layer: A * h1 * W2
        h2 = (A_norm @ h1) @ self.W2

        # Normalize embeddings
        h2 = h2 / (np.linalg.norm(h2, axis=1, keepdims=True) + 1e-8)

        return h2


class VolatilityClusteringAnalyzer:
    """Analyze volatility clustering in asset networks"""

    @staticmethod
    def calculate_volatility_spillover(
        returns_df: pd.DataFrame,
        window: int = 20
    ) -> Dict[str, float]:
        """
        Calculate spillover effects between assets

        Args:
            returns_df: Asset returns (T, n_assets)
            window: Rolling window for volatility

        Returns:
            Spillover indices by symbol
        """
        volatilities = returns_df.rolling(window).std()

        spillover = {}

        for symbol in returns_df.columns:
            # Correlation of volatility with market volatility
            market_vol = volatilities.mean(axis=1)
            symbol_vol = volatilities[symbol]

            # Spillover as correlation of volatilities
            spill = np.corrcoef(symbol_vol.dropna(),
                               market_vol.loc[symbol_vol.index].dropna())[0, 1]

            spillover[symbol] = float(np.nan_to_num(spill, nan=0.0))

        return spillover

    @staticmethod
    def detect_volatility_clusters(
        returns_df: pd.DataFrame,
        threshold: float = 0.6
    ) -> List[List[str]]:
        """
        Detect clusters of highly correlated volatility

        Args:
            returns_df: Asset returns
            threshold: Correlation threshold for clustering

        Returns:
            List of asset clusters
        """
        # Calculate rolling volatility correlation
        volatilities = returns_df.rolling(20).std()
        vol_corr = volatilities.corr()

        # Simple clustering: group assets with high correlation
        clusters = []
        assigned = set()

        for col in vol_corr.columns:
            if col in assigned:
                continue

            cluster = [col]
            assigned.add(col)

            # Find all highly correlated assets
            for other_col in vol_corr.columns:
                if other_col not in assigned and vol_corr.loc[col, other_col] >= threshold:
                    cluster.append(other_col)
                    assigned.add(other_col)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters


class PortfolioRiskPropagation:
    """Model risk propagation through asset networks"""

    def __init__(self, graph: AssetGraph):
        self.graph = graph
        self.adj_matrix = graph.get_adjacency_matrix()

    def calculate_risk_contribution(
        self,
        position_weights: Dict[str, float],
        marginal_var: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate how position risk propagates through network

        Args:
            position_weights: Weight of each asset in portfolio
            marginal_var: Marginal VaR contribution of each asset

        Returns:
            Network-adjusted risk contribution
        """
        weights_vec = np.array([position_weights.get(s, 0.0) for s in self.graph.symbols])
        var_vec = np.array([marginal_var.get(s, 0.0) for s in self.graph.symbols])

        # Direct risk + network spillover
        direct_risk = weights_vec * var_vec

        # Network spillover: A @ (weights * var)
        spillover_risk = self.adj_matrix @ (weights_vec * var_vec)

        # Total risk = direct + spillover factor
        total_risk = direct_risk + 0.3 * spillover_risk

        return {s: float(total_risk[i]) for i, s in enumerate(self.graph.symbols)}

    def get_systemic_risk_score(self) -> float:
        """
        Calculate systemic risk in the network

        Returns:
            Score 0-1, higher = more systemic risk
        """
        # Systemic risk = average node degree / max possible
        degree_dict = self.graph.get_node_degree()
        avg_degree = np.mean(list(degree_dict.values()))
        max_degree = len(self.graph.symbols) - 1

        systemic_score = avg_degree / max_degree if max_degree > 0 else 0.0

        return float(systemic_score)


class ContagionAnalyzer:
    """Analyze contagion risk in financial networks"""

    def __init__(self, graph: AssetGraph, contagion_threshold: float = 0.7):
        self.graph = graph
        self.contagion_threshold = contagion_threshold
        self.adj_matrix = graph.get_adjacency_matrix()

    def simulate_shock_propagation(
        self,
        shocked_asset: str,
        shock_magnitude: float = 0.1,
        steps: int = 5
    ) -> Dict[str, float]:
        """
        Simulate how a shock to one asset propagates through network

        Args:
            shocked_asset: Asset receiving initial shock
            shock_magnitude: Size of initial shock (e.g., 10% price drop)
            steps: Number of propagation steps

        Returns:
            Final shock impact on each asset
        """
        # Initialize shock vector
        n = len(self.graph.symbols)
        shock = np.zeros(n)

        # Apply initial shock
        asset_idx = self.graph.nodes[shocked_asset]['id']
        shock[asset_idx] = shock_magnitude

        # Propagate through network
        for _ in range(steps):
            # Shock propagates through connections
            new_shock = self.adj_matrix @ shock

            # Dampen over time (not all shock propagates)
            shock = 0.6 * new_shock + 0.4 * shock

        return {s: float(shock[self.graph.nodes[s]['id']])
                for s in self.graph.symbols}

    def identify_contagion_hubs(self) -> List[Tuple[str, float]]:
        """
        Identify assets most likely to transmit contagion

        Returns:
            List of (symbol, contagion_potential)
        """
        # Contagion potential = weighted out-degree in correlation network
        degree_dict = self.graph.get_node_degree()

        # Weighted by correlation strength
        contagion_scores = {}

        for sym in self.graph.symbols:
            neighbors = [
                (edge[1] if edge[0] == sym else edge[0], weight)
                for edge, weight in self.graph.edges.items()
                if sym in edge
            ]

            # Contagion score = weighted sum of edge weights
            score = sum(weight for _, weight in neighbors)
            contagion_scores[sym] = score

        # Normalize
        max_score = max(contagion_scores.values()) if contagion_scores else 1.0

        return sorted(
            [(sym, score / (max_score + 1e-8)) for sym, score in contagion_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate sample return data
    n_days = 252
    symbols = ['TECH', 'FINANCE', 'HEALTHCARE', 'ENERGY', 'UTILITIES']

    # Create correlated returns
    base_returns = np.random.randn(n_days, 3)
    returns_data = {
        'TECH': base_returns[:, 0] * 0.02,
        'FINANCE': base_returns[:, 0] * 0.015 + base_returns[:, 1] * 0.01,
        'HEALTHCARE': base_returns[:, 1] * 0.02,
        'ENERGY': base_returns[:, 2] * 0.025,
        'UTILITIES': base_returns[:, 2] * 0.015
    }

    returns_df = pd.DataFrame(returns_data)

    # Build asset graph
    graph = AssetGraph(symbols, correlation_threshold=0.3)
    graph.build_from_returns(returns_df)

    logger.info("Asset Graph Built")
    logger.info(f"Edges: {len(graph.edges)}")
    logger.info(f"Node degrees: {graph.get_node_degree()}")

    # GCN embeddings
    node_features = np.eye(len(symbols))
    gcn = GraphConvolutionalNetwork(input_dim=len(symbols), embedding_dim=8)
    embeddings = gcn.forward(node_features, graph.get_adjacency_matrix())

    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Volatility clustering
    vol_analyzer = VolatilityClusteringAnalyzer()
    spillover = vol_analyzer.calculate_volatility_spillover(returns_df)
    clusters = vol_analyzer.detect_volatility_clusters(returns_df)

    logger.info(f"Volatility spillover: {spillover}")
    logger.info(f"Volatility clusters: {clusters}")

    # Risk propagation
    risk_prop = PortfolioRiskPropagation(graph)
    weights = {s: 0.2 for s in symbols}
    marginal_var = {s: 0.1 for s in symbols}

    risk_contrib = risk_prop.calculate_risk_contribution(weights, marginal_var)
    systemic_risk = risk_prop.get_systemic_risk_score()

    logger.info(f"Risk contribution: {risk_contrib}")
    logger.info(f"Systemic risk score: {systemic_risk:.4f}")

    # Contagion analysis
    contagion = ContagionAnalyzer(graph)
    shock_impact = contagion.simulate_shock_propagation('TECH', shock_magnitude=0.1)
    hubs = contagion.identify_contagion_hubs()

    logger.info(f"Shock propagation from TECH: {shock_impact}")
    logger.info(f"Contagion hubs: {hubs}")
