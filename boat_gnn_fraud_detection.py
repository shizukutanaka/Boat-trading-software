#!/usr/bin/env python3
"""
Graph Neural Networks for Financial Fraud Detection
=====================================================

GNN-based fraud detection in financial transaction networks:
  - Transaction graph construction with metapaths
  - Graph message passing for relationship modeling
  - Anomaly detection via embedding space analysis
  - Fraud pattern learning from historical data
  - Real-time suspicious activity flagging

Based on 2025 research (arXiv:2411.05815, Metapath-GNN, NVIDIA AI Blueprint).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    """Financial transaction"""
    transaction_id: str
    sender: str
    receiver: str
    amount: float
    timestamp: int
    transaction_type: str  # 'transfer', 'payment', 'withdrawal'


@dataclass
class Node:
    """Graph node (account/entity)"""
    node_id: str
    node_type: str  # 'account', 'merchant', 'device'
    features: np.ndarray  # Feature vector
    embedding: Optional[np.ndarray] = None
    fraud_risk: float = 0.0


@dataclass
class Edge:
    """Graph edge (transaction/relationship)"""
    source_id: str
    target_id: str
    edge_type: str  # 'transfer', 'shared_device', 'geographic_proximity'
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)


class GraphLayer:
    """Graph neural network layer"""

    def __init__(self, input_dim: int, output_dim: int, n_heads: int = 4):
        """Initialize graph layer"""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads

        # Attention weights
        self.W_query = np.random.randn(input_dim, output_dim) * 0.01
        self.W_key = np.random.randn(input_dim, output_dim) * 0.01
        self.W_value = np.random.randn(input_dim, output_dim) * 0.01

    def forward(self, node_features: np.ndarray, adjacency: Dict[str, List[str]]) -> np.ndarray:
        """
        Graph neural network forward pass

        Args:
            node_features: (n_nodes, input_dim) features
            adjacency: Adjacency dict for message passing

        Returns:
            (n_nodes, output_dim) aggregated features
        """
        n_nodes = len(node_features)
        output = np.zeros((n_nodes, self.output_dim))

        for i, (node_id, neighbors) in enumerate(adjacency.items()):
            if i >= len(node_features):
                break

            # Query for this node
            query = node_features[i] @ self.W_query

            # Aggregate from neighbors
            if neighbors:
                for neighbor_id in neighbors:
                    # Find neighbor index
                    for j, (nid, _) in enumerate(adjacency.items()):
                        if nid == neighbor_id:
                            key = node_features[j] @ self.W_key
                            value = node_features[j] @ self.W_value

                            # Attention score
                            score = np.dot(query, key) / np.sqrt(self.output_dim)
                            attention = 1.0 / (1.0 + np.exp(-score))  # Sigmoid

                            # Add to output
                            output[i] += attention * value
                            break

        return output


class TransactionGraph:
    """Financial transaction graph"""

    def __init__(self):
        """Initialize transaction graph"""
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        self.node_index: Dict[str, int] = {}

    def add_node(self, node_id: str, node_type: str, features: np.ndarray = None):
        """Add node to graph"""
        if features is None:
            features = np.random.randn(16)  # Default 16-dim features

        node = Node(node_id=node_id, node_type=node_type, features=features)
        self.nodes[node_id] = node
        self.node_index[node_id] = len(self.node_index)

    def add_edge(self, source_id: str, target_id: str, edge_type: str, weight: float = 1.0):
        """Add edge to graph"""
        edge = Edge(source_id=source_id, target_id=target_id, edge_type=edge_type, weight=weight)
        self.edges.append(edge)
        self.adjacency[source_id].append(target_id)

    def get_node_features(self) -> np.ndarray:
        """Get all node features as matrix"""
        feature_list = []
        for node_id in sorted(self.nodes.keys()):
            feature_list.append(self.nodes[node_id].features)

        return np.array(feature_list)

    def compute_local_density(self) -> Dict[str, float]:
        """Compute local density for anomaly detection"""
        density = {}

        for node_id, neighbors in self.adjacency.items():
            if node_id not in self.nodes:
                continue

            # Count connections and sum weights
            n_connections = len(neighbors)
            total_weight = 0.0

            for edge in self.edges:
                if edge.source_id == node_id:
                    total_weight += edge.weight

            # Local density = weighted connectivity
            density[node_id] = total_weight / max(1, n_connections)

        return density

    def detect_metapath_patterns(self, path_type: str = "account-merchant-account") -> List[Tuple[str, float]]:
        """
        Detect metapath patterns for fraud

        Args:
            path_type: Type of metapath to detect

        Returns:
            List of (node_id, anomaly_score)
        """
        patterns = []

        # Simple 2-hop pattern detection
        for node_id in self.nodes:
            if node_id not in self.adjacency:
                continue

            # Count 2-hop paths
            two_hop_count = 0
            for neighbor in self.adjacency[node_id]:
                if neighbor in self.adjacency:
                    two_hop_count += len(self.adjacency[neighbor])

            # Anomaly if unusually high 2-hop connectivity
            anomaly_score = min(1.0, two_hop_count / 100.0)
            patterns.append((node_id, anomaly_score))

        return sorted(patterns, key=lambda x: x[1], reverse=True)


class GraphNeuralNetworkFraudDetector:
    """GNN-based fraud detection"""

    def __init__(self, embedding_dim: int = 32, n_layers: int = 2):
        """Initialize GNN fraud detector"""
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.gnn_layers = [GraphLayer(16 if i == 0 else embedding_dim, embedding_dim)
                           for i in range(n_layers)]

    def compute_node_embeddings(self, graph: TransactionGraph) -> Dict[str, np.ndarray]:
        """
        Compute node embeddings via GNN

        Args:
            graph: Transaction graph

        Returns:
            Dict of node embeddings
        """
        # Get initial features
        node_features = graph.get_node_features()  # (n_nodes, 16)

        # Pass through GNN layers
        for layer in self.gnn_layers:
            node_features = layer.forward(node_features, graph.adjacency)
            node_features = np.maximum(node_features, 0)  # ReLU

        # Store embeddings
        embeddings = {}
        node_ids = sorted(graph.nodes.keys())
        for i, node_id in enumerate(node_ids):
            if i < len(node_features):
                embeddings[node_id] = node_features[i]
                graph.nodes[node_id].embedding = node_features[i]

        return embeddings

    def detect_fraud(self, graph: TransactionGraph, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Detect fraudulent accounts/transactions

        Args:
            graph: Transaction graph
            threshold: Fraud score threshold

        Returns:
            List of (node_id, fraud_score)
        """
        # Compute embeddings
        embeddings = self.compute_node_embeddings(graph)

        # Compute local density
        density = graph.compute_local_density()

        # Compute fraud scores
        fraud_scores = []

        for node_id, embedding in embeddings.items():
            # Embedding-based anomaly (distance from center)
            center = np.mean(list(embeddings.values()), axis=0)
            distance = np.linalg.norm(embedding - center)

            # Density-based anomaly
            local_density = density.get(node_id, 0.0)
            density_anomaly = 1.0 / (1.0 + local_density)

            # Combined fraud score
            fraud_score = (distance * 0.5 + density_anomaly * 0.5) / 2.0
            fraud_score = min(1.0, fraud_score)

            fraud_scores.append((node_id, float(fraud_score)))
            graph.nodes[node_id].fraud_risk = fraud_score

        # Filter by threshold
        suspicious = [(node_id, score) for node_id, score in fraud_scores
                      if score >= threshold]

        return sorted(suspicious, key=lambda x: x[1], reverse=True)

    def rank_suspicious_patterns(self, graph: TransactionGraph) -> List[Tuple[str, str, float]]:
        """
        Rank nodes by suspicious pattern types

        Args:
            graph: Transaction graph

        Returns:
            List of (node_id, pattern_type, score)
        """
        patterns = []

        # Metapath patterns
        metapath_patterns = graph.detect_metapath_patterns()
        for node_id, score in metapath_patterns[:5]:
            patterns.append((node_id, "metapath_anomaly", score))

        # Degree anomalies
        avg_degree = np.mean([len(neighbors) for neighbors in graph.adjacency.values()])
        for node_id, neighbors in graph.adjacency.items():
            degree_anomaly = abs(len(neighbors) - avg_degree) / max(1, avg_degree)
            if degree_anomaly > 1.0:
                patterns.append((node_id, "degree_anomaly", min(1.0, degree_anomaly / 5.0)))

        return sorted(patterns, key=lambda x: x[2], reverse=True)


if __name__ == "__main__":
    logger.info("Graph Neural Network Fraud Detection")
    logger.info("=" * 50)

    np.random.seed(42)

    # Build transaction graph
    logger.info("\nBuilding transaction graph")
    graph = TransactionGraph()

    # Add accounts
    accounts = [f"ACC_{i:03d}" for i in range(20)]
    for acc in accounts:
        features = np.random.randn(16)
        graph.add_node(acc, "account", features)

    logger.info(f"Added {len(accounts)} accounts")

    # Add transactions (edges)
    logger.info("Adding transactions")
    np.random.seed(42)
    for _ in range(50):
        sender = np.random.choice(accounts)
        receiver = np.random.choice(accounts)
        if sender != receiver:
            amount = np.random.exponential(1000)
            graph.add_edge(sender, receiver, "transfer", weight=amount)

    logger.info(f"Added {len(graph.edges)} transactions")

    # Initialize fraud detector
    detector = GraphNeuralNetworkFraudDetector(embedding_dim=32, n_layers=2)

    # Detect fraud
    logger.info("\nDetecting fraudulent accounts")
    suspicious_accounts = detector.detect_fraud(graph, threshold=0.5)

    logger.info("Suspicious Accounts (threshold=0.5):")
    for node_id, fraud_score in suspicious_accounts[:10]:
        logger.info(f"  {node_id}: {fraud_score:.4f}")

    # Rank patterns
    logger.info("\nRanking suspicious patterns")
    pattern_ranking = detector.rank_suspicious_patterns(graph)

    logger.info("Top Suspicious Patterns:")
    for node_id, pattern_type, score in pattern_ranking[:10]:
        logger.info(f"  {node_id} ({pattern_type}): {score:.4f}")

    # Graph metrics
    logger.info("\nGraph Metrics")
    density = graph.compute_local_density()
    avg_density = np.mean(list(density.values()))
    logger.info(f"  Average node density: {avg_density:.4f}")
    logger.info(f"  Total edges: {len(graph.edges)}")
    logger.info(f"  Total nodes: {len(graph.nodes)}")

    logger.info("\nFraud Detection Complete")
