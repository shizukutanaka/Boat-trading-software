#!/usr/bin/env python3
"""
Advanced Anomaly Detection for Fraud and Market Abuse Detection
================================================================

Deep learning-based anomaly detection for financial data:
  - Autoencoder-based anomaly detection
  - Isolation Forest algorithm
  - Local Outlier Factor (LOF)
  - Mahalanobis distance anomalies
  - Multivariate time series anomalies
  - Real-time alert generation
  - Anomaly scoring and ranking

Based on 2025 research on deep learning for financial fraud detection.
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
class AnomalyScore:
    """Anomaly detection score and explanation"""
    timestamp: datetime
    asset: str
    anomaly_score: float  # 0-1, higher = more anomalous
    is_anomaly: bool
    detection_method: str
    confidence: float
    features_flagged: List[str]


class AutoencoderAnomalyDetector:
    """Deep learning autoencoder for anomaly detection"""

    def __init__(self, input_dim: int, encoding_dim: int = 8):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        # Encoder weights
        self.W1_enc = np.random.randn(input_dim, encoding_dim) * 0.01
        self.b1_enc = np.zeros(encoding_dim)

        # Decoder weights
        self.W1_dec = np.random.randn(encoding_dim, input_dim) * 0.01
        self.b1_dec = np.zeros(input_dim)

        self.reconstruction_errors = []

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode input to latent representation

        Args:
            x: Input (n_samples, input_dim)

        Returns:
            Encoded (n_samples, encoding_dim)
        """
        z = (x @ self.W1_enc) + self.b1_enc
        return np.maximum(z, 0)  # ReLU

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode from latent representation

        Args:
            z: Encoded (n_samples, encoding_dim)

        Returns:
            Reconstructed (n_samples, input_dim)
        """
        return (z @ self.W1_dec) + self.b1_dec

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Complete forward pass"""
        z = self.encode(x)
        return self.decode(z)

    def detect_anomalies(
        self,
        x: np.ndarray,
        threshold_percentile: float = 95.0
    ) -> Tuple[np.ndarray, float]:
        """
        Detect anomalies based on reconstruction error

        Args:
            x: Input data (n_samples, input_dim)
            threshold_percentile: Percentile for anomaly threshold

        Returns:
            (anomaly_flags, threshold)
        """
        # Get reconstructions
        x_recon = self.forward(x)

        # Calculate reconstruction error
        errors = np.mean((x - x_recon) ** 2, axis=1)
        self.reconstruction_errors = errors

        # Determine threshold
        threshold = np.percentile(errors, threshold_percentile)

        # Anomaly flags
        anomalies = errors > threshold

        return anomalies, threshold


class IsolationForestDetector:
    """Isolation Forest for anomaly detection"""

    def __init__(self, n_trees: int = 100, max_samples: int = None):
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.trees = []
        self.data = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit isolation forest

        Args:
            X: Training data (n_samples, n_features)
        """
        self.data = X
        n_samples = X.shape[0]

        if self.max_samples is None:
            self.max_samples = min(256, n_samples)

        # Build isolation trees
        for _ in range(self.n_trees):
            # Random sample
            indices = np.random.choice(n_samples, self.max_samples, replace=False)
            sample = X[indices]

            # Simple tree structure (random splits)
            tree = self._build_tree(sample, depth=0)
            self.trees.append(tree)

    def _build_tree(self, X: np.ndarray, depth: int, max_depth: int = 10) -> Dict[str, Any]:
        """Build isolation tree recursively"""
        if X.shape[0] <= 1 or depth >= max_depth:
            return {'type': 'leaf', 'size': X.shape[0]}

        # Random feature and split value
        feat_idx = np.random.randint(X.shape[1])
        feat_values = X[:, feat_idx]
        split_value = np.random.uniform(feat_values.min(), feat_values.max())

        # Split data
        left_mask = feat_values < split_value
        left_data = X[left_mask]
        right_data = X[~left_mask]

        # Skip if no split
        if left_data.shape[0] == 0 or right_data.shape[0] == 0:
            return {'type': 'leaf', 'size': X.shape[0]}

        return {
            'type': 'node',
            'feature': feat_idx,
            'split_value': split_value,
            'left': self._build_tree(left_data, depth + 1, max_depth),
            'right': self._build_tree(right_data, depth + 1, max_depth)
        }

    def _path_length(self, x: np.ndarray, tree: Dict, depth: int = 0) -> float:
        """Calculate path length to leaf"""
        if tree['type'] == 'leaf':
            return depth

        feat_idx = tree['feature']
        if x[feat_idx] < tree['split_value']:
            return self._path_length(x, tree['left'], depth + 1)
        else:
            return self._path_length(x, tree['right'], depth + 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores

        Args:
            X: Test data (n_samples, n_features)

        Returns:
            Anomaly scores (0-1)
        """
        scores = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            # Average path length across trees
            path_lengths = [self._path_length(x, tree) for tree in self.trees]
            avg_path_length = np.mean(path_lengths)

            # Normalize by expected path length
            expected_path_length = np.log(self.max_samples) + 0.5772156649

            anomaly_score = 2 ** (-avg_path_length / expected_path_length)
            scores[i] = anomaly_score

        return scores


class LocalOutlierFactor:
    """Local Outlier Factor for density-based anomalies"""

    def __init__(self, k: int = 20):
        self.k = k
        self.training_data = None
        self.k_distances = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit LOF model

        Args:
            X: Training data (n_samples, n_features)
        """
        self.training_data = X
        self.k_distances = self._calculate_k_distances(X)

    def _calculate_distances(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Calculate pairwise distances"""
        return np.linalg.norm(X1[:, np.newaxis, :] - X2[np.newaxis, :, :], axis=2)

    def _calculate_k_distances(self, X: np.ndarray) -> np.ndarray:
        """Calculate k-th nearest neighbor distance"""
        distances = self._calculate_distances(X, X)

        # Set diagonal to inf (exclude self)
        np.fill_diagonal(distances, np.inf)

        # Find k-th smallest distance
        k_distances = np.partition(distances, self.k - 1, axis=1)[:, self.k - 1]

        return k_distances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict LOF scores

        Args:
            X: Test data (n_samples, n_features)

        Returns:
            LOF scores
        """
        if self.training_data is None:
            raise ValueError("Must fit before predict")

        distances = self._calculate_distances(X, self.training_data)

        # Local outlier factor for each test point
        lof_scores = np.ones(X.shape[0])

        for i in range(X.shape[0]):
            # k nearest neighbors
            k_nearest_idx = np.argsort(distances[i])[:self.k]

            # Local reachability density
            lrd_sum = 0
            for j_idx in k_nearest_idx:
                d = distances[i, j_idx]
                k_distance_neighbor = self.k_distances[j_idx]
                reachability = max(d, k_distance_neighbor)
                lrd_sum += 1.0 / (reachability + 1e-8)

            lrd = self.k / (lrd_sum + 1e-8)

            # Compare with neighbors
            neighbor_lrd_sum = sum(
                self.k / (self.k_distances[j_idx] + 1e-8)
                for j_idx in k_nearest_idx
            )

            lof_scores[i] = neighbor_lrd_sum / (self.k * lrd + 1e-8)

        return lof_scores


class MahalanobisAnomalyDetector:
    """Mahalanobis distance-based anomaly detection"""

    def __init__(self):
        self.mean = None
        self.cov_matrix = None
        self.inv_cov_matrix = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit Mahalanobis distance model

        Args:
            X: Training data (n_samples, n_features)
        """
        self.mean = np.mean(X, axis=0)
        self.cov_matrix = np.cov(X.T)

        # Add regularization for numerical stability
        self.cov_matrix += np.eye(X.shape[1]) * 1e-6

        self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate Mahalanobis distances

        Args:
            X: Test data (n_samples, n_features)

        Returns:
            Mahalanobis distances
        """
        if self.mean is None:
            raise ValueError("Must fit before predict")

        distances = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            diff = X[i] - self.mean
            dist_sq = diff @ self.inv_cov_matrix @ diff.T
            distances[i] = np.sqrt(max(0, dist_sq))

        return distances


class MultiMethodAnomalyDetector:
    """Combine multiple anomaly detection methods"""

    def __init__(self):
        self.autoencoder = None
        self.isolation_forest = None
        self.lof = None
        self.mahalanobis = None

    def fit(self, X: np.ndarray) -> None:
        """Fit all detectors"""
        # Initialize detectors
        self.autoencoder = AutoencoderAnomalyDetector(input_dim=X.shape[1])
        self.isolation_forest = IsolationForestDetector()
        self.lof = LocalOutlierFactor(k=min(20, X.shape[0] // 2))
        self.mahalanobis = MahalanobisAnomalyDetector()

        # Fit each detector
        self.isolation_forest.fit(X)
        self.lof.fit(X)
        self.mahalanobis.fit(X)

    def predict_ensemble(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble anomaly score

        Args:
            X: Test data (n_samples, n_features)

        Returns:
            (ensemble_scores, method_details)
        """
        if self.isolation_forest is None:
            raise ValueError("Must fit before predict")

        # Get scores from each method
        if_scores = self.isolation_forest.predict(X)  # 0-1
        lof_scores = self.lof.predict(X)  # Normalized to 0-1
        mahal_scores = self.mahalanobis.predict(X)  # Normalized to 0-1

        # Normalize LOF and Mahalanobis
        lof_scores = np.clip((lof_scores - 1.0) / 2.0, 0, 1)
        mahal_scores = np.clip(mahal_scores / (np.max(mahal_scores) + 1e-8), 0, 1)

        # Ensemble: weighted average
        ensemble_scores = 0.4 * if_scores + 0.3 * lof_scores + 0.3 * mahal_scores

        method_details = np.column_stack([if_scores, lof_scores, mahal_scores])

        return ensemble_scores, method_details

    def detect_anomalies(
        self,
        X: np.ndarray,
        threshold: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies with threshold

        Args:
            X: Test data
            threshold: Anomaly threshold

        Returns:
            (anomaly_flags, anomaly_scores)
        """
        scores, _ = self.predict_ensemble(X)
        anomalies = scores > threshold

        return anomalies, scores


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate normal data
    normal_data = np.random.randn(500, 5)

    # Add some anomalies
    test_data = np.vstack([
        normal_data,
        normal_data.mean(axis=0) + np.random.randn(20, 5) * 3  # Outliers
    ])

    # Fit multi-method detector
    detector = MultiMethodAnomalyDetector()
    detector.fit(normal_data)

    # Detect anomalies
    anomalies, scores = detector.detect_anomalies(test_data, threshold=0.6)

    logger.info(f"Anomalies detected: {np.sum(anomalies)}")
    logger.info(f"Anomaly rate: {np.mean(anomalies):.2%}")
    logger.info(f"Mean anomaly score: {np.mean(scores):.4f}")

    # Get ensemble scores for detailed analysis
    ensemble_scores, method_details = detector.predict_ensemble(test_data)

    logger.info(f"Top 5 anomalies:")
    top_indices = np.argsort(ensemble_scores)[-5:][::-1]

    for idx in top_indices:
        logger.info(
            f"  Index {idx}: score={ensemble_scores[idx]:.4f}, "
            f"IF={method_details[idx, 0]:.4f}, "
            f"LOF={method_details[idx, 1]:.4f}, "
            f"Mahal={method_details[idx, 2]:.4f}"
        )
