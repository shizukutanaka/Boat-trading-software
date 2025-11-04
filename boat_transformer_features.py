#!/usr/bin/env python3
"""
Advanced Transformer-Based Feature Extraction for Financial Markets
====================================================================

Attention mechanisms and transformer models for feature engineering:
  - Multi-head attention feature extraction
  - Cross-asset dependency modeling
  - Temporal attention patterns
  - Feature importance via attention weights
  - Asset interdependency graphs
  - News impact quantification

Based on 2025 research on transformer models in asset pricing and feature selection.
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
class AttentionScore:
    """Attention-based feature importance"""
    feature_name: str
    attention_weight: float
    temporal_persistence: float
    cross_asset_relevance: float
    feature_importance_score: float


class TransformerFeatureExtractor:
    """Extract features using transformer attention mechanisms"""

    def __init__(self, n_heads: int = 8, d_model: int = 64):
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        # Attention weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01

        self.attention_weights_history = []

    def compute_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scaled dot-product attention

        Args:
            query: Query matrix (seq_len, d_model)
            key: Key matrix (seq_len, d_model)
            value: Value matrix (seq_len, d_model)

        Returns:
            (attended_values, attention_weights)
        """
        # Compute attention scores
        scores = (query @ key.T) / np.sqrt(self.head_dim)

        # Apply softmax
        attention_weights = self._softmax(scores, axis=-1)

        # Apply attention to values
        attended = attention_weights @ value

        return attended, attention_weights

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax function"""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def extract_features(
        self,
        price_data: np.ndarray,
        volume_data: np.ndarray,
        n_features: int = 10
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Extract transformer-based features

        Args:
            price_data: Historical prices (T, n_assets)
            volume_data: Trading volume (T, n_assets)
            n_features: Number of features to extract

        Returns:
            (features, attention_scores)
        """
        T, n_assets = price_data.shape

        # Calculate returns and volatility
        returns = np.diff(price_data, axis=0) / (price_data[:-1] + 1e-8)
        volatility = np.std(returns, axis=0, keepdims=True)

        # Normalize inputs
        returns_norm = returns / (volatility + 1e-8)
        volume_norm = (volume_data[1:] - np.mean(volume_data, axis=0)) / (np.std(volume_data, axis=0) + 1e-8)

        # Attention features
        features_list = []
        attention_dict = {}

        # Multi-head attention across time
        for t in range(20, T - 5):
            # Current window (20-day lookback)
            window = returns_norm[t - 20:t]

            # Query: current price movement
            query = returns_norm[t].reshape(1, -1)

            # Key: historical patterns
            key = window

            # Value: volume and volatility
            value = np.column_stack([volume_norm[t - 20:t], np.ones_like(window)])

            # Compute attention
            attended, att_weights = self.compute_attention(query, key, value)

            # Extract features from attention
            feat = {
                'attention_diversity': np.std(att_weights),
                'max_attention': np.max(att_weights),
                'entropy': self._calculate_entropy(att_weights),
                'volume_attention': np.sum(att_weights * volume_norm[t - 20:t]),
                'price_momentum': np.mean(returns_norm[t - 5:t]),
                'volatility_spike': np.max(np.std(window, axis=0)),
                'mean_reversion_signal': -np.mean(returns_norm[t - 5:t]),
                'cross_asset_corr': np.mean(np.abs(np.corrcoef(returns[t - 20:t].T))),
                'volume_momentum': np.mean(np.diff(volume_norm[t - 20:t])),
                'price_acceleration': np.mean(np.diff(returns_norm[t - 5:t]))
            }

            features_list.append(list(feat.values()))
            self.attention_weights_history.append(att_weights)

            # Track attention scores
            for i, (fname, fval) in enumerate(feat.items()):
                if fname not in attention_dict:
                    attention_dict[fname] = []
                attention_dict[fname].append(fval)

        # Average attention scores
        attention_scores = {
            fname: np.mean(fvals) for fname, fvals in attention_dict.items()
        }

        return np.array(features_list), attention_scores

    @staticmethod
    def _calculate_entropy(weights: np.ndarray) -> float:
        """Calculate entropy of attention distribution"""
        weights = np.clip(weights, 1e-8, 1.0)
        return float(-np.sum(weights * np.log(weights)))


class CrossAssetAttention:
    """Model dependencies between assets using attention"""

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.dependency_matrix = None
        self.attention_graph = {}

    def build_dependency_graph(
        self,
        returns_df: pd.DataFrame,
        window: int = 60
    ) -> np.ndarray:
        """
        Build cross-asset dependency matrix using attention

        Args:
            returns_df: Asset returns (T, n_assets)
            window: Rolling window size

        Returns:
            Dependency matrix (n_assets, n_assets)
        """
        n_assets = len(self.symbols)
        dependency = np.zeros((n_assets, n_assets))

        # For each asset, compute attention to other assets
        for i, sym in enumerate(self.symbols):
            target_returns = returns_df[sym].values[window:]

            for j, other_sym in enumerate(self.symbols):
                if i == j:
                    dependency[i, j] = 1.0
                    continue

                other_returns = returns_df[other_sym].values[window:]

                # Attention-based correlation
                # Higher when other asset predicts target
                correlation = np.corrcoef(other_returns[:-1], target_returns[1:])[0, 1]
                dependency[i, j] = np.clip(correlation, -1, 1)

        self.dependency_matrix = dependency
        return dependency

    def get_influential_assets(
        self,
        target_symbol: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get most influential assets for target

        Args:
            target_symbol: Target asset
            top_k: Number of top influences

        Returns:
            List of (symbol, influence_score)
        """
        if self.dependency_matrix is None:
            raise ValueError("Must build dependency graph first")

        idx = self.symbols.index(target_symbol)
        influences = self.dependency_matrix[idx]

        # Get top influences (excluding self)
        top_indices = np.argsort(np.abs(influences))[-top_k-1:-1]

        return [
            (self.symbols[i], float(influences[i]))
            for i in reversed(top_indices)
        ]


class FeatureImportanceFromAttention:
    """Calculate feature importance from attention mechanisms"""

    @staticmethod
    def aggregate_attention_importance(
        attention_weights_history: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Aggregate attention weights across time

        Args:
            attention_weights_history: List of attention matrices

        Returns:
            Feature importance scores
        """
        if not attention_weights_history:
            return {}

        # Average attention weights across time
        mean_attention = np.mean(attention_weights_history, axis=0)

        # Importance = average attention
        importance = {
            f'feature_{i}': float(mean_attention[i])
            for i in range(len(mean_attention))
        }

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def temporal_attention_analysis(
        attention_weights_history: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Analyze temporal patterns in attention

        Args:
            attention_weights_history: Attention history

        Returns:
            Temporal analysis results
        """
        if not attention_weights_history:
            return {}

        attention_array = np.array(attention_weights_history)

        # Temporal consistency
        consistency = np.zeros(attention_array.shape[1])

        for i in range(1, len(attention_array)):
            consistency += np.abs(attention_array[i] - attention_array[i-1])

        consistency = 1.0 / (1.0 + consistency / len(attention_array))

        # Persistence (how long attention focus is maintained)
        persistence = {}

        for i in range(attention_array.shape[1]):
            feature_attention = attention_array[:, i]
            above_mean = feature_attention > np.mean(feature_attention)

            # Calculate consecutive periods above mean
            runs = np.diff(np.concatenate([[0], above_mean.astype(int), [0]]))
            persistence[f'feature_{i}'] = float(np.mean(np.diff(np.where(runs == -1))))

        return {
            'consistency_scores': {f'feature_{i}': float(consistency[i]) for i in range(len(consistency))},
            'persistence_scores': persistence,
            'mean_attention_entropy': float(np.mean([
                -np.sum(w * np.log(np.clip(w, 1e-8, 1.0)))
                for w in attention_weights_history
            ]))
        }


class NewsImpactQuantifier:
    """Quantify impact of news events using attention"""

    @staticmethod
    def calculate_news_surprise_impact(
        price_before: float,
        price_after: float,
        expected_sentiment: float,
        actual_sentiment: float
    ) -> Dict[str, float]:
        """
        Calculate impact of unexpected news

        Args:
            price_before: Price before news
            price_after: Price after news
            expected_sentiment: Expected sentiment
            actual_sentiment: Actual sentiment

        Returns:
            Impact metrics
        """
        # Price impact
        price_impact = (price_after - price_before) / (price_before + 1e-8)

        # Sentiment surprise
        sentiment_surprise = actual_sentiment - expected_sentiment

        # Correlation of impact with surprise
        surprise_effectiveness = price_impact / (abs(sentiment_surprise) + 1e-8)

        return {
            'price_impact': float(price_impact),
            'sentiment_surprise': float(sentiment_surprise),
            'surprise_effectiveness': float(surprise_effectiveness),
            'attention_trigger_score': abs(float(price_impact * sentiment_surprise))
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate sample data
    n_days = 300
    n_assets = 5

    # Create correlated price data
    base_returns = np.random.randn(n_days, 1) * 0.01
    returns = np.column_stack([
        base_returns.squeeze() + np.random.randn(n_days) * 0.005
        for _ in range(n_assets)
    ])

    prices = np.exp(np.cumsum(returns, axis=0))
    volumes = np.random.exponential(1.0, (n_days, n_assets)) * 1000000

    returns_df = pd.DataFrame(returns, columns=[f'asset_{i}' for i in range(n_assets)])

    # Feature extraction
    extractor = TransformerFeatureExtractor(n_heads=4, d_model=32)
    features, att_scores = extractor.extract_features(prices, volumes)

    logger.info(f"Extracted features shape: {features.shape}")
    logger.info(f"Top attention scores:")
    for fname, score in sorted(att_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"  {fname}: {score:.4f}")

    # Cross-asset attention
    symbols = [f'asset_{i}' for i in range(n_assets)]
    cross_attn = CrossAssetAttention(symbols)
    dep_matrix = cross_attn.build_dependency_graph(returns_df)

    logger.info(f"\nDependency matrix:\n{dep_matrix}")

    # Influential assets
    influences = cross_attn.get_influential_assets('asset_0', top_k=3)

    logger.info(f"Most influential assets for asset_0:")
    for sym, score in influences:
        logger.info(f"  {sym}: {score:.4f}")

    # Temporal analysis
    temporal = FeatureImportanceFromAttention.temporal_attention_analysis(
        extractor.attention_weights_history
    )

    logger.info(f"\nTemporal attention entropy: {temporal['mean_attention_entropy']:.4f}")

    # News impact
    news_impact = NewsImpactQuantifier.calculate_news_surprise_impact(
        price_before=100.0,
        price_after=102.5,
        expected_sentiment=0.3,
        actual_sentiment=0.7
    )

    logger.info(f"\nNews Impact:")
    for metric, value in news_impact.items():
        logger.info(f"  {metric}: {value:.4f}")
