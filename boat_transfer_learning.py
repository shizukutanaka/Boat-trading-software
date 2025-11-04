#!/usr/bin/env python3
"""
Transfer Learning Framework for Cross-Market Generalization
=============================================================

Domain adaptation and transfer learning for financial models:
  - Feature extraction and reuse across markets
  - Domain adaptation techniques
  - Fine-tuning strategies
  - Correlation-based feature transfer
  - Multi-market model training
  - Generalization metrics
  - Cross-domain evaluation

Based on 2025 research on transfer learning for financial time series.
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
class DomainAdaptationMetrics:
    """Metrics for domain adaptation"""
    source_accuracy: float
    target_accuracy: float
    domain_discrepancy: float
    transfer_efficiency: float
    generalization_gap: float


class FeatureExtractor:
    """Extract transferable features from financial time series"""

    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.feature_means = None
        self.feature_stds = None

    def extract_features(self, returns: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from returns

        Args:
            returns: Historical returns (T, n_assets)

        Returns:
            Extracted features (T - lookback, n_features)
        """
        features_list = []

        for t in range(self.lookback, len(returns)):
            window = returns[t - self.lookback:t]

            # Statistical features
            feat = {
                'mean': np.mean(window),
                'std': np.std(window),
                'skew': self._calculate_skewness(window),
                'kurtosis': self._calculate_kurtosis(window),
                'max_drawdown': self._calculate_max_drawdown(window),
                'sharpe': self._calculate_sharpe(window),
                'autocorr_1': np.corrcoef(window[:-1], window[1:])[0, 1] if len(window) > 1 else 0,
                'momentum': np.sum(window),
                'volatility_cluster': self._detect_volatility_cluster(window),
                'trend': self._calculate_trend(window)
            }

            features_list.append(list(feat.values()))

        return np.array(features_list)

    @staticmethod
    def _calculate_skewness(x: np.ndarray) -> float:
        """Calculate skewness"""
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return np.mean(((x - mean) / std) ** 3)

    @staticmethod
    def _calculate_kurtosis(x: np.ndarray) -> float:
        """Calculate excess kurtosis"""
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0.0
        return np.mean(((x - mean) / std) ** 4) - 3

    @staticmethod
    def _calculate_max_drawdown(x: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumsum = np.cumsum(x)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = (cumsum - running_max) / (running_max + 1e-8)
        return float(np.min(drawdown))

    @staticmethod
    def _calculate_sharpe(x: np.ndarray, rf: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        mean_return = np.mean(x)
        std_return = np.std(x)
        if std_return == 0:
            return 0.0
        return (mean_return - rf) / std_return

    @staticmethod
    def _detect_volatility_cluster(x: np.ndarray) -> float:
        """Detect volatility clustering"""
        volatilities = np.abs(x)
        return float(np.corrcoef(volatilities[:-1], volatilities[1:])[0, 1])

    @staticmethod
    def _calculate_trend(x: np.ndarray) -> float:
        """Calculate trend using linear regression slope"""
        y = np.cumsum(x)
        x_idx = np.arange(len(y))
        if len(x_idx) < 2:
            return 0.0
        return float(np.polyfit(x_idx, y, 1)[0])

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to zero mean, unit variance"""
        if self.feature_means is None:
            self.feature_means = np.mean(features, axis=0)
            self.feature_stds = np.std(features, axis=0)
            self.feature_stds[self.feature_stds == 0] = 1.0

        return (features - self.feature_means) / self.feature_stds


class DomainAdaptationModel:
    """Domain adaptation for cross-market transfer"""

    def __init__(self, source_features: np.ndarray, target_features: np.ndarray):
        self.source_features = source_features
        self.target_features = target_features

        self.source_mean = np.mean(source_features, axis=0)
        self.target_mean = np.mean(target_features, axis=0)

        self.adaptation_matrix = None

    def calculate_domain_discrepancy(self) -> float:
        """
        Calculate MMD (Maximum Mean Discrepancy) between domains

        Returns:
            MMD distance
        """
        # Mean difference (simplified MMD)
        mmd_sq = np.sum((self.source_mean - self.target_mean) ** 2)
        return float(np.sqrt(mmd_sq))

    def adapt_features(self, source_features: np.ndarray) -> np.ndarray:
        """
        Adapt source features to target domain

        Args:
            source_features: Features from source domain

        Returns:
            Adapted features
        """
        # Simple linear adaptation: X_adapted = X + (target_mean - source_mean)
        if self.adaptation_matrix is None:
            self.adaptation_matrix = self.target_mean - self.source_mean

        return source_features + self.adaptation_matrix

    def calculate_correlation_transfer(self) -> Dict[str, float]:
        """
        Calculate transferability based on feature correlation

        Returns:
            Feature transferability scores
        """
        n_features = self.source_features.shape[1]
        transferability = {}

        for i in range(n_features):
            # Correlation of each feature between domains
            source_feat = self.source_features[:, i]
            target_feat = self.target_features[:, i]

            # Normalize
            source_feat_norm = (source_feat - np.mean(source_feat)) / (np.std(source_feat) + 1e-8)
            target_feat_norm = (target_feat - np.mean(target_feat)) / (np.std(target_feat) + 1e-8)

            # Correlation
            corr = np.mean(source_feat_norm * target_feat_norm)
            transferability[f'feature_{i}'] = float(corr)

        return transferability


class TransferLearningModel:
    """Complete transfer learning pipeline"""

    def __init__(self):
        self.source_model_weights = None
        self.target_model_weights = None
        self.feature_extractor = None
        self.domain_adapter = None

    def train_source_model(
        self,
        source_returns: np.ndarray,
        source_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Train model on source domain

        Args:
            source_returns: Source market returns
            source_labels: Source labels (e.g., buy/sell signals)

        Returns:
            Training metrics
        """
        # Extract features
        self.feature_extractor = FeatureExtractor(lookback=60)
        features = self.feature_extractor.extract_features(source_returns)
        features = self.feature_extractor.normalize_features(features)

        # Simple linear model: weights = features^T @ labels / (features^T @ features)
        self.source_model_weights = np.linalg.lstsq(features, source_labels, rcond=None)[0]

        # Calculate training accuracy
        predictions = features @ self.source_model_weights
        mse = np.mean((predictions - source_labels) ** 2)
        r_squared = 1 - (np.sum((source_labels - predictions) ** 2) /
                         np.sum((source_labels - np.mean(source_labels)) ** 2))

        return {
            'mse': float(mse),
            'r_squared': float(r_squared),
            'n_features': features.shape[1]
        }

    def transfer_to_target_domain(
        self,
        target_returns: np.ndarray,
        target_labels: np.ndarray,
        fine_tune_percentage: float = 0.2
    ) -> DomainAdaptationMetrics:
        """
        Transfer source model to target domain

        Args:
            target_returns: Target market returns
            target_labels: Target labels
            fine_tune_percentage: % of source weights to fine-tune

        Returns:
            Domain adaptation metrics
        """
        if self.source_model_weights is None:
            raise ValueError("Must train source model first")

        # Extract target features
        target_features = self.feature_extractor.extract_features(target_returns)
        target_features = self.feature_extractor.normalize_features(target_features)

        # Domain adaptation
        source_features = self.feature_extractor.extract_features(target_returns[:len(target_returns)])
        self.domain_adapter = DomainAdaptationModel(source_features, target_features)

        # Evaluate source model on target (before adaptation)
        source_pred = target_features @ self.source_model_weights
        source_acc = 1 - np.mean((source_pred - target_labels) ** 2)

        # Fine-tune on target data
        n_finetune = int(len(self.source_model_weights) * fine_tune_percentage)
        self.target_model_weights = self.source_model_weights.copy()

        # Update weights for most important features
        feature_importance = np.abs(self.source_model_weights)
        top_indices = np.argsort(feature_importance)[-n_finetune:]

        # Simple fine-tuning
        for idx in top_indices:
            self.target_model_weights[idx] = np.linalg.lstsq(
                target_features[:, [idx]], target_labels, rcond=None
            )[0][0]

        # Evaluate adapted model on target
        target_pred = target_features @ self.target_model_weights
        target_acc = 1 - np.mean((target_pred - target_labels) ** 2)

        # Calculate metrics
        domain_disc = self.domain_adapter.calculate_domain_discrepancy()
        transfer_eff = target_acc / (source_acc + 1e-8)

        return DomainAdaptationMetrics(
            source_accuracy=float(source_acc),
            target_accuracy=float(target_acc),
            domain_discrepancy=float(domain_disc),
            transfer_efficiency=float(transfer_eff),
            generalization_gap=float(source_acc - target_acc)
        )

    def predict(self, returns: np.ndarray, use_adapted: bool = True) -> np.ndarray:
        """
        Make predictions

        Args:
            returns: Input returns
            use_adapted: Use adapted (target) weights if available

        Returns:
            Predictions
        """
        features = self.feature_extractor.extract_features(returns)
        features = self.feature_extractor.normalize_features(features)

        weights = self.target_model_weights if use_adapted and self.target_model_weights is not None else self.source_model_weights

        return features @ weights


class MultiMarketTransferLearning:
    """Transfer learning across multiple markets"""

    def __init__(self, markets: List[str]):
        self.markets = markets
        self.models: Dict[str, TransferLearningModel] = {}
        self.source_market = None

    def fit(
        self,
        market_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        source_market: str
    ) -> Dict[str, DomainAdaptationMetrics]:
        """
        Fit transfer learning models

        Args:
            market_data: Dict of market -> (returns, labels)
            source_market: Source domain for transfer

        Returns:
            Metrics for each target market
        """
        self.source_market = source_market

        # Train source model
        source_returns, source_labels = market_data[source_market]
        self.models[source_market] = TransferLearningModel()
        self.models[source_market].train_source_model(source_returns, source_labels)

        # Transfer to target markets
        metrics = {}

        for target_market in self.markets:
            if target_market == source_market:
                continue

            target_returns, target_labels = market_data[target_market]

            self.models[target_market] = TransferLearningModel()
            self.models[target_market].feature_extractor = self.models[source_market].feature_extractor
            self.models[target_market].source_model_weights = self.models[source_market].source_model_weights

            # Transfer
            metrics[target_market] = self.models[target_market].transfer_to_target_domain(
                target_returns, target_labels
            )

        return metrics

    def predict_all(self, market_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Predict on all markets"""
        predictions = {}

        for market in self.markets:
            if market in self.models:
                predictions[market] = self.models[market].predict(market_data[market])

        return predictions


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate source market data
    source_returns = np.random.randn(500, 1) * 0.02
    source_labels = (source_returns.squeeze() > 0).astype(float)

    # Generate target market data (slightly different distribution)
    target_returns = np.random.randn(300, 1) * 0.025 + 0.001
    target_labels = (target_returns.squeeze() > 0).astype(float)

    # Train source model
    source_model = TransferLearningModel()
    source_metrics = source_model.train_source_model(source_returns, source_labels)

    logger.info("Source Model Training:")
    logger.info(f"MSE: {source_metrics['mse']:.6f}")
    logger.info(f"R²: {source_metrics['r_squared']:.4f}")

    # Transfer to target domain
    transfer_metrics = source_model.transfer_to_target_domain(target_returns, target_labels)

    logger.info("\nTransfer Learning Metrics:")
    logger.info(f"Source Accuracy: {transfer_metrics.source_accuracy:.4f}")
    logger.info(f"Target Accuracy: {transfer_metrics.target_accuracy:.4f}")
    logger.info(f"Domain Discrepancy: {transfer_metrics.domain_discrepancy:.4f}")
    logger.info(f"Transfer Efficiency: {transfer_metrics.transfer_efficiency:.4f}")
    logger.info(f"Generalization Gap: {transfer_metrics.generalization_gap:.4f}")

    # Multi-market transfer learning
    market_data = {
        'US_Market': (source_returns, source_labels),
        'EU_Market': (target_returns, target_labels)
    }

    multi_market = MultiMarketTransferLearning(['US_Market', 'EU_Market'])
    transfer_results = multi_market.fit(market_data, source_market='US_Market')

    logger.info("\nMulti-Market Transfer Results:")
    for market, metrics in transfer_results.items():
        logger.info(f"{market}: Target Accuracy = {metrics.target_accuracy:.4f}")
