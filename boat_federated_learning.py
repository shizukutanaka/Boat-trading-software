#!/usr/bin/env python3
"""
Federated Learning for Privacy-Preserving Financial Models
============================================================

Distributed machine learning without centralized data:
  - Federated Averaging (FedAvg) algorithm
  - Local differential privacy mechanisms
  - Model aggregation across institutions
  - Gradient compression and communication efficiency
  - Privacy-preserving risk control
  - Multi-institutional credit scoring

Based on 2025 research on federated learning in finance.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LocalModel:
    """Local model at each institution"""
    institution_id: str
    parameters: np.ndarray
    data_size: int
    gradient: Optional[np.ndarray] = None
    local_accuracy: float = 0.0


@dataclass
class FederatedMetrics:
    """Metrics for federated learning"""
    global_accuracy: float
    communication_rounds: int
    privacy_epsilon: float
    data_utility: float
    convergence_speed: float


class FederatedAveraging:
    """Federated Averaging (FedAvg) algorithm"""

    def __init__(
        self,
        n_institutions: int,
        model_dim: int,
        learning_rate: float = 0.01,
        privacy_epsilon: float = 1.0
    ):
        """
        Initialize FedAvg

        Args:
            n_institutions: Number of participating institutions
            model_dim: Model dimensionality
            learning_rate: Local learning rate
            privacy_epsilon: Differential privacy epsilon
        """
        self.n_institutions = n_institutions
        self.model_dim = model_dim
        self.learning_rate = learning_rate
        self.privacy_epsilon = privacy_epsilon

        # Global model
        self.global_model = np.random.randn(model_dim) * 0.01
        self.local_models: Dict[str, LocalModel] = {}

    def initialize_local_model(
        self,
        institution_id: str,
        data_size: int
    ) -> None:
        """
        Initialize local model at institution

        Args:
            institution_id: Institution identifier
            data_size: Number of samples at institution
        """
        local_model = LocalModel(
            institution_id=institution_id,
            parameters=self.global_model.copy(),
            data_size=data_size
        )
        self.local_models[institution_id] = local_model

    def local_training(
        self,
        institution_id: str,
        X_local: np.ndarray,
        y_local: np.ndarray,
        epochs: int = 5
    ) -> None:
        """
        Train model locally at institution

        Args:
            institution_id: Institution identifier
            X_local: Local training features
            y_local: Local training targets
            epochs: Local training epochs
        """
        local_model = self.local_models[institution_id]
        params = local_model.parameters.copy()

        # Simple SGD training
        for epoch in range(epochs):
            # Compute gradient on local data
            predictions = X_local @ params
            errors = predictions - y_local
            gradient = X_local.T @ errors / len(y_local)

            # Local update
            params = params - self.learning_rate * gradient

            # Local accuracy
            mse = np.mean(errors ** 2)
            local_model.local_accuracy = float(1.0 / (1.0 + mse))

        # Add differential privacy noise
        noise = np.random.laplace(0, 1.0 / self.privacy_epsilon, self.model_dim)
        params_private = params + noise

        local_model.parameters = params_private
        local_model.gradient = gradient

    def aggregate_models(self) -> None:
        """
        Aggregate local models to global model (FedAvg)
        """
        if not self.local_models:
            return

        # Weighted average by data size
        total_samples = sum(m.data_size for m in self.local_models.values())
        global_params = np.zeros(self.model_dim)

        for local_model in self.local_models.values():
            weight = local_model.data_size / total_samples
            global_params += weight * local_model.parameters

        self.global_model = global_params

    def broadcast_model(self) -> np.ndarray:
        """
        Broadcast global model to all institutions

        Returns:
            Global model parameters
        """
        for local_model in self.local_models.values():
            local_model.parameters = self.global_model.copy()

        return self.global_model.copy()

    def federated_training(
        self,
        data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        communication_rounds: int = 10,
        local_epochs: int = 5
    ) -> List[float]:
        """
        Execute federated training

        Args:
            data_dict: {institution_id: (X, y)} pairs
            communication_rounds: Number of communication rounds
            local_epochs: Local training epochs per round

        Returns:
            List of global accuracy per round
        """
        # Initialize local models
        for inst_id, (X, y) in data_dict.items():
            self.initialize_local_model(inst_id, len(X))

        accuracies = []

        for round_num in range(communication_rounds):
            logger.info(f"Communication Round {round_num + 1}/{communication_rounds}")

            # Local training
            for inst_id, (X, y) in data_dict.items():
                self.local_training(inst_id, X, y, epochs=local_epochs)

            # Aggregation
            self.aggregate_models()

            # Broadcast
            self.broadcast_model()

            # Compute global accuracy
            avg_accuracy = np.mean([m.local_accuracy for m in self.local_models.values()])
            accuracies.append(avg_accuracy)

            logger.info(f"  Global Accuracy: {avg_accuracy:.4f}")

        return accuracies


class DifferentialPrivacyMechanism:
    """Differential privacy for federated learning"""

    @staticmethod
    def laplace_mechanism(
        data: np.ndarray,
        epsilon: float,
        sensitivity: float = 1.0
    ) -> np.ndarray:
        """
        Laplace mechanism for differential privacy

        Args:
            data: Sensitive data
            epsilon: Privacy budget
            sensitivity: Global sensitivity

        Returns:
            Differentially private data
        """
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise

    @staticmethod
    def gaussian_mechanism(
        data: np.ndarray,
        epsilon: float,
        delta: float = 1e-5,
        sensitivity: float = 1.0
    ) -> np.ndarray:
        """
        Gaussian mechanism for differential privacy

        Args:
            data: Sensitive data
            epsilon: Privacy budget
            delta: Failure probability
            sensitivity: Global sensitivity

        Returns:
            Differentially private data
        """
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise = np.random.normal(0, sigma, data.shape)
        return data + noise

    @staticmethod
    def gradient_clipping(
        gradients: np.ndarray,
        threshold: float = 1.0
    ) -> np.ndarray:
        """
        Clip gradients by L2 norm

        Args:
            gradients: Gradient vector
            threshold: Clipping threshold

        Returns:
            Clipped gradients
        """
        norm = np.linalg.norm(gradients)
        if norm > threshold:
            return gradients * threshold / norm
        return gradients


class SecureAggregation:
    """Secure aggregation protocol"""

    @staticmethod
    def secret_sharing(
        value: np.ndarray,
        n_shares: int
    ) -> List[np.ndarray]:
        """
        Shamir's secret sharing

        Args:
            value: Value to share
            n_shares: Number of shares

        Returns:
            List of shares
        """
        # Simple XOR sharing (not cryptographically secure, for illustration)
        shares = []
        remaining = value.copy()

        for i in range(n_shares - 1):
            share = np.random.randn(*value.shape)
            shares.append(share)
            remaining = remaining - share

        shares.append(remaining)
        return shares

    @staticmethod
    def reconstruct_secret(
        shares: List[np.ndarray]
    ) -> np.ndarray:
        """
        Reconstruct secret from shares

        Args:
            shares: List of shares

        Returns:
            Reconstructed value
        """
        return np.sum(shares, axis=0)


class PrivacyPreservingRiskControl:
    """Risk control model without sharing raw data"""

    def __init__(
        self,
        n_institutions: int,
        feature_dim: int,
        privacy_epsilon: float = 1.0
    ):
        """
        Initialize privacy-preserving risk model

        Args:
            n_institutions: Number of institutions
            feature_dim: Feature dimension
            privacy_epsilon: Privacy budget
        """
        self.n_institutions = n_institutions
        self.feature_dim = feature_dim
        self.privacy_epsilon = privacy_epsilon

        # Global risk model
        self.global_model = np.random.randn(feature_dim) * 0.01

    def fit(
        self,
        data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        communication_rounds: int = 10
    ) -> Tuple[np.ndarray, float]:
        """
        Train risk model federally

        Args:
            data_dict: {institution_id: (X_risk_features, y_risk)}
            communication_rounds: Number of rounds

        Returns:
            (global_model, privacy_guarantee)
        """
        fed_avg = FederatedAveraging(
            n_institutions=len(data_dict),
            model_dim=self.feature_dim,
            privacy_epsilon=self.privacy_epsilon
        )

        accuracies = fed_avg.federated_training(
            data_dict,
            communication_rounds=communication_rounds,
            local_epochs=5
        )

        self.global_model = fed_avg.global_model

        # Privacy guarantee (epsilon)
        privacy_guarantee = self.privacy_epsilon

        return self.global_model.copy(), privacy_guarantee

    def predict_risk(
        self,
        features: np.ndarray
    ) -> np.ndarray:
        """
        Predict risk using global model

        Args:
            features: Risk features

        Returns:
            Risk predictions
        """
        return features @ self.global_model


class MultiInstitutionalCreditScoring:
    """Federated credit scoring without sharing customer data"""

    def __init__(
        self,
        institutions: List[str],
        feature_dim: int = 20,
        privacy_epsilon: float = 1.0
    ):
        """
        Initialize federated credit scoring

        Args:
            institutions: List of bank names
            feature_dim: Credit score feature dimension
            privacy_epsilon: Privacy budget
        """
        self.institutions = institutions
        self.feature_dim = feature_dim
        self.privacy_epsilon = privacy_epsilon
        self.global_scorecard = np.random.randn(feature_dim) * 0.01

    def local_credit_modeling(
        self,
        bank_id: str,
        customer_features: np.ndarray,
        default_labels: np.ndarray,
        learning_rate: float = 0.01
    ) -> Dict[str, float]:
        """
        Train credit model locally at bank

        Args:
            bank_id: Bank identifier
            customer_features: Customer features (N, feature_dim)
            default_labels: Default indicators
            learning_rate: Learning rate

        Returns:
            Local model performance metrics
        """
        # Local model
        local_model = self.global_scorecard.copy()

        # Logistic regression training
        for iteration in range(10):
            logits = customer_features @ local_model
            predictions = 1 / (1 + np.exp(-logits))
            errors = predictions - default_labels

            # Gradient
            gradient = customer_features.T @ errors / len(default_labels)

            # Update
            local_model = local_model - learning_rate * gradient

        # Add differential privacy
        dp_model = DifferentialPrivacyMechanism.gaussian_mechanism(
            local_model,
            epsilon=self.privacy_epsilon
        )

        # Evaluate
        logits = customer_features @ dp_model
        predictions = 1 / (1 + np.exp(-logits))
        accuracy = np.mean((predictions > 0.5).astype(int) == default_labels)

        return {
            'bank_id': bank_id,
            'accuracy': accuracy,
            'model': dp_model
        }

    def federated_credit_scoring(
        self,
        training_data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """
        Train credit scoring system federally

        Args:
            training_data: {bank_id: (features, default_labels)}

        Returns:
            Global credit scorecard
        """
        # Collect local models
        local_models = []

        for bank_id, (features, labels) in training_data.items():
            result = self.local_credit_modeling(bank_id, features, labels)
            local_models.append(result)

        # Aggregate models
        models_array = np.array([m['model'] for m in local_models])
        self.global_scorecard = np.mean(models_array, axis=0)

        logger.info("Federated Credit Scoring Complete")
        for m in local_models:
            logger.info(f"  {m['bank_id']}: Accuracy={m['accuracy']:.4f}")

        return self.global_scorecard


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Simulated data from 3 institutions
    n_institutions = 3
    institutions = ['Bank_A', 'Bank_B', 'Bank_C']

    # Create synthetic training data
    training_data = {}

    for i, bank in enumerate(institutions):
        n_samples = 100 + i * 50
        X = np.random.randn(n_samples, 10)
        y = (np.random.randn(n_samples) > 0).astype(int)
        training_data[bank] = (X, y)

    # Federated learning
    fed_avg = FederatedAveraging(
        n_institutions=len(institutions),
        model_dim=10,
        privacy_epsilon=1.0
    )

    accuracies = fed_avg.federated_training(
        training_data,
        communication_rounds=5,
        local_epochs=3
    )

    logger.info(f"Final Global Accuracy: {accuracies[-1]:.4f}")

    # Privacy-preserving risk control
    logger.info("\nPrivacy-Preserving Risk Control:")
    risk_model = PrivacyPreservingRiskControl(
        n_institutions=n_institutions,
        feature_dim=10,
        privacy_epsilon=1.0
    )
    global_risk_model, privacy_eps = risk_model.fit(training_data, communication_rounds=5)
    logger.info(f"Privacy Guarantee (ε): {privacy_eps:.4f}")

    # Multi-institutional credit scoring
    logger.info("\nMulti-Institutional Credit Scoring:")
    credit_scorer = MultiInstitutionalCreditScoring(
        institutions=institutions,
        feature_dim=10,
        privacy_epsilon=1.0
    )
    global_scorecard = credit_scorer.federated_credit_scoring(training_data)

    logger.info("Federated Learning Complete")
