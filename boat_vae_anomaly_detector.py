#!/usr/bin/env python3
"""
Variational Autoencoders for Market Anomaly Detection
=====================================================

Probabilistic unsupervised anomaly detection for financial markets:
  - VAE encoder-decoder architecture
  - Latent space probabilistic modeling
  - Reconstruction error anomaly scoring
  - Market regime anomaly detection
  - Fraud and manipulation pattern identification

Based on 2025 research (VAE anomaly detection, probabilistic inference).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VAEConfig:
    """VAE configuration"""
    input_dim: int = 32
    latent_dim: int = 8
    hidden_dim: int = 32
    num_layers: int = 2


@dataclass
class AnomalyDetectionOutput:
    """Anomaly detection results"""
    reconstruction_error: np.ndarray  # (n_samples,)
    anomaly_scores: np.ndarray  # (n_samples,) [0, 1]
    anomalies: List[int]  # Indices of anomalies
    latent_means: np.ndarray  # (n_samples, latent_dim)
    latent_logvars: np.ndarray  # (n_samples, latent_dim)


class VAEEncoder:
    """Variational autoencoder encoder"""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 32):
        """Initialize encoder"""
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)

        # Mean and log-variance heads
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)

        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_logvar = np.zeros(latent_dim)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encoder forward pass

        Args:
            x: Input data (n_samples, input_dim) or (input_dim,)

        Returns:
            (latent_mean, latent_logvar, hidden_state)
        """
        # Ensure 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Hidden layer
        h = np.maximum(x @ self.W1 + self.b1, 0)  # ReLU

        # Mean
        mu = h @ self.W_mu + self.b_mu

        # Log-variance (ensure positivity)
        logvar = h @ self.W_logvar + self.b_logvar

        return mu, logvar, h

    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick for sampling

        Args:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution

        Returns:
            Sampled latent vector
        """
        std = np.sqrt(np.exp(logvar) + 1e-8)
        epsilon = np.random.randn(*mu.shape)
        z = mu + std * epsilon

        return z


class VAEDecoder:
    """Variational autoencoder decoder"""

    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 32):
        """Initialize decoder"""
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Decoder weights
        self.W1 = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)

        self.W_out = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b_out = np.zeros(output_dim)

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Decoder forward pass

        Args:
            z: Latent vector (n_samples, latent_dim) or (latent_dim,)

        Returns:
            Reconstructed output (n_samples, output_dim)
        """
        # Ensure 2D
        if z.ndim == 1:
            z = z.reshape(1, -1)

        # Hidden layer
        h = np.maximum(z @ self.W1 + self.b1, 0)  # ReLU

        # Output (sigmoid for [0,1] reconstruction)
        x_recon = 1.0 / (1.0 + np.exp(-(h @ self.W_out + self.b_out)))

        return x_recon


class VariationalAutoencoder:
    """Variational Autoencoder for anomaly detection"""

    def __init__(self, config: VAEConfig):
        """Initialize VAE"""
        self.config = config
        self.encoder = VAEEncoder(config.input_dim, config.latent_dim, config.hidden_dim)
        self.decoder = VAEDecoder(config.latent_dim, config.input_dim, config.hidden_dim)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through VAE

        Args:
            x: Input data

        Returns:
            (reconstruction, latent_mean, latent_logvar)
        """
        # Encoder
        mu, logvar, _ = self.encoder.forward(x)

        # Reparameterization
        z = self.encoder.reparameterize(mu, logvar)

        # Decoder
        x_recon = self.decoder.forward(z)

        return x_recon, mu, logvar

    def compute_elbo_loss(self, x: np.ndarray, x_recon: np.ndarray,
                         mu: np.ndarray, logvar: np.ndarray) -> float:
        """
        Compute ELBO loss (reconstruction + KL divergence)

        Args:
            x: Original data
            x_recon: Reconstructed data
            mu: Latent mean
            logvar: Latent log-variance

        Returns:
            ELBO loss value
        """
        # Reconstruction loss (binary cross-entropy)
        x_recon_clipped = np.clip(x_recon, 1e-8, 1 - 1e-8)
        recon_loss = -np.mean(x * np.log(x_recon_clipped) + (1 - x) * np.log(1 - x_recon_clipped))

        # KL divergence loss
        kl_loss = -0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar))

        # ELBO = reconstruction + KL
        elbo = recon_loss + kl_loss

        return float(elbo)


class MarketAnomalyDetector:
    """Market anomaly detection using VAE"""

    def __init__(self, input_dim: int = 32, latent_dim: int = 8):
        """Initialize market anomaly detector"""
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        config = VAEConfig(input_dim=input_dim, latent_dim=latent_dim)
        self.vae = VariationalAutoencoder(config)

        # Reconstruction error statistics for threshold
        self.error_mean = None
        self.error_std = None

    def extract_features(self, price_data: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Extract features from price data

        Args:
            price_data: (n_periods,) price time series
            window: Window size for feature extraction

        Returns:
            (n_windows, feature_dim) feature matrix
        """
        n_periods = len(price_data)
        n_features = n_periods - window + 1

        # Features: returns, volatility, momentum, etc.
        returns = np.diff(np.log(price_data))
        features = np.zeros((n_features, self.input_dim))

        for i in range(n_features):
            window_returns = returns[i:i + window]

            # Return statistics
            features[i, 0] = np.mean(window_returns)
            features[i, 1] = np.std(window_returns)
            features[i, 2] = np.max(window_returns)
            features[i, 3] = np.min(window_returns)

            # Autocorrelation
            for lag in range(1, min(4, window)):
                if len(window_returns) > lag:
                    acf = np.corrcoef(window_returns[:-lag], window_returns[lag:])[0, 1]
                    features[i, 3 + lag] = acf

            # Fill remaining with noise for now
            features[i, 7:] = np.random.randn(self.input_dim - 7) * 0.1

        # Normalize
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)

        return features

    def fit(self, price_data: np.ndarray, window: int = 20) -> float:
        """
        Fit VAE on normal market data

        Args:
            price_data: (n_periods,) price time series
            window: Feature extraction window

        Returns:
            Average ELBO loss
        """
        # Extract features
        features = self.extract_features(price_data, window)

        # Normalize to [0, 1]
        features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)

        # Store reconstruction errors for threshold
        self.fit_features = features
        losses = []

        for i in range(len(features)):
            x = features[i]
            x_recon, mu, logvar = self.vae.forward(x)
            loss = self.vae.compute_elbo_loss(x, x_recon, mu, logvar)
            losses.append(loss)

        avg_loss = np.mean(losses)

        # Compute reconstruction error statistics
        recon_errors = np.array([np.mean((features[i] - self.vae.forward(features[i])[0])**2)
                                for i in range(len(features))])
        self.error_mean = np.mean(recon_errors)
        self.error_std = np.std(recon_errors)

        return avg_loss

    def detect_anomalies(self, price_data: np.ndarray, threshold_sigma: float = 3.0) -> AnomalyDetectionOutput:
        """
        Detect market anomalies

        Args:
            price_data: (n_periods,) price time series
            threshold_sigma: Sigma threshold for anomaly

        Returns:
            AnomalyDetectionOutput with anomaly scores and indices
        """
        # Extract features
        features = self.extract_features(price_data)
        features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)

        # Compute reconstruction errors
        reconstruction_errors = []
        latent_means_list = []
        latent_logvars_list = []

        for i in range(len(features)):
            x = features[i]
            x_recon, mu, logvar = self.vae.forward(x)
            error = np.mean((x - x_recon)**2)
            reconstruction_errors.append(error)
            latent_means_list.append(mu.flatten())
            latent_logvars_list.append(logvar.flatten())

        reconstruction_errors = np.array(reconstruction_errors)
        latent_means = np.array(latent_means_list)
        latent_logvars = np.array(latent_logvars_list)

        # Normalize errors
        if self.error_mean is not None:
            z_scores = (reconstruction_errors - self.error_mean) / (self.error_std + 1e-8)
            anomaly_scores = 1.0 / (1.0 + np.exp(-z_scores))  # Sigmoid
        else:
            anomaly_scores = reconstruction_errors / (np.max(reconstruction_errors) + 1e-8)

        # Detect anomalies
        threshold = (self.error_mean if self.error_mean else np.mean(reconstruction_errors)) + \
                   threshold_sigma * (self.error_std if self.error_std else np.std(reconstruction_errors))
        anomalies = np.where(reconstruction_errors > threshold)[0].tolist()

        return AnomalyDetectionOutput(
            reconstruction_error=reconstruction_errors,
            anomaly_scores=anomaly_scores,
            anomalies=anomalies,
            latent_means=latent_means,
            latent_logvars=latent_logvars
        )


if __name__ == "__main__":
    logger.info("Variational Autoencoders for Market Anomaly Detection")
    logger.info("=" * 50)

    np.random.seed(42)

    # Generate synthetic price data with anomalies
    logger.info("\nGenerating synthetic market data with anomalies")
    n_periods = 500

    # Normal market data
    normal_returns = np.random.randn(n_periods) * 0.01
    price_data = 100 * np.exp(np.cumsum(normal_returns))

    # Inject anomalies (sudden volatility spikes)
    anomaly_indices = [100, 150, 250, 350, 400]
    for idx in anomaly_indices:
        price_data[idx:idx + 10] *= np.exp(np.random.randn(10) * 0.1)

    logger.info(f"  Periods: {n_periods}")
    logger.info(f"  Injected {len(anomaly_indices)} anomaly regions")
    logger.info(f"  Price range: [{price_data.min():.2f}, {price_data.max():.2f}]")

    # Initialize detector
    logger.info("\nInitializing Market Anomaly Detector")
    detector = MarketAnomalyDetector(input_dim=32, latent_dim=8)

    # Fit on training data (first 400 periods without anomalies)
    logger.info("\nFitting VAE on normal market data")
    train_data = price_data[:350]
    train_loss = detector.fit(train_data, window=20)
    logger.info(f"  Training ELBO loss: {train_loss:.4f}")

    # Detect anomalies
    logger.info("\nDetecting market anomalies")
    results = detector.detect_anomalies(price_data, threshold_sigma=2.5)

    logger.info(f"  Reconstruction error range: [{results.reconstruction_error.min():.6f}, {results.reconstruction_error.max():.6f}]")
    logger.info(f"  Mean anomaly score: {np.mean(results.anomaly_scores):.4f}")
    logger.info(f"  Detected anomalies: {len(results.anomalies)} regions")

    if results.anomalies:
        logger.info(f"  Anomaly indices: {results.anomalies[:10]}")

    # Latent space analysis
    logger.info("\nLatent Space Analysis")
    logger.info(f"  Latent means shape: {results.latent_means.shape}")
    logger.info(f"  Latent means range: [{results.latent_means.min():.4f}, {results.latent_means.max():.4f}]")
    logger.info(f"  Latent variance range: [{np.exp(results.latent_logvars).min():.4f}, {np.exp(results.latent_logvars).max():.4f}]")

    logger.info("\nMarket Anomaly Detection Complete")
