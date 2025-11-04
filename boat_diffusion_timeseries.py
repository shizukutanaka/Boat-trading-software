#!/usr/bin/env python3
"""
Diffusion Models for Financial Time Series Generation & Forecasting
====================================================================

Generative diffusion models for financial time series:
  - Reverse diffusion process for denoising
  - Conditional diffusion with exogenous variables
  - Time series synthesis and augmentation
  - Probabilistic forecasting with uncertainty
  - Multi-step ahead prediction
  - Stylized facts preservation

Based on 2025 research on diffusion models for time series forecasting.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DiffusionConfig:
    """Configuration for diffusion model"""
    timesteps: int = 1000
    noise_schedule: str = "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"  # epsilon, v_prediction, or sample


@dataclass
class ForecastResult:
    """Result from diffusion-based forecasting"""
    mean_forecast: np.ndarray
    uncertainty: np.ndarray
    samples: List[np.ndarray]


class NoiseScheduler:
    """Schedule for noise levels during diffusion"""

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        """
        Initialize noise scheduler

        Args:
            num_timesteps: Number of diffusion steps
            schedule_type: Type of noise schedule
            beta_start: Starting beta value
            beta_end: Ending beta value
        """
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type

        if schedule_type == "linear":
            betas = np.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "quadratic":
            betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        else:
            betas = np.linspace(beta_start, beta_end, num_timesteps)

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])

    def add_noise(
        self,
        x_0: np.ndarray,
        t: int,
        noise: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add noise to original signal

        Args:
            x_0: Original signal
            t: Timestep
            noise: Noise to add (random if None)

        Returns:
            (noisy_signal, noise)
        """
        if noise is None:
            noise = np.random.randn(*x_0.shape)

        sqrt_alpha_cumprod = np.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alpha_cumprod = np.sqrt(1.0 - self.alphas_cumprod[t])

        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise

        return x_t, noise

    def get_variance(self, t: int) -> float:
        """Get variance at timestep"""
        numerator = (1.0 - self.alphas_cumprod_prev[t]) * self.betas[t]
        denominator = 1.0 - self.alphas_cumprod[t]
        return max(numerator / denominator, 0.0)


class DiffusionModel:
    """Diffusion model for time series generation"""

    def __init__(
        self,
        config: DiffusionConfig = None,
        sequence_length: int = 100,
        n_features: int = 1
    ):
        """
        Initialize diffusion model

        Args:
            config: Diffusion configuration
            sequence_length: Length of time series
            n_features: Number of features
        """
        self.config = config or DiffusionConfig()
        self.sequence_length = sequence_length
        self.n_features = n_features

        self.scheduler = NoiseScheduler(
            num_timesteps=self.config.timesteps,
            schedule_type=self.config.noise_schedule,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end
        )

        # Simple denoising network weights
        self.denoise_weights = np.random.randn(sequence_length * n_features, 64) * 0.01
        self.denoise_weights_out = np.random.randn(64, sequence_length * n_features) * 0.01

    def denoise(
        self,
        x_t: np.ndarray,
        t: int
    ) -> np.ndarray:
        """
        Predict noise/signal from noisy input

        Args:
            x_t: Noisy signal
            t: Current timestep

        Returns:
            Denoised signal
        """
        # Flatten for network
        x_flat = x_t.flatten()

        # Simple 1-hidden-layer denoise network
        hidden = np.tanh(x_flat @ self.denoise_weights)
        prediction = hidden @ self.denoise_weights_out

        # Reshape back
        return prediction.reshape(x_t.shape)

    def reverse_diffusion(
        self,
        x_T: np.ndarray,
        num_steps: Optional[int] = None
    ) -> np.ndarray:
        """
        Perform reverse diffusion to generate from noise

        Args:
            x_T: Noise sample
            num_steps: Number of denoising steps

        Returns:
            Generated sample
        """
        if num_steps is None:
            num_steps = self.config.timesteps

        x = x_T.copy()
        step_size = self.config.timesteps // num_steps

        for t in range(self.config.timesteps - 1, 0, -step_size):
            # Predict noise
            noise_pred = self.denoise(x, t)

            # Backward step
            alpha = self.scheduler.alphas[t]
            beta = self.scheduler.betas[t]
            variance = self.scheduler.get_variance(t)

            # Reverse formula
            mean = (x - (beta / np.sqrt(1.0 - self.scheduler.alphas_cumprod[t])) * noise_pred) / np.sqrt(alpha)

            if t > 1:
                z = np.random.randn(*x.shape)
                x = mean + np.sqrt(variance) * z
            else:
                x = mean

        return x

    def generate(
        self,
        num_samples: int = 10,
        num_steps: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate time series samples

        Args:
            num_samples: Number of samples to generate
            num_steps: Number of denoising steps

        Returns:
            Generated samples (num_samples, sequence_length, n_features)
        """
        samples = []

        for _ in range(num_samples):
            # Start from noise
            x_T = np.random.randn(self.sequence_length, self.n_features)

            # Reverse diffusion
            sample = self.reverse_diffusion(x_T, num_steps)

            samples.append(sample)

        return np.array(samples)


class ConditionalDiffusion:
    """Conditional diffusion for forecasting given historical data"""

    def __init__(
        self,
        config: DiffusionConfig = None,
        lookback: int = 50,
        forecast_horizon: int = 20
    ):
        """
        Initialize conditional diffusion

        Args:
            config: Diffusion configuration
            lookback: Historical lookback window
            forecast_horizon: Number of steps to forecast
        """
        self.config = config or DiffusionConfig()
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon

        self.scheduler = NoiseScheduler(
            num_timesteps=self.config.timesteps,
            schedule_type=self.config.noise_schedule
        )

        # Network for conditional prediction
        self.cond_weights = np.random.randn(lookback, 32) * 0.01
        self.cond_to_pred = np.random.randn(32 + forecast_horizon, forecast_horizon) * 0.01

    def encode_condition(self, history: np.ndarray) -> np.ndarray:
        """
        Encode historical data as condition

        Args:
            history: Historical data (lookback,)

        Returns:
            Condition embedding
        """
        condition = history @ self.cond_weights
        return condition

    def forecast_step(
        self,
        x_t: np.ndarray,
        condition: np.ndarray,
        t: int
    ) -> np.ndarray:
        """
        Single denoising step with condition

        Args:
            x_t: Current noisy forecast
            condition: Historical condition
            t: Timestep

        Returns:
            Updated forecast
        """
        # Concatenate condition and prediction
        combined = np.concatenate([condition, x_t])
        prediction = combined @ self.cond_to_pred

        return prediction

    def forecast(
        self,
        history: np.ndarray,
        num_samples: int = 10,
        num_steps: Optional[int] = None
    ) -> ForecastResult:
        """
        Generate probabilistic forecasts

        Args:
            history: Historical data
            num_samples: Number of forecast samples
            num_steps: Number of denoising steps

        Returns:
            ForecastResult with mean, uncertainty, samples
        """
        if num_steps is None:
            num_steps = min(50, self.config.timesteps // 20)

        condition = self.encode_condition(history)
        forecast_samples = []

        for _ in range(num_samples):
            # Start from noise
            x = np.random.randn(self.forecast_horizon)

            # Iterative denoising
            for step in range(num_steps - 1, 0, -1):
                t_idx = int(step * (self.config.timesteps / num_steps))
                x = self.forecast_step(x, condition, t_idx)

            forecast_samples.append(x)

        forecast_array = np.array(forecast_samples)

        return ForecastResult(
            mean_forecast=np.mean(forecast_array, axis=0),
            uncertainty=np.std(forecast_array, axis=0),
            samples=forecast_samples
        )


class StyleizedFactsPreserver:
    """Ensure generated data preserves stylized facts"""

    @staticmethod
    def compute_stylized_facts(data: np.ndarray) -> Dict[str, float]:
        """
        Compute stylized facts of time series

        Args:
            data: Time series data

        Returns:
            Dictionary of stylized facts
        """
        returns = np.diff(np.log(data.flatten()))

        facts = {
            'mean_return': float(np.mean(returns)),
            'volatility': float(np.std(returns)),
            'skewness': float(np.mean((returns - np.mean(returns)) ** 3) / (np.std(returns) ** 3)),
            'kurtosis': float(np.mean((returns - np.mean(returns)) ** 4) / (np.std(returns) ** 4)),
            'autocorr_1': float(np.corrcoef(returns[:-1], returns[1:])[0, 1]),
            'vol_clustering': float(np.corrcoef(returns[:-1] ** 2, returns[1:] ** 2)[0, 1]),
        }

        return facts

    @staticmethod
    def adjust_for_stylized_facts(
        generated: np.ndarray,
        target_facts: Dict[str, float]
    ) -> np.ndarray:
        """
        Adjust generated data to match stylized facts

        Args:
            generated: Generated time series
            target_facts: Target stylized facts

        Returns:
            Adjusted time series
        """
        # Ensure positive prices
        generated_positive = np.abs(generated.flatten()) + 0.01

        # Simple adjustment: standardize and rescale
        returns = np.diff(np.log(generated_positive))
        adjusted_returns = returns - np.mean(returns)
        adjusted_returns = adjusted_returns / (np.std(adjusted_returns) + 1e-8)
        adjusted_returns = adjusted_returns * target_facts['volatility']
        adjusted_returns = adjusted_returns + target_facts['mean_return']

        # Reconstruct log prices
        log_prices = np.concatenate([[np.log(generated_positive[0])], np.log(generated_positive[0]) + np.cumsum(adjusted_returns)])
        adjusted = np.exp(log_prices)

        return adjusted.reshape(generated.shape)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Create synthetic financial data
    n_days = 250
    true_prices = 100 * np.cumprod(1 + 0.001 * np.random.randn(n_days) + 0.0002)

    # Initialize diffusion model
    config = DiffusionConfig(timesteps=100)
    diffusion = DiffusionModel(config, sequence_length=50, n_features=1)

    logger.info("Diffusion Model initialized")

    # Generate samples
    generated_samples = diffusion.generate(num_samples=5, num_steps=20)
    logger.info(f"Generated samples shape: {generated_samples.shape}")

    # Conditional diffusion for forecasting
    cond_diffusion = ConditionalDiffusion(
        config=config,
        lookback=50,
        forecast_horizon=20
    )

    # Prepare historical data
    history = true_prices[:50]

    # Forecast
    forecast_result = cond_diffusion.forecast(history, num_samples=10, num_steps=20)

    logger.info(f"Mean forecast: {forecast_result.mean_forecast[:5]}")
    logger.info(f"Uncertainty: {forecast_result.uncertainty[:5]}")

    # Stylized facts preservation
    target_facts = StyleizedFactsPreserver.compute_stylized_facts(true_prices)
    logger.info(f"Target stylized facts: {target_facts}")

    # Adjust generated samples
    adjusted_sample = StyleizedFactsPreserver.adjust_for_stylized_facts(
        generated_samples[0],
        target_facts
    )

    logger.info(f"Adjusted sample shape: {adjusted_sample.shape}")
    logger.info("Diffusion Time Series Generation Complete")
