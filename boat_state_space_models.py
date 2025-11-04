#!/usr/bin/env python3
"""
State Space Models for Time Series (Mamba/S4)
==============================================

Structured state space models for efficient time series processing:
  - S4 (Structured State Space Sequence) models
  - Mamba-style selective SSMs with input-dependent parameters
  - Linear-time sequence modeling vs quadratic attention
  - Efficient parallel training and fast inference
  - Long-range dependency modeling

Based on 2025 research (Mamba, S4, comprehensive SSM survey).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SSMState:
    """State space model state"""
    h: np.ndarray  # Hidden state
    output: np.ndarray  # Output


class StateSpaceMatrix:
    """Structured state space matrices A, B, C"""

    def __init__(self, state_dim: int = 32):
        """Initialize state space matrices"""
        self.state_dim = state_dim

        # State transition matrix A (diagonal with complex eigenvalues)
        self.A = np.diag(np.random.uniform(-1, -0.1, state_dim))

        # Input matrix B
        self.B = np.random.randn(state_dim, 1) * 0.01

        # Output matrix C
        self.C = np.random.randn(1, state_dim) * 0.01


class S4Layer:
    """S4 (Structured State Space Sequence) layer"""

    def __init__(self, state_dim: int = 32, seq_len: int = 256):
        """Initialize S4 layer"""
        self.state_dim = state_dim
        self.seq_len = seq_len

        # State space matrices
        self.ssm = StateSpaceMatrix(state_dim)

        # Convolutional kernel (pre-computed via S4 discretization)
        self.kernel = self._compute_convolutional_kernel()

    def _compute_convolutional_kernel(self) -> np.ndarray:
        """
        Compute convolutional kernel from state space model

        S4 converts SSM to efficient convolutional representation
        """
        dt = 1.0  # Discretization step

        # Bilinear discretization
        I = np.eye(self.state_dim)
        Ad = (I + 0.5 * dt * self.ssm.A) @ np.linalg.inv(I - 0.5 * dt * self.ssm.A)
        Bd = dt * np.linalg.inv(I - 0.5 * dt * self.ssm.A) @ self.ssm.B

        # Compute impulse response (convolution kernel)
        kernel = np.zeros(self.seq_len)
        h = np.zeros((self.state_dim, 1))

        for t in range(self.seq_len):
            h = Ad @ h + Bd
            kernel[t] = (self.ssm.C @ h)[0, 0]

        return kernel / (np.linalg.norm(kernel) + 1e-8)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass via convolution (fast parallel computation)

        Args:
            x: Input sequence (seq_len,)

        Returns:
            Output sequence (seq_len,)
        """
        # Pad sequences for valid convolution
        x_padded = np.pad(x, (self.seq_len - 1, 0), mode='constant')

        # Convolutional filtering
        output = np.convolve(x_padded, self.kernel, mode='valid')

        return output[:len(x)]

    def recurrent_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass via recurrence (for inference)

        Args:
            x: Input sequence (seq_len,)

        Returns:
            Output sequence (seq_len,)
        """
        dt = 1.0
        I = np.eye(self.state_dim)

        # Discretized matrices
        Ad = (I + 0.5 * dt * self.ssm.A) @ np.linalg.inv(I - 0.5 * dt * self.ssm.A)
        Bd = dt * np.linalg.inv(I - 0.5 * dt * self.ssm.A) @ self.ssm.B

        h = np.zeros((self.state_dim, 1))
        outputs = []

        for t in range(len(x)):
            h = Ad @ h + Bd * x[t]
            y = (self.ssm.C @ h)[0, 0]
            outputs.append(y)

        return np.array(outputs)


class MambaLayer:
    """Mamba selective SSM with input-dependent parameters"""

    def __init__(self, state_dim: int = 32, seq_len: int = 256):
        """Initialize Mamba layer"""
        self.state_dim = state_dim
        self.seq_len = seq_len

        # Base SSM
        self.ssm = StateSpaceMatrix(state_dim)

        # Parameter projection networks for input-dependency
        self.delta_proj = np.random.randn(1, state_dim) * 0.01
        self.B_proj = np.random.randn(1, state_dim) * 0.01
        self.C_proj = np.random.randn(1, state_dim) * 0.01

    def selective_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with input-dependent (selective) parameters

        Args:
            x: Input sequence (seq_len,)

        Returns:
            Output sequence (seq_len,)
        """
        dt = 1.0
        I = np.eye(self.state_dim)

        h = np.zeros((self.state_dim, 1))
        outputs = []

        for t in range(len(x)):
            # Input-dependent step size (selectivity)
            x_t_scalar = x[t] if np.isscalar(x[t]) else x[t].item()
            delta = 1.0 / (1.0 + np.exp(-x_t_scalar))  # Sigmoid

            # Input-dependent B and C
            B_t = self.ssm.B * np.tanh(x_t_scalar)
            C_t = self.ssm.C * np.tanh(x_t_scalar)

            # Selective state update
            Ad = (I + 0.5 * delta * self.ssm.A) @ np.linalg.inv(I - 0.5 * delta * self.ssm.A)
            h = Ad @ h + B_t * x_t_scalar

            # Output
            y = (C_t @ h)[0, 0]
            outputs.append(y)

        return np.array(outputs)


class StateSpaceTimeSeries:
    """State space model for time series forecasting"""

    def __init__(self, model_type: str = "mamba", state_dim: int = 32):
        """
        Initialize state space time series model

        Args:
            model_type: 's4' or 'mamba'
            state_dim: State dimension
        """
        self.model_type = model_type
        self.state_dim = state_dim

        if model_type == "s4":
            self.layer = S4Layer(state_dim)
        elif model_type == "mamba":
            self.layer = MambaLayer(state_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forecast(self, history: np.ndarray, horizon: int = 10) -> np.ndarray:
        """
        Forecast future values

        Args:
            history: Historical sequence
            horizon: Number of steps to forecast

        Returns:
            Forecasted sequence
        """
        # Normalize history
        mu = np.mean(history)
        sigma = np.std(history) + 1e-8
        history_norm = (history - mu) / sigma

        # Compute features from history
        if self.model_type == "s4":
            output = self.layer.forward(history_norm)
        else:
            output = self.layer.selective_forward(history_norm)

        # Extract trend and seasonality
        trend = np.mean(output[-10:]) if len(output) >= 10 else 0
        seasonal = output[-1] if len(output) > 0 else 0

        # Forecast (simple extrapolation)
        forecast = []
        last_value = history_norm[-1]

        for step in range(horizon):
            next_val = trend * 0.1 + seasonal * 0.9 + np.random.randn() * 0.01
            forecast.append(next_val)
            seasonal = next_val

        forecast = np.array(forecast)

        # Denormalize
        forecast = forecast * sigma + mu

        return forecast

    def efficiency_metrics(self, seq_len: int = 256) -> Dict[str, float]:
        """
        Compute efficiency metrics vs Transformers

        Args:
            seq_len: Sequence length

        Returns:
            Metrics dict
        """
        # SSM complexity: O(seq_len * state_dim)
        ssm_flops = seq_len * self.state_dim

        # Transformer complexity: O(seq_len^2)
        transformer_flops = seq_len ** 2

        # Speedup
        speedup = transformer_flops / (ssm_flops + 1e-8)

        return {
            'ssm_flops': float(ssm_flops),
            'transformer_flops': float(transformer_flops),
            'speedup': float(speedup),
            'memory_efficient': True,
            'parallelizable': True if self.model_type == "s4" else False
        }


if __name__ == "__main__":
    logger.info("State Space Models for Time Series")
    logger.info("=" * 50)

    np.random.seed(42)

    # Generate synthetic time series
    logger.info("\nGenerating synthetic time series")
    n_periods = 256
    trend = np.linspace(0, 1, n_periods)
    seasonality = np.sin(np.arange(n_periods) * 2 * np.pi / 50)
    noise = np.random.randn(n_periods) * 0.1
    time_series = 100 + trend * 10 + seasonality * 5 + noise

    logger.info(f"  Length: {n_periods}")
    logger.info(f"  Range: [{time_series.min():.2f}, {time_series.max():.2f}]")

    # S4 Model
    logger.info("\nS4 Model")
    s4_model = StateSpaceTimeSeries(model_type="s4", state_dim=32)

    logger.info("  Forecasting with S4")
    s4_forecast = s4_model.forecast(time_series, horizon=20)
    logger.info(f"  S4 forecast (5 steps): {s4_forecast[:5]}")
    logger.info(f"  S4 forecast range: [{s4_forecast.min():.2f}, {s4_forecast.max():.2f}]")

    # Efficiency
    s4_metrics = s4_model.efficiency_metrics(seq_len=256)
    logger.info(f"  S4 FLOPs: {s4_metrics['ssm_flops']:.0f}")
    logger.info(f"  Speedup vs Transformer: {s4_metrics['speedup']:.2f}x")

    # Mamba Model
    logger.info("\nMamba Model")
    mamba_model = StateSpaceTimeSeries(model_type="mamba", state_dim=32)

    logger.info("  Forecasting with Mamba")
    mamba_forecast = mamba_model.forecast(time_series, horizon=20)
    logger.info(f"  Mamba forecast (5 steps): {mamba_forecast[:5]}")
    logger.info(f"  Mamba forecast range: [{mamba_forecast.min():.2f}, {mamba_forecast.max():.2f}]")

    # Efficiency
    mamba_metrics = mamba_model.efficiency_metrics(seq_len=256)
    logger.info(f"  Mamba FLOPs: {mamba_metrics['ssm_flops']:.0f}")
    logger.info(f"  Speedup vs Transformer: {mamba_metrics['speedup']:.2f}x")

    # Comparison
    logger.info("\nModel Comparison")
    mae_s4 = np.mean(np.abs(s4_forecast[:10] - time_series[-10:]))
    mae_mamba = np.mean(np.abs(mamba_forecast[:10] - time_series[-10:]))

    logger.info(f"  S4 MAE (vs recent): {mae_s4:.4f}")
    logger.info(f"  Mamba MAE (vs recent): {mae_mamba:.4f}")

    logger.info("\nState Space Models Complete")
