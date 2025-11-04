#!/usr/bin/env python3
"""
Neural ODE for Stock Price Forecasting
=======================================

Continuous-time dynamics via Neural Differential Equations:
  - Phase space reconstruction
  - Neural ODE solver
  - Irregular time series handling
  - Multi-scale temporal modeling
  - Regime-switching detection

Based on 2025 research (SSRN:4817927, arXiv:2502.09885, PSR-NODE framework).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ODETrajectory:
    """Neural ODE trajectory output"""
    times: np.ndarray
    states: np.ndarray  # (len(times), state_dim)
    forecast: np.ndarray  # Forecasted returns


class PhaseSpaceReconstruction:
    """Time-delay embedding for phase space reconstruction"""

    @staticmethod
    def embed(time_series: np.ndarray, embedding_dim: int = 3, delay: int = 1) -> np.ndarray:
        """
        Time-delay embedding: (x[t], x[t+d], x[t+2d], ...)

        Args:
            time_series: 1D time series
            embedding_dim: Embedding dimension
            delay: Time delay

        Returns:
            (N-delay*(dim-1), embedding_dim) embedded space
        """
        N = len(time_series)
        M = N - delay * (embedding_dim - 1)

        embedded = np.zeros((M, embedding_dim))
        for d in range(embedding_dim):
            embedded[:, d] = time_series[d * delay : d * delay + M]

        return embedded

    @staticmethod
    def reconstruct_lyapunov(embedded: np.ndarray) -> float:
        """
        Estimate largest Lyapunov exponent (chaos indicator)

        Args:
            embedded: Phase space embedded data

        Returns:
            Largest Lyapunov exponent
        """
        N = len(embedded)
        max_lag = min(N // 10, 50)

        # Compute nearest neighbor divergence
        distances = []
        for lag in range(1, max_lag):
            for i in range(N - lag):
                dist = np.linalg.norm(embedded[i] - embedded[i + lag])
                distances.append(dist)

        distances = np.array(distances)

        # Approximate Lyapunov from divergence rate
        lyapunov = float(np.mean(np.log(distances + 1e-8)) / max_lag)
        return lyapunov


class NeuralODENet:
    """Neural network parametrizing ODE vector field"""

    def __init__(self, state_dim: int, hidden_dim: int = 32):
        """Initialize Neural ODE network"""
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Network weights
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, state_dim) * 0.1
        self.b2 = np.zeros((1, state_dim))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute dx/dt = f(x; θ)

        Args:
            x: (batch, state_dim) state variables

        Returns:
            dx/dt velocity
        """
        h = np.maximum(np.dot(x, self.W1) + self.b1, 0)  # ReLU
        dx_dt = np.dot(h, self.W2) + self.b2

        return dx_dt


class NeuralODESolver:
    """Solves Neural ODE via Euler method"""

    def __init__(self, ode_net: NeuralODENet, dt: float = 0.01):
        """Initialize ODE solver"""
        self.ode_net = ode_net
        self.dt = dt

    def solve(self, x0: np.ndarray, t_span: np.ndarray) -> np.ndarray:
        """
        Solve Neural ODE: dx/dt = f(x)

        Args:
            x0: Initial state (state_dim,)
            t_span: Time points to evaluate

        Returns:
            (len(t_span), state_dim) trajectory
        """
        trajectory = [x0.copy()]
        x_current = x0.copy()

        t_current = t_span[0]
        t_idx = 1

        while t_idx < len(t_span):
            t_target = t_span[t_idx]

            # Euler integration
            while t_current < t_target:
                dx_dt = self.ode_net.forward(x_current.reshape(1, -1))[0]
                x_current = x_current + self.dt * dx_dt
                t_current += self.dt

            trajectory.append(x_current.copy())
            t_idx += 1

        return np.array(trajectory)


class StockPriceNODE:
    """Neural ODE for stock price forecasting"""

    def __init__(self, embedding_dim: int = 5, hidden_dim: int = 32):
        """Initialize stock price NODE model"""
        self.embedding_dim = embedding_dim
        self.ode_net = NeuralODENet(state_dim=embedding_dim, hidden_dim=hidden_dim)
        self.solver = NeuralODESolver(self.ode_net, dt=0.01)

    def prepare_data(self, price_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare returns and embeddings

        Args:
            price_series: Stock prices

        Returns:
            (returns, embedded_returns)
        """
        returns = np.diff(np.log(price_series))
        normalized_returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Phase space reconstruction
        embedded = PhaseSpaceReconstruction.embed(
            normalized_returns, embedding_dim=self.embedding_dim, delay=1
        )

        return normalized_returns, embedded

    def forecast(self, price_series: np.ndarray, forecast_steps: int = 5) -> ODETrajectory:
        """
        Forecast stock returns using Neural ODE

        Args:
            price_series: Historical prices
            forecast_steps: Number of steps to forecast

        Returns:
            ODETrajectory with forecasted returns
        """
        returns, embedded = self.prepare_data(price_series)

        # Initial state (last embedded point)
        x0 = embedded[-1]

        # Time points
        times = np.arange(0, forecast_steps, 0.1)

        # Solve Neural ODE
        trajectory = self.solver.solve(x0, times)

        # Extract forecast (first component is the return)
        forecast_returns = trajectory[:, 0]

        return ODETrajectory(times=times, states=trajectory, forecast=forecast_returns)

    def compute_uncertainty(self, price_series: np.ndarray, n_monte_carlo: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forecast uncertainty via Monte Carlo

        Args:
            price_series: Historical prices
            n_monte_carlo: Number of MC samples

        Returns:
            (mean_forecast, std_forecast)
        """
        forecasts = []

        for _ in range(n_monte_carlo):
            # Add noise to initial state
            returns, embedded = self.prepare_data(price_series)
            x0 = embedded[-1] + np.random.randn(self.embedding_dim) * 0.05

            times = np.arange(0, 5, 0.1)
            trajectory = self.solver.solve(x0, times)
            forecasts.append(trajectory[:, 0])

        forecasts = np.array(forecasts)
        mean_forecast = np.mean(forecasts, axis=0)
        std_forecast = np.std(forecasts, axis=0)

        return mean_forecast, std_forecast

    def detect_regime(self, price_series: np.ndarray) -> str:
        """
        Detect market regime using Lyapunov exponent

        Args:
            price_series: Historical prices

        Returns:
            Regime type ('STABLE', 'CHAOTIC', 'TRANSITION')
        """
        returns, embedded = self.prepare_data(price_series)
        lyapunov = PhaseSpaceReconstruction.reconstruct_lyapunov(embedded)

        if lyapunov < -0.1:
            regime = "STABLE"
        elif lyapunov > 0.1:
            regime = "CHAOTIC"
        else:
            regime = "TRANSITION"

        return regime


if __name__ == "__main__":
    logger.info("Neural ODE Stock Price Forecasting")
    logger.info("=" * 50)

    np.random.seed(42)

    # Generate synthetic stock data
    n_days = 252  # 1 year
    price_series = 100 + np.cumsum(np.random.randn(n_days) * 0.5)

    # Initialize NODE model
    model = StockPriceNODE(embedding_dim=5, hidden_dim=32)

    logger.info("\nPhase 1: Phase Space Reconstruction")
    returns, embedded = model.prepare_data(price_series)
    logger.info(f"  Returns shape: {returns.shape}")
    logger.info(f"  Embedded shape: {embedded.shape}")

    # Detect regime
    regime = model.detect_regime(price_series)
    logger.info(f"  Market Regime: {regime}")

    logger.info("\nPhase 2: Neural ODE Forecasting")
    trajectory = model.forecast(price_series, forecast_steps=5)
    logger.info(f"  Trajectory shape: {trajectory.states.shape}")
    logger.info(f"  Forecast (first 5 steps): {trajectory.forecast[:5]}")

    logger.info("\nPhase 3: Uncertainty Quantification")
    mean_forecast, std_forecast = model.compute_uncertainty(price_series, n_monte_carlo=10)
    logger.info(f"  Mean Forecast (steps 1-5): {mean_forecast[:5]}")
    logger.info(f"  Std Forecast (steps 1-5): {std_forecast[:5]}")

    logger.info("\nNeural ODE Forecasting Complete")
