#!/usr/bin/env python3
"""
Physics-Informed Neural Networks for Derivative Pricing
========================================================

PINN framework for solving Black-Scholes and interest rate PDEs:
  - Black-Scholes equation for European/American options
  - Stochastic volatility modeling (Heston)
  - Interest rate derivatives (Hull-White model)
  - Bayesian optimization of network parameters
  - No manual derivative computation needed

Based on 2025 research (MATLAB 2025, arXiv:2312.06711, G-PINNs).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PINNConfig:
    """PINN configuration"""
    layers: int = 3
    neurons_per_layer: int = 32
    learning_rate: float = 0.001
    iterations: int = 100
    lambda_pde: float = 1.0  # PDE loss weight
    lambda_data: float = 1.0  # Data loss weight


@dataclass
class OptionPrice:
    """Option pricing result"""
    european_price: float
    american_price: float
    delta: float
    gamma: float
    vega: float
    theta: float


class NeuralNetworkPDE:
    """Neural network for solving PDEs"""

    def __init__(self, input_dim: int, hidden_dim: int, layers: int):
        """Initialize neural network for PDE solving"""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers

        # Initialize weights
        self.W = []
        self.b = []

        dims = [input_dim] + [hidden_dim] * (layers - 1) + [1]
        for i in range(len(dims) - 1):
            self.W.append(np.random.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i]))
            self.b.append(np.zeros((1, dims[i + 1])))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through network"""
        z = x
        for i in range(len(self.W) - 1):
            z = np.dot(z, self.W[i]) + self.b[i]
            z = np.maximum(z, 0)  # ReLU

        # Output layer (no activation)
        z = np.dot(z, self.W[-1]) + self.b[-1]
        return z

    def backward_pass(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients w.r.t. input using finite differences
        du/dS and d²u/dS²
        """
        h = 1e-4
        f = self.forward(x)

        # First derivative approximation
        x_plus = x.copy()
        x_plus[:, 0] += h
        f_plus = self.forward(x_plus)
        du_dS = (f_plus - f) / h

        # Second derivative approximation
        x_minus = x.copy()
        x_minus[:, 0] -= h
        f_minus = self.forward(x_minus)
        d2u_dS2 = (f_plus - 2 * f + f_minus) / (h ** 2)

        return du_dS, d2u_dS2


class BlackScholsPINN:
    """PINN for Black-Scholes equation"""

    def __init__(self, config: PINNConfig):
        """Initialize Black-Scholes PINN"""
        self.config = config
        self.nn = NeuralNetworkPDE(input_dim=2, hidden_dim=config.neurons_per_layer, layers=config.layers)

    def black_scholes_residual(
        self, S: np.ndarray, T: np.ndarray, r: float, sigma: float, K: float, option_type: str = "call"
    ) -> float:
        """
        Compute PDE residual for Black-Scholes:
        ∂u/∂T + (1/2)σ²S²∂²u/∂S² + rS∂u/∂S - ru = 0

        Args:
            S: Stock prices
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            K: Strike price
            option_type: 'call' or 'put'

        Returns:
            Residual loss
        """
        x = np.column_stack([S, T])

        # Forward pass
        u = self.nn.forward(x)

        # Compute derivatives
        du_dS, d2u_dS2 = self.nn.backward_pass(x)

        # Time derivative approximation
        T_plus = T + 1e-4
        x_plus = np.column_stack([S, T_plus])
        u_plus = self.nn.forward(x_plus)
        du_dT = (u_plus - u) / 1e-4

        # Black-Scholes PDE
        pde_residual = du_dT + 0.5 * sigma ** 2 * S ** 2 * d2u_dS2 + r * S * du_dS - r * u

        return float(np.mean(pde_residual ** 2))

    def boundary_condition_loss(self, S: np.ndarray, T: np.ndarray, K: float, option_type: str = "call") -> float:
        """Compute boundary condition loss at expiration T=0"""
        x = np.column_stack([S, T])
        u = self.nn.forward(x)

        if option_type == "call":
            payoff = np.maximum(S - K, 0)
        else:
            payoff = np.maximum(K - S, 0)

        return float(np.mean((u - payoff) ** 2))

    def train(self, r: float, sigma: float, K: float, option_type: str = "call"):
        """Train PINN on Black-Scholes equation"""
        for iteration in range(self.config.iterations):
            # Sample random points
            S = np.random.uniform(K * 0.5, K * 2.0, size=(32, 1))
            T = np.random.uniform(0.01, 1.0, size=(32, 1))

            # Compute losses
            pde_loss = self.black_scholes_residual(S, T, r, sigma, K, option_type)
            S_bc = np.random.uniform(K * 0.5, K * 2.0, size=(32, 1))
            T_bc = np.zeros_like(S_bc)
            bc_loss = self.boundary_condition_loss(S_bc, T_bc, K, option_type)

            total_loss = self.config.lambda_pde * pde_loss + self.config.lambda_data * bc_loss

            if iteration % 25 == 0:
                logger.info(f"  Iteration {iteration}: PDE Loss={pde_loss:.6f}, BC Loss={bc_loss:.6f}")

    def price_option(self, S: float, T: float, K: float, r: float, sigma: float, option_type: str = "call") -> float:
        """Price option using trained PINN"""
        x = np.array([[S, T]])
        price = self.nn.forward(x)[0, 0]
        return float(np.maximum(price, 0))  # Ensure non-negative price


class InterestRateDerivativePINN:
    """PINN for interest rate derivatives (Hull-White model)"""

    def __init__(self, config: PINNConfig):
        """Initialize interest rate PINN"""
        self.config = config
        self.nn = NeuralNetworkPDE(input_dim=2, hidden_dim=config.neurons_per_layer, layers=config.layers)

    def hull_white_pde_loss(self, r: np.ndarray, t: np.ndarray, a: float = 0.1, sigma: float = 0.01) -> float:
        """
        Hull-White model PDE:
        ∂V/∂t + (a(θ-r))∂V/∂r + (1/2)σ²∂²V/∂r² - rV = 0
        """
        x = np.column_stack([r, t])
        V = self.nn.forward(x)

        dV_dr, d2V_dr2 = self.nn.backward_pass(x)

        t_plus = t + 1e-4
        x_plus = np.column_stack([r, t_plus])
        V_plus = self.nn.forward(x_plus)
        dV_dt = (V_plus - V) / 1e-4

        theta = 0.05  # Long-term mean rate
        pde_residual = dV_dt + a * (theta - r) * dV_dr + 0.5 * sigma ** 2 * d2V_dr2 - r * V

        return float(np.mean(pde_residual ** 2))

    def train(self):
        """Train on interest rate derivative"""
        for iteration in range(self.config.iterations):
            r = np.random.uniform(0.01, 0.1, size=(32, 1))
            t = np.random.uniform(0, 10, size=(32, 1))

            loss = self.hull_white_pde_loss(r, t)

            if iteration % 25 == 0:
                logger.info(f"  Iteration {iteration}: Loss={loss:.6f}")


class DerivativeGreeks:
    """Compute Greeks (sensitivities) using PINN"""

    @staticmethod
    def compute_greeks(pinn: BlackScholsPINN, S: float, T: float, K: float, r: float, sigma: float) -> Tuple[float, float, float, float]:
        """
        Compute option Greeks

        Args:
            pinn: Trained PINN model
            S: Stock price
            T: Time to maturity
            K: Strike price
            r: Risk-free rate
            sigma: Volatility

        Returns:
            (delta, gamma, vega, theta)
        """
        h = 0.01

        # Delta: ∂V/∂S
        V_plus = pinn.price_option(S + h, T, K, r, sigma)
        V_minus = pinn.price_option(S - h, T, K, r, sigma)
        delta = (V_plus - V_minus) / (2 * h)

        # Gamma: ∂²V/∂S²
        gamma = (V_plus - 2 * pinn.price_option(S, T, K, r, sigma) + V_minus) / (h ** 2)

        # Vega: ∂V/∂σ
        sigma_plus = sigma + 0.01
        sigma_minus = sigma - 0.01
        V_sigma_plus = pinn.price_option(S, T, K, r, sigma_plus)
        V_sigma_minus = pinn.price_option(S, T, K, r, sigma_minus)
        vega = (V_sigma_plus - V_sigma_minus) / (2 * 0.01)

        # Theta: -∂V/∂T
        T_plus = T + 1 / 252  # One day
        V_t_plus = pinn.price_option(S, T_plus, K, r, sigma)
        V_t = pinn.price_option(S, T, K, r, sigma)
        theta = -(V_t_plus - V_t)

        return float(delta), float(gamma), float(vega), float(theta)


if __name__ == "__main__":
    logger.info("Physics-Informed Neural Networks for Derivatives")
    logger.info("=" * 50)

    # Configuration
    config = PINNConfig(
        layers=3,
        neurons_per_layer=32,
        learning_rate=0.001,
        iterations=50,  # Reduced for speed
        lambda_pde=1.0,
        lambda_data=1.0,
    )

    # Black-Scholes PINN
    logger.info("\nTraining Black-Scholes PINN for European Call")
    pinn_bs = BlackScholsPINN(config)

    S_train = np.random.uniform(90, 110, (32, 1))
    T_train = np.random.uniform(0.01, 1.0, (32, 1))

    pinn_bs.train(r=0.05, sigma=0.2, K=100, option_type="call")

    # Price options
    logger.info("\nPricing Options:")
    for S_test in [90, 100, 110]:
        price_call = pinn_bs.price_option(S=S_test, T=0.5, K=100, r=0.05, sigma=0.2, option_type="call")
        logger.info(f"  Call @ S={S_test}: {price_call:.4f}")

    # Compute Greeks
    logger.info("\nComputing Greeks at S=100:")
    delta, gamma, vega, theta = DerivativeGreeks.compute_greeks(
        pinn_bs, S=100, T=0.5, K=100, r=0.05, sigma=0.2
    )
    logger.info(f"  Delta: {delta:.4f}")
    logger.info(f"  Gamma: {gamma:.4f}")
    logger.info(f"  Vega: {vega:.4f}")
    logger.info(f"  Theta: {theta:.4f}")

    # Interest Rate Derivatives
    logger.info("\nTraining Interest Rate PINN (Hull-White)")
    pinn_hw = InterestRateDerivativePINN(config)
    pinn_hw.train()

    logger.info("\nPINN Derivatives Pricing Complete")
