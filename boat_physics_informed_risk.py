#!/usr/bin/env python3
"""
Physics-Informed Risk Management System
========================================

Practical risk framework embedding physical conservation laws:
  - Mean reversion (spring-like dynamics)
  - Momentum conservation (inertia effects)
  - Energy dissipation (friction/decay)
  - Portfolio heat flow (risk distribution)
  - Stress testing via impulse responses

Based on practical finance: treat portfolios as physical systems.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    var_95: float  # Value at Risk (95%)
    var_99: float  # Value at Risk (99%)
    expected_shortfall: float  # Conditional VaR
    systemic_risk: float  # System-wide risk
    concentration_risk: float  # Diversification measure
    stress_pnl: float  # P&L under stress scenario


class MeanReversionPhysics:
    """Model price mean reversion as spring dynamics"""

    def __init__(self, equilibrium: float, stiffness: float = 0.1):
        """
        Initialize mean reversion model

        Args:
            equilibrium: Long-term equilibrium price
            stiffness: Spring constant (higher = faster reversion)
        """
        self.equilibrium = equilibrium
        self.stiffness = stiffness  # k in F = -k*x

    def expected_return(self, current_price: float) -> float:
        """
        Compute expected return using Hooke's law
        F = -k(x - eq), so return ≈ -k * deviation
        """
        deviation = current_price - self.equilibrium
        return -self.stiffness * deviation / (current_price + 1e-8)

    def forecast_price(self, current_price: float, periods: int = 20) -> np.ndarray:
        """Forecast price path using mean reversion dynamics"""
        prices = np.zeros(periods)
        prices[0] = current_price

        for t in range(1, periods):
            ret = self.expected_return(prices[t-1])
            shock = np.random.randn() * 0.02
            prices[t] = prices[t-1] * (1 + ret + shock)

        return prices


class MomentumInertia:
    """Model momentum as inertial effects"""

    def __init__(self, friction_coeff: float = 0.05):
        """
        Initialize momentum model

        Args:
            friction_coeff: Friction/decay coefficient
        """
        self.friction = friction_coeff

    def momentum_factor(self, returns: np.ndarray, lookback: int = 20) -> float:
        """
        Compute momentum factor (like velocity in physics)
        Momentum = sum(returns) - friction * volatility
        """
        if len(returns) < lookback:
            lookback = len(returns)

        recent_returns = returns[-lookback:]
        momentum = np.sum(recent_returns)
        volatility = np.std(recent_returns)

        # Friction dampens momentum
        adjusted = momentum - self.friction * volatility

        return float(adjusted)

    def stress_impact(self, momentum: float, shock_size: float = 0.05) -> float:
        """
        Impact of market shock on position with momentum
        Shock impact = shock_size * (1 + |momentum|)
        """
        return shock_size * (1 + abs(momentum))


class PortfolioHeatFlow:
    """Model risk distribution as heat flow through portfolio"""

    def __init__(self, n_assets: int):
        """Initialize portfolio heat model"""
        self.n_assets = n_assets

    def heat_diffusion(self, weights: np.ndarray, volatilities: np.ndarray,
                       time_steps: int = 10) -> np.ndarray:
        """
        Simulate risk (heat) diffusion through portfolio
        Higher concentration → higher local risk
        """
        heat = weights * volatilities
        smoothed_heat = heat.copy()

        for _ in range(time_steps):
            # Diffusion: risk spreads to neighbors
            new_heat = smoothed_heat.copy()
            for i in range(self.n_assets):
                left = (i - 1) % self.n_assets
                right = (i + 1) % self.n_assets
                # Heat spreads: each asset affected by neighbors
                new_heat[i] = 0.6 * smoothed_heat[i] + 0.2 * smoothed_heat[left] + 0.2 * smoothed_heat[right]
            smoothed_heat = new_heat

        return smoothed_heat

    def concentration_risk(self, weights: np.ndarray, volatilities: np.ndarray) -> float:
        """
        Measure concentration using Herfindahl index
        Higher weights on high-vol assets increase concentration risk
        """
        weighted_vols = weights * volatilities
        concentration = np.sum(weighted_vols ** 2)
        return float(concentration)


class PhysicsInformedRiskManager:
    """Complete physics-informed risk management framework"""

    def __init__(self, n_assets: int):
        """Initialize risk manager"""
        self.n_assets = n_assets
        self.mean_reversion = MeanReversionPhysics(equilibrium=100.0, stiffness=0.1)
        self.momentum = MomentumInertia(friction_coeff=0.05)
        self.heat_flow = PortfolioHeatFlow(n_assets)

    def compute_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Compute Value at Risk using historical method

        Args:
            returns: Return distribution
            confidence: Confidence level (0.95 for 95% VaR)

        Returns:
            VaR estimate (negative = loss threshold)
        """
        return float(np.quantile(returns, 1 - confidence))

    def compute_expected_shortfall(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Compute Expected Shortfall (CVaR)
        Mean loss given VaR threshold
        """
        var = self.compute_var(returns, confidence)
        return float(np.mean(returns[returns <= var]))

    def stress_test(self, prices: np.ndarray, weights: np.ndarray,
                   shock_magnitude: float = 0.10) -> Dict:
        """
        Stress test portfolio under market shock

        Args:
            prices: Current prices
            weights: Portfolio weights
            shock_magnitude: Size of shock (e.g., 10%)

        Returns:
            Stress test results
        """
        # Apply shock to prices
        shocked_prices = prices * (1 - shock_magnitude)

        # Compute price changes
        price_changes = (shocked_prices - prices) / (prices + 1e-8)

        # Portfolio P&L under stress
        portfolio_pnl = np.sum(weights * price_changes)

        # Volatility increases in stress
        stress_volatility = np.std(price_changes) * 2.5

        return {
            'portfolio_pnl': float(portfolio_pnl),
            'stress_volatility': float(stress_volatility),
            'max_asset_loss': float(np.min(price_changes)),
            'var_increase': float(stress_volatility / np.std(price_changes))
        }

    def analyze_portfolio(self, prices: np.ndarray, weights: np.ndarray,
                         returns: np.ndarray) -> RiskMetrics:
        """
        Complete portfolio risk analysis

        Args:
            prices: Current asset prices (n_assets,)
            weights: Portfolio weights (n_assets,)
            returns: Historical returns (n_periods, n_assets)

        Returns:
            Comprehensive risk metrics
        """
        # Compute volatilities
        volatilities = np.std(returns, axis=0)

        # Portfolio returns
        portfolio_returns = returns @ weights

        # VaR metrics
        var_95 = self.compute_var(portfolio_returns, 0.95)
        var_99 = self.compute_var(portfolio_returns, 0.99)
        es = self.compute_expected_shortfall(portfolio_returns, 0.95)

        # Systemic risk (correlation-based)
        corr_matrix = np.corrcoef(returns.T)
        systemic = np.mean(corr_matrix[~np.eye(len(corr_matrix), dtype=bool)])

        # Concentration risk
        concentration = self.heat_flow.concentration_risk(weights, volatilities)

        # Stress test
        stress_result = self.stress_test(prices, weights, shock_magnitude=0.10)

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=es,
            systemic_risk=float(systemic),
            concentration_risk=float(concentration),
            stress_pnl=stress_result['portfolio_pnl']
        )


if __name__ == "__main__":
    logger.info("Physics-Informed Risk Management System")
    logger.info("=" * 60)

    np.random.seed(42)

    # Generate synthetic market data
    logger.info("\nGenerating synthetic market data")
    n_assets = 8
    n_periods = 200

    prices = 100 * np.ones(n_assets)
    returns_list = []

    for t in range(n_periods):
        shocks = np.random.randn(n_assets) * 0.02
        prices = prices * (1 + shocks)
        returns_list.append(shocks)

    returns = np.array(returns_list)
    weights = np.array([0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10])

    logger.info(f"  Assets: {n_assets}")
    logger.info(f"  Periods: {n_periods}")
    logger.info(f"  Final prices: {prices}")

    # Initialize risk manager
    logger.info("\nInitializing Physics-Informed Risk Manager")
    rm = PhysicsInformedRiskManager(n_assets)

    # Analyze portfolio
    logger.info("\nAnalyzing portfolio risk")
    metrics = rm.analyze_portfolio(prices, weights, returns)

    logger.info(f"\nValue at Risk (VaR):")
    logger.info(f"  95% VaR: {metrics.var_95:.4f} (loss threshold)")
    logger.info(f"  99% VaR: {metrics.var_99:.4f}")
    logger.info(f"  Expected Shortfall: {metrics.expected_shortfall:.4f}")

    logger.info(f"\nRisk Metrics:")
    logger.info(f"  Systemic Risk (correlation): {metrics.systemic_risk:.4f}")
    logger.info(f"  Concentration Risk: {metrics.concentration_risk:.4f}")
    logger.info(f"  Stress P&L (10% shock): {metrics.stress_pnl:.4f}")

    # Mean reversion analysis
    logger.info(f"\nMean Reversion Dynamics (Physics):")
    eq_price = prices[0]
    forecast = rm.mean_reversion.forecast_price(eq_price, periods=20)
    logger.info(f"  Equilibrium: {rm.mean_reversion.equilibrium:.2f}")
    logger.info(f"  Current: {eq_price:.2f}")
    logger.info(f"  Forecast (20 periods): {forecast[-1]:.2f}")
    logger.info(f"  Expected return: {rm.mean_reversion.expected_return(eq_price):.4f}")

    # Momentum analysis
    logger.info(f"\nMomentum Inertia (Physics):")
    momentum = rm.momentum.momentum_factor(returns[:, 0], lookback=20)
    shock_impact = rm.momentum.stress_impact(momentum)
    logger.info(f"  Momentum factor: {momentum:.4f}")
    logger.info(f"  Stress amplification: {shock_impact:.4f}x")

    # Heat flow analysis
    logger.info(f"\nPortfolio Heat Flow (Risk Distribution):")
    heat = rm.heat_flow.heat_diffusion(weights, np.std(returns, axis=0))
    logger.info(f"  Risk concentration by asset:")
    for i in range(min(5, n_assets)):
        logger.info(f"    Asset {i}: {heat[i]:.4f}")

    logger.info("\nPhysics-Informed Risk Management Complete")
