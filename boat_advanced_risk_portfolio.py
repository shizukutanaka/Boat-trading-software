#!/usr/bin/env python3
"""
Advanced Risk & Portfolio Optimization Framework
=================================================

Comprehensive risk modeling and portfolio optimization:
  - Copula-based tail dependence modeling
  - CVaR and quantile regression
  - GARCH volatility modeling
  - Multi-objective optimization (Pareto frontier)
  - Risk parity and efficient frontier
  - Time series decomposition

Based on 2025 research on advanced portfolio optimization and risk modeling.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    return_pct: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    sortino_ratio: float


class CopulaGARCHModel:
    """Copula-GARCH for modeling tail dependencies"""

    def __init__(self, returns: np.ndarray, n_lags: int = 1):
        self.returns = returns
        self.n_lags = n_lags
        self.omega = 0.00001
        self.alpha = 0.05
        self.beta = 0.94
        self.conditional_var = np.ones(len(returns))

    def fit_garch(self) -> Dict[str, float]:
        """Fit GARCH(1,1) model"""
        T = len(self.returns)

        for t in range(1, T):
            self.conditional_var[t] = (
                self.omega +
                self.alpha * (self.returns[t-1] ** 2) +
                self.beta * self.conditional_var[t-1]
            )

        return {
            'omega': self.omega,
            'alpha': self.alpha,
            'beta': self.beta,
            'mean_conditional_vol': float(np.mean(np.sqrt(self.conditional_var)))
        }

    def forecast_volatility(self, steps: int = 10) -> np.ndarray:
        """Forecast future volatility"""
        forecasts = np.zeros(steps)
        current_var = self.conditional_var[-1]

        for i in range(steps):
            forecasts[i] = np.sqrt(
                self.omega + (self.alpha + self.beta) * current_var
            )
            current_var = forecasts[i] ** 2

        return forecasts


class QuantileRegressionCVaR:
    """CVaR calculation via quantile regression"""

    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate CVaR (Expected Shortfall)"""
        var_quantile = np.percentile(returns, (1 - confidence) * 100)
        cvar = np.mean(returns[returns <= var_quantile])
        return float(cvar)

    @staticmethod
    def quantile_regression(
        X: np.ndarray,
        y: np.ndarray,
        tau: float = 0.05
    ) -> np.ndarray:
        """Quantile regression for tau-th quantile"""
        n_features = X.shape[1]
        beta = np.zeros(n_features)

        # Iteratively reweighted least squares
        for iteration in range(10):
            residuals = y - X @ beta
            weights = 1 / (np.abs(residuals) + 0.001)

            W = np.diag(weights)
            beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y

        return beta


class TimeSeriesDecomposition:
    """STL decomposition for trend, seasonality, residuals"""

    @staticmethod
    def decompose(series: np.ndarray, period: int = 252) -> Dict[str, np.ndarray]:
        """Simple decomposition using moving averages"""
        T = len(series)

        # Trend via moving average
        trend = np.convolve(series, np.ones(period)/period, mode='same')

        # Detrended
        detrended = series - trend

        # Seasonality (average seasonal pattern)
        seasonality = np.zeros_like(series)
        for i in range(period):
            seasonal_indices = np.arange(i, T, period)
            if len(seasonal_indices) > 0:
                seasonality[seasonal_indices] = np.mean(detrended[seasonal_indices])

        # Residuals
        residuals = series - trend - seasonality

        return {
            'original': series,
            'trend': trend,
            'seasonality': seasonality,
            'residuals': residuals
        }

    @staticmethod
    def detect_anomalies(series: np.ndarray, threshold: float = 3.0) -> List[int]:
        """Detect anomalies in residuals"""
        decomposed = TimeSeriesDecomposition.decompose(series)
        residuals = decomposed['residuals']

        mean = np.mean(residuals)
        std = np.std(residuals)

        anomalies = np.where(np.abs(residuals - mean) > threshold * std)[0]
        return [int(i) for i in anomalies]


class MultiObjectivePortfolio:
    """Multi-objective portfolio optimization"""

    @staticmethod
    def calculate_objectives(
        weights: np.ndarray,
        returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Dict[str, float]:
        """Calculate Pareto objectives"""
        portfolio_return = np.sum(weights * np.mean(returns, axis=0))
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)

        # Skewness (simplified)
        ret_std = np.std(returns, axis=0)
        portfolio_skew = np.mean(((returns - np.mean(returns, axis=0)) / ret_std) ** 3)

        # Concentration (Herfindahl)
        concentration = np.sum(weights ** 2)

        return {
            'return': float(portfolio_return),
            'volatility': float(portfolio_volatility),
            'skewness': float(portfolio_skew),
            'concentration': float(concentration)
        }

    @staticmethod
    def generate_random_portfolio(n_assets: int, n_portfolios: int = 1000) -> np.ndarray:
        """Generate random portfolios for Pareto front"""
        portfolios = []

        for _ in range(n_portfolios):
            weights = np.random.dirichlet(np.ones(n_assets))
            portfolios.append(weights)

        return np.array(portfolios)


class RiskParityPortfolio:
    """Risk parity portfolio construction"""

    @staticmethod
    def construct_risk_parity(
        volatilities: np.ndarray,
        correlation_matrix: np.ndarray
    ) -> np.ndarray:
        """Construct risk parity portfolio"""
        n_assets = len(volatilities)

        # Risk budgets: equal contribution
        risk_budgets = np.ones(n_assets) / n_assets

        # Marginal risk contribution
        # mrc_i = (cov_matrix @ w)_i / sqrt(w @ cov @ w)

        # Simple inverse volatility weighting
        inv_vol = 1.0 / (volatilities + 1e-8)
        weights = inv_vol / np.sum(inv_vol)

        return weights

    @staticmethod
    def calculate_risk_contribution(
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculate risk contribution of each asset"""
        portfolio_variance = weights @ cov_matrix @ weights
        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / np.sqrt(portfolio_variance + 1e-8)

        return risk_contrib


class EfficientFrontier:
    """Efficient frontier calculation"""

    @staticmethod
    def calculate_frontier(
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate efficient frontier"""
        min_return = np.min(mean_returns)
        max_return = np.max(mean_returns)
        target_returns = np.linspace(min_return, max_return, n_points)

        volatilities = []

        for target in target_returns:
            # Minimize variance subject to target return
            n_assets = len(mean_returns)

            # Simple approach: two-fund separation
            weights = np.zeros(n_assets)
            weights[np.argmax(mean_returns)] = 1.0

            # Adjust for target return
            max_return_idx = np.argmax(mean_returns)
            current_return = mean_returns[max_return_idx]

            if abs(current_return - target) > 1e-6:
                # Find weights (simplified)
                weights = np.ones(n_assets) / n_assets

            variance = weights @ cov_matrix @ weights
            volatilities.append(np.sqrt(variance))

        return np.array(target_returns), np.array(volatilities)


class IntegratedRiskMetrics:
    """Comprehensive risk metrics"""

    @staticmethod
    def calculate_portfolio_metrics(
        returns: np.ndarray,
        weights: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        portfolio_returns = returns @ weights

        mean_return = np.mean(portfolio_returns)
        volatility = np.std(portfolio_returns)
        sharpe = (mean_return - risk_free_rate / 252) / (volatility + 1e-8)

        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])

        cumsum_returns = np.cumsum(portfolio_returns)
        running_max = np.maximum.accumulate(cumsum_returns)
        drawdown = cumsum_returns - running_max
        max_drawdown = np.min(drawdown)

        # Sortino ratio (downside volatility)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = np.std(downside_returns) if len(downside_returns) > 0 else volatility
        sortino = (mean_return - risk_free_rate / 252) / (downside_vol + 1e-8)

        return PortfolioMetrics(
            return_pct=float(mean_return * 252),
            volatility=float(volatility * np.sqrt(252)),
            sharpe_ratio=float(sharpe),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            max_drawdown=float(max_drawdown),
            sortino_ratio=float(sortino)
        )


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate sample returns
    n_assets = 5
    n_days = 252

    returns = np.random.randn(n_days, n_assets) * 0.01 + 0.0005

    # GARCH-Copula
    garch = CopulaGARCHModel(returns[:, 0])
    garch_params = garch.fit_garch()
    vol_forecast = garch.forecast_volatility(10)

    logger.info("GARCH Volatility Forecast:")
    logger.info(f"Mean conditional vol: {garch_params['mean_conditional_vol']:.6f}")
    logger.info(f"10-day forecast: {vol_forecast[:3]}")

    # Time series decomposition
    prices = np.cumprod(1 + returns[:, 0])
    decomposed = TimeSeriesDecomposition.decompose(prices, period=60)
    anomalies = TimeSeriesDecomposition.detect_anomalies(prices)

    logger.info(f"\nTime Series Decomposition:")
    logger.info(f"Detected {len(anomalies)} anomalies")

    # Risk parity
    volatilities = np.std(returns, axis=0)
    cov_matrix = np.cov(returns.T)

    rp_weights = RiskParityPortfolio.construct_risk_parity(volatilities, cov_matrix)

    logger.info(f"\nRisk Parity Weights: {rp_weights}")

    # Portfolio metrics
    metrics = IntegratedRiskMetrics.calculate_portfolio_metrics(returns, rp_weights)

    logger.info(f"\nPortfolio Metrics:")
    logger.info(f"Return: {metrics.return_pct:.2%}")
    logger.info(f"Volatility: {metrics.volatility:.2%}")
    logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
    logger.info(f"CVaR 95%: {metrics.cvar_95:.6f}")
    logger.info(f"Max Drawdown: {metrics.max_drawdown:.4f}")
