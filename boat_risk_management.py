#!/usr/bin/env python3
"""
Risk Management System for Boat Trading Platform
=================================================

Comprehensive risk management for quantitative trading:
  - Value at Risk (VaR) - Historical, Monte Carlo, Parametric methods
  - Conditional Value at Risk (CVaR/Expected Shortfall)
  - Portfolio correlation and covariance analysis
  - Value-weighted risk metrics
  - Stress testing and scenario analysis
  - Risk limits and constraints
  - Position sizing based on risk budgets

Based on 2025 risk management best practices in quantitative finance.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VaRMethod(Enum):
    """Value at Risk calculation methods"""
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    PARAMETRIC = "parametric"  # Normal distribution based


class RiskMetric(Enum):
    """Risk metrics"""
    VALUE_AT_RISK = "var"
    CONDITIONAL_VALUE_AT_RISK = "cvar"
    EXPECTED_SHORTFALL = "es"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe"
    SORTINO_RATIO = "sortino"
    CALMAR_RATIO = "calmar"


@dataclass
class Position:
    """Trading position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    notional_value: float = 0.0
    position_pct: float = 0.0  # % of portfolio

    def __post_init__(self):
        self.notional_value = self.quantity * self.current_price
        if self.quantity < 0:
            self.notional_value = abs(self.notional_value)


@dataclass
class RiskParameters:
    """Risk management parameters"""
    confidence_level: float = 0.95  # 95% for 1-day VaR
    holding_period: int = 1  # days
    var_method: VaRMethod = VaRMethod.HISTORICAL
    lookback_days: int = 252  # 1 year of trading days

    # Position limits
    max_position_size: float = 0.10  # Max 10% per position
    max_sector_exposure: float = 0.30  # Max 30% per sector
    max_leverage: float = 2.0  # Max 2x leverage

    # Risk limits
    max_daily_loss: float = 0.02  # Max 2% daily loss
    max_monthly_loss: float = 0.05  # Max 5% monthly loss
    max_drawdown: float = 0.15  # Max 15% drawdown

    # Monte Carlo settings
    monte_carlo_sims: int = 10000

    # Risk-free rate for Sharpe/Sortino
    risk_free_rate: float = 0.03


@dataclass
class RiskMetrics:
    """Calculated risk metrics"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    portfolio_value: float = 0.0

    # VaR metrics
    var_95: float = 0.0  # 95% confidence
    var_99: float = 0.0  # 99% confidence
    cvar_95: float = 0.0  # Conditional VaR at 95%
    cvar_99: float = 0.0  # Conditional VaR at 99%

    # Drawdown metrics
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    drawdown_recovery_days: int = 0

    # Return metrics
    daily_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Correlation metrics
    correlation_matrix: Optional[np.ndarray] = None
    portfolio_beta: float = 0.0

    # Position metrics
    num_positions: int = 0
    concentration: float = 0.0  # Herfindahl index
    gross_leverage: float = 0.0


@dataclass
class StressScenario:
    """Stress test scenario"""
    name: str
    description: str
    return_shocks: Dict[str, float]  # Symbol -> return shock percentage
    volatility_multiplier: float = 1.0  # Multiply volatility
    correlation_changes: Optional[Dict[Tuple[str, str], float]] = None


class ValueAtRiskCalculator:
    """Calculate Value at Risk using multiple methods"""

    def __init__(self, params: RiskParameters):
        self.params = params

    def historical_var(\n        self,\n        returns: np.ndarray,\n        confidence: float = 0.95\n    ) -> float:\n        \"\"\"\n        Calculate VaR using historical simulation\n        \n        Args:\n            returns: Array of historical returns\n            confidence: Confidence level (e.g., 0.95)\n            \n        Returns:\n            VaR value\n        \"\"\"\n        percentile = (1 - confidence) * 100\n        var = np.percentile(returns, percentile)\n        return float(var)\n
    def parametric_var(\n        self,\n        returns: np.ndarray,\n        confidence: float = 0.95\n    ) -> float:\n        \"\"\"\n        Calculate VaR assuming normal distribution\n        \n        Args:\n            returns: Array of historical returns\n            confidence: Confidence level\n            \n        Returns:\n            VaR value\n        \"\"\"\n        mean_return = np.mean(returns)\n        std_return = np.std(returns)\n        \n        # Z-score for confidence level\n        z_score = stats.norm.ppf(1 - confidence)\n        var = mean_return + z_score * std_return\n        \n        return float(var)\n    \n    def monte_carlo_var(\n        self,\n        returns: np.ndarray,\n        confidence: float = 0.95,\n        sims: int = 10000\n    ) -> float:\n        \"\"\"\n        Calculate VaR using Monte Carlo simulation\n        \n        Args:\n            returns: Array of historical returns\n            confidence: Confidence level\n            sims: Number of simulations\n            \n        Returns:\n            VaR value\n        \"\"\"\n        mean_return = np.mean(returns)\n        std_return = np.std(returns)\n        \n        # Simulate future returns\n        simulated_returns = np.random.normal(\n            mean_return,\n            std_return,\n            sims\n        )\n        \n        percentile = (1 - confidence) * 100\n        var = np.percentile(simulated_returns, percentile)\n        \n        return float(var)\n
    def expected_shortfall(\n        self,\n        returns: np.ndarray,\n        confidence: float = 0.95\n    ) -> float:\n        \"\"\"\n        Calculate Conditional VaR (Expected Shortfall)\n        \n        Args:\n            returns: Array of historical returns\n            confidence: Confidence level\n            \n        Returns:\n            CVaR/ES value\n        \"\"\"\n        var = self.historical_var(returns, confidence)\n        # Average of returns worse than VaR\n        worse_returns = returns[returns <= var]\n        \n        if len(worse_returns) > 0:\n            cvar = np.mean(worse_returns)\n        else:\n            cvar = var\n        \n        return float(cvar)\n
    def calculate(\n        self,\n        returns: np.ndarray,\n        confidence: float = 0.95\n    ) -> Tuple[float, float]:\n        \"\"\"\n        Calculate VaR and CVaR\n        \n        Args:\n            returns: Array of returns\n            confidence: Confidence level\n            \n        Returns:\n            (VaR, CVaR)\n        \"\"\"\n        if self.params.var_method == VaRMethod.HISTORICAL:\n            var = self.historical_var(returns, confidence)\n        elif self.params.var_method == VaRMethod.PARAMETRIC:\n            var = self.parametric_var(returns, confidence)\n        else:  # MONTE_CARLO\n            var = self.monte_carlo_var(\n                returns,\n                confidence,\n                self.params.monte_carlo_sims\n            )\n        \n        cvar = self.expected_shortfall(returns, confidence)\n        return var, cvar


class PortfolioRiskAnalyzer:
    """Analyze portfolio-level risk metrics"""

    def __init__(self, params: RiskParameters):
        self.params = params
        self.var_calculator = ValueAtRiskCalculator(params)
        self.price_history: Dict[str, List[float]] = {}\
        self.return_history: Dict[str, List[float]] = {}
    \n    def add_price_data(\n        self,\n        symbol: str,\n        prices: List[float]\n    ) -> None:\n        \"\"\"\n        Add price history for asset\n        \n        Args:\n            symbol: Asset symbol\n            prices: List of historical prices\n        \"\"\"\n        self.price_history[symbol] = prices\n        \n        # Calculate returns\n        prices_array = np.array(prices)\n        returns = np.diff(prices_array) / prices_array[:-1]\n        self.return_history[symbol] = returns.tolist()\n
    def calculate_correlation_matrix(\n        self,\n        symbols: List[str]\n    ) -> np.ndarray:\n        \"\"\"\n        Calculate correlation matrix between assets\n        \n        Args:\n            symbols: List of asset symbols\n            \n        Returns:\n            Correlation matrix\n        \"\"\"\n        returns_data = []\n        for symbol in symbols:\n            if symbol in self.return_history:\n                returns_data.append(self.return_history[symbol])\n        \n        if not returns_data:\n            return np.eye(len(symbols))\n        \n        # Pad to same length\n        min_len = min(len(r) for r in returns_data)\n        returns_data = [r[-min_len:] for r in returns_data]\n        \n        returns_array = np.array(returns_data)\n        correlation = np.corrcoef(returns_array)\n        \n        return correlation\n
    def calculate_portfolio_metrics(\n        self,\n        positions: List[Position],\n        price_history: Dict[str, List[float]]\n    ) -> RiskMetrics:\n        \"\"\"\n        Calculate comprehensive portfolio risk metrics\n        \n        Args:\n            positions: List of open positions\n            price_history: Historical prices by symbol\n            \n        Returns:\n            RiskMetrics object\n        \"\"\"\n        metrics = RiskMetrics()\n        \n        # Calculate portfolio value\n        total_value = sum(p.notional_value for p in positions)\n        metrics.portfolio_value = total_value\n        metrics.num_positions = len(positions)\n        \n        # Calculate portfolio returns\n        portfolio_returns = []\n        symbols = [p.symbol for p in positions]\n        \n        for symbol in symbols:\n            if symbol in self.return_history:\n                returns = np.array(self.return_history[symbol])\n                weight = next(\n                    (p.notional_value / total_value for p in positions if p.symbol == symbol),\n                    0\n                )\n                portfolio_returns.append(returns * weight)\n        \n        if portfolio_returns:\n            portfolio_returns = np.sum(portfolio_returns, axis=0)\n        else:\n            portfolio_returns = np.array([0.0])\n        \n        # VaR metrics\n        if len(portfolio_returns) > 0:\n            var_95, cvar_95 = self.var_calculator.calculate(\n                portfolio_returns,\n                0.95\n            )\n            var_99, cvar_99 = self.var_calculator.calculate(\n                portfolio_returns,\n                0.99\n            )\n            \n            metrics.var_95 = float(var_95)\n            metrics.var_99 = float(var_99)\n            metrics.cvar_95 = float(cvar_95)\n            metrics.cvar_99 = float(cvar_99)\n        \n        # Return metrics\n        if len(portfolio_returns) > 0:\n            metrics.daily_return = float(np.mean(portfolio_returns))\n            metrics.annual_volatility = float(np.std(portfolio_returns) * np.sqrt(252))\n        \n        # Sharpe ratio\n        if metrics.annual_volatility > 0:\n            excess_return = metrics.daily_return * 252 - self.params.risk_free_rate\n            metrics.sharpe_ratio = excess_return / metrics.annual_volatility\n        \n        # Drawdown metrics\n        if len(portfolio_returns) > 0:\n            cumulative = np.cumprod(1 + portfolio_returns)\n            running_max = np.maximum.accumulate(cumulative)\n            drawdown = (cumulative - running_max) / running_max\n            metrics.max_drawdown = float(np.min(drawdown))\n            metrics.current_drawdown = float(drawdown[-1])\n        \n        # Concentration (Herfindahl index)\n        if total_value > 0:\n            weights = [p.notional_value / total_value for p in positions]\n            metrics.concentration = float(np.sum(np.array(weights) ** 2))\n        \n        # Correlation matrix\n        if symbols:\n            metrics.correlation_matrix = self.calculate_correlation_matrix(symbols)\n        \n        return metrics

    def calculate_position_limit_usage(\n        self,\n        positions: List[Position],\n        total_portfolio_value: float\n    ) -> Dict[str, float]:\n        \"\"\"\n        Calculate position limit usage\n        \n        Args:\n            positions: List of positions\n            total_portfolio_value: Total portfolio value\n            \n        Returns:\n            Dictionary of limit usage percentages\n        \"\"\"\n        usage = {}\n        \n        for position in positions:\n            position_pct = position.notional_value / total_portfolio_value if total_portfolio_value > 0 else 0\n            limit_usage = position_pct / self.params.max_position_size\n            usage[position.symbol] = limit_usage\n        \n        return usage


class RiskLimitMonitor:
    """Monitor adherence to risk limits"""

    def __init__(self, params: RiskParameters):
        self.params = params
        self.violations: List[Dict[str, Any]] = []

    def check_position_limits(\n        self,\n        positions: List[Position],\n        total_value: float\n    ) -> List[str]:\n        \"\"\"\n        Check position size limits\n        \n        Args:\n            positions: List of positions\n            total_value: Total portfolio value\n            \n        Returns:\n            List of violation messages\n        \"\"\"\n        violations = []\n        \n        for position in positions:\n            position_pct = position.notional_value / total_value if total_value > 0 else 0\n            \n            if position_pct > self.params.max_position_size:\n                violations.append(\n                    f\"Position {position.symbol} exceeds max size: \"\n                    f\"{position_pct:.2%} > {self.params.max_position_size:.2%}\"\n                )\n        \n        return violations\n
    def check_leverage_limits(\n        self,\n        gross_exposure: float,\n        net_exposure: float,\n        total_value: float\n    ) -> List[str]:\n        \"\"\"\n        Check leverage limits\n        \n        Args:\n            gross_exposure: Sum of absolute position values\n            net_exposure: Sum of signed position values\n            total_value: Total portfolio value\n            \n        Returns:\n            List of violation messages\n        \"\"\"\n        violations = []\n        \n        if total_value > 0:\n            gross_leverage = gross_exposure / total_value\n            \n            if gross_leverage > self.params.max_leverage:\n                violations.append(\n                    f\"Gross leverage exceeds limit: \"\n                    f\"{gross_leverage:.2f}x > {self.params.max_leverage:.2f}x\"\n                )\n        \n        return violations\n
    def check_drawdown_limits(\n        self,\n        current_value: float,\n        peak_value: float\n    ) -> List[str]:\n        \"\"\"\n        Check drawdown limits\n        \n        Args:\n            current_value: Current portfolio value\n            peak_value: Peak portfolio value\n            \n        Returns:\n            List of violation messages\n        \"\"\"\n        violations = []\n        \n        if peak_value > 0:\n            drawdown = (current_value - peak_value) / peak_value\n            \n            if drawdown < -self.params.max_drawdown:\n                violations.append(\n                    f\"Drawdown exceeds limit: \"\n                    f\"{drawdown:.2%} < {-self.params.max_drawdown:.2%}\"\n                )\n        \n        return violations

    def check_all_limits(\n        self,\n        positions: List[Position],\n        metrics: RiskMetrics\n    ) -> Dict[str, List[str]]:\n        \"\"\"\n        Check all risk limits\n        \n        Args:\n            positions: List of positions\n            metrics: Calculated risk metrics\n            \n        Returns:\n            Dictionary of violations by category\n        \"\"\"\n        total_value = metrics.portfolio_value\n        gross_exposure = sum(p.notional_value for p in positions)\n        \n        return {\n            'position_limits': self.check_position_limits(positions, total_value),\n            'leverage_limits': self.check_leverage_limits(\n                gross_exposure,\n                sum(p.quantity * p.current_price for p in positions),\n                total_value\n            ),\n            'drawdown_limits': self.check_drawdown_limits(\n                total_value,\n                total_value / (1 + abs(metrics.max_drawdown))\n            )\n        }


class StressTester:
    \"\"\"Stress test portfolio under extreme scenarios\"\"\"\n    \n    def __init__(self, analyzer: PortfolioRiskAnalyzer):\n        self.analyzer = analyzer\n    \n    def apply_scenario(\n        self,\n        positions: List[Position],\n        scenario: StressScenario\n    ) -> Dict[str, Any]:\n        \"\"\"\n        Apply stress scenario to portfolio\n        \n        Args:\n            positions: List of positions\n            scenario: Stress scenario\n            \n        Returns:\n            Portfolio impact analysis\n        \"\"\"\n        impact = {\n            'scenario': scenario.name,\n            'total_pnl': 0.0,\n            'by_position': {},\n            'portfolio_loss_pct': 0.0\n        }\n        \n        initial_value = sum(p.notional_value for p in positions)\n        \n        for position in positions:\n            # Apply return shock\n            shock = scenario.return_shocks.get(position.symbol, 0.0)\n            new_price = position.current_price * (1 + shock)\n            pnl = (new_price - position.current_price) * position.quantity\n            \n            impact['by_position'][position.symbol] = {\n                'return_shock': shock,\n                'new_price': new_price,\n                'pnl': pnl\n            }\n            impact['total_pnl'] += pnl\n        \n        if initial_value > 0:\n            impact['portfolio_loss_pct'] = impact['total_pnl'] / initial_value\n        \n        return impact


class RiskBudgetOptimizer:
    \"\"\"Optimize position sizing based on risk budget\"\"\"\n    \n    def __init__(self, params: RiskParameters):\n        self.params = params\n    \n    def optimize_positions(\n        self,\n        symbols: List[str],\n        expected_returns: Dict[str, float],\n        volatilities: Dict[str, float],\n        correlation_matrix: np.ndarray,\n        risk_budget: float\n    ) -> Dict[str, float]:\n        \"\"\"\n        Optimize position weights based on risk budget\n        \n        Args:\n            symbols: Asset symbols\n            expected_returns: Expected returns by symbol\n            volatilities: Volatilities by symbol\n            correlation_matrix: Correlation matrix\n            risk_budget: Total risk budget (volatility target)\n            \n        Returns:\n            Optimal position weights\n        \"\"\"\n        n = len(symbols)\n        \n        # Initial guess: equal weight\n        x0 = np.array([1.0 / n] * n)\n        \n        # Constraints\n        constraints = [\n            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Sum to 1\n            {'type': 'ineq', 'fun': lambda x: x},  # Non-negative\n            {'type': 'ineq', 'fun': lambda x: 1.0 - x}  # Max 100%\n        ]\n        \n        # Objective: maximize return per unit of risk\n        def objective(weights):\n            portfolio_return = np.sum(weights * np.array([expected_returns[s] for s in symbols]))\n            portfolio_vol = np.sqrt(\n                weights @ correlation_matrix *\n                np.outer(\n                    np.array([volatilities[s] for s in symbols]),\n                    np.array([volatilities[s] for s in symbols])\n                ) @ weights\n            )\n            \n            if portfolio_vol > 0:\n                return -portfolio_return / portfolio_vol  # Minimize negative Sharpe\n            else:\n                return 0.0\n        \n        result = minimize(objective, x0, constraints=constraints, method='SLSQP')\n        \n        if result.success:\n            return {symbol: float(weight) for symbol, weight in zip(symbols, result.x)}\n        else:\n            return {symbol: 1.0/n for symbol in symbols}


if __name__ == \"__main__\":\n    # Example usage\n    params = RiskParameters(\n        confidence_level=0.95,\n        var_method=VaRMethod.HISTORICAL\n    )\n    \n    analyzer = PortfolioRiskAnalyzer(params)\n    \n    # Add sample price data\n    np.random.seed(42)\n    symbol1_prices = 100 + np.cumsum(np.random.randn(252) * 2)\n    symbol2_prices = 50 + np.cumsum(np.random.randn(252) * 1.5)\n    \n    analyzer.add_price_data(\"STOCK1\", symbol1_prices.tolist())\n    analyzer.add_price_data(\"STOCK2\", symbol2_prices.tolist())\n    \n    # Create sample positions\n    positions = [\n        Position(\"STOCK1\", 10, 100, symbol1_prices[-1]),\n        Position(\"STOCK2\", 5, 50, symbol2_prices[-1])\n    ]\n    \n    # Calculate metrics\n    metrics = analyzer.calculate_portfolio_metrics(positions, {})\n    logger.info(f\"VaR (95%): {metrics.var_95:.4f}\")\n    logger.info(f\"CVaR (95%): {metrics.cvar_95:.4f}\")\n    logger.info(f\"Sharpe Ratio: {metrics.sharpe_ratio:.2f}\")\n    logger.info(f\"Max Drawdown: {metrics.max_drawdown:.2%}\")\n    \n    # Check risk limits\n    monitor = RiskLimitMonitor(params)\n    violations = monitor.check_all_limits(positions, metrics)\n    logger.info(f\"Risk violations: {violations}\")\n