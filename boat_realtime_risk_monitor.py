"""
BOAT - Real-Time Portfolio Risk Monitor
=======================================

Production-ready real-time risk monitoring system with VaR/CVaR calculations
and position limit enforcement.

Features:
- Value at Risk (VaR) calculation (Historical, Parametric, Monte Carlo)
- Conditional Value at Risk (CVaR/Expected Shortfall)
- Position limit monitoring and alerts
- Real-time risk exposure tracking
- Portfolio-level risk aggregation
- Risk limit breach detection

Based on 2025 research:
- CVaR for daily portfolio risk management
- Real-time risk monitoring systems
- Position limit frameworks
- Expected Shortfall for tail risk

Design Philosophy (Carmack/Martin/Pike):
- Practical VaR implementations
- Fast daily recalculation
- Clear limit enforcement
- No complex dependencies
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from datetime import datetime


class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class VaRMethod(Enum):
    """VaR calculation methods"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


@dataclass
class PositionRisk:
    """Risk metrics for a single position"""
    symbol: str
    quantity: int
    current_price: float
    market_value: float
    var_1day: float
    cvar_1day: float
    volatility: float
    portfolio_weight: float


@dataclass
class PortfolioRisk:
    """Aggregated portfolio risk metrics"""
    total_value: float
    var_1day: float
    var_5day: float
    cvar_1day: float
    cvar_5day: float
    volatility: float
    positions: List[PositionRisk]
    correlation_adjusted_var: float
    diversification_benefit: float


@dataclass
class RiskLimit:
    """Risk limit definition"""
    name: str
    limit_type: str  # var, cvar, position_size, concentration
    threshold: float
    current_value: float
    breach: bool
    severity: RiskLevel


@dataclass
class RiskAlert:
    """Risk limit breach alert"""
    timestamp: int
    alert_type: str
    severity: RiskLevel
    message: str
    current_value: float
    limit_value: float
    action_required: str


class VaRCalculator:
    """
    Value at Risk calculator with multiple methods.

    Implements Historical, Parametric, and Monte Carlo VaR calculations.
    """

    @staticmethod
    def historical_var(
        returns: np.ndarray,
        confidence_level: float = 0.95,
        horizon: int = 1
    ) -> float:
        """
        Calculate VaR using historical simulation method.

        Args:
            returns: Historical returns
            confidence_level: Confidence level (0.95 = 95%)
            horizon: Time horizon in days

        Returns:
            VaR value (negative = loss)
        """
        if len(returns) < 20:
            return 0.0

        # Adjust returns for horizon
        if horizon > 1:
            returns_scaled = returns * np.sqrt(horizon)
        else:
            returns_scaled = returns

        # Calculate VaR as percentile
        var = np.percentile(returns_scaled, (1 - confidence_level) * 100)

        return abs(var)  # Return positive value for loss

    @staticmethod
    def parametric_var(
        returns: np.ndarray,
        confidence_level: float = 0.95,
        horizon: int = 1
    ) -> float:
        """
        Calculate VaR using parametric method (assumes normal distribution).

        Args:
            returns: Historical returns
            confidence_level: Confidence level
            horizon: Time horizon in days

        Returns:
            VaR value
        """
        if len(returns) < 10:
            return 0.0

        # Calculate mean and std
        mean = np.mean(returns)
        std = np.std(returns)

        # Z-score for confidence level
        from scipy.stats import norm
        z_score = norm.ppf(1 - confidence_level)

        # VaR formula: μ + z*σ*√t
        var = -(mean * horizon + z_score * std * np.sqrt(horizon))

        return abs(var)

    @staticmethod
    def monte_carlo_var(
        returns: np.ndarray,
        confidence_level: float = 0.95,
        horizon: int = 1,
        n_simulations: int = 10000
    ) -> float:
        """
        Calculate VaR using Monte Carlo simulation.

        Args:
            returns: Historical returns
            confidence_level: Confidence level
            horizon: Time horizon in days
            n_simulations: Number of simulations

        Returns:
            VaR value
        """
        if len(returns) < 10:
            return 0.0

        mean = np.mean(returns)
        std = np.std(returns)

        # Simulate returns
        simulated_returns = np.random.normal(
            mean * horizon,
            std * np.sqrt(horizon),
            n_simulations
        )

        # Calculate VaR
        var = np.percentile(simulated_returns, (1 - confidence_level) * 100)

        return abs(var)

    @staticmethod
    def calculate_cvar(
        returns: np.ndarray,
        confidence_level: float = 0.95,
        horizon: int = 1
    ) -> float:
        """
        Calculate CVaR (Conditional VaR / Expected Shortfall).

        Args:
            returns: Historical returns
            confidence_level: Confidence level
            horizon: Time horizon in days

        Returns:
            CVaR value (average loss beyond VaR)
        """
        if len(returns) < 20:
            return 0.0

        # Adjust returns for horizon
        if horizon > 1:
            returns_scaled = returns * np.sqrt(horizon)
        else:
            returns_scaled = returns

        # Find VaR threshold
        var_threshold = np.percentile(returns_scaled, (1 - confidence_level) * 100)

        # Calculate average of returns below VaR
        tail_losses = returns_scaled[returns_scaled <= var_threshold]

        if len(tail_losses) > 0:
            cvar = abs(np.mean(tail_losses))
        else:
            cvar = abs(var_threshold)

        return cvar


class RealtimeRiskMonitor:
    """
    Real-time portfolio risk monitoring system.

    Monitors VaR, CVaR, position limits, and generates alerts.
    """

    def __init__(
        self,
        portfolio_value: float = 1000000.0,
        confidence_level: float = 0.95,
        var_method: VaRMethod = VaRMethod.HISTORICAL
    ):
        """
        Initialize risk monitor.

        Args:
            portfolio_value: Total portfolio value
            confidence_level: VaR/CVaR confidence level
            var_method: VaR calculation method
        """
        self.portfolio_value = portfolio_value
        self.confidence_level = confidence_level
        self.var_method = var_method

        # Risk limits
        self.risk_limits: Dict[str, RiskLimit] = {}
        self.setup_default_limits()

        # Alert history
        self.alerts: List[RiskAlert] = []

        # Position tracking
        self.positions: Dict[str, Dict] = {}

    def setup_default_limits(self):
        """Setup default risk limits"""
        # Portfolio-level limits
        self.add_risk_limit(
            "portfolio_var_1day",
            "var",
            self.portfolio_value * 0.02,  # 2% daily VaR limit
            RiskLevel.HIGH
        )

        self.add_risk_limit(
            "portfolio_cvar_1day",
            "cvar",
            self.portfolio_value * 0.03,  # 3% daily CVaR limit
            RiskLevel.CRITICAL
        )

        self.add_risk_limit(
            "max_position_size",
            "position_size",
            self.portfolio_value * 0.20,  # 20% max per position
            RiskLevel.MODERATE
        )

        self.add_risk_limit(
            "sector_concentration",
            "concentration",
            0.30,  # 30% max sector concentration
            RiskLevel.MODERATE
        )

    def add_risk_limit(
        self,
        name: str,
        limit_type: str,
        threshold: float,
        severity: RiskLevel
    ):
        """Add a risk limit"""
        self.risk_limits[name] = RiskLimit(
            name=name,
            limit_type=limit_type,
            threshold=threshold,
            current_value=0.0,
            breach=False,
            severity=severity
        )

    def calculate_position_risk(
        self,
        symbol: str,
        quantity: int,
        current_price: float,
        historical_returns: np.ndarray
    ) -> PositionRisk:
        """
        Calculate risk metrics for a position.

        Args:
            symbol: Stock symbol
            quantity: Position size
            current_price: Current price
            historical_returns: Historical returns

        Returns:
            Position risk metrics
        """
        market_value = quantity * current_price

        # Calculate VaR
        if self.var_method == VaRMethod.HISTORICAL:
            var_pct = VaRCalculator.historical_var(historical_returns, self.confidence_level)
        elif self.var_method == VaRMethod.PARAMETRIC:
            var_pct = VaRCalculator.parametric_var(historical_returns, self.confidence_level)
        else:
            var_pct = VaRCalculator.monte_carlo_var(historical_returns, self.confidence_level)

        # Calculate CVaR
        cvar_pct = VaRCalculator.calculate_cvar(historical_returns, self.confidence_level)

        # Convert to dollar amounts
        var_dollar = market_value * var_pct
        cvar_dollar = market_value * cvar_pct

        # Volatility
        volatility = np.std(historical_returns)

        return PositionRisk(
            symbol=symbol,
            quantity=quantity,
            current_price=current_price,
            market_value=market_value,
            var_1day=var_dollar,
            cvar_1day=cvar_dollar,
            volatility=volatility,
            portfolio_weight=market_value / self.portfolio_value
        )

    def calculate_portfolio_risk(
        self,
        positions: List[PositionRisk],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> PortfolioRisk:
        """
        Calculate aggregated portfolio risk.

        Args:
            positions: List of position risks
            correlation_matrix: Asset correlation matrix

        Returns:
            Portfolio risk metrics
        """
        if not positions:
            return PortfolioRisk(
                total_value=0.0,
                var_1day=0.0,
                var_5day=0.0,
                cvar_1day=0.0,
                cvar_5day=0.0,
                volatility=0.0,
                positions=[],
                correlation_adjusted_var=0.0,
                diversification_benefit=0.0
            )

        total_value = sum(p.market_value for p in positions)

        # Simple aggregation (no correlation)
        simple_var = sum(p.var_1day for p in positions)
        simple_cvar = sum(p.cvar_1day for p in positions)

        # Correlation-adjusted VaR
        if correlation_matrix is not None and len(positions) > 1:
            # Portfolio variance with correlation
            weights = np.array([p.market_value / total_value for p in positions])
            volatilities = np.array([p.volatility for p in positions])

            # Covariance matrix
            cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix

            # Portfolio variance
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_variance)

            # Correlation-adjusted VaR (approximate)
            correlation_adjusted_var = total_value * portfolio_vol * 2.33  # 99% confidence
        else:
            correlation_adjusted_var = simple_var
            portfolio_vol = np.sqrt(sum((p.volatility * p.market_value)**2 for p in positions)) / total_value

        # Diversification benefit
        diversification_benefit = simple_var - correlation_adjusted_var

        # Multi-day VaR (scale by sqrt(time))
        var_5day = correlation_adjusted_var * np.sqrt(5)
        cvar_5day = simple_cvar * np.sqrt(5)

        return PortfolioRisk(
            total_value=total_value,
            var_1day=correlation_adjusted_var,
            var_5day=var_5day,
            cvar_1day=simple_cvar,
            cvar_5day=cvar_5day,
            volatility=portfolio_vol,
            positions=positions,
            correlation_adjusted_var=correlation_adjusted_var,
            diversification_benefit=max(diversification_benefit, 0.0)
        )

    def check_risk_limits(
        self,
        portfolio_risk: PortfolioRisk
    ) -> List[RiskAlert]:
        """
        Check all risk limits and generate alerts.

        Args:
            portfolio_risk: Portfolio risk metrics

        Returns:
            List of risk alerts
        """
        alerts = []
        timestamp = int(datetime.now().timestamp())

        # Check VaR limit
        var_limit = self.risk_limits.get("portfolio_var_1day")
        if var_limit:
            var_limit.current_value = portfolio_risk.var_1day
            if portfolio_risk.var_1day > var_limit.threshold:
                var_limit.breach = True
                alerts.append(RiskAlert(
                    timestamp=timestamp,
                    alert_type="VaR Breach",
                    severity=var_limit.severity,
                    message=f"Portfolio VaR ${portfolio_risk.var_1day:,.0f} exceeds limit ${var_limit.threshold:,.0f}",
                    current_value=portfolio_risk.var_1day,
                    limit_value=var_limit.threshold,
                    action_required="Reduce position sizes or hedge exposure"
                ))

        # Check CVaR limit
        cvar_limit = self.risk_limits.get("portfolio_cvar_1day")
        if cvar_limit:
            cvar_limit.current_value = portfolio_risk.cvar_1day
            if portfolio_risk.cvar_1day > cvar_limit.threshold:
                cvar_limit.breach = True
                alerts.append(RiskAlert(
                    timestamp=timestamp,
                    alert_type="CVaR Breach",
                    severity=cvar_limit.severity,
                    message=f"Portfolio CVaR ${portfolio_risk.cvar_1day:,.0f} exceeds limit ${cvar_limit.threshold:,.0f}",
                    current_value=portfolio_risk.cvar_1day,
                    limit_value=cvar_limit.threshold,
                    action_required="CRITICAL: Reduce exposure immediately"
                ))

        # Check position size limits
        position_limit = self.risk_limits.get("max_position_size")
        if position_limit:
            for pos in portfolio_risk.positions:
                if pos.market_value > position_limit.threshold:
                    alerts.append(RiskAlert(
                        timestamp=timestamp,
                        alert_type="Position Size Breach",
                        severity=position_limit.severity,
                        message=f"{pos.symbol} position ${pos.market_value:,.0f} exceeds limit ${position_limit.threshold:,.0f}",
                        current_value=pos.market_value,
                        limit_value=position_limit.threshold,
                        action_required=f"Reduce {pos.symbol} position"
                    ))

        self.alerts.extend(alerts)
        return alerts

    def get_risk_summary(self, portfolio_risk: PortfolioRisk) -> Dict[str, Any]:
        """Get human-readable risk summary"""
        return {
            'portfolio_value': portfolio_risk.total_value,
            'var_1day_dollar': portfolio_risk.var_1day,
            'var_1day_percent': portfolio_risk.var_1day / portfolio_risk.total_value,
            'cvar_1day_dollar': portfolio_risk.cvar_1day,
            'cvar_1day_percent': portfolio_risk.cvar_1day / portfolio_risk.total_value,
            'var_5day_dollar': portfolio_risk.var_5day,
            'volatility': portfolio_risk.volatility,
            'num_positions': len(portfolio_risk.positions),
            'diversification_benefit': portfolio_risk.diversification_benefit,
            'active_alerts': len([a for a in self.alerts if a.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]])
        }


def test_realtime_risk_monitor():
    """Test Real-Time Portfolio Risk Monitor"""
    print("=" * 70)
    print("Testing Real-Time Portfolio Risk Monitor")
    print("=" * 70)

    # Initialize monitor
    monitor = RealtimeRiskMonitor(
        portfolio_value=1000000,
        confidence_level=0.95,
        var_method=VaRMethod.HISTORICAL
    )

    print("\n1. Risk Limits Configuration:")
    print("-" * 70)
    print(f"Portfolio Value: ${monitor.portfolio_value:,.0f}")
    print(f"Confidence Level: {monitor.confidence_level:.0%}")
    print(f"VaR Method: {monitor.var_method.value}")

    print("\nConfigured Limits:")
    for name, limit in monitor.risk_limits.items():
        if limit.limit_type == "var" or limit.limit_type == "cvar":
            print(f"  {name}: ${limit.threshold:,.0f} ({limit.severity.value})")
        elif limit.limit_type == "position_size":
            print(f"  {name}: ${limit.threshold:,.0f} ({limit.severity.value})")
        else:
            print(f"  {name}: {limit.threshold:.1%} ({limit.severity.value})")

    print("\n2. VaR Calculation Methods:")
    print("-" * 70)

    # Generate synthetic returns
    np.random.seed(42)
    returns = np.random.normal(-0.0005, 0.02, 252)  # Slightly negative drift

    print("Testing on 252 days of returns...")
    print(f"Mean return: {np.mean(returns):.4%}")
    print(f"Volatility: {np.std(returns):.4%}")

    # Compare methods
    hist_var = VaRCalculator.historical_var(returns, 0.95)
    param_var = VaRCalculator.parametric_var(returns, 0.95)
    mc_var = VaRCalculator.monte_carlo_var(returns, 0.95, n_simulations=10000)

    print(f"\nVaR (95% confidence, 1-day):")
    print(f"  Historical:  {hist_var:.4%}")
    print(f"  Parametric:  {param_var:.4%}")
    print(f"  Monte Carlo: {mc_var:.4%}")

    # CVaR
    cvar = VaRCalculator.calculate_cvar(returns, 0.95)
    print(f"\nCVaR (Expected Shortfall): {cvar:.4%}")
    print(f"CVaR/VaR Ratio: {cvar/hist_var:.2f}x")

    print("\n3. Position Risk Calculation:")
    print("-" * 70)

    # Test positions
    test_positions = [
        ("AAPL", 1000, 150.0),
        ("MSFT", 800, 300.0),
        ("GOOGL", 500, 100.0),
    ]

    position_risks = []

    for symbol, quantity, price in test_positions:
        # Generate different returns for each symbol
        pos_returns = np.random.normal(-0.0005, 0.02 * (1 + hash(symbol) % 10 / 20), 252)

        pos_risk = monitor.calculate_position_risk(
            symbol, quantity, price, pos_returns
        )
        position_risks.append(pos_risk)

        print(f"\n{symbol}:")
        print(f"  Position: {quantity} shares @ ${price:.2f}")
        print(f"  Market Value: ${pos_risk.market_value:,.0f}")
        print(f"  VaR (1-day): ${pos_risk.var_1day:,.0f} ({pos_risk.var_1day/pos_risk.market_value:.2%})")
        print(f"  CVaR (1-day): ${pos_risk.cvar_1day:,.0f} ({pos_risk.cvar_1day/pos_risk.market_value:.2%})")
        print(f"  Volatility: {pos_risk.volatility:.2%}")
        print(f"  Portfolio Weight: {pos_risk.portfolio_weight:.1%}")

    print("\n4. Portfolio Risk Aggregation:")
    print("-" * 70)

    # Create correlation matrix
    n_assets = len(position_risks)
    correlation_matrix = np.eye(n_assets)
    correlation_matrix[0, 1] = correlation_matrix[1, 0] = 0.7  # AAPL-MSFT
    correlation_matrix[0, 2] = correlation_matrix[2, 0] = 0.6  # AAPL-GOOGL
    correlation_matrix[1, 2] = correlation_matrix[2, 1] = 0.65  # MSFT-GOOGL

    portfolio_risk = monitor.calculate_portfolio_risk(position_risks, correlation_matrix)

    print(f"Total Portfolio Value: ${portfolio_risk.total_value:,.0f}")
    print(f"\nRisk Metrics:")
    print(f"  VaR (1-day): ${portfolio_risk.var_1day:,.0f} ({portfolio_risk.var_1day/portfolio_risk.total_value:.2%})")
    print(f"  VaR (5-day): ${portfolio_risk.var_5day:,.0f} ({portfolio_risk.var_5day/portfolio_risk.total_value:.2%})")
    print(f"  CVaR (1-day): ${portfolio_risk.cvar_1day:,.0f} ({portfolio_risk.cvar_1day/portfolio_risk.total_value:.2%})")
    print(f"  CVaR (5-day): ${portfolio_risk.cvar_5day:,.0f}")
    print(f"  Portfolio Volatility: {portfolio_risk.volatility:.2%}")
    print(f"  Diversification Benefit: ${portfolio_risk.diversification_benefit:,.0f}")

    print("\n5. Risk Limit Monitoring:")
    print("-" * 70)

    alerts = monitor.check_risk_limits(portfolio_risk)

    if alerts:
        print(f"ALERTS GENERATED: {len(alerts)}")
        for alert in alerts:
            print(f"\n[{alert.severity.value.upper()}] {alert.alert_type}")
            print(f"  {alert.message}")
            print(f"  Action: {alert.action_required}")
    else:
        print("No limit breaches detected. Portfolio within risk limits.")

    print("\n6. Risk Summary:")
    print("-" * 70)

    summary = monitor.get_risk_summary(portfolio_risk)

    print(f"Portfolio Value: ${summary['portfolio_value']:,.0f}")
    print(f"1-Day VaR: ${summary['var_1day_dollar']:,.0f} ({summary['var_1day_percent']:.2%})")
    print(f"1-Day CVaR: ${summary['cvar_1day_dollar']:,.0f} ({summary['cvar_1day_percent']:.2%})")
    print(f"5-Day VaR: ${summary['var_5day_dollar']:,.0f}")
    print(f"Number of Positions: {summary['num_positions']}")
    print(f"Diversification Benefit: ${summary['diversification_benefit']:,.0f}")
    print(f"Active Critical Alerts: {summary['active_alerts']}")

    print("\n7. Stress Testing:")
    print("-" * 70)

    # Simulate market stress (2x volatility)
    stress_returns = np.random.normal(-0.001, 0.04, 252)
    stress_var = VaRCalculator.historical_var(stress_returns, 0.95)
    stress_cvar = VaRCalculator.calculate_cvar(stress_returns, 0.95)

    print("Market Stress Scenario (2x volatility, negative drift):")
    print(f"  Normal VaR: {hist_var:.2%} -> Stress VaR: {stress_var:.2%}")
    print(f"  Normal CVaR: {cvar:.2%} -> Stress CVaR: {stress_cvar:.2%}")
    print(f"  Portfolio Impact: ${portfolio_risk.total_value * stress_var:,.0f} potential loss")

    print("\n[SUCCESS] Real-Time Risk Monitor test completed successfully!")


if __name__ == "__main__":
    test_realtime_risk_monitor()
