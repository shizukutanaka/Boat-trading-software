"""
BOAT - Trading Performance Analytics
====================================

Production-ready performance analytics with comprehensive metrics.

Features:
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk focus)
- Calmar Ratio (drawdown-adjusted return)
- Maximum Drawdown calculation
- Win rate and profit factor
- Trade analysis and statistics
- Equity curve visualization data

Based on 2025 research:
- Sharpe, Sortino, Calmar ratios
- Risk-adjusted performance metrics
- Drawdown analysis frameworks
- Professional performance reporting

Design Philosophy (Carmack/Martin/Pike):
- Standard industry metrics
- Clear calculation methods
- Practical performance insights
- Fast computation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Trade:
    """Trade record"""
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # 'long' or 'short'
    pnl: float
    pnl_percent: float
    symbol: str
    strategy: str = ""


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Return metrics
    total_return: float
    annualized_return: float
    cagr: float

    # Risk metrics
    volatility: float
    downside_deviation: float
    max_drawdown: float
    max_drawdown_duration: int

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float

    # Additional metrics
    best_trade: float
    worst_trade: float
    avg_trade_duration: float
    longest_winning_streak: int
    longest_losing_streak: int


@dataclass
class EquityPoint:
    """Equity curve point"""
    timestamp: int
    equity: float
    drawdown: float
    drawdown_percent: float


class PerformanceAnalyzer:
    """
    Trading performance analyzer.

    Calculates comprehensive performance metrics for trading strategies.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize analyzer.

        Args:
            initial_capital: Starting capital
            risk_free_rate: Risk-free rate for Sharpe/Sortino
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        self.trades: List[Trade] = []
        self.equity_curve: List[EquityPoint] = []

    def add_trade(self, trade: Trade):
        """Add a trade to the record"""
        self.trades.append(trade)

    def calculate_equity_curve(self) -> List[EquityPoint]:
        """
        Calculate equity curve from trades.

        Returns:
            List of equity curve points
        """
        if not self.trades:
            return []

        equity = self.initial_capital
        peak = equity
        equity_curve = []

        # Sort trades by exit time
        sorted_trades = sorted(self.trades, key=lambda t: t.exit_time)

        for trade in sorted_trades:
            equity += trade.pnl

            # Track peak and drawdown
            if equity > peak:
                peak = equity

            drawdown = peak - equity
            drawdown_pct = drawdown / peak if peak > 0 else 0

            equity_curve.append(EquityPoint(
                timestamp=trade.exit_time,
                equity=equity,
                drawdown=drawdown,
                drawdown_percent=drawdown_pct
            ))

        self.equity_curve = equity_curve
        return equity_curve

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Array of returns
            periods_per_year: Trading periods per year

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / periods_per_year
        sharpe = np.mean(excess_returns) / (np.std(returns) + 1e-8) * np.sqrt(periods_per_year)

        return sharpe

    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino ratio (downside deviation focus).

        Args:
            returns: Array of returns
            periods_per_year: Trading periods per year

        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / periods_per_year

        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
        else:
            downside_std = 0.0

        if downside_std == 0:
            return 0.0

        sortino = np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)

        return sortino

    def calculate_calmar_ratio(
        self,
        annualized_return: float,
        max_drawdown: float
    ) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).

        Args:
            annualized_return: Annualized return
            max_drawdown: Maximum drawdown

        Returns:
            Calmar ratio
        """
        if max_drawdown == 0:
            return 0.0

        return annualized_return / max_drawdown

    def calculate_max_drawdown(
        self,
        equity_curve: List[EquityPoint]
    ) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and duration.

        Args:
            equity_curve: Equity curve points

        Returns:
            Tuple of (max_drawdown, duration_in_periods)
        """
        if not equity_curve:
            return (0.0, 0)

        max_dd = 0.0
        max_dd_duration = 0
        current_duration = 0
        in_drawdown = False

        for point in equity_curve:
            if point.drawdown_percent > max_dd:
                max_dd = point.drawdown_percent

            if point.drawdown > 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 0
                current_duration += 1
            else:
                if in_drawdown:
                    max_dd_duration = max(max_dd_duration, current_duration)
                    in_drawdown = False

        return (max_dd, max_dd_duration)

    def analyze_performance(self) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Returns:
            Performance metrics
        """
        if not self.trades:
            return PerformanceMetrics(
                total_return=0.0, annualized_return=0.0, cagr=0.0,
                volatility=0.0, downside_deviation=0.0,
                max_drawdown=0.0, max_drawdown_duration=0,
                sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0.0, avg_win=0.0, avg_loss=0.0,
                profit_factor=0.0, expectancy=0.0,
                best_trade=0.0, worst_trade=0.0,
                avg_trade_duration=0.0,
                longest_winning_streak=0, longest_losing_streak=0
            )

        # Calculate equity curve
        equity_curve = self.calculate_equity_curve()

        # Return metrics
        final_equity = equity_curve[-1].equity
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # Time span in years
        first_trade = min(t.entry_time for t in self.trades)
        last_trade = max(t.exit_time for t in self.trades)
        days = (last_trade - first_trade) / 86400  # Seconds to days
        years = days / 365 if days > 0 else 1

        annualized_return = total_return / years if years > 0 else 0
        cagr = (final_equity / self.initial_capital) ** (1 / years) - 1 if years > 0 else 0

        # Calculate returns series
        pnl_series = np.array([t.pnl for t in self.trades])
        equity_series = np.cumsum(pnl_series) + self.initial_capital
        returns = np.diff(equity_series) / equity_series[:-1]

        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0

        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0

        max_dd, max_dd_duration = self.calculate_max_drawdown(equity_curve)

        # Risk-adjusted ratios
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        calmar = self.calculate_calmar_ratio(annualized_return, max_dd)

        # Trade statistics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        losing_trades = sum(1 for t in self.trades if t.pnl <= 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in self.trades if t.pnl <= 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = sum(losses) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        expectancy = (win_rate * avg_win - (1 - win_rate) * avg_loss) if total_trades > 0 else 0

        # Additional metrics
        best_trade = max((t.pnl for t in self.trades), default=0)
        worst_trade = min((t.pnl for t in self.trades), default=0)

        durations = [(t.exit_time - t.entry_time) / 86400 for t in self.trades]  # Days
        avg_duration = np.mean(durations) if durations else 0

        # Streaks
        longest_win_streak = 0
        longest_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0

        for trade in sorted(self.trades, key=lambda t: t.exit_time):
            if trade.pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
                longest_win_streak = max(longest_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                longest_loss_streak = max(longest_loss_streak, current_loss_streak)

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cagr=cagr,
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_trade_duration=avg_duration,
            longest_winning_streak=longest_win_streak,
            longest_losing_streak=longest_loss_streak
        )


def test_performance_analytics():
    """Test Trading Performance Analytics"""
    print("=" * 70)
    print("Testing Trading Performance Analytics")
    print("=" * 70)

    # Initialize analyzer
    analyzer = PerformanceAnalyzer(initial_capital=100000, risk_free_rate=0.02)

    # Generate synthetic trades
    np.random.seed(42)
    n_trades = 50

    print("\n1. Generating Synthetic Trade History:")
    print("-" * 70)

    base_time = int(datetime(2024, 1, 1).timestamp())

    for i in range(n_trades):
        # Random trade characteristics
        win_prob = 0.55  # 55% win rate
        is_winner = np.random.random() < win_prob

        entry_time = base_time + i * 86400 * 7  # Weekly trades
        exit_time = entry_time + np.random.randint(1, 10) * 86400  # 1-10 days

        entry_price = 100 + np.random.randn() * 10
        if is_winner:
            pnl_pct = np.random.uniform(0.01, 0.08)  # 1-8% win
        else:
            pnl_pct = np.random.uniform(-0.06, -0.01)  # 1-6% loss

        exit_price = entry_price * (1 + pnl_pct)
        quantity = 100
        pnl = (exit_price - entry_price) * quantity

        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            side='long',
            pnl=pnl,
            pnl_percent=pnl_pct,
            symbol='TEST',
            strategy='test_strategy'
        )

        analyzer.add_trade(trade)

    print(f"Generated {n_trades} synthetic trades")
    print(f"Time period: ~{n_trades // 4} weeks")

    print("\n2. Performance Metrics:")
    print("-" * 70)

    metrics = analyzer.analyze_performance()

    print("Return Metrics:")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  Annualized Return: {metrics.annualized_return:.2%}")
    print(f"  CAGR: {metrics.cagr:.2%}")

    print("\nRisk Metrics:")
    print(f"  Volatility: {metrics.volatility:.2%}")
    print(f"  Downside Deviation: {metrics.downside_deviation:.2%}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Max DD Duration: {metrics.max_drawdown_duration} trades")

    print("\nRisk-Adjusted Metrics:")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:.3f}")
    print(f"  Calmar Ratio: {metrics.calmar_ratio:.3f}")

    print("\n3. Trade Statistics:")
    print("-" * 70)

    print(f"Total Trades: {metrics.total_trades}")
    print(f"Winning Trades: {metrics.winning_trades}")
    print(f"Losing Trades: {metrics.losing_trades}")
    print(f"Win Rate: {metrics.win_rate:.1%}")

    print(f"\nAverage Win: ${metrics.avg_win:.2f}")
    print(f"Average Loss: ${metrics.avg_loss:.2f}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"Expectancy: ${metrics.expectancy:.2f} per trade")

    print(f"\nBest Trade: ${metrics.best_trade:.2f}")
    print(f"Worst Trade: ${metrics.worst_trade:.2f}")
    print(f"Average Trade Duration: {metrics.avg_trade_duration:.1f} days")

    print("\n4. Streaks:")
    print("-" * 70)

    print(f"Longest Winning Streak: {metrics.longest_winning_streak} trades")
    print(f"Longest Losing Streak: {metrics.longest_losing_streak} trades")

    print("\n5. Equity Curve Analysis:")
    print("-" * 70)

    equity_curve = analyzer.equity_curve

    print(f"Initial Capital: ${analyzer.initial_capital:,.0f}")
    print(f"Final Equity: ${equity_curve[-1].equity:,.0f}")
    print(f"Peak Equity: ${max(p.equity for p in equity_curve):,.0f}")
    print(f"Lowest Equity: ${min(p.equity for p in equity_curve):,.0f}")

    # Show equity progression
    print("\nEquity Progression (sample points):")
    print(f"{'Trade #':<10} {'Equity':<15} {'Drawdown':<15}")
    print("-" * 40)

    for i in [0, len(equity_curve)//4, len(equity_curve)//2, -1]:
        point = equity_curve[i]
        print(f"{i:<10} ${point.equity:<14,.0f} {point.drawdown_percent:<15.2%}")

    print("\n6. Risk-Adjusted Performance Interpretation:")
    print("-" * 70)

    print("Sharpe Ratio Interpretation:")
    if metrics.sharpe_ratio > 2:
        print("  Excellent (>2): Outstanding risk-adjusted performance")
    elif metrics.sharpe_ratio > 1:
        print("  Good (1-2): Acceptable risk-adjusted returns")
    elif metrics.sharpe_ratio > 0:
        print("  Poor (0-1): Below par risk-adjusted returns")
    else:
        print("  Negative: Returns below risk-free rate")

    print(f"\nSortino Ratio: {metrics.sortino_ratio:.3f}")
    print("  (Higher is better - focuses on downside risk)")

    print(f"\nCalmar Ratio: {metrics.calmar_ratio:.3f}")
    print("  (Return per unit of maximum drawdown)")

    print("\n7. Strategy Quality Assessment:")
    print("-" * 70)

    quality_score = 0

    # Win rate
    if metrics.win_rate >= 0.5:
        quality_score += 1
        print("[OK] Win rate >= 50%")
    else:
        print("[WARN] Win rate < 50%")

    # Profit factor
    if metrics.profit_factor > 1.5:
        quality_score += 1
        print("[OK] Profit factor > 1.5")
    elif metrics.profit_factor > 1.0:
        print("[WARN] Profit factor marginal (1.0-1.5)")
    else:
        print("[FAIL] Profit factor < 1.0 (losing strategy)")

    # Sharpe ratio
    if metrics.sharpe_ratio > 1.0:
        quality_score += 1
        print("[OK] Sharpe ratio > 1.0")
    else:
        print("[WARN] Sharpe ratio < 1.0")

    # Max drawdown
    if metrics.max_drawdown < 0.20:
        quality_score += 1
        print("[OK] Max drawdown < 20%")
    else:
        print("[WARN] Max drawdown >= 20%")

    print(f"\nOverall Quality Score: {quality_score}/4")
    if quality_score >= 3:
        print("Strategy Assessment: GOOD")
    elif quality_score >= 2:
        print("Strategy Assessment: ACCEPTABLE")
    else:
        print("Strategy Assessment: NEEDS IMPROVEMENT")

    print("\n[SUCCESS] Performance Analytics test completed successfully!")


if __name__ == "__main__":
    test_performance_analytics()
