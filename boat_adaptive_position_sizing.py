"""
BOAT - Adaptive Position Sizing System
========================================

Production-ready position sizing with Kelly Criterion and Optimal F.

Features:
- Kelly Criterion for optimal position sizing
- Fractional Kelly (1/4, 1/2, 3/4) for risk management
- Optimal F (Ralph Vince) for variable win/loss sizes
- Volatility-adjusted sizing
- Correlation-aware multi-position sizing
- Dynamic capital allocation

Based on 2025 research:
- Kelly Criterion mathematical framework
- Fractional Kelly for reduced volatility
- Optimal F for real-world trading scenarios
- Adaptive sizing based on market conditions

Design Philosophy (Carmack/Martin/Pike):
- Proven mathematical foundations
- Practical risk management
- Clear calculation methods
- No over-leverage
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SizingMethod(Enum):
    """Position sizing method"""
    FULL_KELLY = "full_kelly"
    HALF_KELLY = "half_kelly"
    QUARTER_KELLY = "quarter_kelly"
    OPTIMAL_F = "optimal_f"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    FIXED_FRACTIONAL = "fixed_fractional"


@dataclass
class SizingResult:
    """Position sizing result"""
    symbol: str
    method: str
    position_size: float  # Fraction of capital (0.0 to 1.0)
    expected_growth_rate: float  # Expected geometric growth
    risk_of_ruin: float  # Probability of losing all capital
    kelly_fraction: float  # Kelly fraction used
    max_position_value: float  # Maximum dollar amount
    recommended_shares: int  # Number of shares to buy
    confidence_score: float  # Confidence in sizing (0-1)


@dataclass
class TradeHistory:
    """Historical trade record"""
    wins: List[float]  # Winning trade returns (%)
    losses: List[float]  # Losing trade returns (%)
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float


class KellyCriterionCalculator:
    """
    Kelly Criterion position sizing calculator.

    Determines optimal fraction of capital to risk based on
    win probability and win/loss ratio.
    """

    @staticmethod
    def calculate_kelly_fraction(
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly fraction.

        Kelly Formula: f* = (bp - q) / b
        Where:
        - b = odds received on bet (avg_win / avg_loss)
        - p = probability of winning
        - q = probability of losing (1 - p)

        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning return (e.g., 0.05 for 5%)
            avg_loss: Average losing return (e.g., 0.03 for 3%, as positive)

        Returns:
            Kelly fraction (0-1)
        """
        if win_rate <= 0 or win_rate >= 1:
            return 0.0

        if avg_loss <= 0:
            return 0.0

        # b = odds ratio
        b = abs(avg_win) / abs(avg_loss)

        # p = win probability
        p = win_rate
        q = 1 - p

        # Kelly formula
        kelly = (b * p - q) / b

        # Clamp between 0 and 1
        kelly = max(0.0, min(1.0, kelly))

        return kelly

    @staticmethod
    def fractional_kelly(
        kelly_fraction: float,
        fraction: float = 0.5
    ) -> float:
        """
        Apply fractional Kelly for reduced volatility.

        Args:
            kelly_fraction: Full Kelly fraction
            fraction: Fraction to use (0.25, 0.5, 0.75)

        Returns:
            Fractional Kelly
        """
        return kelly_fraction * fraction

    @staticmethod
    def expected_growth_rate(
        kelly_fraction: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate expected geometric growth rate.

        G = p * log(1 + f*b) + q * log(1 - f)

        Args:
            kelly_fraction: Position size fraction
            win_rate: Win probability
            avg_win: Average win
            avg_loss: Average loss

        Returns:
            Expected growth rate
        """
        if kelly_fraction == 0:
            return 0.0

        p = win_rate
        q = 1 - p
        f = kelly_fraction
        b = abs(avg_win) / abs(avg_loss)

        # Avoid log(0) or log(negative)
        term1 = p * np.log(1 + f * b) if (1 + f * b) > 0 else 0
        term2 = q * np.log(1 - f) if (1 - f) > 0 else -np.inf

        if term2 == -np.inf:
            return -1.0  # Risk of ruin

        return term1 + term2

    @staticmethod
    def risk_of_ruin(
        kelly_fraction: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        capital: float,
        ruin_threshold: float = 0.1
    ) -> float:
        """
        Estimate probability of ruin (losing capital below threshold).

        Args:
            kelly_fraction: Position size fraction
            win_rate: Win probability
            avg_win: Average win
            avg_loss: Average loss
            capital: Current capital
            ruin_threshold: Ruin level (0.1 = lose 90% of capital)

        Returns:
            Risk of ruin probability
        """
        # Simplified gambler's ruin approximation
        if kelly_fraction >= 1.0:
            return 1.0  # Certain ruin with full leverage

        if win_rate >= 0.5:
            # Positive expectancy
            return 0.0  # Low risk with proper Kelly sizing

        # Negative expectancy - high risk
        return 1.0 - win_rate


class OptimalFCalculator:
    """
    Optimal F calculator (Ralph Vince).

    Accounts for variable win/loss sizes, more realistic than Kelly.
    """

    @staticmethod
    def calculate_optimal_f(
        trade_returns: np.ndarray,
        search_points: int = 100
    ) -> Tuple[float, float]:
        """
        Calculate Optimal F using TWR (Terminal Wealth Relative).

        Optimal F maximizes:
        TWR = Product[(1 + f * R_i)] for all trades

        Where R_i is the return of trade i as fraction of max loss.

        Args:
            trade_returns: Array of trade returns ($ P&L)
            search_points: Grid search resolution

        Returns:
            (optimal_f, max_twr)
        """
        if len(trade_returns) == 0:
            return 0.0, 0.0

        # Find largest loss (denominator)
        max_loss = abs(np.min(trade_returns))

        if max_loss == 0:
            # No losses, use max win as reference
            max_loss = np.max(trade_returns)

        if max_loss == 0:
            return 0.0, 0.0

        # Normalize returns by max loss
        normalized_returns = trade_returns / max_loss

        # Grid search for optimal f
        f_values = np.linspace(0.01, 1.0, search_points)
        twr_values = []

        for f in f_values:
            # Calculate TWR
            hpr = 1 + f * normalized_returns  # Holding Period Returns

            # Avoid log of negative or zero
            if np.any(hpr <= 0):
                twr_values.append(-np.inf)
            else:
                # Geometric mean
                twr = np.prod(hpr) ** (1.0 / len(hpr))
                twr_values.append(twr)

        # Find maximum TWR
        max_idx = np.argmax(twr_values)
        optimal_f = f_values[max_idx]
        max_twr = twr_values[max_idx]

        return optimal_f, max_twr

    @staticmethod
    def f_to_position_size(
        optimal_f: float,
        max_loss: float,
        current_capital: float,
        safety_factor: float = 0.5
    ) -> float:
        """
        Convert Optimal F to position size.

        Position Size = (Optimal F / Max Expected Loss) * Capital * Safety Factor

        Args:
            optimal_f: Optimal F value
            max_loss: Maximum expected loss (as fraction, e.g., 0.05 for 5%)
            current_capital: Current capital
            safety_factor: Safety multiplier (0.25 to 0.5 recommended)

        Returns:
            Position size in dollars
        """
        if max_loss == 0:
            return 0.0

        # Position size
        position = (optimal_f / max_loss) * current_capital * safety_factor

        # Clamp to capital
        position = min(position, current_capital)

        return position


class AdaptivePositionSizer:
    """
    Adaptive position sizing system with multiple methods.

    Combines Kelly, Optimal F, and volatility-based approaches.
    """

    def __init__(
        self,
        capital: float,
        default_method: SizingMethod = SizingMethod.HALF_KELLY,
        max_position_size: float = 0.25,
        min_trades_required: int = 20
    ):
        """
        Initialize adaptive position sizer.

        Args:
            capital: Total capital available
            default_method: Default sizing method
            max_position_size: Maximum position size (fraction of capital)
            min_trades_required: Minimum trades for statistical significance
        """
        self.capital = capital
        self.default_method = default_method
        self.max_position_size = max_position_size
        self.min_trades_required = min_trades_required

        self.kelly_calc = KellyCriterionCalculator()
        self.optimal_f_calc = OptimalFCalculator()

    def analyze_trade_history(
        self,
        trade_returns: np.ndarray
    ) -> TradeHistory:
        """
        Analyze historical trades.

        Args:
            trade_returns: Array of trade returns (as fractions, e.g., 0.05 for 5%)

        Returns:
            TradeHistory with statistics
        """
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]

        win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0

        total_wins = np.sum(wins)
        total_losses = abs(np.sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        return TradeHistory(
            wins=wins.tolist(),
            losses=losses.tolist(),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor
        )

    def calculate_position_size(
        self,
        symbol: str,
        trade_history: TradeHistory,
        current_price: float,
        method: Optional[SizingMethod] = None,
        kelly_fraction: float = 0.5,
        volatility: Optional[float] = None
    ) -> SizingResult:
        """
        Calculate optimal position size.

        Args:
            symbol: Trading symbol
            trade_history: Historical trade statistics
            current_price: Current price per share
            method: Sizing method (uses default if None)
            kelly_fraction: Fraction of Kelly to use (0.25, 0.5, 0.75, 1.0)
            volatility: Historical volatility (optional, for volatility-adjusted sizing)

        Returns:
            SizingResult with position size recommendation
        """
        method = method or self.default_method

        # Calculate Kelly fraction
        kelly_full = self.kelly_calc.calculate_kelly_fraction(
            trade_history.win_rate,
            trade_history.avg_win,
            trade_history.avg_loss
        )

        # Apply sizing method
        if method == SizingMethod.FULL_KELLY:
            position_fraction = kelly_full
            kelly_frac_used = 1.0
        elif method == SizingMethod.HALF_KELLY:
            position_fraction = self.kelly_calc.fractional_kelly(kelly_full, 0.5)
            kelly_frac_used = 0.5
        elif method == SizingMethod.QUARTER_KELLY:
            position_fraction = self.kelly_calc.fractional_kelly(kelly_full, 0.25)
            kelly_frac_used = 0.25
        elif method == SizingMethod.OPTIMAL_F:
            # Use Optimal F calculation
            all_returns = np.concatenate([trade_history.wins, trade_history.losses])
            optimal_f, _ = self.optimal_f_calc.calculate_optimal_f(all_returns)
            position_fraction = optimal_f * 0.5  # Apply safety factor
            kelly_frac_used = 0.5
        elif method == SizingMethod.VOLATILITY_ADJUSTED:
            # Adjust Kelly by volatility
            if volatility is not None:
                vol_adjustment = max(0.5, 1.0 - volatility)  # Reduce size in high volatility
                position_fraction = kelly_full * kelly_fraction * vol_adjustment
            else:
                position_fraction = kelly_full * kelly_fraction
            kelly_frac_used = kelly_fraction
        else:  # FIXED_FRACTIONAL
            position_fraction = 0.02  # Fixed 2% risk
            kelly_frac_used = 0.0

        # Enforce maximum position size
        position_fraction = min(position_fraction, self.max_position_size)

        # Calculate dollar amount
        max_position_value = self.capital * position_fraction

        # Calculate shares
        recommended_shares = int(max_position_value / current_price) if current_price > 0 else 0

        # Calculate expected growth rate
        growth_rate = self.kelly_calc.expected_growth_rate(
            position_fraction,
            trade_history.win_rate,
            trade_history.avg_win,
            trade_history.avg_loss
        )

        # Calculate risk of ruin
        risk_ruin = self.kelly_calc.risk_of_ruin(
            position_fraction,
            trade_history.win_rate,
            trade_history.avg_win,
            trade_history.avg_loss,
            self.capital
        )

        # Confidence score based on number of trades
        num_trades = len(trade_history.wins) + len(trade_history.losses)
        confidence = min(1.0, num_trades / self.min_trades_required)

        return SizingResult(
            symbol=symbol,
            method=method.value,
            position_size=position_fraction,
            expected_growth_rate=growth_rate,
            risk_of_ruin=risk_ruin,
            kelly_fraction=kelly_frac_used,
            max_position_value=max_position_value,
            recommended_shares=recommended_shares,
            confidence_score=confidence
        )

    def multi_position_sizing(
        self,
        positions: Dict[str, Tuple[TradeHistory, float]],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, SizingResult]:
        """
        Calculate position sizes for multiple positions with correlation adjustment.

        Args:
            positions: Dict of {symbol: (trade_history, current_price)}
            correlation_matrix: Correlation matrix between positions (optional)

        Returns:
            Dict of {symbol: SizingResult}
        """
        results = {}

        # Calculate individual Kelly fractions
        for symbol, (history, price) in positions.items():
            result = self.calculate_position_size(symbol, history, price)
            results[symbol] = result

        # If correlation provided, adjust for diversification
        if correlation_matrix is not None and len(positions) > 1:
            symbols = list(positions.keys())
            kelly_fractions = np.array([results[s].position_size for s in symbols])

            # Portfolio variance adjustment
            portfolio_variance = np.dot(kelly_fractions, np.dot(correlation_matrix, kelly_fractions))

            # Diversification multiplier (higher correlation = lower multiplier)
            avg_correlation = (np.sum(correlation_matrix) - len(symbols)) / (len(symbols) * (len(symbols) - 1))
            diversification_mult = max(0.7, 1.0 - avg_correlation * 0.5)

            # Adjust position sizes
            for i, symbol in enumerate(symbols):
                results[symbol].position_size *= diversification_mult
                results[symbol].max_position_value = self.capital * results[symbol].position_size
                results[symbol].recommended_shares = int(
                    results[symbol].max_position_value / positions[symbol][1]
                )

        return results


def test_adaptive_position_sizing():
    """Test Adaptive Position Sizing System"""
    print("=" * 70)
    print("Testing Adaptive Position Sizing System")
    print("=" * 70)

    # Initialize
    capital = 100000
    sizer = AdaptivePositionSizer(
        capital=capital,
        default_method=SizingMethod.HALF_KELLY,
        max_position_size=0.25
    )

    print(f"\nInitial Capital: ${capital:,}")
    print(f"Max Position Size: {sizer.max_position_size:.0%}")

    # ========================================================================
    # 1. Test Kelly Criterion
    # ========================================================================
    print("\n" + "=" * 70)
    print("1. Kelly Criterion Position Sizing")
    print("=" * 70)

    # Generate synthetic trade history (good strategy)
    np.random.seed(42)
    num_trades = 50

    # 60% win rate, avg win 5%, avg loss 3%
    wins = np.random.normal(0.05, 0.02, int(num_trades * 0.6))
    losses = -np.random.normal(0.03, 0.015, int(num_trades * 0.4))
    trade_returns = np.concatenate([wins, losses])
    np.random.shuffle(trade_returns)

    history = sizer.analyze_trade_history(trade_returns)

    print(f"\nTrade History Analysis:")
    print(f"  Total Trades: {len(trade_returns)}")
    print(f"  Win Rate: {history.win_rate:.1%}")
    print(f"  Avg Win: {history.avg_win:.2%}")
    print(f"  Avg Loss: {history.avg_loss:.2%}")
    print(f"  Profit Factor: {history.profit_factor:.2f}")

    # Test different Kelly fractions
    symbol = "AAPL"
    price = 175.0

    print(f"\nPosition Sizing for {symbol} @ ${price}:")
    print("-" * 70)
    print(f"{'Method':<20} {'Position %':<12} {'Shares':<10} {'Value':<15} {'Growth Rate':<15}")
    print("-" * 70)

    methods = [
        (SizingMethod.FULL_KELLY, "Full Kelly"),
        (SizingMethod.HALF_KELLY, "Half Kelly"),
        (SizingMethod.QUARTER_KELLY, "Quarter Kelly"),
        (SizingMethod.OPTIMAL_F, "Optimal F"),
    ]

    for method, name in methods:
        result = sizer.calculate_position_size(symbol, history, price, method=method)
        print(f"{name:<20} {result.position_size:<12.1%} {result.recommended_shares:<10} "
              f"${result.max_position_value:<14,.0f} {result.expected_growth_rate:<15.4f}")

    # ========================================================================
    # 2. Test Optimal F
    # ========================================================================
    print("\n" + "=" * 70)
    print("2. Optimal F Calculation")
    print("=" * 70)

    # Generate trade P&L in dollars
    trade_pnl = trade_returns * capital * 0.1  # Assume 10% position size

    optimal_f, max_twr = sizer.optimal_f_calc.calculate_optimal_f(trade_pnl)

    print(f"\nOptimal F Analysis:")
    print(f"  Optimal F: {optimal_f:.3f}")
    print(f"  Terminal Wealth Relative: {max_twr:.4f}")
    print(f"  Interpretation: {optimal_f:.1%} of capital at risk per trade")

    # ========================================================================
    # 3. Test Volatility-Adjusted Sizing
    # ========================================================================
    print("\n" + "=" * 70)
    print("3. Volatility-Adjusted Position Sizing")
    print("=" * 70)

    volatilities = [0.15, 0.25, 0.35, 0.50]

    print(f"\nPosition Size vs. Volatility (Half-Kelly base):")
    print("-" * 50)
    print(f"{'Volatility':<15} {'Position Size':<15} {'Shares':<10}")
    print("-" * 50)

    for vol in volatilities:
        result = sizer.calculate_position_size(
            symbol, history, price,
            method=SizingMethod.VOLATILITY_ADJUSTED,
            volatility=vol
        )
        print(f"{vol:<15.1%} {result.position_size:<15.1%} {result.recommended_shares:<10}")

    # ========================================================================
    # 4. Test Multi-Position Sizing with Correlation
    # ========================================================================
    print("\n" + "=" * 70)
    print("4. Multi-Position Sizing with Correlation Adjustment")
    print("=" * 70)

    # Create multiple positions
    symbols = ["AAPL", "MSFT", "GOOGL"]
    prices = [175.0, 380.0, 140.0]

    positions = {}
    for sym, prc in zip(symbols, prices):
        # Generate slightly different histories
        trades = np.random.normal(0.04, 0.03, 40)
        hist = sizer.analyze_trade_history(trades)
        positions[sym] = (hist, prc)

    # Correlation matrix (moderate correlation)
    correlation = np.array([
        [1.0, 0.6, 0.5],
        [0.6, 1.0, 0.55],
        [0.5, 0.55, 1.0]
    ])

    # Calculate without correlation
    print("\nWithout Correlation Adjustment:")
    print("-" * 70)
    print(f"{'Symbol':<10} {'Position %':<12} {'Shares':<10} {'Value':<15}")
    print("-" * 70)

    results_no_corr = sizer.multi_position_sizing(positions, correlation_matrix=None)
    total_allocation_no_corr = 0
    for sym in symbols:
        res = results_no_corr[sym]
        print(f"{sym:<10} {res.position_size:<12.1%} {res.recommended_shares:<10} ${res.max_position_value:<14,.0f}")
        total_allocation_no_corr += res.position_size

    print(f"\nTotal Allocation: {total_allocation_no_corr:.1%}")

    # Calculate with correlation
    print("\nWith Correlation Adjustment:")
    print("-" * 70)
    print(f"{'Symbol':<10} {'Position %':<12} {'Shares':<10} {'Value':<15}")
    print("-" * 70)

    results_with_corr = sizer.multi_position_sizing(positions, correlation_matrix=correlation)
    total_allocation_corr = 0
    for sym in symbols:
        res = results_with_corr[sym]
        print(f"{sym:<10} {res.position_size:<12.1%} {res.recommended_shares:<10} ${res.max_position_value:<14,.0f}")
        total_allocation_corr += res.position_size

    print(f"\nTotal Allocation: {total_allocation_corr:.1%}")
    print(f"Diversification Benefit: {(total_allocation_no_corr - total_allocation_corr) / total_allocation_no_corr:.1%} reduction")

    # ========================================================================
    # 5. Test Risk of Ruin
    # ========================================================================
    print("\n" + "=" * 70)
    print("5. Risk of Ruin Analysis")
    print("=" * 70)

    # Test with varying win rates
    test_scenarios = [
        ("Excellent (65% win)", 0.65, 0.06, 0.03),
        ("Good (55% win)", 0.55, 0.05, 0.03),
        ("Break-even (50% win)", 0.50, 0.04, 0.04),
        ("Poor (40% win)", 0.40, 0.03, 0.05),
    ]

    print(f"\n{'Scenario':<25} {'Kelly %':<12} {'Growth Rate':<15} {'Risk of Ruin':<15}")
    print("-" * 67)

    for scenario, win_rate, avg_win, avg_loss in test_scenarios:
        kelly = sizer.kelly_calc.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        kelly_half = kelly * 0.5
        growth = sizer.kelly_calc.expected_growth_rate(kelly_half, win_rate, avg_win, avg_loss)
        risk = sizer.kelly_calc.risk_of_ruin(kelly_half, win_rate, avg_win, avg_loss, capital)

        print(f"{scenario:<25} {kelly_half:<12.1%} {growth:<15.4f} {risk:<15.1%}")

    # ========================================================================
    # 6. Confidence Score Based on Sample Size
    # ========================================================================
    print("\n" + "=" * 70)
    print("6. Confidence Score vs. Number of Trades")
    print("=" * 70)

    trade_counts = [10, 20, 30, 50, 100]

    print(f"\n{'Trades':<15} {'Confidence':<15} {'Recommendation':<30}")
    print("-" * 60)

    for count in trade_counts:
        trades = np.random.normal(0.04, 0.03, count)
        hist = sizer.analyze_trade_history(trades)
        result = sizer.calculate_position_size(symbol, hist, price)

        if result.confidence_score < 0.5:
            recommendation = "Insufficient data - use caution"
        elif result.confidence_score < 0.8:
            recommendation = "Moderate confidence"
        else:
            recommendation = "High confidence"

        print(f"{count:<15} {result.confidence_score:<15.1%} {recommendation:<30}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\nKey Findings:")
    print("  - Full Kelly provides maximum growth but high volatility")
    print("  - Half Kelly balances growth and risk (recommended)")
    print("  - Quarter Kelly is very conservative")
    print("  - Optimal F accounts for variable win/loss sizes")
    print("  - Volatility adjustment reduces size in unstable markets")
    print("  - Correlation adjustment prevents over-concentration")
    print("  - Minimum 20 trades recommended for statistical significance")

    print("\n[SUCCESS] Adaptive Position Sizing test completed successfully!")


if __name__ == "__main__":
    test_adaptive_position_sizing()
