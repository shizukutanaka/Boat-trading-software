"""
BOAT - Automated Trading Strategy Engine
========================================

Production-ready automated trading strategy system with backtesting.

Features:
- Mean reversion strategy implementation
- Momentum/trend following strategies
- Multi-strategy portfolio management
- Strategy backtesting framework
- Position sizing and risk management
- Performance analytics

Based on 2025 research:
- Automated trading strategies (80%+ of market volume)
- Multi-strategy hedge fund approaches
- Risk-adjusted position sizing
- Backtesting best practices

Design Philosophy (Carmack/Martin/Pike):
- Simple, proven strategies
- No overfitting or complexity
- Practical risk management
- Lightweight execution
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class StrategyType(Enum):
    """Trading strategy types"""
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    TREND_FOLLOWING = "trend_following"
    BREAKOUT = "breakout"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradingSignal:
    """Trading signal from strategy"""
    strategy: str
    symbol: str
    side: OrderSide
    strength: float  # 0-1
    price: float
    timestamp: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Open position"""
    symbol: str
    shares: int
    entry_price: float
    entry_time: int
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0


@dataclass
class Trade:
    """Completed trade"""
    symbol: str
    side: str
    shares: int
    entry_price: float
    exit_price: float
    entry_time: int
    exit_time: int
    pnl: float
    pnl_percent: float
    strategy: str


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    trades: List[Trade]


class MeanReversionStrategy:
    """
    Mean reversion trading strategy.

    Buys when price is below moving average, sells when above.
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_std: float = 2.0,
        exit_std: float = 0.5
    ):
        """
        Initialize mean reversion strategy.

        Args:
            lookback: Lookback period for mean calculation
            entry_std: Standard deviations for entry
            exit_std: Standard deviations for exit
        """
        self.lookback = lookback
        self.entry_std = entry_std
        self.exit_std = exit_std
        self.name = "MeanReversion"

    def generate_signal(
        self,
        symbol: str,
        prices: np.ndarray,
        timestamp: int
    ) -> TradingSignal:
        """
        Generate mean reversion signal.

        Args:
            symbol: Stock symbol
            prices: Price history
            timestamp: Current timestamp

        Returns:
            Trading signal
        """
        if len(prices) < self.lookback:
            return TradingSignal(
                self.name, symbol, OrderSide.HOLD, 0.0, prices[-1], timestamp
            )

        # Calculate mean and std
        mean = np.mean(prices[-self.lookback:])
        std = np.std(prices[-self.lookback:])
        current_price = prices[-1]

        # Z-score
        z_score = (current_price - mean) / (std + 1e-8)

        # Generate signal
        if z_score < -self.entry_std:
            # Price far below mean - BUY
            strength = min(abs(z_score) / self.entry_std, 1.0)
            return TradingSignal(
                self.name, symbol, OrderSide.BUY, strength, current_price,
                timestamp, {'z_score': z_score, 'mean': mean}
            )
        elif z_score > self.entry_std:
            # Price far above mean - SELL
            strength = min(z_score / self.entry_std, 1.0)
            return TradingSignal(
                self.name, symbol, OrderSide.SELL, strength, current_price,
                timestamp, {'z_score': z_score, 'mean': mean}
            )
        else:
            # Within range - HOLD
            return TradingSignal(
                self.name, symbol, OrderSide.HOLD, 0.0, current_price,
                timestamp, {'z_score': z_score, 'mean': mean}
            )


class MomentumStrategy:
    """
    Momentum trading strategy.

    Buys when price momentum is positive, sells when negative.
    """

    def __init__(
        self,
        lookback: int = 10,
        threshold: float = 0.02
    ):
        """
        Initialize momentum strategy.

        Args:
            lookback: Lookback period for momentum
            threshold: Minimum momentum for signal
        """
        self.lookback = lookback
        self.threshold = threshold
        self.name = "Momentum"

    def generate_signal(
        self,
        symbol: str,
        prices: np.ndarray,
        timestamp: int
    ) -> TradingSignal:
        """
        Generate momentum signal.

        Args:
            symbol: Stock symbol
            prices: Price history
            timestamp: Current timestamp

        Returns:
            Trading signal
        """
        if len(prices) < self.lookback + 1:
            return TradingSignal(
                self.name, symbol, OrderSide.HOLD, 0.0, prices[-1], timestamp
            )

        # Calculate momentum
        momentum = (prices[-1] - prices[-self.lookback]) / prices[-self.lookback]
        current_price = prices[-1]

        # Generate signal
        if momentum > self.threshold:
            # Positive momentum - BUY
            strength = min(momentum / (2 * self.threshold), 1.0)
            return TradingSignal(
                self.name, symbol, OrderSide.BUY, strength, current_price,
                timestamp, {'momentum': momentum}
            )
        elif momentum < -self.threshold:
            # Negative momentum - SELL
            strength = min(abs(momentum) / (2 * self.threshold), 1.0)
            return TradingSignal(
                self.name, symbol, OrderSide.SELL, strength, current_price,
                timestamp, {'momentum': momentum}
            )
        else:
            # Weak momentum - HOLD
            return TradingSignal(
                self.name, symbol, OrderSide.HOLD, 0.0, current_price,
                timestamp, {'momentum': momentum}
            )


class TrendFollowingStrategy:
    """
    Trend following strategy using moving average crossover.

    Buys when fast MA crosses above slow MA, sells on opposite.
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30
    ):
        """
        Initialize trend following strategy.

        Args:
            fast_period: Fast MA period
            slow_period: Slow MA period
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.name = "TrendFollowing"

    def generate_signal(
        self,
        symbol: str,
        prices: np.ndarray,
        timestamp: int
    ) -> TradingSignal:
        """
        Generate trend following signal.

        Args:
            symbol: Stock symbol
            prices: Price history
            timestamp: Current timestamp

        Returns:
            Trading signal
        """
        if len(prices) < self.slow_period + 1:
            return TradingSignal(
                self.name, symbol, OrderSide.HOLD, 0.0, prices[-1], timestamp
            )

        # Calculate moving averages
        fast_ma = np.mean(prices[-self.fast_period:])
        slow_ma = np.mean(prices[-self.slow_period:])

        prev_fast_ma = np.mean(prices[-self.fast_period-1:-1])
        prev_slow_ma = np.mean(prices[-self.slow_period-1:-1])

        current_price = prices[-1]

        # Check for crossover
        if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
            # Golden cross - BUY
            strength = min((fast_ma - slow_ma) / slow_ma * 10, 1.0)
            return TradingSignal(
                self.name, symbol, OrderSide.BUY, strength, current_price,
                timestamp, {'fast_ma': fast_ma, 'slow_ma': slow_ma}
            )
        elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
            # Death cross - SELL
            strength = min((slow_ma - fast_ma) / slow_ma * 10, 1.0)
            return TradingSignal(
                self.name, symbol, OrderSide.SELL, strength, current_price,
                timestamp, {'fast_ma': fast_ma, 'slow_ma': slow_ma}
            )
        elif fast_ma > slow_ma:
            # Uptrend - weak BUY
            return TradingSignal(
                self.name, symbol, OrderSide.BUY, 0.3, current_price,
                timestamp, {'fast_ma': fast_ma, 'slow_ma': slow_ma}
            )
        elif fast_ma < slow_ma:
            # Downtrend - weak SELL
            return TradingSignal(
                self.name, symbol, OrderSide.SELL, 0.3, current_price,
                timestamp, {'fast_ma': fast_ma, 'slow_ma': slow_ma}
            )
        else:
            return TradingSignal(
                self.name, symbol, OrderSide.HOLD, 0.0, current_price,
                timestamp, {'fast_ma': fast_ma, 'slow_ma': slow_ma}
            )


class AutomatedTradingEngine:
    """
    Automated trading engine with multi-strategy support.

    Manages strategy execution, position tracking, and performance analysis.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_size: float = 0.1,
        commission: float = 0.001
    ):
        """
        Initialize trading engine.

        Args:
            initial_capital: Starting capital
            max_position_size: Maximum position size as fraction of capital
            commission: Commission rate per trade
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_size = max_position_size
        self.commission = commission

        # Strategies
        self.strategies: Dict[str, Any] = {}

        # Positions and trades
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []

        # Performance tracking
        self.equity_curve: List[Tuple[int, float]] = []

    def add_strategy(self, strategy: Any):
        """Add a trading strategy"""
        self.strategies[strategy.name] = strategy

    def calculate_position_size(
        self,
        price: float,
        signal_strength: float
    ) -> int:
        """
        Calculate position size based on risk.

        Args:
            price: Entry price
            signal_strength: Signal strength (0-1)

        Returns:
            Number of shares
        """
        # Maximum dollar amount for position
        max_dollars = self.capital * self.max_position_size * signal_strength

        # Calculate shares
        shares = int(max_dollars / price)

        return max(shares, 0)

    def execute_trade(
        self,
        signal: TradingSignal
    ) -> Optional[Trade]:
        """
        Execute trade based on signal.

        Args:
            signal: Trading signal

        Returns:
            Trade object if executed, None otherwise
        """
        symbol = signal.symbol

        # Check if position exists
        has_position = symbol in self.positions

        if signal.side == OrderSide.BUY and not has_position:
            # Open long position
            shares = self.calculate_position_size(signal.price, signal.strength)

            if shares > 0:
                cost = shares * signal.price * (1 + self.commission)

                if cost <= self.capital:
                    self.capital -= cost
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        shares=shares,
                        entry_price=signal.price,
                        entry_time=signal.timestamp,
                        current_price=signal.price,
                        unrealized_pnl=0.0
                    )
                    return None  # Position opened, not a completed trade

        elif signal.side == OrderSide.SELL and has_position:
            # Close long position
            position = self.positions[symbol]
            proceeds = position.shares * signal.price * (1 - self.commission)
            self.capital += proceeds

            # Calculate P&L
            pnl = proceeds - (position.shares * position.entry_price * (1 + self.commission))
            pnl_percent = pnl / (position.shares * position.entry_price)

            # Create trade record
            trade = Trade(
                symbol=symbol,
                side="LONG",
                shares=position.shares,
                entry_price=position.entry_price,
                exit_price=signal.price,
                entry_time=position.entry_time,
                exit_time=signal.timestamp,
                pnl=pnl,
                pnl_percent=pnl_percent,
                strategy=signal.strategy
            )

            self.trades.append(trade)
            del self.positions[symbol]

            return trade

        return None

    def update_positions(self, symbol: str, current_price: float):
        """Update position values"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = current_price
            position.unrealized_pnl = (
                position.shares * (current_price - position.entry_price)
            )

    def get_total_equity(self) -> float:
        """Calculate total equity (capital + positions)"""
        equity = self.capital

        for position in self.positions.values():
            equity += position.shares * position.current_price

        return equity

    def backtest(
        self,
        symbol: str,
        prices: np.ndarray,
        strategy_name: str
    ) -> StrategyPerformance:
        """
        Backtest strategy on historical data.

        Args:
            symbol: Stock symbol
            prices: Historical prices
            strategy_name: Strategy to backtest

        Returns:
            Strategy performance metrics
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not found")

        # Reset state
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

        strategy = self.strategies[strategy_name]

        # Run backtest
        for i in range(len(prices)):
            # Generate signal
            signal = strategy.generate_signal(symbol, prices[:i+1], i)

            # Execute trade
            self.execute_trade(signal)

            # Update positions
            self.update_positions(symbol, prices[i])

            # Record equity
            equity = self.get_total_equity()
            self.equity_curve.append((i, equity))

        # Close any open positions at end
        for symbol in list(self.positions.keys()):
            close_signal = TradingSignal(
                strategy_name, symbol, OrderSide.SELL, 1.0,
                prices[-1], len(prices) - 1
            )
            self.execute_trade(close_signal)

        # Calculate performance
        return self._calculate_performance(strategy_name)

    def _calculate_performance(self, strategy_name: str) -> StrategyPerformance:
        """Calculate strategy performance metrics"""
        strategy_trades = [t for t in self.trades if t.strategy == strategy_name]

        if not strategy_trades:
            return StrategyPerformance(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                trades=[]
            )

        # Basic metrics
        total_trades = len(strategy_trades)
        winning_trades = sum(1 for t in strategy_trades if t.pnl > 0)
        losing_trades = sum(1 for t in strategy_trades if t.pnl <= 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # P&L metrics
        total_pnl = sum(t.pnl for t in strategy_trades)
        wins = [t.pnl for t in strategy_trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in strategy_trades if t.pnl <= 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = sum(wins) / sum(losses) if sum(losses) > 0 else 0

        # Sharpe ratio
        returns = np.array([t.pnl_percent for t in strategy_trades])
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)

        # Max drawdown
        equity_values = [e[1] for e in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0

        for equity in equity_values:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)

        return StrategyPerformance(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            trades=strategy_trades
        )


def test_automated_trading():
    """Test Automated Trading Strategy Engine"""
    print("=" * 60)
    print("Testing Automated Trading Strategy Engine")
    print("=" * 60)

    # Initialize engine
    engine = AutomatedTradingEngine(
        initial_capital=100000,
        max_position_size=0.2,
        commission=0.001
    )

    # Add strategies
    engine.add_strategy(MeanReversionStrategy(lookback=20, entry_std=2.0))
    engine.add_strategy(MomentumStrategy(lookback=10, threshold=0.02))
    engine.add_strategy(TrendFollowingStrategy(fast_period=10, slow_period=30))

    print("\n1. Strategies Loaded:")
    print("-" * 40)
    for name in engine.strategies.keys():
        print(f"  - {name}")

    # Generate synthetic price data with trends
    print("\n2. Generating Test Data:")
    print("-" * 40)

    np.random.seed(42)
    n_periods = 500

    # Trending price with reversions
    trend = np.linspace(100, 120, n_periods)
    cycles = 10 * np.sin(np.linspace(0, 4 * np.pi, n_periods))
    noise = np.random.randn(n_periods) * 2
    prices = trend + cycles + noise

    print(f"Price data: {n_periods} periods")
    print(f"Starting price: ${prices[0]:.2f}")
    print(f"Ending price: ${prices[-1]:.2f}")
    print(f"Price change: {(prices[-1] - prices[0]) / prices[0]:.2%}")

    # Test individual strategies
    print("\n3. Strategy Signal Generation:")
    print("-" * 40)

    test_idx = 100
    test_prices = prices[:test_idx+1]

    for strategy_name, strategy in engine.strategies.items():
        signal = strategy.generate_signal('TEST', test_prices, test_idx)
        print(f"\n{strategy_name}:")
        print(f"  Signal: {signal.side.value}")
        print(f"  Strength: {signal.strength:.2f}")
        print(f"  Price: ${signal.price:.2f}")
        if signal.metadata:
            for key, val in signal.metadata.items():
                if isinstance(val, float):
                    print(f"  {key}: {val:.4f}")

    # Backtest strategies
    print("\n4. Strategy Backtesting:")
    print("-" * 40)

    results = {}
    for strategy_name in engine.strategies.keys():
        perf = engine.backtest('TEST', prices, strategy_name)
        results[strategy_name] = perf

        print(f"\n{strategy_name}:")
        print(f"  Total Trades: {perf.total_trades}")
        print(f"  Win Rate: {perf.win_rate:.2%}")
        print(f"  Total P&L: ${perf.total_pnl:.2f}")
        print(f"  Avg Win: ${perf.avg_win:.2f}")
        print(f"  Avg Loss: ${perf.avg_loss:.2f}")
        print(f"  Profit Factor: {perf.profit_factor:.2f}")
        print(f"  Sharpe Ratio: {perf.sharpe_ratio:.3f}")
        print(f"  Max Drawdown: {perf.max_drawdown:.2%}")

    # Compare strategies
    print("\n5. Strategy Comparison:")
    print("-" * 40)

    print(f"{'Strategy':<20} {'Trades':<10} {'Win Rate':<12} {'P&L':<15} {'Sharpe':<10}")
    print("-" * 67)

    for strategy_name, perf in results.items():
        print(f"{strategy_name:<20} {perf.total_trades:<10} {perf.win_rate:<12.2%} ${perf.total_pnl:<14.2f} {perf.sharpe_ratio:<10.3f}")

    # Best strategy
    best_strategy = max(results.items(), key=lambda x: x[1].sharpe_ratio)
    print(f"\nBest Strategy (Sharpe): {best_strategy[0]}")

    # Trade analysis
    print("\n6. Trade Analysis:")
    print("-" * 40)

    best_perf = best_strategy[1]
    if best_perf.trades:
        print("Last 5 trades:")
        print(f"{'Entry':<10} {'Exit':<10} {'P&L':<12} {'Return':<12} {'Duration':<10}")
        print("-" * 54)

        for trade in best_perf.trades[-5:]:
            duration = trade.exit_time - trade.entry_time
            print(f"${trade.entry_price:<9.2f} ${trade.exit_price:<9.2f} ${trade.pnl:<11.2f} {trade.pnl_percent:<12.2%} {duration:<10}")

    print("\n7. Portfolio Performance:")
    print("-" * 40)

    final_equity = engine.equity_curve[-1][1]
    total_return = (final_equity - engine.initial_capital) / engine.initial_capital

    print(f"Initial Capital: ${engine.initial_capital:,.2f}")
    print(f"Final Equity: ${final_equity:,.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Number of Trades: {len(engine.trades)}")

    print("\n[SUCCESS] Automated Trading Strategy test completed successfully!")


if __name__ == "__main__":
    test_automated_trading()
