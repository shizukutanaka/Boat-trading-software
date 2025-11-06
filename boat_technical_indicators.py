"""
BOAT - Technical Indicators System
==================================

Production-ready technical indicator calculations for automated trading.
Implements RSI, MACD, Moving Averages, and signal generation.

Features:
- RSI (Relative Strength Index) calculation
- MACD (Moving Average Convergence Divergence)
- Multiple Moving Average types (SMA, EMA)
- Crossover signal detection
- Multi-indicator signal aggregation

Based on 2025 practical trading research.
No external dependencies beyond NumPy.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class MAType(Enum):
    """Moving average types"""
    SMA = "sma"  # Simple Moving Average
    EMA = "ema"  # Exponential Moving Average


@dataclass
class IndicatorSignal:
    """Signal from a single indicator"""
    indicator: str
    signal: Signal
    strength: float  # 0-1
    value: float
    timestamp: int


@dataclass
class AggregatedSignal:
    """Aggregated signal from multiple indicators"""
    final_signal: Signal
    confidence: float  # 0-1
    contributing_signals: List[IndicatorSignal]
    agreement_ratio: float


class TechnicalIndicators:
    """
    Technical indicator calculator for trading signals.

    Implements standard indicators without external dependencies.
    """

    @staticmethod
    def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Simple Moving Average.

        Args:
            prices: Price array
            period: Moving average period

        Returns:
            SMA values
        """
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        sma = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            sma[i] = np.mean(prices[i - period + 1:i + 1])

        return sma

    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average.

        Args:
            prices: Price array
            period: Moving average period

        Returns:
            EMA values
        """
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        ema = np.full(len(prices), np.nan)
        multiplier = 2 / (period + 1)

        # Initialize with SMA
        ema[period - 1] = np.mean(prices[:period])

        # Calculate EMA
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]

        return ema

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate Relative Strength Index.

        Args:
            prices: Price array
            period: RSI period (default 14)

        Returns:
            RSI values (0-100)
        """
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)

        # Calculate price changes
        deltas = np.diff(prices)

        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Calculate initial averages
        avg_gain = np.full(len(prices), np.nan)
        avg_loss = np.full(len(prices), np.nan)

        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        # Calculate smoothed averages
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

        # Calculate RSI
        rsi = np.full(len(prices), np.nan)
        for i in range(period, len(prices)):
            if avg_loss[i] == 0:
                rsi[i] = 100
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_macd(
        prices: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            prices: Price array
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line period (default 9)

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        # Calculate EMAs
        fast_ema = TechnicalIndicators.calculate_ema(prices, fast_period)
        slow_ema = TechnicalIndicators.calculate_ema(prices, slow_period)

        # MACD line
        macd_line = fast_ema - slow_ema

        # Signal line (EMA of MACD)
        valid_macd = macd_line[~np.isnan(macd_line)]
        if len(valid_macd) >= signal_period:
            signal_line = TechnicalIndicators.calculate_ema(
                macd_line[~np.isnan(macd_line)], signal_period
            )

            # Pad signal line to match original length
            signal_full = np.full(len(macd_line), np.nan)
            signal_full[len(macd_line) - len(signal_line):] = signal_line
        else:
            signal_full = np.full(len(macd_line), np.nan)

        # Histogram
        histogram = macd_line - signal_full

        return macd_line, signal_full, histogram

    @staticmethod
    def detect_ma_crossover(
        short_ma: np.ndarray,
        long_ma: np.ndarray
    ) -> List[Tuple[int, Signal]]:
        """
        Detect moving average crossovers.

        Args:
            short_ma: Short period MA
            long_ma: Long period MA

        Returns:
            List of (index, signal) tuples
        """
        crossovers = []

        for i in range(1, len(short_ma)):
            if np.isnan(short_ma[i]) or np.isnan(long_ma[i]):
                continue
            if np.isnan(short_ma[i - 1]) or np.isnan(long_ma[i - 1]):
                continue

            # Golden cross (bullish)
            if short_ma[i - 1] <= long_ma[i - 1] and short_ma[i] > long_ma[i]:
                crossovers.append((i, Signal.BUY))

            # Death cross (bearish)
            elif short_ma[i - 1] >= long_ma[i - 1] and short_ma[i] < long_ma[i]:
                crossovers.append((i, Signal.SELL))

        return crossovers

    @staticmethod
    def generate_rsi_signal(rsi: float, overbought: float = 70, oversold: float = 30) -> IndicatorSignal:
        """Generate signal from RSI value"""
        if np.isnan(rsi):
            return IndicatorSignal("RSI", Signal.HOLD, 0.0, rsi, 0)

        if rsi > overbought:
            strength = min((rsi - overbought) / 30, 1.0)
            return IndicatorSignal("RSI", Signal.SELL, strength, rsi, 0)
        elif rsi < oversold:
            strength = min((oversold - rsi) / 30, 1.0)
            return IndicatorSignal("RSI", Signal.BUY, strength, rsi, 0)
        else:
            return IndicatorSignal("RSI", Signal.HOLD, 0.0, rsi, 0)

    @staticmethod
    def generate_macd_signal(
        macd: float,
        signal: float,
        prev_macd: float,
        prev_signal: float
    ) -> IndicatorSignal:
        """Generate signal from MACD crossover"""
        if any(np.isnan([macd, signal, prev_macd, prev_signal])):
            return IndicatorSignal("MACD", Signal.HOLD, 0.0, macd, 0)

        # Bullish crossover
        if prev_macd <= prev_signal and macd > signal:
            strength = min(abs(macd - signal) / abs(macd + 1e-8), 1.0)
            return IndicatorSignal("MACD", Signal.BUY, strength, macd, 0)

        # Bearish crossover
        elif prev_macd >= prev_signal and macd < signal:
            strength = min(abs(signal - macd) / abs(signal + 1e-8), 1.0)
            return IndicatorSignal("MACD", Signal.SELL, strength, macd, 0)

        # Continuation
        elif macd > signal:
            return IndicatorSignal("MACD", Signal.BUY, 0.3, macd, 0)
        elif macd < signal:
            return IndicatorSignal("MACD", Signal.SELL, 0.3, macd, 0)
        else:
            return IndicatorSignal("MACD", Signal.HOLD, 0.0, macd, 0)

    @staticmethod
    def aggregate_signals(signals: List[IndicatorSignal]) -> AggregatedSignal:
        """
        Aggregate multiple indicator signals.

        Args:
            signals: List of indicator signals

        Returns:
            Aggregated signal with confidence
        """
        if not signals:
            return AggregatedSignal(Signal.HOLD, 0.0, [], 0.0)

        # Count votes
        buy_votes = sum(1 for s in signals if s.signal == Signal.BUY)
        sell_votes = sum(1 for s in signals if s.signal == Signal.SELL)

        # Calculate weighted confidence
        buy_confidence = sum(s.strength for s in signals if s.signal == Signal.BUY)
        sell_confidence = sum(s.strength for s in signals if s.signal == Signal.SELL)

        # Determine final signal
        if buy_votes > sell_votes:
            final_signal = Signal.BUY
            confidence = buy_confidence / len(signals)
            agreement = buy_votes / len(signals)
        elif sell_votes > buy_votes:
            final_signal = Signal.SELL
            confidence = sell_confidence / len(signals)
            agreement = sell_votes / len(signals)
        else:
            final_signal = Signal.HOLD
            confidence = 0.0
            agreement = 0.0

        return AggregatedSignal(
            final_signal=final_signal,
            confidence=confidence,
            contributing_signals=signals,
            agreement_ratio=agreement
        )


def test_technical_indicators():
    """Test Technical Indicators System"""
    print("=" * 60)
    print("Testing Technical Indicators System")
    print("=" * 60)

    # Generate synthetic price data
    np.random.seed(42)
    n_periods = 100
    trend = np.linspace(100, 110, n_periods)
    noise = np.random.randn(n_periods) * 2
    prices = trend + noise

    print("\n1. Moving Averages:")
    print("-" * 40)

    sma_20 = TechnicalIndicators.calculate_sma(prices, 20)
    sma_50 = TechnicalIndicators.calculate_sma(prices, 50)
    ema_20 = TechnicalIndicators.calculate_ema(prices, 20)

    print(f"Current Price: ${prices[-1]:.2f}")
    print(f"SMA(20): ${sma_20[-1]:.2f}")
    print(f"SMA(50): ${sma_50[-1]:.2f}")
    print(f"EMA(20): ${ema_20[-1]:.2f}")

    # Detect crossovers
    crossovers = TechnicalIndicators.detect_ma_crossover(sma_20, sma_50)
    print(f"\nMA Crossovers detected: {len(crossovers)}")
    if crossovers:
        print("Last 3 crossovers:")
        for idx, signal in crossovers[-3:]:
            print(f"  Day {idx}: {signal.value} (Price: ${prices[idx]:.2f})")

    print("\n2. RSI (Relative Strength Index):")
    print("-" * 40)

    rsi = TechnicalIndicators.calculate_rsi(prices, period=14)
    current_rsi = rsi[-1]

    print(f"Current RSI: {current_rsi:.2f}")
    print(f"Status: ", end="")
    if current_rsi > 70:
        print("OVERBOUGHT (>70)")
    elif current_rsi < 30:
        print("OVERSOLD (<30)")
    else:
        print("NEUTRAL (30-70)")

    rsi_signal = TechnicalIndicators.generate_rsi_signal(current_rsi)
    print(f"RSI Signal: {rsi_signal.signal.value.upper()} (strength: {rsi_signal.strength:.2f})")

    print("\n3. MACD (Moving Average Convergence Divergence):")
    print("-" * 40)

    macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(prices)

    print(f"MACD Line: {macd_line[-1]:.4f}")
    print(f"Signal Line: {signal_line[-1]:.4f}")
    print(f"Histogram: {histogram[-1]:.4f}")

    # Generate MACD signal
    if not np.isnan(macd_line[-2]):
        macd_signal = TechnicalIndicators.generate_macd_signal(
            macd_line[-1], signal_line[-1],
            macd_line[-2], signal_line[-2]
        )
        print(f"MACD Signal: {macd_signal.signal.value.upper()} (strength: {macd_signal.strength:.2f})")

    print("\n4. Signal Aggregation:")
    print("-" * 40)

    # Collect all signals
    all_signals = []

    # MA crossover signal
    if len(crossovers) > 0:
        last_cross_idx, last_cross_signal = crossovers[-1]
        if len(prices) - last_cross_idx < 10:  # Recent crossover
            all_signals.append(
                IndicatorSignal("MA_CROSS", last_cross_signal, 0.8, 0, last_cross_idx)
            )

    all_signals.append(rsi_signal)
    if not np.isnan(macd_line[-2]):
        all_signals.append(macd_signal)

    aggregated = TechnicalIndicators.aggregate_signals(all_signals)

    print(f"Contributing Indicators: {len(all_signals)}")
    for sig in all_signals:
        print(f"  {sig.indicator}: {sig.signal.value} (strength: {sig.strength:.2f})")

    print(f"\nAggregated Signal: {aggregated.final_signal.value.upper()}")
    print(f"Confidence: {aggregated.confidence:.2%}")
    print(f"Agreement Ratio: {aggregated.agreement_ratio:.2%}")

    print("\n5. Backtesting MA Crossover Strategy:")
    print("-" * 40)

    position = 0  # 0 = no position, 1 = long
    entry_price = 0
    trades = []

    for idx, signal in crossovers:
        if signal == Signal.BUY and position == 0:
            position = 1
            entry_price = prices[idx]
            trades.append(('BUY', idx, entry_price))
        elif signal == Signal.SELL and position == 1:
            exit_price = prices[idx]
            pnl = exit_price - entry_price
            trades.append(('SELL', idx, exit_price, pnl))
            position = 0

    print(f"Total Trades: {len([t for t in trades if t[0] == 'SELL'])}")
    if any(len(t) > 3 for t in trades):
        total_pnl = sum(t[3] for t in trades if len(t) > 3)
        print(f"Total P&L: ${total_pnl:.2f}")
        winning_trades = sum(1 for t in trades if len(t) > 3 and t[3] > 0)
        print(f"Win Rate: {winning_trades / len([t for t in trades if len(t) > 3]):.2%}")

    print("\n[SUCCESS] Technical Indicators test completed successfully!")


if __name__ == "__main__":
    test_technical_indicators()
