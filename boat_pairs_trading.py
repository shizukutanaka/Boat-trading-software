#!/usr/bin/env python3
"""
Statistical Arbitrage and Pairs Trading Framework
==================================================

Cointegration-based pairs trading strategies:
  - Cointegration testing (Johansen, Engle-Granger)
  - Mean reversion pair identification
  - Dynamic hedge ratio calculation
  - Pair formation and monitoring
  - Entry/exit signal generation
  - Risk management for pairs

Based on 2025 research on cointegrated pairs trading and statistical arbitrage.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PairTradingSignal:
    """Pairs trading signal"""
    symbol1: str
    symbol2: str
    signal: str  # 'long_pair', 'short_pair', 'close'
    z_score: float
    hedge_ratio: float
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5


class CointegrationTester:
    """Test for cointegration between asset pairs"""

    @staticmethod
    def engle_granger_test(
        series1: np.ndarray,
        series2: np.ndarray
    ) -> Dict[str, float]:
        """
        Engle-Granger two-step cointegration test

        Args:
            series1: First price series (T,)
            series2: Second price series (T,)

        Returns:
            Test results with t-statistic
        """
        # Step 1: Regress series1 on series2
        X = np.column_stack([np.ones(len(series2)), series2])
        beta = np.linalg.lstsq(X, series1, rcond=None)[0]

        # Residuals (error correction term)
        residuals = series1 - (beta[0] + beta[1] * series2)

        # Step 2: Test residual stationarity using ADF-like test
        # Simplified: check if residuals mean-revert
        n = len(residuals)

        # Calculate test statistic (simplified ADF)
        residuals_lagged = residuals[:-1]
        residuals_diff = np.diff(residuals)

        X_adf = np.column_stack([np.ones(len(residuals_diff)), residuals_lagged])
        beta_adf = np.linalg.lstsq(X_adf, residuals_diff, rcond=None)[0]

        # Residual sum of squares
        rss = np.sum((residuals_diff - X_adf @ beta_adf) ** 2)
        sigma_sq = rss / (len(residuals_diff) - 2)

        # t-statistic for mean reversion coefficient
        var_beta = sigma_sq / np.sum((residuals_lagged - np.mean(residuals_lagged)) ** 2)
        t_stat = beta_adf[1] / np.sqrt(var_beta)

        # Critical values (approximate)
        critical_values = {
            '90%': -2.57,
            '95%': -2.86,
            '99%': -3.43
        }

        is_cointegrated = t_stat < critical_values['95%']

        return {
            'coefficient': float(beta[1]),
            'intercept': float(beta[0]),
            't_statistic': float(t_stat),
            'is_cointegrated': is_cointegrated,
            'critical_values': critical_values,
            'residuals': residuals
        }

    @staticmethod
    def johansen_test(
        prices: np.ndarray,
        det_order: int = 0
    ) -> Dict[str, Any]:
        """
        Johansen cointegration test (simplified)

        Args:
            prices: Price matrix (T, n_assets)
            det_order: Deterministic order

        Returns:
            Test results with eigenvalues
        """
        T, n = prices.shape

        # Calculate differences
        diffs = np.diff(prices, axis=0)

        # Covariance matrices
        cov_diffs = np.cov(diffs.T)

        # Levels lagged
        prices_lagged = prices[:-1]
        cov_levels = np.cov(prices_lagged.T)

        # Cross-covariance
        cov_cross = np.cov(diffs.T, prices_lagged.T)[:n, n:]

        # Eigenvalues for trace test
        try:
            M = np.linalg.inv(cov_levels) @ cov_cross @ np.linalg.inv(cov_diffs) @ cov_cross.T
            eigenvalues = np.linalg.eigvalsh(M)
            eigenvalues = np.sort(eigenvalues)[::-1]
        except:
            eigenvalues = np.ones(n)

        return {
            'eigenvalues': eigenvalues,
            'n_cointegrating': int(np.sum(eigenvalues > 0.5))
        }


class PairSelection:
    """Identify and select trading pairs"""

    @staticmethod
    def find_cointegrated_pairs(
        prices_df: pd.DataFrame,
        min_correlation: float = 0.7,
        lookback: int = 252
    ) -> List[Tuple[str, str, float]]:
        """
        Find cointegrated pairs from price dataframe

        Args:
            prices_df: DataFrame with prices (T, n_assets)
            min_correlation: Minimum correlation threshold
            lookback: Historical lookback period

        Returns:
            List of (symbol1, symbol2, cointegration_score)
        """
        symbols = prices_df.columns
        pairs = []

        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i >= j:
                    continue

                # Get prices
                price1 = prices_df[sym1].tail(lookback).values
                price2 = prices_df[sym2].tail(lookback).values

                # Check correlation
                corr = np.corrcoef(price1, price2)[0, 1]

                if abs(corr) < min_correlation:
                    continue

                # Cointegration test
                results = CointegrationTester.engle_granger_test(price1, price2)

                if results['is_cointegrated']:
                    # Score based on t-statistic
                    score = abs(results['t_statistic']) / 2.86  # Normalized by 95% critical value
                    pairs.append((sym1, sym2, score))

        # Sort by score
        pairs.sort(key=lambda x: x[2], reverse=True)

        return pairs


class HedgeRatioCalculator:
    """Calculate dynamic hedge ratios for pairs"""

    @staticmethod
    def calculate_hedge_ratio(
        price1: np.ndarray,
        price2: np.ndarray,
        window: int = 60
    ) -> float:
        """
        Calculate optimal hedge ratio via regression

        Args:
            price1: First asset prices (T,)
            price2: Second asset prices (T,)
            window: Rolling window

        Returns:
            Hedge ratio
        """
        # Simple regression: price1 = beta * price2
        X = np.column_stack([np.ones(len(price2)), price2])
        beta = np.linalg.lstsq(X, price1, rcond=None)[0]

        return float(beta[1])

    @staticmethod
    def calculate_spread(
        price1: np.ndarray,
        price2: np.ndarray,
        hedge_ratio: float
    ) -> np.ndarray:
        """
        Calculate spread (price1 - hedge_ratio * price2)

        Args:
            price1: First asset prices
            price2: Second asset prices
            hedge_ratio: Hedge ratio

        Returns:
            Spread time series
        """
        return price1 - hedge_ratio * price2

    @staticmethod
    def calculate_zscore(
        spread: np.ndarray,
        window: int = 60
    ) -> np.ndarray:
        """
        Calculate z-score of spread

        Args:
            spread: Spread time series
            window: Rolling window

        Returns:
            Z-scores
        """
        rolling_mean = pd.Series(spread).rolling(window).mean().values
        rolling_std = pd.Series(spread).rolling(window).std().values

        return (spread - rolling_mean) / (rolling_std + 1e-8)


class PairsTradingStrategy:
    """Complete pairs trading strategy"""

    def __init__(
        self,
        symbol1: str,
        symbol2: str,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5
    ):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

        self.spread_history = []
        self.zscore_history = []
        self.hedge_ratio_history = []

    def calculate_signal(
        self,
        price1: np.ndarray,
        price2: np.ndarray,
        lookback: int = 60
    ) -> Optional[PairTradingSignal]:
        """
        Generate trading signal

        Args:
            price1: First asset prices (T,)
            price2: Second asset prices (T,)
            lookback: Historical window

        Returns:
            Trading signal or None
        """
        # Use recent data
        p1 = price1[-lookback:] if len(price1) > lookback else price1
        p2 = price2[-lookback:] if len(price2) > lookback else price2

        # Calculate hedge ratio
        hedge_ratio = HedgeRatioCalculator.calculate_hedge_ratio(p1, p2)

        # Calculate spread
        spread = HedgeRatioCalculator.calculate_spread(p1, p2, hedge_ratio)

        # Calculate z-score
        zscore = HedgeRatioCalculator.calculate_zscore(spread)

        # Store history
        self.spread_history.append(spread[-1])
        self.zscore_history.append(zscore[-1])
        self.hedge_ratio_history.append(hedge_ratio)

        current_zscore = zscore[-1]

        # Generate signal
        if len(self.zscore_history) > 1:
            prev_zscore = self.zscore_history[-2]

            # Long pair: price1 relatively cheap vs price2
            if current_zscore < -self.entry_threshold and prev_zscore >= -self.entry_threshold:
                return PairTradingSignal(
                    symbol1=self.symbol1,
                    symbol2=self.symbol2,
                    signal='long_pair',
                    z_score=float(current_zscore),
                    hedge_ratio=hedge_ratio,
                    entry_threshold=self.entry_threshold
                )

            # Short pair: price1 relatively expensive vs price2
            if current_zscore > self.entry_threshold and prev_zscore <= self.entry_threshold:
                return PairTradingSignal(
                    symbol1=self.symbol1,
                    symbol2=self.symbol2,
                    signal='short_pair',
                    z_score=float(current_zscore),
                    hedge_ratio=hedge_ratio,
                    entry_threshold=self.entry_threshold
                )

            # Exit signal
            if abs(current_zscore) < self.exit_threshold:
                return PairTradingSignal(
                    symbol1=self.symbol1,
                    symbol2=self.symbol2,
                    signal='close',
                    z_score=float(current_zscore),
                    hedge_ratio=hedge_ratio
                )

        return None


class PairsRiskManagement:
    """Risk management for pairs trading"""

    @staticmethod
    def calculate_pair_pnl(
        entry_spread: float,
        current_spread: float,
        position_size: float
    ) -> float:
        """
        Calculate P&L from pairs trade

        Args:
            entry_spread: Spread at entry
            current_spread: Current spread
            position_size: Size of position

        Returns:
            P&L
        """
        spread_change = current_spread - entry_spread
        return position_size * spread_change

    @staticmethod
    def calculate_pair_drawdown(
        spread_history: List[float],
        entry_spread: float
    ) -> float:
        """
        Calculate maximum drawdown from entry

        Args:
            spread_history: History of spreads
            entry_spread: Spread at entry

        Returns:
            Maximum drawdown
        """
        spreads_from_entry = np.array(spread_history) - entry_spread
        running_max = np.maximum.accumulate(spreads_from_entry)
        drawdown = spreads_from_entry - running_max

        return float(np.min(drawdown))


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate cointegrated price series
    T = 300
    base = np.cumsum(np.random.randn(T) * 0.01)
    price1 = 100 * np.exp(base)
    price2 = 100 * np.exp(0.95 * base + np.random.randn(T) * 0.005)

    prices_df = pd.DataFrame({
        'ASSET1': price1,
        'ASSET2': price2
    })

    # Test cointegration
    results = CointegrationTester.engle_granger_test(price1, price2)

    logger.info("Cointegration Test:")
    logger.info(f"t-statistic: {results['t_statistic']:.4f}")
    logger.info(f"Is cointegrated: {results['is_cointegrated']}")
    logger.info(f"Hedge ratio: {results['coefficient']:.4f}")

    # Pairs trading
    strategy = PairsTradingStrategy('ASSET1', 'ASSET2')
    signal = strategy.calculate_signal(price1, price2, lookback=60)

    if signal:
        logger.info(f"\nTrading Signal:")
        logger.info(f"Signal: {signal.signal}")
        logger.info(f"Z-score: {signal.z_score:.4f}")
        logger.info(f"Hedge ratio: {signal.hedge_ratio:.4f}")

    # Calculate spread and z-score
    hedge_ratio = HedgeRatioCalculator.calculate_hedge_ratio(price1, price2)
    spread = HedgeRatioCalculator.calculate_spread(price1, price2, hedge_ratio)
    zscore = HedgeRatioCalculator.calculate_zscore(spread)

    logger.info(f"\nSpread Statistics:")
    logger.info(f"Mean: {np.mean(spread):.4f}")
    logger.info(f"Std: {np.std(spread):.4f}")
    logger.info(f"Current Z-score: {zscore[-1]:.4f}")
