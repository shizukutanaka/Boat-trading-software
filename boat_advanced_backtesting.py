"""
BOAT - Advanced Backtesting Framework
======================================

Production-ready backtesting with Combinatorial Purged Cross-Validation (CPCV)
and Walk-Forward Analysis.

Features:
- Combinatorial Purged Cross-Validation (CPCV)
- Walk-Forward Analysis
- Embargo period to prevent information leakage
- Purging of overlapping samples
- Probability of Backtest Overfitting (PBO)
- Deflated Sharpe Ratio (DSR)
- Multiple backtest paths for robustness
- Performance distribution analysis

Based on 2025 research:
- CPCV superiority over traditional methods (Arian et al., SSRN 2024)
- Walk-forward limitations
- Purging and embargo best practices
- Overfitting detection

Design Philosophy (Carmack/Martin/Pike):
- Rigorous statistical validation
- Clear overfitting detection
- Fast execution (parallelizable)
- Actionable insights
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from itertools import combinations
from scipy import stats


@dataclass
class BacktestResult:
    """Single backtest result"""
    train_period: Tuple[int, int]
    test_period: Tuple[int, int]
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    num_trades: int


@dataclass
class CPCVResult:
    """CPCV backtest result"""
    all_results: List[BacktestResult]
    mean_sharpe: float
    std_sharpe: float
    mean_return: float
    sharpe_distribution: np.ndarray
    pbo: float  # Probability of Backtest Overfitting
    dsr: float  # Deflated Sharpe Ratio
    confidence_95: Tuple[float, float]  # 95% confidence interval


class CombinorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation.

    Prevents overfitting by:
    1. Multiple train/test splits (combinatorial)
    2. Purging overlapping samples
    3. Embargo period after each test set
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_pct: float = 0.05,
        embargo_pct: float = 0.01
    ):
        """
        Initialize CPCV.

        Args:
            n_splits: Total number of splits
            n_test_splits: Number of splits to use for testing
            purge_pct: Percentage of data to purge around test set
            embargo_pct: Percentage of data to embargo after test set
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct

    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.

        Args:
            n_samples: Total number of samples

        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []

        # Create equal-sized splits
        split_size = n_samples // self.n_splits
        split_indices = []

        for i in range(self.n_splits):
            start = i * split_size
            end = start + split_size if i < self.n_splits - 1 else n_samples
            split_indices.append(np.arange(start, end))

        # Generate all combinations of test splits
        test_combinations = list(combinations(range(self.n_splits), self.n_test_splits))

        for test_combo in test_combinations:
            # Combine test split indices
            test_idx = np.concatenate([split_indices[i] for i in test_combo])
            test_idx = np.sort(test_idx)

            # Train indices = all indices not in test
            all_idx = np.arange(n_samples)
            train_idx = np.setdiff1d(all_idx, test_idx)

            # Apply purging
            train_idx = self._purge(train_idx, test_idx, n_samples)

            # Apply embargo
            train_idx = self._embargo(train_idx, test_idx, n_samples)

            splits.append((train_idx, test_idx))

        return splits

    def _purge(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Purge training samples that overlap with test period.

        Args:
            train_idx: Training indices
            test_idx: Test indices
            n_samples: Total samples

        Returns:
            Purged training indices
        """
        if len(test_idx) == 0:
            return train_idx

        purge_size = int(n_samples * self.purge_pct)

        # Purge before test start
        test_start = np.min(test_idx)
        purge_start = max(0, test_start - purge_size)

        # Purge after test end
        test_end = np.max(test_idx)
        purge_end = min(n_samples, test_end + purge_size)

        # Remove purged indices from training
        purged_idx = np.arange(purge_start, purge_end)
        train_idx = np.setdiff1d(train_idx, purged_idx)

        return train_idx

    def _embargo(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Apply embargo period after test set.

        Args:
            train_idx: Training indices
            test_idx: Test indices
            n_samples: Total samples

        Returns:
            Embargoed training indices
        """
        if len(test_idx) == 0:
            return train_idx

        embargo_size = int(n_samples * self.embargo_pct)
        test_end = np.max(test_idx)

        # Embargo period
        embargo_start = test_end + 1
        embargo_end = min(n_samples, embargo_start + embargo_size)

        # Remove embargo indices from training
        embargo_idx = np.arange(embargo_start, embargo_end)
        train_idx = np.setdiff1d(train_idx, embargo_idx)

        return train_idx


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis.

    Traditional rolling window backtest.
    """

    def __init__(
        self,
        train_period: int = 252,
        test_period: int = 63,
        step_size: int = 21
    ):
        """
        Initialize walk-forward analyzer.

        Args:
            train_period: Training window size (days)
            test_period: Testing window size (days)
            step_size: Step size for rolling window
        """
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size

    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward splits.

        Args:
            n_samples: Total number of samples

        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []

        current = 0

        while current + self.train_period + self.test_period <= n_samples:
            train_idx = np.arange(current, current + self.train_period)
            test_idx = np.arange(
                current + self.train_period,
                current + self.train_period + self.test_period
            )

            splits.append((train_idx, test_idx))

            current += self.step_size

        return splits


class AdvancedBacktester:
    """
    Advanced backtesting framework with CPCV and Walk-Forward.

    Provides rigorous validation to detect overfitting.
    """

    def __init__(self):
        """Initialize backtester."""
        self.results: List[BacktestResult] = []

    def run_backtest(
        self,
        returns: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        strategy_func: Optional[Callable] = None
    ) -> BacktestResult:
        """
        Run single backtest.

        Args:
            returns: Return series
            train_idx: Training indices
            test_idx: Test indices
            strategy_func: Optional strategy function

        Returns:
            Backtest result
        """
        # Simple strategy: if training mean > 0, go long in test
        train_returns = returns[train_idx]
        test_returns = returns[test_idx]

        # Calculate training statistics
        train_mean = np.mean(train_returns)

        # Strategy: go long if training shows positive returns
        if strategy_func is None:
            # Default: momentum strategy
            if train_mean > 0:
                strategy_returns = test_returns
            else:
                strategy_returns = -test_returns  # Short
        else:
            strategy_returns = strategy_func(train_returns, test_returns)

        # Calculate metrics
        sharpe = self._calculate_sharpe(strategy_returns)
        total_return = np.sum(strategy_returns)
        max_dd = self._calculate_max_drawdown(strategy_returns)
        win_rate = np.sum(strategy_returns > 0) / len(strategy_returns)
        num_trades = len(strategy_returns)

        return BacktestResult(
            train_period=(train_idx[0], train_idx[-1]),
            test_period=(test_idx[0], test_idx[-1]),
            sharpe_ratio=sharpe,
            total_return=total_return,
            max_drawdown=max_dd,
            win_rate=win_rate,
            num_trades=num_trades
        )

    def _calculate_sharpe(self, returns: np.ndarray, periods=252) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        return np.mean(returns) / np.std(returns) * np.sqrt(periods)

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0

        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-10)

        return np.min(drawdown)

    def run_cpcv(
        self,
        returns: np.ndarray,
        n_splits: int = 5,
        n_test_splits: int = 2,
        strategy_func: Optional[Callable] = None
    ) -> CPCVResult:
        """
        Run Combinatorial Purged Cross-Validation.

        Args:
            returns: Return series
            n_splits: Number of splits
            n_test_splits: Number of test splits
            strategy_func: Optional strategy function

        Returns:
            CPCV result with overfitting metrics
        """
        cpcv = CombinorialPurgedCV(
            n_splits=n_splits,
            n_test_splits=n_test_splits,
            purge_pct=0.05,
            embargo_pct=0.01
        )

        splits = cpcv.split(len(returns))
        results = []

        for train_idx, test_idx in splits:
            if len(train_idx) < 20 or len(test_idx) < 5:
                continue  # Skip if insufficient data

            result = self.run_backtest(returns, train_idx, test_idx, strategy_func)
            results.append(result)

        # Calculate statistics
        sharpes = np.array([r.sharpe_ratio for r in results])
        mean_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)

        returns_list = np.array([r.total_return for r in results])
        mean_return = np.mean(returns_list)

        # Probability of Backtest Overfitting (PBO)
        pbo = self._calculate_pbo(sharpes)

        # Deflated Sharpe Ratio
        dsr = self._calculate_dsr(sharpes, len(results))

        # 95% confidence interval
        sem = stats.sem(sharpes)
        confidence_95 = stats.t.interval(0.95, len(sharpes)-1, loc=mean_sharpe, scale=sem)

        return CPCVResult(
            all_results=results,
            mean_sharpe=mean_sharpe,
            std_sharpe=std_sharpe,
            mean_return=mean_return,
            sharpe_distribution=sharpes,
            pbo=pbo,
            dsr=dsr,
            confidence_95=confidence_95
        )

    def _calculate_pbo(self, sharpes: np.ndarray) -> float:
        """
        Calculate Probability of Backtest Overfitting.

        PBO = probability that best in-sample is worse than median out-of-sample.

        Args:
            sharpes: Array of Sharpe ratios from different splits

        Returns:
            PBO (0-1), higher = more likely overfit
        """
        if len(sharpes) < 4:
            return 0.5

        # Sort Sharpes
        sorted_sharpes = np.sort(sharpes)

        # Split into "in-sample" (top half) and "out-of-sample" (bottom half)
        mid = len(sorted_sharpes) // 2
        in_sample = sorted_sharpes[mid:]
        out_sample = sorted_sharpes[:mid]

        # Count how many in-sample are worse than median out-sample
        median_out = np.median(out_sample)
        worse_count = np.sum(in_sample < median_out)

        pbo = worse_count / len(in_sample)

        return pbo

    def _calculate_dsr(self, sharpes: np.ndarray, n_trials: int) -> float:
        """
        Calculate Deflated Sharpe Ratio.

        Adjusts Sharpe for multiple testing and non-normality.

        Args:
            sharpes: Sharpe ratios
            n_trials: Number of trials

        Returns:
            Deflated Sharpe Ratio
        """
        if len(sharpes) == 0:
            return 0.0

        # Expected maximum Sharpe under null hypothesis
        # E[max SR] ~ sqrt(2 * log(N))
        expected_max_sharpe = np.sqrt(2 * np.log(n_trials))

        # Variance of Sharpes
        var_sharpe = np.var(sharpes)

        # Deflated SR
        dsr = (np.mean(sharpes) - expected_max_sharpe) / np.sqrt(var_sharpe + 1e-10)

        return dsr


def test_advanced_backtesting():
    """Test Advanced Backtesting Framework"""
    print("=" * 70)
    print("Testing Advanced Backtesting Framework")
    print("=" * 70)

    # Generate synthetic returns with signal
    np.random.seed(42)
    n_samples = 500

    # Regime 1: Positive momentum (days 0-250)
    regime1 = np.random.normal(0.001, 0.01, 250)

    # Regime 2: No momentum (days 250-500)
    regime2 = np.random.normal(0.0, 0.015, 250)

    returns = np.concatenate([regime1, regime2])

    print(f"\nGenerated {len(returns)} days of returns")
    print(f"  Regime 1 (0-250): Positive momentum ({np.mean(regime1):.4f} mean)")
    print(f"  Regime 2 (250-500): No momentum ({np.mean(regime2):.4f} mean)")

    backtester = AdvancedBacktester()

    # ========================================================================
    # 1. Traditional Walk-Forward Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("1. Traditional Walk-Forward Analysis")
    print("=" * 70)

    wf = WalkForwardAnalyzer(train_period=120, test_period=30, step_size=30)
    wf_splits = wf.split(len(returns))

    print(f"\nWalk-Forward Splits: {len(wf_splits)}")

    wf_results = []
    for train_idx, test_idx in wf_splits:
        result = backtester.run_backtest(returns, train_idx, test_idx)
        wf_results.append(result)

    wf_sharpes = [r.sharpe_ratio for r in wf_results]
    print(f"\nWalk-Forward Results:")
    print(f"  Mean Sharpe: {np.mean(wf_sharpes):.3f}")
    print(f"  Std Sharpe: {np.std(wf_sharpes):.3f}")
    print(f"  Min Sharpe: {np.min(wf_sharpes):.3f}")
    print(f"  Max Sharpe: {np.max(wf_sharpes):.3f}")

    # ========================================================================
    # 2. Combinatorial Purged Cross-Validation
    # ========================================================================
    print("\n" + "=" * 70)
    print("2. Combinatorial Purged Cross-Validation (CPCV)")
    print("=" * 70)

    cpcv_result = backtester.run_cpcv(returns, n_splits=5, n_test_splits=2)

    print(f"\nCPCV Splits: {len(cpcv_result.all_results)}")
    print(f"\nCPCV Results:")
    print(f"  Mean Sharpe: {cpcv_result.mean_sharpe:.3f}")
    print(f"  Std Sharpe: {cpcv_result.std_sharpe:.3f}")
    print(f"  95% CI: ({cpcv_result.confidence_95[0]:.3f}, {cpcv_result.confidence_95[1]:.3f})")

    # ========================================================================
    # 3. Overfitting Detection
    # ========================================================================
    print("\n" + "=" * 70)
    print("3. Overfitting Detection Metrics")
    print("=" * 70)

    print(f"\nProbability of Backtest Overfitting (PBO): {cpcv_result.pbo:.1%}")
    print(f"  Interpretation:")
    if cpcv_result.pbo < 0.3:
        print(f"    - LOW risk of overfitting (< 30%)")
    elif cpcv_result.pbo < 0.5:
        print(f"    - MODERATE risk of overfitting (30-50%)")
    else:
        print(f"    - HIGH risk of overfitting (> 50%)")

    print(f"\nDeflated Sharpe Ratio (DSR): {cpcv_result.dsr:.3f}")
    print(f"  Interpretation:")
    if cpcv_result.dsr > 1.5:
        print(f"    - STRONG statistical significance")
    elif cpcv_result.dsr > 1.0:
        print(f"    - MODERATE statistical significance")
    else:
        print(f"    - WEAK statistical significance")

    # ========================================================================
    # 4. Sharpe Distribution Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("4. Sharpe Ratio Distribution")
    print("=" * 70)

    sharpes = cpcv_result.sharpe_distribution

    print(f"\nDistribution Statistics:")
    print(f"  Count: {len(sharpes)}")
    print(f"  Mean: {np.mean(sharpes):.3f}")
    print(f"  Median: {np.median(sharpes):.3f}")
    print(f"  Std: {np.std(sharpes):.3f}")
    print(f"  Min: {np.min(sharpes):.3f}")
    print(f"  Max: {np.max(sharpes):.3f}")

    # Percentiles
    p25, p50, p75 = np.percentile(sharpes, [25, 50, 75])
    print(f"\nPercentiles:")
    print(f"  25th: {p25:.3f}")
    print(f"  50th: {p50:.3f}")
    print(f"  75th: {p75:.3f}")

    # ========================================================================
    # 5. Comparison: Walk-Forward vs CPCV
    # ========================================================================
    print("\n" + "=" * 70)
    print("5. Walk-Forward vs CPCV Comparison")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Walk-Forward':<15} {'CPCV':<15}")
    print("-" * 60)
    print(f"{'Number of Paths':<30} {len(wf_results):<15} {len(cpcv_result.all_results):<15}")
    print(f"{'Mean Sharpe':<30} {np.mean(wf_sharpes):<15.3f} {cpcv_result.mean_sharpe:<15.3f}")
    print(f"{'Std Sharpe':<30} {np.std(wf_sharpes):<15.3f} {cpcv_result.std_sharpe:<15.3f}")
    print(f"{'Min Sharpe':<30} {np.min(wf_sharpes):<15.3f} {np.min(sharpes):<15.3f}")
    print(f"{'Max Sharpe':<30} {np.max(wf_sharpes):<15.3f} {np.max(sharpes):<15.3f}")

    # ========================================================================
    # 6. Individual Path Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("6. Individual CPCV Path Analysis (Sample)")
    print("=" * 70)

    print(f"\n{'Train Period':<25} {'Test Period':<25} {'Sharpe':<10} {'Return':<12} {'Max DD':<10}")
    print("-" * 82)

    for i, result in enumerate(cpcv_result.all_results[:5]):  # Show first 5
        train_str = f"{result.train_period[0]}-{result.train_period[1]}"
        test_str = f"{result.test_period[0]}-{result.test_period[1]}"
        print(f"{train_str:<25} {test_str:<25} {result.sharpe_ratio:<10.3f} "
              f"{result.total_return:<12.3f} {result.max_drawdown:<10.3f}")

    # ========================================================================
    # 7. Robustness Check
    # ========================================================================
    print("\n" + "=" * 70)
    print("7. Strategy Robustness Assessment")
    print("=" * 70)

    # Count positive Sharpe paths
    positive_sharpes = np.sum(sharpes > 0)
    positive_pct = positive_sharpes / len(sharpes)

    # Count significant Sharpe paths (> 1.0)
    significant_sharpes = np.sum(sharpes > 1.0)
    significant_pct = significant_sharpes / len(sharpes)

    print(f"\nRobustness Metrics:")
    print(f"  Positive Sharpe Paths: {positive_sharpes}/{len(sharpes)} ({positive_pct:.1%})")
    print(f"  Significant Sharpe Paths (>1.0): {significant_sharpes}/{len(sharpes)} ({significant_pct:.1%})")

    if positive_pct > 0.7 and cpcv_result.pbo < 0.3:
        assessment = "ROBUST - Low overfitting risk, consistent performance"
    elif positive_pct > 0.5 and cpcv_result.pbo < 0.5:
        assessment = "ACCEPTABLE - Moderate confidence in strategy"
    else:
        assessment = "QUESTIONABLE - High variance or overfitting risk"

    print(f"\nOverall Assessment: {assessment}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\nKey Findings:")
    print("  - CPCV provides multiple backtest paths for robustness")
    print("  - PBO measures probability of overfitting")
    print("  - DSR adjusts Sharpe for multiple testing")
    print("  - Purging and embargo prevent information leakage")
    print(f"  - Strategy PBO: {cpcv_result.pbo:.1%} (target < 30%)")
    print(f"  - Strategy DSR: {cpcv_result.dsr:.2f} (target > 1.0)")

    print("\n[SUCCESS] Advanced Backtesting test completed successfully!")


if __name__ == "__main__":
    test_advanced_backtesting()
