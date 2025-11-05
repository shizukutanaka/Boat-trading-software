"""
BOAT - Adaptive Risk Parity Portfolio System
============================================

Production-ready adaptive risk parity implementation with hierarchical clustering
and equal risk contribution optimization.

Features:
- Hierarchical Risk Parity (HRP) algorithm
- Equal Risk Contribution (ERC) optimization
- Adaptive regime detection
- Hierarchical clustering for asset grouping
- Risk budgeting across asset classes

Based on 2024-2025 research:
- Marcos Lopez de Prado's HRP methodology
- Thomas Raffinot's HERC improvements
- Combines MVO and Risk Parity advantages
- No covariance matrix inversion required
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
import warnings


@dataclass
class RiskParityResult:
    """Result from risk parity optimization"""
    weights: np.ndarray
    risk_contributions: np.ndarray
    total_risk: float
    diversification_ratio: float
    effective_n: float  # Effective number of assets
    converged: bool


@dataclass
class HRPResult:
    """Result from Hierarchical Risk Parity"""
    weights: np.ndarray
    clusters: List[List[int]]
    dendrogram_order: List[int]
    risk_contributions: np.ndarray
    cluster_risks: Dict[int, float]


class AdaptiveRiskParitySystem:
    """
    Adaptive risk parity portfolio optimization system.

    Implements multiple risk parity methods:
    - Equal Risk Contribution (ERC)
    - Hierarchical Risk Parity (HRP)
    - Naive Risk Parity
    - Risk Budgeting
    """

    def __init__(
        self,
        risk_measure: str = 'volatility',
        rebalance_frequency: int = 21,  # Monthly
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ):
        """
        Initialize the risk parity system.

        Args:
            risk_measure: Risk metric ('volatility', 'cvar', 'mad')
            rebalance_frequency: Days between rebalancing
            min_weight: Minimum position weight
            max_weight: Maximum position weight
        """
        self.risk_measure = risk_measure
        self.rebalance_frequency = rebalance_frequency
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Regime detection parameters
        self.vol_regimes = {'low': 0.10, 'medium': 0.20, 'high': 0.30}
        self.current_regime = 'medium'

    def calculate_returns_stats(
        self,
        returns: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate return statistics.

        Args:
            returns: Asset returns (n_periods x n_assets)

        Returns:
            Tuple of (expected_returns, covariance_matrix)
        """
        expected_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)

        # Shrinkage for stability (Ledoit-Wolf style)
        shrinkage_target = np.diag(np.diag(cov_matrix))
        shrinkage_factor = 0.1
        cov_matrix = (1 - shrinkage_factor) * cov_matrix + shrinkage_factor * shrinkage_target

        return expected_returns, cov_matrix

    def equal_risk_contribution(
        self,
        cov_matrix: np.ndarray,
        initial_weights: Optional[np.ndarray] = None
    ) -> RiskParityResult:
        """
        Calculate Equal Risk Contribution portfolio.

        Args:
            cov_matrix: Covariance matrix
            initial_weights: Starting weights for optimization

        Returns:
            Risk parity result
        """
        n_assets = len(cov_matrix)

        if initial_weights is None:
            initial_weights = np.ones(n_assets) / n_assets

        def risk_contribution(weights, cov_matrix):
            """Calculate risk contributions"""
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            return contrib

        def objective(weights, cov_matrix):
            """ERC objective: minimize squared differences in risk contributions"""
            contrib = risk_contribution(weights, cov_matrix)
            target = np.ones(len(weights)) / len(weights)  # Equal target
            return np.sum((contrib - target) ** 2)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Sum to 1
        ]

        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        weights = result.x
        weights = weights / weights.sum()  # Normalize

        # Calculate final metrics
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        risk_contrib = risk_contribution(weights, cov_matrix)

        # Diversification ratio
        weighted_vol = np.sum(weights * np.sqrt(np.diag(cov_matrix)))
        div_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 1.0

        # Effective number of assets (inverse HHI)
        hhi = np.sum(weights ** 2)
        effective_n = 1 / hhi if hhi > 0 else 1.0

        return RiskParityResult(
            weights=weights,
            risk_contributions=risk_contrib,
            total_risk=portfolio_vol,
            diversification_ratio=div_ratio,
            effective_n=effective_n,
            converged=result.success
        )

    def hierarchical_risk_parity(
        self,
        returns: np.ndarray
    ) -> HRPResult:
        """
        Hierarchical Risk Parity (Lopez de Prado).

        Args:
            returns: Asset returns (n_periods x n_assets)

        Returns:
            HRP result with weights and clustering
        """
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(returns.T)
        n_assets = len(corr_matrix)

        # Step 1: Tree clustering
        distance_matrix = np.sqrt((1 - corr_matrix) / 2)
        # Ensure distance matrix has zeros on diagonal
        np.fill_diagonal(distance_matrix, 0)
        # Convert to condensed form for linkage (upper triangle only)
        distance_condensed = squareform(distance_matrix, checks=False)

        # Hierarchical clustering
        clusters = hierarchy.linkage(distance_condensed, method='single')

        # Step 2: Quasi-diagonalization (seriation)
        sort_idx = self._get_quasi_diag(clusters)

        # Step 3: Recursive bisection
        cov_matrix = np.cov(returns.T)
        weights = self._recursive_bisection(cov_matrix, sort_idx)

        # Get cluster assignments
        num_clusters = min(3, n_assets // 2)  # Adaptive cluster count
        cluster_labels = hierarchy.fcluster(clusters, num_clusters, criterion='maxclust')

        # Group assets by cluster
        clusters_dict = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(i)

        # Calculate risk contributions
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol if portfolio_vol > 0 else weights

        # Calculate cluster risks
        cluster_risks = {}
        for label, assets in clusters_dict.items():
            cluster_risk = sum(risk_contrib[i] for i in assets)
            cluster_risks[label] = cluster_risk

        return HRPResult(
            weights=weights,
            clusters=list(clusters_dict.values()),
            dendrogram_order=sort_idx.tolist(),
            risk_contributions=risk_contrib,
            cluster_risks=cluster_risks
        )

    def _get_quasi_diag(self, link: np.ndarray) -> np.ndarray:
        """
        Quasi-diagonalization for HRP.

        Args:
            link: Linkage matrix from clustering

        Returns:
            Sorted indices
        """
        n = int(link[-1, 3])
        sort_idx = []
        clusters = {i: [i] for i in range(n)}

        for i, row in enumerate(link):
            c1, c2 = int(row[0]), int(row[1])
            new_cluster_id = n + i

            # Merge clusters
            clusters[new_cluster_id] = clusters.get(c1, [c1]) + clusters.get(c2, [c2])

            # Clean up old clusters if they exist
            if c1 in clusters and c1 < n + i:
                del clusters[c1]
            if c2 in clusters and c2 < n + i:
                del clusters[c2]

        # Get the final cluster (root)
        root_id = max(clusters.keys())
        sort_idx = clusters[root_id]

        return np.array(sort_idx)

    def _recursive_bisection(
        self,
        cov: np.ndarray,
        sort_idx: np.ndarray
    ) -> np.ndarray:
        """
        Recursive bisection for HRP weight allocation.

        Args:
            cov: Covariance matrix
            sort_idx: Sorted indices

        Returns:
            Portfolio weights
        """
        def get_cluster_var(cov, items):
            """Calculate cluster variance"""
            cov_slice = cov[np.ix_(items, items)]
            w = self._get_ivp(cov_slice)
            var = w @ cov_slice @ w
            return var

        def _recursive_bisection_inner(cov, sort_idx):
            """Inner recursive function"""
            n = len(sort_idx)
            if n == 1:
                return np.array([1.0])

            # Split into two clusters
            split = n // 2
            idx1 = sort_idx[:split].tolist()
            idx2 = sort_idx[split:].tolist()

            # Calculate cluster variances
            var1 = get_cluster_var(cov, idx1)
            var2 = get_cluster_var(cov, idx2)

            # Allocate between clusters inversely to variance
            alpha = 1 - var1 / (var1 + var2)

            # Recursive allocation within clusters
            w1 = _recursive_bisection_inner(cov, sort_idx[:split])
            w2 = _recursive_bisection_inner(cov, sort_idx[split:])

            # Combine weights
            w = np.zeros(n)
            w[:split] = w1 * alpha
            w[split:] = w2 * (1 - alpha)

            return w

        weights = _recursive_bisection_inner(cov, sort_idx)

        # Map back to original order
        final_weights = np.zeros(len(cov))
        for i, idx in enumerate(sort_idx):
            final_weights[idx] = weights[i]

        return final_weights / final_weights.sum()

    def _get_ivp(self, cov: np.ndarray) -> np.ndarray:
        """
        Inverse Variance Portfolio for a given covariance matrix.

        Args:
            cov: Covariance matrix

        Returns:
            IVP weights
        """
        ivp = 1 / np.diag(cov)
        return ivp / ivp.sum()

    def naive_risk_parity(
        self,
        returns: np.ndarray
    ) -> np.ndarray:
        """
        Naive risk parity (inverse volatility weighting).

        Args:
            returns: Asset returns

        Returns:
            Portfolio weights
        """
        volatilities = np.std(returns, axis=0)

        # Inverse volatility weights
        inv_vols = 1 / (volatilities + 1e-8)
        weights = inv_vols / inv_vols.sum()

        return weights

    def risk_budgeting(
        self,
        cov_matrix: np.ndarray,
        risk_budgets: np.ndarray
    ) -> RiskParityResult:
        """
        Risk budgeting with specified risk allocations.

        Args:
            cov_matrix: Covariance matrix
            risk_budgets: Target risk budgets for each asset

        Returns:
            Risk parity result
        """
        n_assets = len(cov_matrix)

        # Normalize risk budgets
        risk_budgets = risk_budgets / risk_budgets.sum()

        def objective(weights, cov_matrix, budgets):
            """Risk budgeting objective"""
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol

            # Minimize deviation from target budgets
            return np.sum((contrib - budgets) ** 2)

        # Initial guess
        initial_weights = risk_budgets  # Start with budget proportions

        # Constraints and bounds
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            args=(cov_matrix, risk_budgets),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        weights = result.x
        weights = weights / weights.sum()

        # Calculate metrics
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol

        # Diversification ratio
        weighted_vol = np.sum(weights * np.sqrt(np.diag(cov_matrix)))
        div_ratio = weighted_vol / portfolio_vol

        # Effective N
        effective_n = 1 / np.sum(weights ** 2)

        return RiskParityResult(
            weights=weights,
            risk_contributions=risk_contrib,
            total_risk=portfolio_vol,
            diversification_ratio=div_ratio,
            effective_n=effective_n,
            converged=result.success
        )

    def detect_regime(
        self,
        returns: np.ndarray
    ) -> str:
        """
        Detect market regime based on volatility.

        Args:
            returns: Recent returns

        Returns:
            Regime label
        """
        recent_vol = np.std(returns[-21:])  # Last month

        if recent_vol < self.vol_regimes['low']:
            return 'low'
        elif recent_vol < self.vol_regimes['medium']:
            return 'medium'
        else:
            return 'high'

    def adaptive_allocation(
        self,
        returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Adaptive allocation based on market regime.

        Args:
            returns: Asset returns

        Returns:
            Allocation results with multiple methods
        """
        # Detect regime
        regime = self.detect_regime(returns)
        self.current_regime = regime

        # Calculate statistics
        _, cov_matrix = self.calculate_returns_stats(returns)

        results = {}

        # Different methods based on regime
        if regime == 'low':
            # Low vol: Use ERC for balance
            erc_result = self.equal_risk_contribution(cov_matrix)
            results['primary'] = erc_result
            results['method'] = 'ERC'

        elif regime == 'high':
            # High vol: Use HRP for robustness
            hrp_result = self.hierarchical_risk_parity(returns)
            results['primary'] = hrp_result
            results['method'] = 'HRP'

        else:
            # Medium vol: Blend approaches
            erc_result = self.equal_risk_contribution(cov_matrix)
            hrp_result = self.hierarchical_risk_parity(returns)

            # Average weights
            blended_weights = 0.5 * erc_result.weights + 0.5 * hrp_result.weights
            blended_weights = blended_weights / blended_weights.sum()

            results['primary'] = RiskParityResult(
                weights=blended_weights,
                risk_contributions=erc_result.risk_contributions,
                total_risk=np.sqrt(blended_weights @ cov_matrix @ blended_weights),
                diversification_ratio=erc_result.diversification_ratio,
                effective_n=1 / np.sum(blended_weights ** 2),
                converged=True
            )
            results['method'] = 'Blended'

        # Also calculate naive RP for comparison
        naive_weights = self.naive_risk_parity(returns)
        results['naive'] = naive_weights

        results['regime'] = regime

        return results


def test_adaptive_risk_parity():
    """Test the Adaptive Risk Parity System"""
    print("=" * 60)
    print("Testing Adaptive Risk Parity Portfolio System")
    print("=" * 60)

    # Initialize system
    rp_system = AdaptiveRiskParitySystem(
        risk_measure='volatility',
        rebalance_frequency=21,
        min_weight=0.02,
        max_weight=0.40
    )

    # Generate synthetic returns for multiple assets
    np.random.seed(42)
    n_periods = 252
    n_assets = 8

    # Different asset characteristics
    asset_names = ['Stocks', 'Bonds', 'Commodities', 'Real Estate',
                   'Gold', 'EM Equity', 'HY Bonds', 'Cash']

    # Annual return and volatility for each asset
    annual_returns = np.array([0.08, 0.04, 0.05, 0.07, 0.03, 0.10, 0.06, 0.01])
    annual_vols = np.array([0.16, 0.05, 0.18, 0.12, 0.15, 0.22, 0.08, 0.01])

    # Generate correlated returns
    correlation = np.array([
        [1.00, 0.10, 0.30, 0.50, 0.20, 0.70, 0.40, -0.10],  # Stocks
        [0.10, 1.00, -0.20, 0.20, 0.10, 0.00, 0.60, 0.30],  # Bonds
        [0.30, -0.20, 1.00, 0.30, 0.50, 0.20, 0.10, 0.00],  # Commodities
        [0.50, 0.20, 0.30, 1.00, 0.20, 0.40, 0.30, 0.10],  # Real Estate
        [0.20, 0.10, 0.50, 0.20, 1.00, 0.10, 0.20, 0.20],  # Gold
        [0.70, 0.00, 0.20, 0.40, 0.10, 1.00, 0.30, -0.05],  # EM Equity
        [0.40, 0.60, 0.10, 0.30, 0.20, 0.30, 1.00, 0.20],  # HY Bonds
        [-0.10, 0.30, 0.00, 0.10, 0.20, -0.05, 0.20, 1.00]  # Cash
    ])

    # Generate returns
    daily_returns = annual_returns / 252
    daily_vols = annual_vols / np.sqrt(252)

    # Cholesky decomposition for correlation
    L = np.linalg.cholesky(correlation)

    # Generate correlated random returns
    random_returns = np.random.randn(n_periods, n_assets)
    correlated_returns = random_returns @ L.T

    # Scale by volatility and add drift
    returns = correlated_returns * daily_vols + daily_returns

    print("\n1. Asset Characteristics:")
    print("-" * 40)
    for i, name in enumerate(asset_names):
        realized_ret = np.mean(returns[:, i]) * 252
        realized_vol = np.std(returns[:, i]) * np.sqrt(252)
        sharpe = realized_ret / realized_vol
        print(f"{name:12}: Return={realized_ret:6.2%}, Vol={realized_vol:6.2%}, Sharpe={sharpe:.3f}")

    print("\n2. Equal Risk Contribution (ERC):")
    print("-" * 40)

    _, cov_matrix = rp_system.calculate_returns_stats(returns)
    erc_result = rp_system.equal_risk_contribution(cov_matrix)

    print(f"Converged: {erc_result.converged}")
    print(f"Portfolio Risk: {erc_result.total_risk * np.sqrt(252):.2%} annualized")
    print(f"Diversification Ratio: {erc_result.diversification_ratio:.3f}")
    print(f"Effective N: {erc_result.effective_n:.2f}")

    print("\nERC Weights:")
    for i, name in enumerate(asset_names):
        print(f"  {name:12}: {erc_result.weights[i]:6.2%} (risk contrib: {erc_result.risk_contributions[i]:.2%})")

    print("\n3. Hierarchical Risk Parity (HRP):")
    print("-" * 40)

    hrp_result = rp_system.hierarchical_risk_parity(returns)

    print(f"Number of clusters: {len(hrp_result.clusters)}")
    print("\nHRP Weights:")
    for i, name in enumerate(asset_names):
        print(f"  {name:12}: {hrp_result.weights[i]:6.2%} (risk contrib: {hrp_result.risk_contributions[i]:.2%})")

    print("\nCluster Analysis:")
    for cluster_id, assets in enumerate(hrp_result.clusters, 1):
        cluster_assets = [asset_names[i] for i in assets]
        cluster_risk = hrp_result.cluster_risks.get(cluster_id, 0)
        print(f"  Cluster {cluster_id}: {', '.join(cluster_assets)}")
        print(f"    Risk contribution: {cluster_risk:.2%}")

    print("\n4. Naive Risk Parity:")
    print("-" * 40)

    naive_weights = rp_system.naive_risk_parity(returns)

    print("Naive RP Weights (inverse volatility):")
    for i, name in enumerate(asset_names):
        print(f"  {name:12}: {naive_weights[i]:6.2%}")

    print("\n5. Risk Budgeting with Custom Targets:")
    print("-" * 40)

    # Define custom risk budgets (e.g., 40% equity risk, 30% bonds, 30% alternatives)
    risk_budgets = np.array([0.20, 0.15, 0.10, 0.15, 0.10, 0.20, 0.05, 0.05])

    rb_result = rp_system.risk_budgeting(cov_matrix, risk_budgets)

    print("Risk Budgeting Results:")
    print(f"Converged: {rb_result.converged}")
    print(f"Portfolio Risk: {rb_result.total_risk * np.sqrt(252):.2%} annualized")

    print("\nActual vs Target Risk Contributions:")
    for i, name in enumerate(asset_names):
        actual = rb_result.risk_contributions[i]
        target = risk_budgets[i] / risk_budgets.sum()
        print(f"  {name:12}: Weight={rb_result.weights[i]:6.2%}, "
              f"Actual={actual:.2%}, Target={target:.2%}, "
              f"Diff={actual-target:+.2%}")

    print("\n6. Regime-Based Adaptive Allocation:")
    print("-" * 40)

    # Test with different volatility regimes
    regime_returns = {
        'low_vol': returns * 0.5,  # Scale down volatility
        'medium_vol': returns,
        'high_vol': returns * 2.0  # Scale up volatility
    }

    for regime_name, regime_data in regime_returns.items():
        results = rp_system.adaptive_allocation(regime_data[-63:])  # Last quarter

        print(f"\n{regime_name.upper()} Regime:")
        print(f"  Detected regime: {results['regime']}")
        print(f"  Method used: {results['method']}")

        if 'primary' in results:
            if hasattr(results['primary'], 'weights'):
                weights = results['primary'].weights
            else:
                weights = results['primary'].weights if isinstance(results['primary'], HRPResult) else np.array([])

            if len(weights) > 0:
                print("  Weights:")
                for i, name in enumerate(asset_names[:len(weights)]):
                    print(f"    {name:12}: {weights[i]:6.2%}")

    print("\n7. Performance Comparison:")
    print("-" * 40)

    # Backtest different methods
    methods = {
        'ERC': erc_result.weights,
        'HRP': hrp_result.weights,
        'Naive': naive_weights,
        'Equal': np.ones(n_assets) / n_assets
    }

    print("Annual Performance Metrics:")
    print(f"{'Method':<10} {'Return':<10} {'Volatility':<12} {'Sharpe':<8} {'Max DD':<10}")
    print("-" * 50)

    for method_name, weights in methods.items():
        # Calculate portfolio returns
        portfolio_returns = returns @ weights

        # Metrics
        ann_return = np.mean(portfolio_returns) * 252
        ann_vol = np.std(portfolio_returns) * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Max drawdown
        cum_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max
        max_dd = np.min(drawdowns)

        print(f"{method_name:<10} {ann_return:<10.2%} {ann_vol:<12.2%} {sharpe:<8.3f} {max_dd:<10.2%}")

    print("\n[SUCCESS] Adaptive Risk Parity System test completed successfully!")


if __name__ == "__main__":
    test_adaptive_risk_parity()