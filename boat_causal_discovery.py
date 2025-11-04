#!/usr/bin/env python3
"""
Causal Discovery and Market Structure Analysis
===============================================

Algorithms for discovering causal relationships in financial markets:
  - Granger causality with significance testing
  - PC algorithm for causal graphs
  - Convergent Cross-Mapping (CCM)
  - Dynamic causal graphs
  - Causal strength quantification
  - Market leader/follower identification

Based on 2025 research on causal inference in financial markets.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from datetime import datetime
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CausalEdge:
    """Causal edge between two variables"""
    source: str
    target: str
    strength: float
    pvalue: float
    lag: int = 0


@dataclass
class CausalGraph:
    """Causal graph representation"""
    nodes: Set[str]
    edges: List[CausalEdge]
    adjacency_matrix: np.ndarray


class GrangerCausalityTest:
    """Granger causality testing framework"""

    def __init__(self, max_lag: int = 5, significance_level: float = 0.05):
        """
        Initialize Granger causality tester

        Args:
            max_lag: Maximum lag to test
            significance_level: Significance level for F-test
        """
        self.max_lag = max_lag
        self.significance_level = significance_level

    def test_causality(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        lag: int = 1
    ) -> Tuple[float, float]:
        """
        Test if X Granger-causes Y

        Args:
            X: Source time series (N,)
            Y: Target time series (N,)
            lag: Lag to test

        Returns:
            (F_statistic, p_value)
        """
        N = len(Y)

        # Restricted model: Y regressed on lagged Y only
        Y_lagged = np.column_stack([Y[i:N - lag + i] for i in range(lag)])
        Y_restricted = Y[lag:N]

        X_restricted = np.column_stack([np.ones(len(Y_restricted)), Y_lagged])
        beta_r = np.linalg.lstsq(X_restricted, Y_restricted, rcond=None)[0]
        residuals_r = Y_restricted - np.dot(X_restricted, beta_r)
        ssr_r = np.sum(residuals_r ** 2)

        # Unrestricted model: Y regressed on lagged Y and lagged X
        X_lagged = np.column_stack([X[i:N - lag + i] for i in range(lag)])
        X_unrestricted = np.column_stack([np.ones(len(Y_restricted)), Y_lagged, X_lagged])
        beta_u = np.linalg.lstsq(X_unrestricted, Y_restricted, rcond=None)[0]
        residuals_u = Y_restricted - np.dot(X_unrestricted, beta_u)
        ssr_u = np.sum(residuals_u ** 2)

        # F-statistic
        dof_restricted = X_restricted.shape[1]
        dof_unrestricted = X_unrestricted.shape[1]

        f_stat = ((ssr_r - ssr_u) / (dof_unrestricted - dof_restricted)) / (ssr_u / (len(Y_restricted) - dof_unrestricted))

        # P-value
        pvalue = 1 - stats.f.cdf(f_stat, dof_unrestricted - dof_restricted, len(Y_restricted) - dof_unrestricted)

        return float(f_stat), float(pvalue)

    def find_optimal_lag(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> int:
        """
        Find optimal lag using AIC criterion

        Args:
            X: Source time series
            Y: Target time series

        Returns:
            Optimal lag
        """
        aic_scores = []

        for lag in range(1, self.max_lag + 1):
            _, pvalue = self.test_causality(X, Y, lag)
            # Simple AIC: penalize complexity
            aic = -2 * np.log(pvalue + 1e-8) + 2 * lag
            aic_scores.append(aic)

        optimal_lag = np.argmin(aic_scores) + 1
        return int(optimal_lag)

    def causality_network(
        self,
        data: pd.DataFrame
    ) -> CausalGraph:
        """
        Build causal network from multivariate time series

        Args:
            data: DataFrame with multiple time series columns

        Returns:
            CausalGraph
        """
        columns = data.columns.tolist()
        n_vars = len(columns)
        edges = []

        # Test causality between all pairs
        for i, source in enumerate(columns):
            for j, target in enumerate(columns):
                if i == j:
                    continue

                f_stat, pvalue = self.test_causality(
                    data[source].values,
                    data[target].values,
                    lag=1
                )

                if pvalue < self.significance_level:
                    strength = 1 - pvalue  # Strength inversely related to p-value
                    edge = CausalEdge(
                        source=source,
                        target=target,
                        strength=strength,
                        pvalue=pvalue,
                        lag=1
                    )
                    edges.append(edge)

        # Build adjacency matrix
        adjacency = np.zeros((n_vars, n_vars))
        for edge in edges:
            i = columns.index(edge.source)
            j = columns.index(edge.target)
            adjacency[i, j] = edge.strength

        return CausalGraph(
            nodes=set(columns),
            edges=edges,
            adjacency_matrix=adjacency
        )


class ConvergentCrossMapping:
    """Convergent Cross-Mapping for causality in time series"""

    def __init__(self, max_embedding_dim: int = 10):
        """
        Initialize CCM

        Args:
            max_embedding_dim: Maximum embedding dimension
        """
        self.max_embedding_dim = max_embedding_dim

    def compute_embedding(
        self,
        time_series: np.ndarray,
        embedding_dim: int,
        delay: int = 1
    ) -> np.ndarray:
        """
        Compute time-delay embedding

        Args:
            time_series: Input time series
            embedding_dim: Embedding dimension
            delay: Time delay

        Returns:
            Embedded time series (M, embedding_dim)
        """
        N = len(time_series)
        M = N - (embedding_dim - 1) * delay

        embedded = np.zeros((M, embedding_dim))
        for d in range(embedding_dim):
            embedded[:, d] = time_series[d * delay:d * delay + M]

        return embedded

    def compute_cross_map_skill(
        self,
        X_embedded: np.ndarray,
        Y: np.ndarray,
        library_size: int = None
    ) -> float:
        """
        Compute cross-mapping skill (prediction accuracy)

        Args:
            X_embedded: Embedded time series
            Y: Target time series
            library_size: Size of library set

        Returns:
            Cross-mapping skill (correlation)
        """
        if library_size is None:
            library_size = len(X_embedded)

        # Random selection of library
        indices = np.random.choice(len(X_embedded), size=library_size, replace=False)
        X_lib = X_embedded[indices]
        Y_lib = Y[indices]

        # Simplex projection: find nearest neighbors
        Y_pred = []

        for i in range(len(X_embedded)):
            x_point = X_embedded[i]

            # Compute distances to library
            distances = np.linalg.norm(X_lib - x_point, axis=1)
            nearest_indices = np.argsort(distances)[:3]  # 3-nearest neighbors

            # Weighted average prediction
            weights = 1.0 / (distances[nearest_indices] + 1e-8)
            weights /= np.sum(weights)
            y_pred = np.sum(weights * Y_lib[nearest_indices])

            Y_pred.append(y_pred)

        Y_pred = np.array(Y_pred)

        # Compute correlation skill
        correlation = np.corrcoef(Y, Y_pred)[0, 1]

        return float(correlation)

    def test_causality_ccm(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        embedding_dim: int = 3
    ) -> Tuple[float, float]:
        """
        Test causality using CCM

        Returns:
            (X->Y skill, Y->X skill)
        """
        X_embedded = self.compute_embedding(X, embedding_dim)
        Y_embedded = self.compute_embedding(Y, embedding_dim)

        # X causes Y if X embedded predicts Y better
        xy_skill = self.compute_cross_map_skill(X_embedded, Y[embedding_dim - 1:])

        # Y causes X if Y embedded predicts X better
        yx_skill = self.compute_cross_map_skill(Y_embedded, X[embedding_dim - 1:])

        return xy_skill, yx_skill


class PCAlgorithmCausality:
    """PC (Peter-Clark) Algorithm for causal discovery"""

    def __init__(self, alpha: float = 0.05):
        """
        Initialize PC algorithm

        Args:
            alpha: Significance level for conditional independence
        """
        self.alpha = alpha

    def conditional_independence_test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray = None
    ) -> Tuple[float, bool]:
        """
        Test conditional independence X ⊥ Y | Z

        Args:
            X: First variable
            Y: Second variable
            Z: Conditioning set (can be None)

        Returns:
            (p_value, is_independent)
        """
        if Z is None or len(Z) == 0:
            # Simple Pearson correlation test
            correlation = np.corrcoef(X, Y)[0, 1]
            n = len(X)

            # Fisher z-transformation
            z = 0.5 * np.log((1 + correlation) / (1 - correlation + 1e-8))
            test_stat = z * np.sqrt(n - 3)
            pvalue = 2 * (1 - stats.norm.cdf(np.abs(test_stat)))

        else:
            # Partial correlation test
            # Residuals of X after regressing on Z
            if Z.ndim == 1:
                Z = Z.reshape(-1, 1)

            X_resid = X - np.dot(np.linalg.lstsq(Z, X.reshape(-1, 1), rcond=None)[0].T, Z.T).flatten()
            Y_resid = Y - np.dot(np.linalg.lstsq(Z, Y.reshape(-1, 1), rcond=None)[0].T, Z.T).flatten()

            partial_corr = np.corrcoef(X_resid, Y_resid)[0, 1]
            n = len(X)
            z = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr + 1e-8))
            test_stat = z * np.sqrt(n - len(Z) - 3)
            pvalue = 2 * (1 - stats.norm.cdf(np.abs(test_stat)))

        is_independent = pvalue > self.alpha

        return float(pvalue), bool(is_independent)

    def learn_causal_graph(
        self,
        data: pd.DataFrame
    ) -> CausalGraph:
        """
        Learn causal graph using PC algorithm

        Args:
            data: DataFrame with variables

        Returns:
            CausalGraph
        """
        columns = data.columns.tolist()
        n_vars = len(columns)

        # Initialize fully connected undirected graph
        adj = np.ones((n_vars, n_vars))
        np.fill_diagonal(adj, 0)

        # Skeleton phase: remove edges where conditioning set exists
        separation_sets = {}

        for depth in range(n_vars - 1):
            to_remove = []

            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if adj[i, j] == 0:
                        continue

                    # Find neighbors
                    neighbors_i = set(np.where(adj[i, :] > 0)[0]) - {j}
                    neighbors_j = set(np.where(adj[j, :] > 0)[0]) - {i}
                    neighbors = neighbors_i | neighbors_j

                    if len(neighbors) < depth + 1:
                        continue

                    # Test conditional independence
                    for subset in self._powerset(neighbors, depth + 1):
                        X = data.iloc[:, i].values
                        Y = data.iloc[:, j].values
                        Z = data.iloc[:, list(subset)].values if len(subset) > 0 else None

                        pvalue, is_indep = self.conditional_independence_test(X, Y, Z)

                        if is_indep:
                            to_remove.append((i, j))
                            separation_sets[(i, j)] = subset
                            break

            # Remove edges
            for i, j in to_remove:
                adj[i, j] = 0
                adj[j, i] = 0

        return CausalGraph(
            nodes=set(columns),
            edges=[],
            adjacency_matrix=adj
        )

    def _powerset(self, s: Set, max_size: int) -> List[Tuple]:
        """Generate subsets up to max_size"""
        from itertools import combinations

        result = []
        for size in range(min(max_size + 1, len(s) + 1)):
            result.extend(combinations(s, size))

        return result


class DynamicCausalNetwork:
    """Dynamic causal network tracking over time"""

    def __init__(self, window_size: int = 100):
        """
        Initialize dynamic causal network

        Args:
            window_size: Rolling window size
        """
        self.window_size = window_size
        self.tester = GrangerCausalityTest()

    def compute_rolling_causality(
        self,
        data: pd.DataFrame,
        step: int = 10
    ) -> Dict[Tuple[str, str], List[float]]:
        """
        Compute causality strength over rolling windows

        Args:
            data: DataFrame with time series
            step: Rolling window step

        Returns:
            Dictionary of (source, target) -> [strengths over time]
        """
        columns = data.columns.tolist()
        causality_strengths = {}

        # Initialize
        for i, source in enumerate(columns):
            for j, target in enumerate(columns):
                if i != j:
                    causality_strengths[(source, target)] = []

        # Rolling windows
        for start in range(0, len(data) - self.window_size, step):
            end = start + self.window_size
            window_data = data.iloc[start:end]

            # Test causality in window
            for i, source in enumerate(columns):
                for j, target in enumerate(columns):
                    if i == j:
                        continue

                    f_stat, pvalue = self.tester.test_causality(
                        window_data[source].values,
                        window_data[target].values,
                        lag=1
                    )

                    strength = 1 - pvalue  # Strength inversely related to p-value
                    causality_strengths[(source, target)].append(strength)

        return causality_strengths

    def identify_leaders(
        self,
        causality_strengths: Dict[Tuple[str, str], List[float]]
    ) -> Dict[str, float]:
        """
        Identify market leaders based on causality

        Args:
            causality_strengths: Rolling causality strengths

        Returns:
            Dictionary of variable -> leadership score
        """
        variables = set()
        for source, target in causality_strengths.keys():
            variables.add(source)
            variables.add(target)

        leadership_scores = {}

        for var in variables:
            # Outgoing causality: how much var causes others
            outgoing = []
            for (source, target), strengths in causality_strengths.items():
                if source == var:
                    outgoing.extend(strengths)

            # Incoming causality: how much others cause var
            incoming = []
            for (source, target), strengths in causality_strengths.items():
                if target == var:
                    incoming.extend(strengths)

            # Leadership: outgoing - incoming
            out_score = np.mean(outgoing) if outgoing else 0.0
            in_score = np.mean(incoming) if incoming else 0.0

            leadership_scores[var] = float(out_score - in_score)

        return leadership_scores


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Create synthetic multivariate time series
    n_samples = 500
    t = np.arange(n_samples)

    # X influences Y, Y influences Z
    X = np.sin(t / 50) + np.random.randn(n_samples) * 0.1
    Y = np.roll(X, 5) + np.cos(t / 50) + np.random.randn(n_samples) * 0.1
    Z = np.roll(Y, 3) + np.sin(t / 30) + np.random.randn(n_samples) * 0.1

    data = pd.DataFrame({
        'X': X,
        'Y': Y,
        'Z': Z
    })

    # Granger causality
    granger = GrangerCausalityTest()
    causal_graph = granger.causality_network(data)

    logger.info("Granger Causality Network:")
    for edge in causal_graph.edges:
        logger.info(f"  {edge.source} -> {edge.target}: strength={edge.strength:.4f}, p={edge.pvalue:.4e}")

    # CCM
    ccm = ConvergentCrossMapping()
    xy_skill, yx_skill = ccm.test_causality_ccm(X, Y, embedding_dim=3)
    logger.info(f"\nCCM X->Y skill: {xy_skill:.4f}")
    logger.info(f"CCM Y->X skill: {yx_skill:.4f}")

    # PC Algorithm
    pc = PCAlgorithmCausality()
    pc_graph = pc.learn_causal_graph(data)
    logger.info(f"\nPC Algorithm discovered edges: {np.sum(pc_graph.adjacency_matrix > 0)}")

    # Dynamic causality
    dynamic = DynamicCausalNetwork(window_size=100)
    rolling_causality = dynamic.compute_rolling_causality(data, step=20)
    leaders = dynamic.identify_leaders(rolling_causality)

    logger.info("\nMarket Leaders:")
    for var, score in sorted(leaders.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {var}: {score:.4f}")
