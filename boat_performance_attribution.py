#!/usr/bin/env python3
"""
Portfolio Performance Attribution Analysis
===========================================

Decompose portfolio returns into sources and factors:
  - Brinson attribution analysis
  - Factor-based performance attribution
  - Sector contribution analysis
  - Security selection vs allocation
  - Risk factor decomposition
  - Performance persistence analysis

Based on 2025 research on performance attribution and factor analysis.
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
class AttributionResult:
    """Attribution analysis result"""
    total_return: float
    benchmark_return: float
    active_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    attribution_by_position: Dict[str, float]


class BrinsonAttribution:
    """Brinson-Fachler performance attribution"""

    @staticmethod
    def analyze_attribution(
        portfolio_weights: Dict[str, float],
        benchmark_weights: Dict[str, float],
        portfolio_returns: Dict[str, float],
        benchmark_returns: Dict[str, float]
    ) -> AttributionResult:
        """
        Analyze portfolio attribution

        Args:
            portfolio_weights: Portfolio weights
            benchmark_weights: Benchmark weights
            portfolio_returns: Holding period returns
            benchmark_returns: Benchmark security returns

        Returns:
            Attribution result
        """
        positions = list(portfolio_weights.keys())

        # Ensure all positions in both
        for pos in positions:
            if pos not in benchmark_weights:
                benchmark_weights[pos] = 0.0
            if pos not in portfolio_returns:
                portfolio_returns[pos] = 0.0
            if pos not in benchmark_returns:
                benchmark_returns[pos] = 0.0

        allocation_effect = {}
        selection_effect = {}
        interaction_effect = {}

        for pos in positions:
            pw = portfolio_weights.get(pos, 0.0)
            bw = benchmark_weights.get(pos, 0.0)
            pr = portfolio_returns.get(pos, 0.0)
            br = benchmark_returns.get(pos, 0.0)

            # Allocation effect: (portfolio weight - benchmark weight) * benchmark return
            allocation_effect[pos] = (pw - bw) * br

            # Selection effect: benchmark weight * (portfolio return - benchmark return)
            selection_effect[pos] = bw * (pr - br)

            # Interaction effect: (portfolio weight - benchmark weight) * (portfolio return - benchmark return)
            interaction_effect[pos] = (pw - bw) * (pr - br)

        # Total returns
        total_return = sum(w * r for w, r in zip(portfolio_weights.values(), portfolio_returns.values()))
        benchmark_return = sum(w * r for w, r in zip(benchmark_weights.values(), benchmark_returns.values()))
        active_return = total_return - benchmark_return

        # Effect totals
        total_allocation = sum(allocation_effect.values())
        total_selection = sum(selection_effect.values())
        total_interaction = sum(interaction_effect.values())

        attribution_by_position = {
            pos: allocation_effect.get(pos, 0.0) + selection_effect.get(pos, 0.0) + interaction_effect.get(pos, 0.0)
            for pos in positions
        }

        return AttributionResult(
            total_return=float(total_return),
            benchmark_return=float(benchmark_return),
            active_return=float(active_return),
            allocation_effect=float(total_allocation),
            selection_effect=float(total_selection),
            interaction_effect=float(total_interaction),
            attribution_by_position=attribution_by_position
        )


class FactorAttribution:
    """Factor-based performance attribution"""

    def __init__(self, factors: List[str]):
        self.factors = factors
        self.factor_exposures = {}
        self.factor_returns = {}
        self.factor_loadings = None

    def calculate_factor_attribution(
        self,
        portfolio_returns: np.ndarray,
        factor_returns: np.ndarray,
        securities: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate attribution to factors

        Args:
            portfolio_returns: Portfolio returns (T,)
            factor_returns: Factor returns (T, n_factors)
            securities: Security returns for exposure calculation (T, n_securities)

        Returns:
            Attribution by factor
        """
        T = len(portfolio_returns)

        # Calculate factor loadings (exposures) via regression
        X = np.column_stack([np.ones(T), factor_returns])
        beta = np.linalg.lstsq(X, portfolio_returns, rcond=None)[0]

        # Factor contributions
        contributions = {}

        contributions['alpha'] = float(beta[0])  # Intercept

        for i, factor_name in enumerate(self.factors):
            factor_contribution = beta[i + 1] * np.mean(factor_returns[:, i])
            contributions[factor_name] = float(factor_contribution)

        # Residual
        predicted = X @ beta
        residual = np.mean(portfolio_returns - predicted)
        contributions['residual'] = float(residual)

        self.factor_loadings = beta

        return contributions


class SectorAttribution:
    """Sector-level performance attribution"""

    @staticmethod
    def analyze_sector_contribution(
        sector_weights: Dict[str, float],
        sector_returns: Dict[str, float],
        security_sector_mapping: Dict[str, str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze contribution by sector

        Args:
            sector_weights: Sector portfolio weights
            sector_returns: Sector returns
            security_sector_mapping: Security to sector mapping

        Returns:
            Attribution by sector
        """
        sector_attribution = {}

        for sector, weight in sector_weights.items():
            sector_return = sector_returns.get(sector, 0.0)
            contribution = weight * sector_return

            sector_attribution[sector] = {
                'weight': weight,
                'return': float(sector_return),
                'contribution': float(contribution),
                'count_securities': sum(1 for s in security_sector_mapping.values() if s == sector)
            }

        return sector_attribution


class SelectionVsAllocation:
    """Separate selection skill from allocation skill"""

    @staticmethod
    def decompose_performance(
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        portfolio_weights: np.ndarray,
        benchmark_weights: np.ndarray,
        security_returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Decompose performance into selection and allocation

        Args:
            portfolio_returns: Portfolio returns (T,)
            benchmark_returns: Benchmark returns (T,)
            portfolio_weights: Portfolio weights (T, n_securities)
            benchmark_weights: Benchmark weights (T, n_securities)
            security_returns: Security returns (T, n_securities)

        Returns:
            Decomposition results
        """
        T = len(portfolio_returns)

        # Return from selection: use benchmark weights with portfolio returns
        selection_return = np.mean(np.sum(benchmark_weights * security_returns, axis=1))

        # Return from allocation: use portfolio weights with security excess returns
        excess_returns = security_returns - benchmark_weights

        allocation_return = np.mean(np.sum(portfolio_weights * excess_returns, axis=1))

        # Total active return
        total_active = np.mean(portfolio_returns) - np.mean(benchmark_returns)

        return {
            'total_active_return': float(total_active),
            'selection_return': float(selection_return),
            'allocation_return': float(allocation_return),
            'interaction_return': float(total_active - selection_return - allocation_return),
            'selection_pct': float(selection_return / (abs(total_active) + 1e-8)),
            'allocation_pct': float(allocation_return / (abs(total_active) + 1e-8))
        }


class RiskFactorDecomposition:
    """Decompose portfolio risk into factors"""

    @staticmethod
    def decompose_risk(
        factor_exposures: np.ndarray,
        factor_volatility: np.ndarray,
        factor_correlation_matrix: np.ndarray,
        idiosyncratic_risk: float
    ) -> Dict[str, float]:
        """
        Decompose portfolio risk

        Args:
            factor_exposures: Exposure to each factor
            factor_volatility: Factor volatilities
            factor_correlation_matrix: Factor correlations
            idiosyncratic_risk: Idiosyncratic risk

        Returns:
            Risk decomposition
        """
        n_factors = len(factor_exposures)

        # Systematic risk from each factor
        factor_risks = {}

        for i in range(n_factors):
            # Marginal contribution of factor i
            exposure = factor_exposures[i]
            vol = factor_volatility[i]
            marginal_risk = exposure * vol

            factor_risks[f'factor_{i}'] = float(marginal_risk ** 2)

        # Interaction risk from correlation
        interaction_risk = 0.0

        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                cov = (factor_exposures[i] * factor_volatility[i] *
                      factor_exposures[j] * factor_volatility[j] *
                      factor_correlation_matrix[i, j])
                interaction_risk += 2 * cov

        # Idiosyncratic risk
        idio_risk = idiosyncratic_risk ** 2

        # Total risk
        systematic_risk = sum(factor_risks.values()) + interaction_risk
        total_risk = systematic_risk + idio_risk

        return {
            'systematic_risk': float(systematic_risk),
            'idiosyncratic_risk': float(idio_risk),
            'total_risk': float(total_risk),
            'systematic_pct': float(systematic_risk / (total_risk + 1e-8)),
            'idiosyncratic_pct': float(idio_risk / (total_risk + 1e-8)),
            'factor_risks': factor_risks
        }


class PerformancePersistence:
    """Analyze performance persistence"""

    @staticmethod
    def analyze_persistence(
        returns_history: np.ndarray,
        window_size: int = 60
    ) -> Dict[str, float]:
        """
        Analyze performance persistence across periods

        Args:
            returns_history: Historical returns (T,)
            window_size: Window for period returns

        Returns:
            Persistence metrics
        """
        n_periods = len(returns_history) // window_size

        # Calculate returns by period
        period_returns = []

        for i in range(n_periods):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            period_return = np.mean(returns_history[start_idx:end_idx])
            period_returns.append(period_return)

        # Calculate autocorrelations
        persistence = {}

        for lag in [1, 2, 3, 4]:
            if lag < len(period_returns):
                corr = np.corrcoef(period_returns[:-lag], period_returns[lag:])[0, 1]
                persistence[f'autocorr_lag_{lag}'] = float(corr)

        # Win rate (% of periods with positive returns)
        win_rate = float(np.mean(np.array(period_returns) > 0))
        persistence['win_rate'] = win_rate

        # Consecutive wins
        consecutive_wins = 0
        max_consecutive_wins = 0

        for ret in period_returns:
            if ret > 0:
                consecutive_wins += 1
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_wins = 0

        persistence['max_consecutive_wins'] = float(max_consecutive_wins)
        persistence['avg_period_return'] = float(np.mean(period_returns))
        persistence['std_period_return'] = float(np.std(period_returns))

        return persistence


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Brinson attribution
    portfolio_weights = {'A': 0.40, 'B': 0.35, 'C': 0.25}
    benchmark_weights = {'A': 0.50, 'B': 0.30, 'C': 0.20}
    portfolio_returns = {'A': 0.12, 'B': 0.08, 'C': 0.10}
    benchmark_returns = {'A': 0.10, 'B': 0.07, 'C': 0.09}

    brinson = BrinsonAttribution.analyze_attribution(
        portfolio_weights, benchmark_weights,
        portfolio_returns, benchmark_returns
    )

    logger.info("Brinson Attribution:")
    logger.info(f"Total Return: {brinson.total_return:.4f}")
    logger.info(f"Benchmark Return: {brinson.benchmark_return:.4f}")
    logger.info(f"Active Return: {brinson.active_return:.4f}")
    logger.info(f"Allocation Effect: {brinson.allocation_effect:.6f}")
    logger.info(f"Selection Effect: {brinson.selection_effect:.6f}")

    # Factor attribution
    T = 252
    portfolio_ret = np.random.randn(T) * 0.01 + 0.0005
    factor_ret = np.random.randn(T, 3) * 0.005

    factor_attr = FactorAttribution(['Value', 'Momentum', 'Quality'])
    contributions = factor_attr.calculate_factor_attribution(portfolio_ret, factor_ret)

    logger.info("\nFactor Attribution:")
    for factor, contrib in contributions.items():
        logger.info(f"  {factor}: {contrib:.6f}")

    # Performance persistence
    returns_hist = np.random.randn(500) * 0.01
    persistence = PerformancePersistence.analyze_persistence(returns_hist, window_size=60)

    logger.info("\nPerformance Persistence:")
    for metric, value in persistence.items():
        logger.info(f"  {metric}: {value:.4f}")
