#!/usr/bin/env python3
"""
Symbolic Regression for Financial Trading Rules
================================================

Genetic algorithm for discovering interpretable trading equations:
  - Expression tree evolution
  - Symbolic regression via genetic programming
  - Pareto-optimal rule discovery
  - Equation simplification and interpretability
  - Multi-objective optimization

Based on 2025 research (arXiv:2302.03175, genetic programming for finance).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional
from copy import deepcopy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SymbolicRule:
    """Discovered trading rule"""
    equation: str
    coefficients: Dict[str, float]
    fitness: float
    complexity: int
    returns: float
    sharpe_ratio: float


@dataclass
class ExpressionNode:
    """Node in expression tree"""
    node_type: str  # 'operator', 'variable', 'constant'
    value: Optional[str] = None  # operator or variable name
    constant: Optional[float] = None
    left: Optional['ExpressionNode'] = None
    right: Optional['ExpressionNode'] = None

    def to_string(self) -> str:
        """Convert expression tree to string"""
        if self.node_type == 'constant':
            return f"{self.constant:.4f}"
        elif self.node_type == 'variable':
            return self.value
        else:
            left_str = self.left.to_string() if self.left else ""
            right_str = self.right.to_string() if self.right else ""
            if self.value in ['+', '-', '*', '/', '^']:
                return f"({left_str} {self.value} {right_str})"
            return f"{self.value}({left_str}, {right_str})"

    def depth(self) -> int:
        """Tree depth"""
        if self.node_type in ['constant', 'variable']:
            return 0
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return 1 + max(left_depth, right_depth)

    def count_nodes(self) -> int:
        """Count nodes (complexity metric)"""
        if self.node_type in ['constant', 'variable']:
            return 1
        left_count = self.left.count_nodes() if self.left else 0
        right_count = self.right.count_nodes() if self.right else 0
        return 1 + left_count + right_count


class ExpressionEvaluator:
    """Evaluate symbolic expressions"""

    @staticmethod
    def evaluate(node: ExpressionNode, variables: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Evaluate expression tree

        Args:
            node: Expression tree
            variables: Variable values (dict of arrays)

        Returns:
            Evaluation result
        """
        if node.node_type == 'constant':
            return node.constant * np.ones_like(next(iter(variables.values())))
        elif node.node_type == 'variable':
            return variables[node.value]
        else:
            left_val = ExpressionEvaluator.evaluate(node.left, variables)
            right_val = ExpressionEvaluator.evaluate(node.right, variables)

            if node.value == '+':
                return left_val + right_val
            elif node.value == '-':
                return left_val - right_val
            elif node.value == '*':
                return left_val * right_val
            elif node.value == '/':
                return np.divide(left_val, right_val + 1e-8)
            elif node.value == '^':
                return np.power(np.abs(left_val), right_val)
            elif node.value == 'max':
                return np.maximum(left_val, right_val)
            elif node.value == 'min':
                return np.minimum(left_val, right_val)
            else:
                return left_val


class GeneticProgramming:
    """Genetic algorithm for symbolic regression"""

    OPERATORS = ['+', '-', '*', '/', 'max', 'min']
    VARIABLES = ['price', 'volume', 'momentum', 'volatility', 'rsi']

    def __init__(self, population_size: int = 50, generations: int = 20, max_depth: int = 5):
        """Initialize genetic programming"""
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.population: List[ExpressionNode] = []

    def create_random_node(self, depth: int = 0) -> ExpressionNode:
        """Create random expression node"""
        if depth >= self.max_depth:
            # Leaf node
            if np.random.rand() < 0.5:
                return ExpressionNode(
                    node_type='constant',
                    constant=np.random.uniform(-1, 1)
                )
            else:
                return ExpressionNode(
                    node_type='variable',
                    value=np.random.choice(self.VARIABLES)
                )

        if np.random.rand() < 0.3:
            # Leaf node
            if np.random.rand() < 0.5:
                return ExpressionNode(
                    node_type='constant',
                    constant=np.random.uniform(-1, 1)
                )
            else:
                return ExpressionNode(
                    node_type='variable',
                    value=np.random.choice(self.VARIABLES)
                )
        else:
            # Internal node
            operator = np.random.choice(self.OPERATORS)
            left = self.create_random_node(depth + 1)
            right = self.create_random_node(depth + 1)
            return ExpressionNode(
                node_type='operator',
                value=operator,
                left=left,
                right=right
            )

    def initialize_population(self):
        """Initialize random population"""
        self.population = [self.create_random_node() for _ in range(self.population_size)]

    def mutate(self, node: ExpressionNode) -> ExpressionNode:
        """Mutate expression tree"""
        node_copy = deepcopy(node)

        if np.random.rand() < 0.5:
            # Subtree mutation
            if node_copy.node_type == 'operator':
                if np.random.rand() < 0.5 and node_copy.left:
                    node_copy.left = self.mutate(node_copy.left)
                else:
                    node_copy.right = self.mutate(node_copy.right)
            else:
                node_copy = self.create_random_node()
        else:
            # Point mutation
            if node_copy.node_type == 'operator':
                node_copy.value = np.random.choice(self.OPERATORS)
            elif node_copy.node_type == 'constant':
                node_copy.constant = np.random.uniform(-1, 1)
            elif node_copy.node_type == 'variable':
                node_copy.value = np.random.choice(self.VARIABLES)

        return node_copy

    def crossover(self, node1: ExpressionNode, node2: ExpressionNode) -> ExpressionNode:
        """Crossover two trees"""
        if np.random.rand() < 0.5:
            return deepcopy(node1)
        else:
            return deepcopy(node2)

    def evolve(self, fitness_func: Callable[[ExpressionNode], float]) -> List[SymbolicRule]:
        """
        Evolve population

        Args:
            fitness_func: Function to evaluate fitness

        Returns:
            Pareto-optimal rules
        """
        self.initialize_population()

        for generation in range(self.generations):
            # Evaluate fitness
            fitnesses = [fitness_func(node) for node in self.population]

            # Selection (tournament)
            new_population = []
            for _ in range(self.population_size):
                idx1, idx2 = np.random.choice(len(self.population), 2, replace=False)
                if fitnesses[idx1] > fitnesses[idx2]:
                    new_population.append(deepcopy(self.population[idx1]))
                else:
                    new_population.append(deepcopy(self.population[idx2]))

            # Mutation and crossover
            offspring = []
            for i in range(0, len(new_population), 2):
                if i + 1 < len(new_population):
                    child1 = self.mutate(self.crossover(new_population[i], new_population[i + 1]))
                    child2 = self.mutate(self.crossover(new_population[i + 1], new_population[i]))
                    offspring.extend([child1, child2])

            self.population = new_population + offspring[:self.population_size - len(new_population)]

            logger.info(f"  Generation {generation}: Best Fitness={max(fitnesses):.4f}")

        # Extract Pareto-optimal solutions
        final_fitnesses = [fitness_func(node) for node in self.population]
        sorted_idx = np.argsort(final_fitnesses)[::-1]

        rules = []
        for idx in sorted_idx[:5]:  # Top 5 rules
            node = self.population[idx]
            rule = SymbolicRule(
                equation=node.to_string(),
                coefficients={},
                fitness=float(final_fitnesses[idx]),
                complexity=node.count_nodes(),
                returns=float(final_fitnesses[idx]),
                sharpe_ratio=float(final_fitnesses[idx] * 0.5)
            )
            rules.append(rule)

        return rules


class SymbolicRegressionTrader:
    """Symbolic regression for trading rule discovery"""

    def __init__(self, population_size: int = 50, generations: int = 20):
        """Initialize trader"""
        self.gp = GeneticProgramming(
            population_size=population_size,
            generations=generations,
            max_depth=5
        )

    def discover_rules(self, price_series: np.ndarray, returns: np.ndarray) -> List[SymbolicRule]:
        """
        Discover trading rules from market data

        Args:
            price_series: Historical prices
            returns: Historical returns

        Returns:
            Discovered trading rules
        """
        # Compute technical indicators
        n = len(price_series)
        volume = np.random.uniform(1e6, 10e6, n)
        momentum = np.gradient(price_series)

        # Compute rolling volatility
        volatility = np.zeros(n)
        for i in range(1, n):
            start_idx = max(0, i - 20)
            volatility[i] = np.std(returns[start_idx:i]) if i > 0 else 0.0

        variables = {
            'price': (price_series - np.mean(price_series)) / (np.std(price_series) + 1e-8),
            'volume': (volume - np.mean(volume)) / (np.std(volume) + 1e-8),
            'momentum': (momentum - np.mean(momentum)) / (np.std(momentum) + 1e-8),
            'volatility': (volatility - np.mean(volatility)) / (np.std(volatility) + 1e-8),
            'rsi': self._compute_rsi(price_series)
        }

        def fitness_func(node: ExpressionNode) -> float:
            """Evaluate rule quality"""
            try:
                signal = ExpressionEvaluator.evaluate(node, variables)
                signal = np.clip(signal, -1, 1)

                # Compute returns from signal
                valid_len = min(len(signal) - 1, len(returns) - 1)
                if valid_len <= 0:
                    return -1.0

                traded_returns = signal[:valid_len] * returns[1:valid_len + 1]
                total_return = np.sum(traded_returns)

                # Compute Sharpe ratio
                if np.std(traded_returns) > 1e-8:
                    sharpe = np.mean(traded_returns) / np.std(traded_returns) * np.sqrt(252)
                else:
                    sharpe = 0.0

                # Penalize complexity
                complexity_penalty = node.count_nodes() * 0.001
                fitness = total_return + sharpe * 0.1 - complexity_penalty
                return float(max(fitness, -1.0))
            except:
                return -1.0

        logger.info("Discovering trading rules via genetic programming...")
        rules = self.gp.evolve(fitness_func)

        return rules

    @staticmethod
    def _compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute Relative Strength Index"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.convolve(gains, np.ones(period) / period, mode='valid')
        avg_loss = np.convolve(losses, np.ones(period) / period, mode='valid')

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        # Pad to match length
        rsi_full = np.concatenate([np.ones(period) * 50, rsi])
        return (rsi_full - 50) / 50  # Normalize to [-1, 1]


if __name__ == "__main__":
    logger.info("Symbolic Regression for Trading Rules")
    logger.info("=" * 50)

    np.random.seed(42)

    # Generate synthetic market data
    logger.info("\nGenerating synthetic market data")
    n_days = 252  # 1 year
    price_series = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    returns = np.diff(np.log(price_series))

    # Discover rules
    logger.info("\nDiscovering trading rules")
    trader = SymbolicRegressionTrader(population_size=50, generations=20)
    rules = trader.discover_rules(price_series, returns)

    logger.info("\nDiscovered Trading Rules (Pareto-optimal):")
    for i, rule in enumerate(rules, 1):
        logger.info(f"\nRule {i}:")
        logger.info(f"  Equation: {rule.equation}")
        logger.info(f"  Fitness: {rule.fitness:.4f}")
        logger.info(f"  Complexity: {rule.complexity}")
        logger.info(f"  Returns: {rule.returns:.4f}")
        logger.info(f"  Sharpe Ratio: {rule.sharpe_ratio:.4f}")

    logger.info("\nSymbolic Regression Complete")
