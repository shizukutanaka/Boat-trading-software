#!/usr/bin/env python3
"""
Reinforcement Learning Policy Gradient Methods for Trading
===========================================================

Actor-Critic and Policy Gradient algorithms for financial trading:
  - Policy Gradient (Vanilla PG) with baseline
  - Advantage Actor-Critic (A2C)
  - Proximal Policy Optimization (PPO)
  - Risk-aware reward functions
  - Experience buffer management
  - Value function estimation

Based on 2025 research on RL in algorithmic trading and portfolio optimization.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience for RL training"""
    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float = 0.0
    value: float = 0.0


@dataclass
class PolicyMetrics:
    """Policy performance metrics"""
    returns: List[float]
    episode_length: int
    total_reward: float
    sharpe_ratio: float
    max_drawdown: float


class PolicyNetwork:
    """Neural network policy for continuous action space"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, learning_rate: float = 0.001):
        """
        Initialize policy network

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for updates
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Simple 2-layer network weights
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, action_dim) * 0.01
        self.b2 = np.zeros(action_dim)

        # Value network for baseline
        self.V1 = np.random.randn(state_dim, hidden_dim) * 0.01
        self.v1 = np.zeros(hidden_dim)
        self.V2 = np.random.randn(hidden_dim, 1) * 0.01
        self.v2 = np.zeros(1)

    def forward(self, state: np.ndarray) -> Tuple[float, float]:
        """
        Forward pass through policy and value networks

        Args:
            state: Current state

        Returns:
            (action_mean, value)
        """
        # Policy network: state -> hidden -> action_mean
        hidden = np.tanh(np.dot(state, self.W1) + self.b1)
        action_mean = np.tanh(np.dot(hidden, self.W2) + self.b2)[0]

        # Value network: state -> hidden -> value
        v_hidden = np.tanh(np.dot(state, self.V1) + self.v1)
        value = np.dot(v_hidden, self.V2)[0] + self.v2[0]

        return action_mean, value

    def sample_action(self, state: np.ndarray, std: float = 0.1) -> Tuple[float, float]:
        """
        Sample action from policy with noise

        Args:
            state: Current state
            std: Standard deviation of action noise

        Returns:
            (action, log_prob)
        """
        action_mean, _ = self.forward(state)
        action = action_mean + np.random.randn() * std
        action = np.clip(action, -1.0, 1.0)

        # Log probability of action
        log_prob = -0.5 * ((action - action_mean) / std) ** 2 - 0.5 * np.log(2 * np.pi * std ** 2)

        return action, log_prob


class VanillaPolicyGradient:
    """Vanilla Policy Gradient (REINFORCE with baseline)"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        learning_rate: float = 0.001,
        discount_gamma: float = 0.99
    ):
        """
        Initialize Vanilla PG

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            learning_rate: Learning rate
            discount_gamma: Discount factor
        """
        self.policy = PolicyNetwork(state_dim, action_dim, learning_rate=learning_rate)
        self.discount_gamma = discount_gamma
        self.experience_buffer: List[Experience] = []

    def collect_experience(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Collect single experience

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        """
        action_mean, value = self.policy.forward(state)
        _, log_prob = self.policy.sample_action(state)

        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value
        )
        self.experience_buffer.append(exp)

    def compute_returns(self) -> np.ndarray:
        """
        Compute discounted returns

        Returns:
            Array of discounted returns
        """
        returns = []
        G = 0.0

        # Reverse iterate to compute returns
        for exp in reversed(self.experience_buffer):
            if exp.done:
                G = 0.0
            G = exp.reward + self.discount_gamma * G
            returns.insert(0, G)

        # Normalize returns
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        return returns

    def update_policy(self, learning_rate: float = 0.001) -> float:
        """
        Update policy using collected experiences

        Args:
            learning_rate: Learning rate for gradient update

        Returns:
            Average policy loss
        """
        if not self.experience_buffer:
            return 0.0

        returns = self.compute_returns()

        # Compute baseline (mean value)
        values = np.array([exp.value for exp in self.experience_buffer])
        baseline = np.mean(values)

        # Compute advantages
        advantages = returns - baseline

        # Policy gradient update
        policy_loss = 0.0
        for i, (exp, advantage) in enumerate(zip(self.experience_buffer, advantages)):
            # Policy gradient: -log(π) * advantage
            loss = -exp.log_prob * advantage
            policy_loss += loss

        policy_loss /= len(self.experience_buffer)

        # Value function update (baseline)
        value_loss = 0.0
        for i, (exp, ret) in enumerate(zip(self.experience_buffer, returns)):
            value_loss += (exp.value - ret) ** 2

        value_loss /= len(self.experience_buffer)

        # Clear buffer for next episode
        self.experience_buffer = []

        return float(policy_loss + value_loss)


class ActorCriticAgent:
    """Advantage Actor-Critic (A2C) for trading"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        learning_rate: float = 0.001,
        discount_gamma: float = 0.99,
        entropy_coeff: float = 0.01
    ):
        """
        Initialize A2C agent

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            learning_rate: Learning rate
            discount_gamma: Discount factor
            entropy_coeff: Entropy regularization coefficient
        """
        self.policy = PolicyNetwork(state_dim, action_dim, learning_rate=learning_rate)
        self.discount_gamma = discount_gamma
        self.entropy_coeff = entropy_coeff
        self.experience_buffer: List[Experience] = []

    def collect_experience(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Collect single experience"""
        action_mean, value = self.policy.forward(state)
        _, log_prob = self.policy.sample_action(state)

        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value
        )
        self.experience_buffer.append(exp)

    def compute_td_targets(self) -> np.ndarray:
        """
        Compute TD targets for value function

        Returns:
            Array of TD targets
        """
        targets = []

        for i, exp in enumerate(self.experience_buffer):
            if exp.done:
                target = exp.reward
            else:
                _, next_value = self.policy.forward(exp.next_state)
                target = exp.reward + self.discount_gamma * next_value

            targets.append(target)

        return np.array(targets)

    def update_policy(self, learning_rate: float = 0.001) -> Tuple[float, float]:
        """
        Update actor and critic using TD advantage

        Args:
            learning_rate: Learning rate

        Returns:
            (actor_loss, critic_loss)
        """
        if not self.experience_buffer:
            return 0.0, 0.0

        td_targets = self.compute_td_targets()
        values = np.array([exp.value for exp in self.experience_buffer])

        # TD Advantage
        advantages = td_targets - values

        # Actor loss with entropy regularization
        actor_loss = 0.0
        for exp, advantage in zip(self.experience_buffer, advantages):
            actor_loss += -exp.log_prob * advantage

        actor_loss /= len(self.experience_buffer)

        # Critic loss (MSE)
        critic_loss = np.mean((values - td_targets) ** 2)

        # Clear buffer
        self.experience_buffer = []

        return float(actor_loss), float(critic_loss)


class ProximalPolicyOptimization:
    """Proximal Policy Optimization (PPO) for trading"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        learning_rate: float = 0.001,
        discount_gamma: float = 0.99,
        epsilon_clip: float = 0.2,
        epochs: int = 3
    ):
        """
        Initialize PPO agent

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            learning_rate: Learning rate
            discount_gamma: Discount factor
            epsilon_clip: Clipping parameter (0.1-0.3)
            epochs: Number of epochs per update
        """
        self.policy = PolicyNetwork(state_dim, action_dim, learning_rate=learning_rate)
        self.discount_gamma = discount_gamma
        self.epsilon_clip = epsilon_clip
        self.epochs = epochs
        self.experience_buffer: List[Experience] = []

    def collect_experience(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Collect single experience"""
        action_mean, value = self.policy.forward(state)
        _, log_prob = self.policy.sample_action(state)

        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value
        )
        self.experience_buffer.append(exp)

    def compute_gae(
        self, lambda_gae: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE)

        Args:
            lambda_gae: GAE lambda parameter

        Returns:
            (advantages, td_targets)
        """
        advantages = []
        gae = 0.0

        for i in range(len(self.experience_buffer) - 1, -1, -1):
            exp = self.experience_buffer[i]

            if exp.done:
                next_value = 0.0
            else:
                _, next_value = self.policy.forward(exp.next_state)

            delta = exp.reward + self.discount_gamma * next_value - exp.value
            gae = delta + self.discount_gamma * lambda_gae * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        td_targets = advantages + np.array([exp.value for exp in self.experience_buffer])

        return advantages, td_targets

    def update_policy(self) -> float:
        """
        Update policy using PPO objective

        Returns:
            Average policy loss
        """
        if not self.experience_buffer:
            return 0.0

        advantages, td_targets = self.compute_gae()

        # Store old log probs
        old_log_probs = np.array([exp.log_prob for exp in self.experience_buffer])

        total_loss = 0.0

        # Multi-epoch update
        for epoch in range(self.epochs):
            # Shuffle experiences
            indices = np.random.permutation(len(self.experience_buffer))

            epoch_loss = 0.0
            for idx in indices:
                exp = self.experience_buffer[idx]

                # Recompute current log prob
                _, current_log_prob = self.policy.sample_action(exp.state)

                # Importance sampling ratio
                ratio = np.exp(current_log_prob - old_log_probs[idx])

                # PPO surrogate loss
                surr1 = ratio * advantages[idx]
                surr2 = np.clip(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages[idx]

                loss = -np.minimum(surr1, surr2)
                epoch_loss += loss

            total_loss += epoch_loss / len(indices)

        # Clear buffer
        self.experience_buffer = []

        return float(total_loss / self.epochs)


class RiskAwareRewardFunction:
    """Risk-aware reward shaping for trading"""

    @staticmethod
    def compute_risk_adjusted_reward(
        raw_return: float,
        portfolio_variance: float,
        max_drawdown: float,
        risk_free_rate: float = 0.02,
        lambda_return: float = 1.0,
        lambda_variance: float = 0.1,
        lambda_drawdown: float = 0.05
    ) -> float:
        """
        Compute risk-adjusted reward

        Args:
            raw_return: Raw trading return
            portfolio_variance: Portfolio variance
            max_drawdown: Maximum drawdown
            risk_free_rate: Risk-free rate
            lambda_return: Return weight
            lambda_variance: Variance penalty weight
            lambda_drawdown: Drawdown penalty weight

        Returns:
            Risk-adjusted reward
        """
        # Reward components
        return_reward = lambda_return * raw_return
        variance_penalty = lambda_variance * portfolio_variance
        drawdown_penalty = lambda_drawdown * max_drawdown

        # Combined reward
        reward = return_reward - variance_penalty - drawdown_penalty

        return reward

    @staticmethod
    def compute_sharpe_reward(
        returns: np.ndarray,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Compute Sharpe ratio based reward

        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Sharpe-based reward
        """
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns) * periods_per_year
        std_return = np.std(returns) * np.sqrt(periods_per_year)

        sharpe = (mean_return - risk_free_rate) / (std_return + 1e-8)

        return float(sharpe)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Environment parameters
    state_dim = 5
    action_dim = 1
    episode_length = 100

    # Initialize agents
    vpg = VanillaPolicyGradient(state_dim=state_dim, action_dim=action_dim)
    a2c = ActorCriticAgent(state_dim=state_dim, action_dim=action_dim)
    ppo = ProximalPolicyOptimization(state_dim=state_dim, action_dim=action_dim)

    # Simulate trading episodes
    for episode in range(5):
        state = np.random.randn(state_dim)
        episode_return = 0.0
        returns_list = []

        for step in range(episode_length):
            # Sample action from policy
            action, log_prob = ppo.policy.sample_action(state)

            # Simulate next state and reward
            next_state = state + np.random.randn(state_dim) * 0.1
            raw_return = action * np.random.randn()
            reward = RiskAwareRewardFunction.compute_risk_adjusted_reward(
                raw_return=raw_return,
                portfolio_variance=np.std([raw_return]),
                max_drawdown=-0.02
            )

            done = (step == episode_length - 1)

            # Collect experience
            ppo.collect_experience(state, action, reward, next_state, done)

            episode_return += reward
            returns_list.append(reward)
            state = next_state

        # Update policy
        ppo_loss = ppo.update_policy()

        # Compute Sharpe reward
        sharpe = RiskAwareRewardFunction.compute_sharpe_reward(np.array(returns_list))

        logger.info(f"Episode {episode}: Return={episode_return:.4f}, PPO Loss={ppo_loss:.4f}, Sharpe={sharpe:.4f}")

    logger.info("RL Training Complete")
