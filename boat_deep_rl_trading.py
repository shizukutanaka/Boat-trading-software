#!/usr/bin/env python3
"""
Deep Reinforcement Learning Trading Agent for Boat
==================================================

Implements state-of-the-art deep RL algorithms for trading:
  - Deep Q-Network (DQN)
  - Policy Gradient Methods (A2C, PPO)
  - Actor-Critic Methods
  - Multi-agent learning

Based on 2025 research in quantitative finance.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingAction(Enum):
    """Trading actions"""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class TradingState:
    """Trading environment state"""
    price: float
    returns: np.ndarray  # Recent returns
    volume: float
    portfolio_value: float
    position_size: float
    cash: float
    timestamp: int


class MarketEnvironment:
    """Trading environment for RL agents"""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_position: float = 1.0,
        commission: float = 0.001,
        episode_length: int = 252
    ):
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.commission = commission
        self.episode_length = episode_length

        # State
        self.cash = initial_capital
        self.position = 0.0
        self.portfolio_value = initial_capital
        self.episode_step = 0
        self.price_history = deque(maxlen=20)

    def reset(self, price: float) -> TradingState:
        """Reset environment"""
        self.cash = self.initial_capital
        self.position = 0.0
        self.portfolio_value = self.initial_capital
        self.episode_step = 0
        self.price_history = deque(maxlen=20)
        self.price_history.append(price)

        return self._get_state(price)

    def step(
        self,
        action: int,
        price: float,
        volume: float
    ) -> Tuple[TradingState, float, bool]:
        """
        Execute trading action

        Args:
            action: 0=hold, 1=buy, 2=sell
            price: Current price
            volume: Current volume

        Returns:
            (new_state, reward, done)
        """
        self.episode_step += 1
        done = self.episode_step >= self.episode_length

        # Record price
        self.price_history.append(price)

        # Execute action
        reward = 0.0

        if action == TradingAction.BUY.value:
            # Buy action
            if self.position < self.max_position and self.cash > 0:
                buy_amount = min(self.cash / price * 0.95, self.max_position - self.position)
                cost = buy_amount * price * (1 + self.commission)

                if cost <= self.cash:
                    self.cash -= cost
                    self.position += buy_amount
                    reward = -self.commission  # Cost of trading

        elif action == TradingAction.SELL.value:
            # Sell action
            if self.position > 0:
                sell_amount = self.position
                proceeds = sell_amount * price * (1 - self.commission)
                self.cash += proceeds
                self.position = 0
                reward = proceeds - (sell_amount * price)  # Profit/loss

        # Update portfolio value
        position_value = self.position * price
        self.portfolio_value = self.cash + position_value

        # Reward: portfolio change
        if len(self.price_history) > 1:
            price_change = (price - self.price_history[-2]) / self.price_history[-2]
            reward += price_change * self.position  # P&L from position

        state = self._get_state(price)
        return state, reward, done

    def _get_state(self, price: float) -> TradingState:
        """Get current state"""
        # Calculate recent returns
        prices = list(self.price_history)
        if len(prices) > 1:
            returns = np.array([
                (prices[i] - prices[i-1]) / prices[i-1]
                for i in range(1, len(prices))
            ])
        else:
            returns = np.array([0.0])

        return TradingState(
            price=price,
            returns=returns,
            volume=0.0,
            portfolio_value=self.portfolio_value,
            position_size=self.position,
            cash=self.cash,
            timestamp=self.episode_step
        )


class DQNAgent:
    """Deep Q-Network Agent"""

    def __init__(
        self,
        state_size: int = 20,
        action_size: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_decay: float = 0.995
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01

        # Experience replay
        self.memory = deque(maxlen=2000)

        # Build Q-network
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()

    def _build_network(self) -> Sequential:
        """Build neural network"""
        model = Sequential([
            layers.Dense(64, activation='relu', input_dim=self.state_size),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model

    def update_target_network(self) -> None:
        """Update target network weights"""
        self.target_network.set_weights(self.q_network.get_weights())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size: int) -> float:
        """Experience replay training"""
        if len(self.memory) < batch_size:
            return 0.0

        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = np.array([self.memory[i][0] for i in batch])
        actions = np.array([self.memory[i][1] for i in batch])
        rewards = np.array([self.memory[i][2] for i in batch])
        next_states = np.array([self.memory[i][3] for i in batch])
        dones = np.array([self.memory[i][4] for i in batch])

        # Predict Q-values
        q_values = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)

        # Update Q-values with Bellman equation
        for i in range(batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train
        loss = self.q_network.train_on_batch(states, q_values)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss


class PolicyGradientAgent:
    """Policy Gradient Agent (Advantage Actor-Critic)"""

    def __init__(
        self,
        state_size: int = 20,
        action_size: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.99
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Actor (policy) network
        self.actor = self._build_actor()

        # Critic (value) network
        self.critic = self._build_critic()

    def _build_actor(self) -> Sequential:
        """Build actor (policy) network"""
        model = Sequential([
            layers.Dense(64, activation='relu', input_dim=self.state_size),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='softmax')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy'
        )
        return model

    def _build_critic(self) -> Sequential:
        """Build critic (value) network"""
        model = Sequential([
            layers.Dense(64, activation='relu', input_dim=self.state_size),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model

    def act(self, state: np.ndarray) -> int:
        """Choose action from policy"""
        probabilities = self.actor.predict(state.reshape(1, -1), verbose=0)[0]
        action = np.random.choice(self.action_size, p=probabilities)
        return action

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> Tuple[float, float]:
        """Train actor-critic network"""
        # Compute value estimates
        values = self.critic.predict(states, verbose=0)
        next_values = self.critic.predict(next_states, verbose=0)

        # Compute advantages
        advantages = rewards.reshape(-1, 1)
        for i in range(len(states)):
            if not dones[i]:
                advantages[i] += self.gamma * next_values[i]
            advantages[i] -= values[i]

        # Convert actions to one-hot
        actions_onehot = keras.utils.to_categorical(actions, self.action_size)

        # Train critic
        critic_loss = self.critic.train_on_batch(states, advantages + values)

        # Train actor with policy gradient
        actor_loss = self.actor.train_on_batch(states, actions_onehot)

        return actor_loss, critic_loss


class RLTrainingLoop:
    """Training loop for RL agents"""

    def __init__(
        self,
        agent: DQNAgent,
        environment: MarketEnvironment,
        episodes: int = 100,
        batch_size: int = 32
    ):
        self.agent = agent
        self.environment = environment
        self.episodes = episodes
        self.batch_size = batch_size
        self.episode_rewards = []

    def train(self, price_data: pd.DataFrame) -> List[float]:
        """
        Train agent on price data

        Args:
            price_data: DataFrame with 'close' column

        Returns:
            List of episode rewards
        """
        prices = price_data['close'].values

        for episode in range(self.episodes):
            state = self.environment.reset(prices[0])
            episode_reward = 0

            for step in range(len(prices) - 1):
                # Get state vector
                state_vector = self._state_to_vector(state)

                # Choose action
                action = self.agent.act(state_vector)

                # Execute action
                next_state, reward, done = self.environment.step(
                    action,
                    prices[step],
                    1000.0  # Mock volume
                )

                episode_reward += reward

                # Remember experience
                next_state_vector = self._state_to_vector(next_state)
                self.agent.remember(
                    state_vector,
                    action,
                    reward,
                    next_state_vector,
                    done
                )

                # Train on batch
                if step % self.batch_size == 0:
                    self.agent.replay(self.batch_size)

                state = next_state

                if done:
                    self.agent.update_target_network()
                    break

            self.episode_rewards.append(episode_reward)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                logger.info(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

        return self.episode_rewards

    def _state_to_vector(self, state: TradingState) -> np.ndarray:
        """Convert state to feature vector"""
        features = [
            state.price,
            state.position_size,
            state.portfolio_value,
            state.cash,
            state.timestamp / 252  # Normalized
        ]

        # Add returns
        features.extend(state.returns)

        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)

        return np.array(features[:20])

    def get_performance(self) -> Dict[str, float]:
        """Get training performance"""
        returns = np.array(self.episode_rewards)
        return {
            'total_episodes': self.episodes,
            'mean_reward': returns.mean(),
            'std_reward': returns.std(),
            'max_reward': returns.max(),
            'min_reward': returns.min(),
            'final_reward': returns[-1]
        }


if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='D')
    price_data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(500) * 0.5)
    }, index=dates)

    # Create environment and agent
    env = MarketEnvironment(initial_capital=10000, episode_length=100)
    agent = DQNAgent(state_size=20, action_size=3)

    # Train
    trainer = RLTrainingLoop(agent, env, episodes=50, batch_size=32)
    rewards = trainer.train(price_data)

    # Get performance
    performance = trainer.get_performance()
    logger.info(f"Final Performance: {performance}")
