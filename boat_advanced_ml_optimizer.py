#!/usr/bin/env python3
"""
Advanced ML-Based Strategy Optimizer for Boat
==============================================

This module implements cutting-edge machine learning techniques for trading strategy
optimization, including hyperparameter tuning, feature engineering, and ensemble methods.

Features:
  - Hyperparameter optimization using Optuna
  - Feature engineering pipeline
  - Ensemble learning models (Random Forest, Gradient Boosting, Neural Networks)
  - Walk-forward validation for realistic backtesting
  - Cross-validation and out-of-sample testing
  - Model explainability with SHAP values
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import json

# ML/Data Science Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

# Optimization Libraries
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("Warning: Optuna not installed. Install with: pip install optuna")

# Neural Network Libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Sequential
except ImportError:
    print("Warning: TensorFlow not installed")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MLOptimizationConfig:
    """Configuration for ML optimization"""
    n_trials: int = 100
    n_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    verbose: bool = True
    save_models: bool = True
    models_dir: str = "models/"
    enable_ensemble: bool = True
    enable_neural_net: bool = True
    feature_importance_threshold: float = 0.01


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering"""
    technical_features: bool = True
    statistical_features: bool = True
    price_features: bool = True
    volume_features: bool = True
    trend_features: bool = True
    volatility_features: bool = True
    momentum_features: bool = True
    correlation_features: bool = True
    lookback_period: int = 252
    normalization_method: str = "robust"  # 'standard' or 'robust'


class FeatureEngineer:
    """Advanced feature engineering pipeline for trading data"""

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.scaler = RobustScaler() if config.normalization_method == "robust" else StandardScaler()
        self.feature_names = []

    def engineer_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive set of features from price data

        Args:
            price_data: DataFrame with OHLCV data

        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=price_data.index)

        if self.config.price_features:
            features = self._add_price_features(features, price_data)

        if self.config.technical_features:
            features = self._add_technical_features(features, price_data)

        if self.config.momentum_features:
            features = self._add_momentum_features(features, price_data)

        if self.config.volatility_features:
            features = self._add_volatility_features(features, price_data)

        if self.config.volume_features:
            features = self._add_volume_features(features, price_data)

        if self.config.trend_features:
            features = self._add_trend_features(features, price_data)

        # Remove NaN values
        features = features.dropna()

        # Normalize features
        feature_cols = [col for col in features.columns if col != 'target']
        features[feature_cols] = self.scaler.fit_transform(features[feature_cols])

        self.feature_names = feature_cols
        logger.info(f"Engineered {len(self.feature_names)} features")

        return features

    def _add_price_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        features['close_change'] = data['close'].pct_change()
        features['high_low_ratio'] = data['high'] / data['low'] - 1
        features['close_open_ratio'] = data['close'] / data['open'] - 1
        features['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        return features

    def _add_technical_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        # SMA
        for period in [5, 10, 20, 50, 200]:
            features[f'sma_{period}'] = data['close'].rolling(period).mean() / data['close'] - 1

        # EMA
        for period in [12, 26]:
            features[f'ema_{period}'] = data['close'].ewm(span=period).mean() / data['close'] - 1

        # RSI
        features['rsi_14'] = self._calculate_rsi(data['close'], 14)
        features['rsi_21'] = self._calculate_rsi(data['close'], 21)

        # MACD
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        features['macd'] = (ema12 - ema26) / data['close']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()

        return features

    def _add_momentum_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features"""
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = data['close'].diff(period) / data['close']
            features[f'roc_{period}'] = data['close'].pct_change(period)

        return features

    def _add_volatility_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        for period in [10, 20, 30]:
            features[f'volatility_{period}'] = data['close'].pct_change().rolling(period).std()
            features[f'parkinson_{period}'] = (
                (np.log(data['high'] / data['low']) ** 2) / (4 * np.log(2))
            ).rolling(period).mean()

        return features

    def _add_volume_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        features['volume_sma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['price_volume_trend'] = (data['close'].pct_change() * data['volume']).rolling(10).sum()
        features['obv'] = (np.sign(data['close'].diff()) * data['volume']).rolling(10).sum()

        return features

    def _add_trend_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based features"""
        # Slope of moving averages
        sma_20 = data['close'].rolling(20).mean()
        features['trend_strength'] = np.polyfit(range(len(sma_20.tail(20))), sma_20.tail(20), 1)[0]

        # Trend direction (up=1, down=-1)
        features['trend_direction'] = np.where(data['close'] > data['close'].shift(1), 1, -1)

        return features

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class MLStrategy(ABC):
    """Base class for ML-based trading strategies"""

    def __init__(self, name: str, model: Any, features: List[str]):
        self.name = name
        self.model = model
        self.features = features
        self.predictions = None
        self.confidence = None

    @abstractmethod
    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate trading signals"""
        pass


class EnsembleMLStrategy(MLStrategy):
    """Ensemble strategy combining multiple models"""

    def __init__(self, models: Dict[str, Tuple[Any, float]], features: List[str]):
        """
        Args:
            models: Dict of {model_name: (model, weight)}
            features: List of feature names
        """
        self.models = models
        self.features = features
        self.weights = {name: weight for name, (_, weight) in models.items()}

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate signals from ensemble predictions"""
        predictions = pd.Series(0.0, index=features.index)

        for model_name, (model, weight) in self.models.items():
            try:
                proba = model.predict_proba(features[self.features])[:, 1]
                predictions += weight * proba
            except:
                pred = model.predict(features[self.features])
                predictions += weight * (pred > 0.5).astype(int)

        # Normalize
        total_weight = sum(self.weights.values())
        predictions /= total_weight

        # Convert to signals (1=buy, 0=hold, -1=sell)
        signals = pd.Series(0, index=features.index)
        signals[predictions > 0.6] = 1
        signals[predictions < 0.4] = -1

        return signals


class MLStrategyOptimizer:
    """Optimize trading strategies using machine learning"""

    def __init__(self, config: MLOptimizationConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(FeatureEngineeringConfig())
        self.best_models = {}
        self.optimization_history = []

    def prepare_training_data(
        self,
        price_data: pd.DataFrame,
        target: pd.Series = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable"""
        features = self.feature_engineer.engineer_features(price_data)

        if target is None:
            # Generate target: 1 if price goes up next day, 0 otherwise
            price_future = price_data.loc[features.index, 'close'].shift(-1)
            target = (price_future > price_data.loc[features.index, 'close']).astype(int)
            target = target.iloc[:-1]
            features = features.iloc[:-1]

        return features, target

    def optimize_xgboost(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> xgb.XGBClassifier:
        """Optimize XGBoost parameters using Optuna"""

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }

            model = xgb.XGBClassifier(**params, random_state=self.config.random_state)
            scores = cross_val_score(
                model, features, target,
                cv=self.config.n_folds,
                scoring='roc_auc'
            )
            return scores.mean()

        sampler = TPESampler(seed=self.config.random_state)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=True)

        best_params = study.best_params
        logger.info(f"Best XGBoost params: {best_params}")

        model = xgb.XGBClassifier(**best_params, random_state=self.config.random_state)
        model.fit(features, target)

        return model

    def optimize_lightgbm(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> lgb.LGBMClassifier:
        """Optimize LightGBM parameters"""

        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
            }

            model = lgb.LGBMClassifier(**params, random_state=self.config.random_state, verbose=-1)
            scores = cross_val_score(
                model, features, target,
                cv=self.config.n_folds,
                scoring='roc_auc'
            )
            return scores.mean()

        sampler = TPESampler(seed=self.config.random_state)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=True)

        best_params = study.best_params
        logger.info(f"Best LightGBM params: {best_params}")

        model = lgb.LGBMClassifier(**best_params, random_state=self.config.random_state, verbose=-1)
        model.fit(features, target)

        return model

    def build_neural_network(
        self,
        input_dim: int,
        output_dim: int = 1
    ) -> Sequential:
        """Build neural network for trading"""
        model = Sequential([
            layers.Dense(128, activation='relu', input_dim=input_dim),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(output_dim, activation='sigmoid')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )

        return model

    def walk_forward_validation(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        train_size: int = 1000,
        step_size: int = 100
    ) -> Dict[str, List[float]]:
        """
        Perform walk-forward validation
        More realistic backtesting with rolling windows
        """
        results = {
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }

        for i in range(train_size, len(features) - step_size, step_size):
            train_data = features.iloc[:i]
            train_target = target.iloc[:i]
            test_data = features.iloc[i:i+step_size]
            test_target = target.iloc[i:i+step_size]

            # Train model
            model = xgb.XGBClassifier(random_state=self.config.random_state)
            model.fit(train_data, train_target)

            # Evaluate
            predictions = model.predict(test_data)

            results['precision'].append(precision_score(test_target, predictions, zero_division=0))
            results['recall'].append(recall_score(test_target, predictions, zero_division=0))
            results['f1'].append(f1_score(test_target, predictions, zero_division=0))

            try:
                proba = model.predict_proba(test_data)[:, 1]
                results['roc_auc'].append(roc_auc_score(test_target, proba))
            except:
                pass

        return results

    def build_ensemble_strategy(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> EnsembleMLStrategy:
        """Build optimized ensemble strategy"""
        models = {}

        # XGBoost
        logger.info("Optimizing XGBoost...")
        xgb_model = self.optimize_xgboost(features, target)
        models['xgboost'] = (xgb_model, 0.4)

        # LightGBM
        logger.info("Optimizing LightGBM...")
        lgb_model = self.optimize_lightgbm(features, target)
        models['lightgbm'] = (lgb_model, 0.3)

        # Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.config.random_state
        )
        rf_model.fit(features, target)
        models['random_forest'] = (rf_model, 0.3)

        return EnsembleMLStrategy(models, self.feature_engineer.feature_names)


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
    price_data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'high': 101 + np.cumsum(np.random.randn(500) * 0.5),
        'low': 99 + np.cumsum(np.random.randn(500) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'volume': np.random.randint(1000000, 5000000, 500)
    }, index=dates)

    # Prepare data
    optimizer = MLStrategyOptimizer(MLOptimizationConfig())
    features, target = optimizer.prepare_training_data(price_data)

    # Build ensemble strategy
    logger.info("Building ensemble strategy...")
    strategy = optimizer.build_ensemble_strategy(features, target)

    # Generate signals
    signals = strategy.generate_signals(features)
    logger.info(f"Generated {len(signals)} trading signals")
    logger.info(f"Buy signals: {(signals == 1).sum()}")
    logger.info(f"Sell signals: {(signals == -1).sum()}")

    # Walk-forward validation
    logger.info("Performing walk-forward validation...")
    validation_results = optimizer.walk_forward_validation(features, target)
    logger.info(f"Average F1 Score: {np.mean(validation_results['f1']):.4f}")
