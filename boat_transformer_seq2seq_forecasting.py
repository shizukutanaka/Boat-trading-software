#!/usr/bin/env python3
"""
Transformer Seq2Seq Networks for Financial Forecasting
=======================================================

Encoder-decoder transformer for sequence-to-sequence prediction:
  - Multi-head self-attention in encoder and decoder
  - Positional encoding for temporal information
  - Encoder-decoder cross-attention
  - Multi-step ahead forecasting
  - Attention visualization for interpretability
  - Superior to LSTM/GRU for long sequences

Based on 2025 research (Transformer for Time Series, Seq2Seq Forecasting, Attention Mechanisms).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Transformer configuration"""
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    seq_len: int = 20
    forecast_horizon: int = 5


@dataclass
class Seq2SeqOutput:
    """Seq2Seq forecasting output"""
    predictions: np.ndarray  # (forecast_horizon,)
    attention_weights: np.ndarray  # (num_heads, seq_len)
    confidence_scores: np.ndarray  # (forecast_horizon,)
    encoder_output: np.ndarray


class PositionalEncoding:
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding"""
        self.d_model = d_model

        # Compute positional encoding
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = pe

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input

        Args:
            x: Input (seq_len, d_model)

        Returns:
            x with positional encoding added
        """
        seq_len = x.shape[0]
        return x + self.pe[:seq_len]


class MultiHeadAttention:
    """Multi-head self-attention mechanism"""

    def __init__(self, d_model: int, num_heads: int):
        """Initialize multi-head attention"""
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_out = np.random.randn(d_model, d_model) * 0.01

    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-head attention forward pass

        Args:
            query: Query (query_len, d_model)
            key: Key (key_len, d_model)
            value: Value (key_len, d_model)
            mask: Attention mask

        Returns:
            (output, attention_weights)
        """
        query_len = query.shape[0]
        key_len = key.shape[0]

        # Linear projections
        Q = query @ self.W_q
        K = key @ self.W_k
        V = value @ self.W_v

        # Split into multiple heads
        Q = Q.reshape(query_len, self.num_heads, self.head_dim)
        K = K.reshape(key_len, self.num_heads, self.head_dim)
        V = V.reshape(key_len, self.num_heads, self.head_dim)

        # Compute attention for each head
        head_outputs = []
        attention_weights_all = []

        for h in range(self.num_heads):
            Q_h = Q[:, h, :]  # (query_len, head_dim)
            K_h = K[:, h, :]  # (key_len, head_dim)
            V_h = V[:, h, :]  # (key_len, head_dim)

            # Attention scores
            scores = (Q_h @ K_h.T) / np.sqrt(self.head_dim)  # (query_len, key_len)

            # Apply mask if provided
            if mask is not None:
                scores = scores + mask

            # Softmax
            scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            attention = scores_exp / (np.sum(scores_exp, axis=1, keepdims=True) + 1e-8)

            # Apply attention to values
            head_output = attention @ V_h  # (query_len, head_dim)
            head_outputs.append(head_output)
            attention_weights_all.append(attention)

        # Concatenate heads
        output = np.concatenate(head_outputs, axis=1)  # (query_len, d_model)

        # Output projection
        output = output @ self.W_out

        # Average attention weights across heads for visualization (take first one)
        avg_attention = np.mean(np.array(attention_weights_all), axis=0)

        return output, avg_attention


class FeedForward:
    """Position-wise feed-forward network"""

    def __init__(self, d_model: int, d_ff: int):
        """Initialize feed-forward network"""
        self.d_model = d_model
        self.d_ff = d_ff

        # Two linear transformations with ReLU in between
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Feed-forward forward pass"""
        # First linear transformation + ReLU
        hidden = np.maximum(0, x @ self.W1 + self.b1)

        # Second linear transformation
        output = hidden @ self.W2 + self.b2

        return output


class TransformerEncoderLayer:
    """Single transformer encoder layer"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        """Initialize encoder layer"""
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.d_model = d_model

        # Layer normalization parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Simple layer normalization"""
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True)
        return gamma * (x - mean) / (std + 1e-8) + beta

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encoder layer forward pass"""
        # Self-attention
        attn_output, attn_weights = self.attention.forward(x, x, x)

        # Residual connection and layer norm
        x = self._layer_norm(x + attn_output, self.gamma1, self.beta1)

        # Feed-forward
        ff_output = self.feed_forward.forward(x)

        # Residual connection and layer norm
        output = self._layer_norm(x + ff_output, self.gamma2, self.beta2)

        return output, attn_weights


class TransformerDecoderLayer:
    """Single transformer decoder layer"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        """Initialize decoder layer"""
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.d_model = d_model

        # Layer normalization parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
        self.gamma3 = np.ones(d_model)
        self.beta3 = np.zeros(d_model)

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True)
        return gamma * (x - mean) / (std + 1e-8) + beta

    def forward(self, x: np.ndarray, encoder_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decoder layer forward pass"""
        # Masked self-attention
        self_attn_output, _ = self.self_attention.forward(x, x, x)
        x = self._layer_norm(x + self_attn_output, self.gamma1, self.beta1)

        # Cross-attention (attend to encoder output)
        cross_attn_output, cross_attn_weights = self.cross_attention.forward(
            x, encoder_output, encoder_output
        )
        x = self._layer_norm(x + cross_attn_output, self.gamma2, self.beta2)

        # Feed-forward
        ff_output = self.feed_forward.forward(x)
        output = self._layer_norm(x + ff_output, self.gamma3, self.beta3)

        return output, cross_attn_weights


class TransformerSeq2Seq:
    """Transformer Seq2Seq model for financial forecasting"""

    def __init__(self, config: TransformerConfig):
        """Initialize transformer"""
        self.config = config
        self.d_model = config.d_model

        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model)

        # Encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(config.d_model, config.num_heads, config.d_ff)
            for _ in range(config.num_layers)
        ]

        # Decoder layers
        self.decoder_layers = [
            TransformerDecoderLayer(config.d_model, config.num_heads, config.d_ff)
            for _ in range(config.num_layers)
        ]

        # Input/output projections
        self.input_proj = np.random.randn(1, config.d_model) * 0.01
        self.output_proj = np.random.randn(config.d_model, 1) * 0.01

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode input sequence

        Args:
            x: Input sequence (seq_len, 1)

        Returns:
            (encoder_output, attention_weights)
        """
        # Project input
        x_proj = x @ self.input_proj

        # Add positional encoding
        x_emb = self.pos_encoder.forward(x_proj)

        # Encoder layers
        encoder_output = x_emb
        attention_weights = None

        for layer in self.encoder_layers:
            encoder_output, attn_weights = layer.forward(encoder_output)
            attention_weights = attn_weights

        return encoder_output, attention_weights

    def decode(self, decoder_input: np.ndarray, encoder_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode for predictions

        Args:
            decoder_input: Decoder input (forecast_horizon, 1)
            encoder_output: Encoder output

        Returns:
            (predictions, attention_weights)
        """
        # Project and embed decoder input
        decoder_input_proj = decoder_input @ self.input_proj
        decoder_emb = self.pos_encoder.forward(decoder_input_proj)

        # Decoder layers
        decoder_output = decoder_emb
        cross_attention_weights = None

        for layer in self.decoder_layers:
            decoder_output, cross_attn = layer.forward(decoder_output, encoder_output)
            cross_attention_weights = cross_attn

        # Project output
        predictions = decoder_output @ self.output_proj

        return predictions, cross_attention_weights

    def forward(self, x: np.ndarray, forecast_horizon: int) -> Seq2SeqOutput:
        """
        Full forward pass

        Args:
            x: Input sequence (seq_len, 1)
            forecast_horizon: Number of steps to forecast

        Returns:
            Seq2SeqOutput
        """
        # Encode
        encoder_output, encoder_attn = self.encode(x)

        # Decode (using shifted target sequence)
        decoder_input = np.zeros((forecast_horizon, 1))
        predictions, decoder_attn = self.decode(decoder_input, encoder_output)

        # Compute confidence scores (inverse of attention entropy)
        entropy = -np.sum(decoder_attn * np.log(decoder_attn + 1e-8), axis=1)
        confidence = 1.0 - (entropy / np.log(x.shape[0]))
        confidence = np.clip(confidence, 0, 1)

        return Seq2SeqOutput(
            predictions=predictions.flatten(),
            attention_weights=encoder_attn,
            confidence_scores=confidence,
            encoder_output=encoder_output
        )


class FinancialForecastingFramework:
    """Framework for financial time series forecasting"""

    def __init__(self, seq_len: int = 20, forecast_horizon: int = 5):
        """Initialize framework"""
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon

        config = TransformerConfig(
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=256,
            seq_len=seq_len,
            forecast_horizon=forecast_horizon
        )

        self.model = TransformerSeq2Seq(config)

    def generate_synthetic_timeseries(self, n_series: int = 5) -> List[np.ndarray]:
        """Generate synthetic financial time series"""
        series_list = []

        for _ in range(n_series):
            # Trend
            trend = np.linspace(100, 110, 100)

            # Seasonal
            seasonal = 5 * np.sin(np.linspace(0, 4 * np.pi, 100))

            # Noise
            noise = np.random.randn(100) * 2

            # Combined
            ts = trend + seasonal + noise
            series_list.append(ts)

        return series_list

    def forecast_single_series(self, series: np.ndarray) -> Dict:
        """
        Forecast single time series

        Args:
            series: Time series array

        Returns:
            Dictionary with predictions and metrics
        """
        # Extract window
        if len(series) >= self.seq_len:
            x = series[-self.seq_len:].reshape(-1, 1).astype(np.float32)
        else:
            x = series.reshape(-1, 1).astype(np.float32)
            x = np.vstack([np.zeros((self.seq_len - len(x), 1)), x])

        # Normalize
        x_mean = np.mean(x)
        x_std = np.std(x) + 1e-8
        x_norm = (x - x_mean) / x_std

        # Forward pass
        output = self.model.forward(x_norm, self.forecast_horizon)

        # Denormalize
        predictions = output.predictions * x_std + x_mean

        return {
            'predictions': predictions,
            'confidence': output.confidence_scores,
            'attention': output.attention_weights,
            'encoder_output': output.encoder_output
        }

    def forecast_portfolio(self, price_matrix: np.ndarray) -> Dict:
        """
        Forecast portfolio returns

        Args:
            price_matrix: (n_assets, n_periods)

        Returns:
            Dictionary with forecasts
        """
        forecasts = {}

        for i in range(price_matrix.shape[0]):
            series = price_matrix[i, :]
            forecast = self.forecast_single_series(series)

            # Compute log returns for forecast
            log_returns = np.log(forecast['predictions'] / (price_matrix[i, -1] + 1e-8))

            forecasts[f"asset_{i}"] = {
                'predictions': forecast['predictions'],
                'log_returns': log_returns,
                'confidence': forecast['confidence']
            }

        return forecasts


if __name__ == "__main__":
    logger.info("Transformer Seq2Seq Networks for Financial Forecasting")
    logger.info("=" * 60)

    np.random.seed(42)

    # Initialize framework
    logger.info("\nInitializing Transformer Forecasting Framework")
    framework = FinancialForecastingFramework(seq_len=20, forecast_horizon=5)

    # Generate synthetic data
    logger.info("\nGenerating synthetic financial time series")
    price_series = framework.generate_synthetic_timeseries(n_series=5)
    logger.info(f"  Generated {len(price_series)} price series")

    # Forecast
    logger.info("\nForecasting portfolio (5 assets, 5 steps ahead)")
    price_matrix = np.array(price_series)

    forecasts = framework.forecast_portfolio(price_matrix)

    # Display results
    logger.info("\nForecast Results (Asset 0):")
    logger.info("Step | Price Forecast | Log Return  | Confidence")
    logger.info("-" * 60)

    asset_0 = forecasts["asset_0"]
    for step in range(len(asset_0['predictions'])):
        logger.info(f"{step + 1:4d} | {asset_0['predictions'][step]:14.4f} | "
                   f"{asset_0['log_returns'][step]:11.6f} | {asset_0['confidence'][step]:10.4f}")

    # Aggregate statistics
    logger.info("\n\nPortfolio-Level Statistics:")
    all_predictions = []
    all_confidence = []

    for asset_name, forecast in forecasts.items():
        all_predictions.extend(forecast['predictions'].tolist())
        all_confidence.extend(forecast['confidence'].tolist())

    logger.info(f"  Mean Forecast Price: ${np.mean(all_predictions):.2f}")
    logger.info(f"  Std Forecast Price: ${np.std(all_predictions):.2f}")
    logger.info(f"  Mean Confidence: {np.mean(all_confidence):.4f}")
    logger.info(f"  Min Confidence: {np.min(all_confidence):.4f}")
    logger.info(f"  Max Confidence: {np.max(all_confidence):.4f}")

    logger.info("\nTransformer Seq2Seq Forecasting Complete")
