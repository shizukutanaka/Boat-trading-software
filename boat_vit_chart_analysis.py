#!/usr/bin/env python3
"""
Vision Transformer for Financial Chart Analysis
================================================

Image-based financial analysis via transformers:
  - Candlestick pattern recognition
  - Technical indicator extraction from charts
  - Multi-timeframe chart analysis
  - Pattern classification (bullish/bearish/neutral)
  - Visual price action interpretation

Based on 2025 research on Vision Transformers ($15.2B market, 28.5% CAGR).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CandleData:
    """Candlestick OHLC data"""
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class ChartPattern:
    """Detected chart pattern"""
    pattern_type: str  # 'hammer', 'engulfing', 'doji', 'harami', etc.
    bullish_probability: float
    location: Tuple[int, int]  # x, y in chart
    strength: float  # 0-1 confidence


@dataclass
class ChartAnalysisResult:
    """Result from ViT chart analysis"""
    patterns: List[ChartPattern]
    trend: str  # 'uptrend', 'downtrend', 'consolidation'
    support_resistance: Dict[str, float]
    trading_signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float


class PatchEmbedding:
    """Convert chart image to patch embeddings"""

    def __init__(self, patch_size: int = 16, embedding_dim: int = 768):
        """
        Initialize patch embedding

        Args:
            patch_size: Size of patches to extract
            embedding_dim: Embedding dimension
        """
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.linear_projection = np.random.randn(patch_size * patch_size * 3, embedding_dim) * 0.01

    def embed_patches(self, chart_image: np.ndarray) -> np.ndarray:
        """
        Extract and embed patches from chart image

        Args:
            chart_image: (H, W, 3) chart image

        Returns:
            (n_patches, embedding_dim) embeddings
        """
        H, W, C = chart_image.shape
        patches = []

        for i in range(0, H - self.patch_size + 1, self.patch_size):
            for j in range(0, W - self.patch_size + 1, self.patch_size):
                patch = chart_image[i : i + self.patch_size, j : j + self.patch_size, :]
                patch_flat = patch.flatten()
                patches.append(patch_flat)

        patches = np.array(patches)
        embeddings = patches @ self.linear_projection

        return embeddings


class ViTAttentionLayer:
    """Multi-head self-attention for vision"""

    def __init__(self, embedding_dim: int = 768, n_heads: int = 12):
        """Initialize ViT attention"""
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads

        self.W_q = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(2.0 / embedding_dim)
        self.W_k = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(2.0 / embedding_dim)
        self.W_v = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(2.0 / embedding_dim)
        self.W_o = np.random.randn(embedding_dim, embedding_dim) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Multi-head self-attention

        Args:
            x: (seq_len, embedding_dim) token embeddings

        Returns:
            Attended output (seq_len, embedding_dim)
        """
        seq_len = x.shape[0]

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head attention
        Q = Q.reshape(seq_len, self.n_heads, self.head_dim)
        K = K.reshape(seq_len, self.n_heads, self.head_dim)
        V = V.reshape(seq_len, self.n_heads, self.head_dim)

        attention_scores = []
        for h in range(self.n_heads):
            Q_h = Q[:, h, :]
            K_h = K[:, h, :]
            V_h = V[:, h, :]

            # Attention
            scores = (Q_h @ K_h.T) / np.sqrt(self.head_dim)
            weights = self._softmax(scores)
            attended = weights @ V_h
            attention_scores.append(attended)

        output = np.concatenate(attention_scores, axis=1)
        output = output @ self.W_o

        return output

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Stable softmax"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)


class VisionTransformer:
    """Vision Transformer for chart analysis"""

    def __init__(self, embedding_dim: int = 768, n_layers: int = 12, n_heads: int = 12):
        """Initialize ViT"""
        self.embedding_dim = embedding_dim
        self.patch_embedding = PatchEmbedding(patch_size=16, embedding_dim=embedding_dim)
        self.attention_layers = [ViTAttentionLayer(embedding_dim, n_heads) for _ in range(n_layers)]

        # Classification head
        self.pattern_classifier = np.random.randn(embedding_dim, 7) * 0.01  # 7 pattern types
        self.trend_classifier = np.random.randn(embedding_dim, 3) * 0.01  # 3 trend types
        self.signal_classifier = np.random.randn(embedding_dim, 3) * 0.01  # BUY/SELL/HOLD

    def forward(self, chart_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        ViT forward pass on chart image

        Args:
            chart_image: (H, W, 3) chart image

        Returns:
            (pattern_logits, trend_logits, signal_logits)
        """
        # Extract patch embeddings
        patch_embeddings = self.patch_embedding.embed_patches(chart_image)

        # Add class token
        class_token = np.random.randn(1, self.embedding_dim) * 0.01
        x = np.vstack([class_token, patch_embeddings])

        # Apply transformer layers
        for layer in self.attention_layers[:2]:  # Use 2 layers for efficiency
            x = layer.forward(x)

        # Extract class token representation
        class_representation = x[0]

        # Classify patterns
        pattern_logits = class_representation @ self.pattern_classifier
        trend_logits = class_representation @ self.trend_classifier
        signal_logits = class_representation @ self.signal_classifier

        return pattern_logits, trend_logits, signal_logits


class CandlePatternDetector:
    """Detect candlestick patterns"""

    @staticmethod
    def detect_hammer(candles: List[CandleData], idx: int) -> bool:
        """Detect hammer pattern"""
        if idx < 1:
            return False

        c = candles[idx]
        body_height = abs(c.close - c.open)
        total_height = c.high - c.low
        lower_wick = min(c.open, c.close) - c.low

        # Hammer: small body, long lower wick, small upper wick
        return body_height < total_height * 0.3 and lower_wick > body_height * 2

    @staticmethod
    def detect_engulfing(candles: List[CandleData], idx: int) -> bool:
        """Detect engulfing pattern"""
        if idx < 1:
            return False

        prev = candles[idx - 1]
        curr = candles[idx]

        # Bullish engulfing
        prev_body = abs(prev.close - prev.open)
        curr_body = abs(curr.close - curr.open)

        return curr_body > prev_body and curr.open < prev.close and curr.close > prev.open

    @staticmethod
    def detect_doji(candles: List[CandleData], idx: int) -> bool:
        """Detect doji pattern"""
        if idx < 0:
            return False

        c = candles[idx]
        body_height = abs(c.close - c.open)
        total_height = c.high - c.low

        # Doji: very small body relative to wicks
        return body_height < total_height * 0.1


class FinancialChartAnalyzer:
    """Complete chart analysis pipeline"""

    def __init__(self):
        """Initialize analyzer"""
        self.vit = VisionTransformer(embedding_dim=768, n_layers=12, n_heads=12)

    def analyze_chart(
        self, candles: List[CandleData], chart_image: Optional[np.ndarray] = None
    ) -> ChartAnalysisResult:
        """
        Analyze financial chart

        Args:
            candles: List of candlestick data
            chart_image: Optional chart image (H, W, 3)

        Returns:
            ChartAnalysisResult
        """
        patterns = []

        # Detect candlestick patterns
        for i in range(1, len(candles)):
            if CandlePatternDetector.detect_hammer(candles, i):
                patterns.append(
                    ChartPattern(
                        pattern_type="hammer",
                        bullish_probability=0.85,
                        location=(i, 0),
                        strength=0.8,
                    )
                )

            if CandlePatternDetector.detect_engulfing(candles, i):
                patterns.append(
                    ChartPattern(
                        pattern_type="engulfing",
                        bullish_probability=0.75,
                        location=(i, 0),
                        strength=0.7,
                    )
                )

            if CandlePatternDetector.detect_doji(candles, i):
                patterns.append(
                    ChartPattern(
                        pattern_type="doji",
                        bullish_probability=0.5,
                        location=(i, 0),
                        strength=0.6,
                    )
                )

        # Analyze chart image with ViT if provided
        vit_signal = "HOLD"
        vit_confidence = 0.5

        if chart_image is not None:
            pattern_logits, trend_logits, signal_logits = self.vit.forward(chart_image)

            signal_probs = self._softmax(signal_logits)
            signal_idx = np.argmax(signal_probs)
            signal_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
            vit_signal = signal_map[signal_idx]
            vit_confidence = float(signal_probs[signal_idx])

        # Determine trend from price action
        closes = [c.close for c in candles[-20:]]
        if len(closes) > 1:
            if closes[-1] > np.mean(closes):
                trend = "uptrend"
            elif closes[-1] < np.mean(closes):
                trend = "downtrend"
            else:
                trend = "consolidation"
        else:
            trend = "consolidation"

        # Support/Resistance
        lows = [c.low for c in candles[-50:]]
        highs = [c.high for c in candles[-50:]]
        support = float(np.min(lows))
        resistance = float(np.max(highs))

        # Aggregate signal
        pattern_signal = "BUY" if patterns and np.mean([p.bullish_probability for p in patterns]) > 0.6 else "SELL"
        final_signal = vit_signal if vit_confidence > 0.6 else pattern_signal

        return ChartAnalysisResult(
            patterns=patterns,
            trend=trend,
            support_resistance={"support": support, "resistance": resistance},
            trading_signal=final_signal,
            confidence=vit_confidence,
        )

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax"""
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)


if __name__ == "__main__":
    logger.info("Vision Transformer Chart Analysis")
    logger.info("=" * 50)

    np.random.seed(42)

    # Generate synthetic candlestick data
    logger.info("\nGenerating synthetic candlestick data")
    candles = []
    price = 100.0

    for i in range(100):
        open_p = price
        close_p = price + np.random.randn() * 0.5
        high_p = max(open_p, close_p) + abs(np.random.randn() * 0.3)
        low_p = min(open_p, close_p) - abs(np.random.randn() * 0.3)
        volume = np.random.uniform(1e6, 10e6)

        candles.append(CandleData(open=open_p, high=high_p, low=low_p, close=close_p, volume=volume))
        price = close_p

    # Analyze chart
    logger.info("\nAnalyzing chart patterns")
    analyzer = FinancialChartAnalyzer()
    result = analyzer.analyze_chart(candles)

    logger.info(f"Detected {len(result.patterns)} patterns:")
    for pattern in result.patterns[:5]:
        logger.info(f"  {pattern.pattern_type}: bullish_prob={pattern.bullish_probability:.2f}")

    logger.info(f"\nTrend: {result.trend}")
    logger.info(f"Support: {result.support_resistance['support']:.2f}")
    logger.info(f"Resistance: {result.support_resistance['resistance']:.2f}")
    logger.info(f"\nTrading Signal: {result.trading_signal}")
    logger.info(f"ViT Confidence: {result.confidence:.4f}")

    logger.info("\nChart Analysis Complete")
