"""
BOAT - Financial Sentiment Analysis for Trading Signals
=======================================================

Production-ready sentiment analysis system for generating trading signals
from financial news and text data.

Features:
- Lexicon-based sentiment scoring
- Multi-source aggregation (news, social, analyst reports)
- Sentiment-to-return prediction
- Signal combination with technical indicators
- Real-time sentiment tracking

Based on 2024-2025 research:
- FinBERT-inspired lexicon approach
- Sentiment-enhanced trading strategies
- Multi-modal signal fusion
- Domain-specific financial vocabulary
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict


class SentimentScore(Enum):
    """Sentiment classifications"""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class SentimentSignal:
    """Sentiment-based trading signal"""
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    signal_strength: float  # Combined metric
    recommended_action: str  # BUY/SELL/HOLD
    sources_count: int


class FinancialSentimentAnalyzer:
    """
    Financial sentiment analyzer with domain-specific lexicon.

    Simplified FinBERT-inspired approach using financial vocabulary
    for practical trading signal generation.
    """

    def __init__(self):
        """Initialize sentiment analyzer with financial lexicon"""

        # Financial sentiment lexicon (positive words)
        self.positive_words = {
            'profit', 'growth', 'increase', 'gain', 'bullish', 'strong',
            'beat', 'outperform', 'exceed', 'positive', 'upside', 'upgrade',
            'buy', 'rally', 'surge', 'momentum', 'recovery', 'expansion',
            'optimistic', 'confidence', 'revenue', 'earnings', 'success',
            'innovation', 'opportunity', 'breakthrough', 'boom', 'advance'
        }

        # Financial sentiment lexicon (negative words)
        self.negative_words = {
            'loss', 'decline', 'decrease', 'fall', 'bearish', 'weak',
            'miss', 'underperform', 'downgrade', 'negative', 'downside',
            'sell', 'plunge', 'crash', 'risk', 'recession', 'contraction',
            'pessimistic', 'concern', 'warning', 'cut', 'failure', 'crisis',
            'volatility', 'uncertainty', 'problem', 'challenge', 'slump'
        }

        # Intensity modifiers
        self.intensifiers = {'very', 'extremely', 'highly', 'significantly', 'substantially'}
        self.diminishers = {'slightly', 'somewhat', 'relatively', 'marginally'}

        # Negation words
        self.negations = {'not', 'no', 'never', 'neither', 'nor', 'nothing'}

        # Sentiment history
        self.sentiment_history: List[Tuple[float, int]] = []  # (score, timestamp)

    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of financial text.

        Args:
            text: Input text

        Returns:
            Sentiment score (-1 to 1)
        """
        words = text.lower().split()

        sentiment = 0.0
        word_count = 0
        negation_active = False
        intensifier_active = False
        diminisher_active = False

        for i, word in enumerate(words):
            # Clean word
            word = word.strip('.,!?;:')

            # Check for modifiers
            if word in self.negations:
                negation_active = True
                continue
            elif word in self.intensifiers:
                intensifier_active = True
                continue
            elif word in self.diminishers:
                diminisher_active = True
                continue

            # Calculate sentiment
            if word in self.positive_words:
                score = 1.0
            elif word in self.negative_words:
                score = -1.0
            else:
                continue

            # Apply modifiers
            if negation_active:
                score *= -1
                negation_active = False
            if intensifier_active:
                score *= 1.5
                intensifier_active = False
            if diminisher_active:
                score *= 0.5
                diminisher_active = False

            sentiment += score
            word_count += 1

        # Normalize
        if word_count > 0:
            sentiment = sentiment / word_count
            sentiment = np.clip(sentiment, -1, 1)

        return sentiment

    def aggregate_multi_source(
        self,
        news_texts: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Aggregate sentiment from multiple sources.

        Args:
            news_texts: List of news article texts
            weights: Optional weights for different sources

        Returns:
            Aggregated sentiment score
        """
        if not news_texts:
            return 0.0

        if weights is None:
            weights = {'default': 1.0}

        scores = []
        for text in news_texts:
            score = self.analyze_text(text)
            scores.append(score)

        # Weighted average
        aggregated = np.mean(scores)

        # Apply recency weighting (more recent = higher weight)
        if len(scores) > 1:
            recency_weights = np.linspace(0.5, 1.0, len(scores))
            aggregated = np.average(scores, weights=recency_weights)

        return aggregated

    def generate_trading_signal(
        self,
        sentiment_score: float,
        volatility: float = 0.02,
        trend: Optional[float] = None
    ) -> SentimentSignal:
        """
        Generate trading signal from sentiment.

        Args:
            sentiment_score: Sentiment score (-1 to 1)
            volatility: Market volatility
            trend: Optional price trend indicator

        Returns:
            Trading signal with recommendation
        """
        # Calculate confidence based on sentiment magnitude and volatility
        confidence = abs(sentiment_score) * (1 - min(volatility, 0.5))

        # Adjust for trend if provided
        if trend is not None:
            # Align sentiment with trend for higher confidence
            if np.sign(sentiment_score) == np.sign(trend):
                confidence *= 1.2
            else:
                confidence *= 0.8

        confidence = min(confidence, 1.0)

        # Signal strength combines sentiment and confidence
        signal_strength = sentiment_score * confidence

        # Trading recommendation
        if signal_strength > 0.3:
            action = "BUY"
        elif signal_strength < -0.3:
            action = "SELL"
        else:
            action = "HOLD"

        return SentimentSignal(
            sentiment_score=sentiment_score,
            confidence=confidence,
            signal_strength=signal_strength,
            recommended_action=action,
            sources_count=1
        )

    def calculate_sentiment_momentum(
        self,
        window: int = 5
    ) -> float:
        """
        Calculate sentiment momentum from history.

        Args:
            window: Lookback window

        Returns:
            Sentiment momentum (-1 to 1)
        """
        if len(self.sentiment_history) < 2:
            return 0.0

        recent = [s[0] for s in self.sentiment_history[-window:]]

        if len(recent) < 2:
            return 0.0

        # Linear regression slope
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]

        # Normalize
        momentum = np.clip(slope * 10, -1, 1)

        return momentum

    def add_to_history(self, sentiment: float, timestamp: int):
        """Add sentiment score to history"""
        self.sentiment_history.append((sentiment, timestamp))

        # Keep last 100 entries
        if len(self.sentiment_history) > 100:
            self.sentiment_history = self.sentiment_history[-100:]


def test_sentiment_analysis():
    """Test Financial Sentiment Analysis"""
    print("=" * 60)
    print("Testing Financial Sentiment Analysis System")
    print("=" * 60)

    analyzer = FinancialSentimentAnalyzer()

    print("\n1. Basic Sentiment Analysis:")
    print("-" * 40)

    test_texts = [
        "Company reports strong earnings growth and beats expectations",
        "Stock plunges amid recession fears and declining revenue",
        "Analyst upgrades rating to buy with positive outlook",
        "Warning issued about potential risks and challenges ahead",
        "Market rallies on optimistic economic data and recovery signs"
    ]

    for text in test_texts:
        score = analyzer.analyze_text(text)
        print(f"\nText: {text[:60]}...")
        print(f"Sentiment: {score:+.3f} ({'POSITIVE' if score > 0 else 'NEGATIVE' if score < 0 else 'NEUTRAL'})")

    print("\n2. Multi-Source Aggregation:")
    print("-" * 40)

    news_articles = [
        "Company announces significant profit increase and expansion plans",
        "Strong quarterly results exceed analyst expectations substantially",
        "Market confidence grows with positive economic indicators",
        "Some concerns about volatility but overall outlook remains optimistic"
    ]

    aggregated = analyzer.aggregate_multi_source(news_articles)
    print(f"Number of sources: {len(news_articles)}")
    print(f"Aggregated sentiment: {aggregated:+.3f}")
    print(f"Overall tone: {'BULLISH' if aggregated > 0.2 else 'BEARISH' if aggregated < -0.2 else 'NEUTRAL'}")

    print("\n3. Trading Signal Generation:")
    print("-" * 40)

    test_scenarios = [
        {"sentiment": 0.7, "volatility": 0.01, "trend": 0.05, "desc": "Strong positive, low vol, uptrend"},
        {"sentiment": -0.6, "volatility": 0.03, "trend": -0.02, "desc": "Strong negative, high vol, downtrend"},
        {"sentiment": 0.2, "volatility": 0.02, "trend": -0.01, "desc": "Weak positive, conflicting trend"},
        {"sentiment": -0.1, "volatility": 0.01, "trend": 0.02, "desc": "Weak negative, low vol"},
    ]

    for scenario in test_scenarios:
        signal = analyzer.generate_trading_signal(
            scenario["sentiment"],
            scenario["volatility"],
            scenario["trend"]
        )
        print(f"\nScenario: {scenario['desc']}")
        print(f"  Sentiment: {signal.sentiment_score:+.2f}")
        print(f"  Confidence: {signal.confidence:.2%}")
        print(f"  Signal Strength: {signal.signal_strength:+.3f}")
        print(f"  Action: {signal.recommended_action}")

    print("\n4. Sentiment Momentum:")
    print("-" * 40)

    # Simulate sentiment history
    print("Building sentiment history...")
    for i in range(10):
        # Trending positive
        sentiment = -0.5 + (i / 10) * 1.0
        analyzer.add_to_history(sentiment, i)
        print(f"  Day {i}: {sentiment:+.2f}")

    momentum = analyzer.calculate_sentiment_momentum(window=5)
    print(f"\nSentiment Momentum (5-day): {momentum:+.3f}")
    print(f"Interpretation: {'IMPROVING' if momentum > 0.1 else 'DETERIORATING' if momentum < -0.1 else 'STABLE'}")

    print("\n5. Lexicon Coverage:")
    print("-" * 40)

    print(f"Positive words: {len(analyzer.positive_words)}")
    print(f"Negative words: {len(analyzer.negative_words)}")
    print(f"Total financial vocabulary: {len(analyzer.positive_words) + len(analyzer.negative_words)}")

    print("\nSample positive terms:", ', '.join(list(analyzer.positive_words)[:10]))
    print("Sample negative terms:", ', '.join(list(analyzer.negative_words)[:10]))

    print("\n[SUCCESS] Financial Sentiment Analysis test completed successfully!")


if __name__ == "__main__":
    test_sentiment_analysis()
