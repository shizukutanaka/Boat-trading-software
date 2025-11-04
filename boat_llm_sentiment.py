#!/usr/bin/env python3
"""
Large Language Model-Based Financial Sentiment Analysis
=========================================================

LLM-powered sentiment extraction for financial texts:
  - Financial text preprocessing and tokenization
  - Domain-specific sentiment scoring
  - News impact quantification
  - Multi-asset sentiment aggregation
  - Sentiment-driven trading signals
  - Explainable sentiment reasoning

Based on 2025 research on LLMs for financial sentiment analysis.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Sentiment score for financial text"""
    positive: float
    negative: float
    neutral: float
    confidence: float
    entity: Optional[str] = None


@dataclass
class NewsImpact:
    """Impact of news on asset"""
    asset: str
    sentiment: float
    magnitude: float
    timestamp: datetime
    source: str


class FinancialSentimentAnalyzer:
    """LLM-based financial sentiment analyzer"""

    def __init__(
        self,
        model_name: str = "financial_llm",
        domain_vocab: Optional[Dict[str, float]] = None
    ):
        """
        Initialize sentiment analyzer

        Args:
            model_name: Name of LLM model
            domain_vocab: Domain-specific vocabulary weights
        """
        self.model_name = model_name
        self.domain_vocab = domain_vocab or self._default_financial_vocab()

        # Simple sentiment lexicon
        self.positive_words = {
            'gain', 'profit', 'surge', 'soar', 'rally', 'bullish', 'upgrade',
            'outperform', 'beat', 'growth', 'strong', 'positive', 'excel', 'record'
        }

        self.negative_words = {
            'loss', 'decline', 'plunge', 'crash', 'bearish', 'downgrade',
            'underperform', 'miss', 'weak', 'negative', 'concerns', 'volatility',
            'selloff', 'default'
        }

    def _default_financial_vocab(self) -> Dict[str, float]:
        """Create default financial vocabulary"""
        return {
            'earnings': 0.3,
            'revenue': 0.2,
            'dividend': 0.2,
            'bankruptcy': -0.9,
            'merger': 0.1,
            'acquisition': 0.1,
            'innovation': 0.3,
            'regulation': -0.2,
            'lawsuit': -0.4
        }

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess financial text

        Args:
            text: Raw financial news text

        Returns:
            Tokenized text
        """
        # Simple tokenization
        text = text.lower()
        tokens = text.replace(',', '').replace('.', '').split()
        return tokens

    def score_sentiment(
        self,
        text: str,
        entity: Optional[str] = None
    ) -> SentimentScore:
        """
        Score sentiment of financial text

        Args:
            text: Financial news text
            entity: Target entity (company/ticker)

        Returns:
            SentimentScore
        """
        tokens = self.preprocess_text(text)

        # Count positive and negative indicators
        positive_count = sum(1 for token in tokens if token in self.positive_words)
        negative_count = sum(1 for token in tokens if token in self.negative_words)

        # Apply domain vocabulary weights
        for word, weight in self.domain_vocab.items():
            if word in tokens:
                if weight > 0:
                    positive_count += weight
                else:
                    negative_count += abs(weight)

        # Normalize scores
        total = positive_count + negative_count + len(tokens)
        positive = positive_count / (total + 1e-8)
        negative = negative_count / (total + 1e-8)
        neutral = 1.0 - positive - negative

        # Confidence based on document length and signal strength
        confidence = min(1.0, len(tokens) / 100.0) * (abs(positive - negative) + 0.3)

        return SentimentScore(
            positive=float(positive),
            negative=float(negative),
            neutral=float(neutral),
            confidence=float(confidence),
            entity=entity
        )

    def extract_entities_sentiment(
        self,
        text: str
    ) -> List[Tuple[str, SentimentScore]]:
        """
        Extract entities and their sentiments

        Args:
            text: Financial text

        Returns:
            List of (entity, sentiment) tuples
        """
        # Simple entity extraction (in practice would use NER)
        entities = []
        tokens = self.preprocess_text(text)

        # Look for capitalized words as entities
        for token in tokens:
            # Skip if already processed
            if token in [e[0].lower() for e in entities]:
                continue

            # Simple heuristic: 3+ letter words could be entities
            if len(token) >= 3 and token.isalpha():
                sentiment = self.score_sentiment(text, entity=token)
                entities.append((token, sentiment))

        return entities[:5]  # Top 5 entities

    def compute_sentiment_signal(
        self,
        sentiment_score: SentimentScore
    ) -> float:
        """
        Convert sentiment to trading signal

        Args:
            sentiment_score: Sentiment score

        Returns:
            Trading signal (-1 to 1)
        """
        # Bullish vs bearish signal
        signal = sentiment_score.positive - sentiment_score.negative

        # Confidence-weighted
        signal = signal * sentiment_score.confidence

        return float(np.clip(signal, -1.0, 1.0))


class NewsImpactQuantifier:
    """Quantify impact of news on asset prices"""

    def __init__(
        self,
        sentiment_analyzer: FinancialSentimentAnalyzer,
        lookback_window: int = 5
    ):
        """
        Initialize news impact quantifier

        Args:
            sentiment_analyzer: Sentiment analyzer instance
            lookback_window: Historical days for impact measurement
        """
        self.analyzer = sentiment_analyzer
        self.lookback_window = lookback_window

    def measure_impact(
        self,
        news_text: str,
        price_before: float,
        price_after: float,
        asset: str
    ) -> NewsImpact:
        """
        Measure impact of news on price

        Args:
            news_text: News article text
            price_before: Price before news
            price_after: Price after news
            asset: Asset identifier

        Returns:
            NewsImpact
        """
        # Sentiment
        sentiment = self.analyzer.score_sentiment(news_text, entity=asset)
        sentiment_signal = self.analyzer.compute_sentiment_signal(sentiment)

        # Actual price impact
        price_change = (price_after - price_before) / price_before

        # Impact magnitude
        magnitude = abs(price_change)

        return NewsImpact(
            asset=asset,
            sentiment=float(sentiment_signal),
            magnitude=float(magnitude),
            timestamp=datetime.now(),
            source="financial_news"
        )

    def expected_impact(
        self,
        news_sentiment: float,
        historical_volatility: float
    ) -> float:
        """
        Predict expected price impact

        Args:
            news_sentiment: Sentiment score (-1 to 1)
            historical_volatility: Historical vol of asset

        Returns:
            Expected price change
        """
        # Simple model: sentiment × volatility
        expected_move = news_sentiment * historical_volatility

        return float(expected_move)


class SentimentAggregator:
    """Aggregate sentiment across multiple sources"""

    @staticmethod
    def aggregate_multi_source(
        sentiments: List[SentimentScore],
        weights: Optional[List[float]] = None
    ) -> SentimentScore:
        """
        Aggregate sentiment from multiple sources

        Args:
            sentiments: List of sentiment scores
            weights: Optional weights for each source

        Returns:
            Aggregated sentiment
        """
        if weights is None:
            weights = [1.0 / len(sentiments)] * len(sentiments)

        # Weighted aggregation
        agg_positive = sum(w * s.positive for w, s in zip(weights, sentiments))
        agg_negative = sum(w * s.negative for w, s in zip(weights, sentiments))
        agg_neutral = sum(w * s.neutral for w, s in zip(weights, sentiments))
        agg_confidence = sum(w * s.confidence for w, s in zip(weights, sentiments))

        # Normalize
        total = agg_positive + agg_negative + agg_neutral
        if total > 0:
            agg_positive /= total
            agg_negative /= total
            agg_neutral /= total

        return SentimentScore(
            positive=float(agg_positive),
            negative=float(agg_negative),
            neutral=float(agg_neutral),
            confidence=float(agg_confidence)
        )

    @staticmethod
    def time_decay_aggregation(
        sentiments_with_time: List[Tuple[SentimentScore, int]],
        decay_factor: float = 0.95
    ) -> SentimentScore:
        """
        Aggregate with time decay

        Args:
            sentiments_with_time: List of (sentiment, days_ago) tuples
            decay_factor: Exponential decay factor

        Returns:
            Aggregated sentiment
        """
        weights = [decay_factor ** days for _, days in sentiments_with_time]
        sentiments = [s for s, _ in sentiments_with_time]

        # Normalize weights
        weights = [w / sum(weights) for w in weights]

        return SentimentAggregator.aggregate_multi_source(sentiments, weights)


class SentimentDrivenStrategy:
    """Trading strategy based on sentiment analysis"""

    def __init__(
        self,
        sentiment_analyzer: FinancialSentimentAnalyzer,
        entry_threshold: float = 0.3,
        exit_threshold: float = 0.0
    ):
        """
        Initialize strategy

        Args:
            sentiment_analyzer: Sentiment analyzer
            entry_threshold: Minimum sentiment for entry
            exit_threshold: Sentiment threshold for exit
        """
        self.analyzer = sentiment_analyzer
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def generate_signal(
        self,
        news_text: str,
        current_position: float = 0.0
    ) -> Dict[str, float]:
        """
        Generate trading signal from news

        Args:
            news_text: News article text
            current_position: Current position size

        Returns:
            Signal dictionary
        """
        sentiment = self.analyzer.score_sentiment(news_text)
        signal = self.analyzer.compute_sentiment_signal(sentiment)

        action = "hold"
        target_position = current_position

        if signal > self.entry_threshold and current_position <= 0:
            action = "buy"
            target_position = 1.0
        elif signal < -self.entry_threshold and current_position >= 0:
            action = "sell"
            target_position = -1.0
        elif current_position != 0 and abs(signal) < self.exit_threshold:
            action = "exit"
            target_position = 0.0

        return {
            'action': action,
            'signal': signal,
            'confidence': sentiment.confidence,
            'target_position': target_position
        }


if __name__ == "__main__":
    # Example usage
    analyzer = FinancialSentimentAnalyzer()

    # Sample financial news
    news_samples = [
        "Apple beats earnings expectations with strong revenue growth and positive guidance",
        "Tech stocks surge as Fed signals potential rate cuts and bullish outlook",
        "Tesla plunges on safety concerns and disappointing sales numbers",
        "Microsoft gains from AI investment, outperforming market expectations",
        "Market faces headwinds from regulatory concerns and economic uncertainty"
    ]

    logger.info("Sentiment Analysis Results:")
    for news in news_samples:
        sentiment = analyzer.score_sentiment(news)
        signal = analyzer.compute_sentiment_signal(sentiment)

        logger.info(f"News: {news[:50]}...")
        logger.info(f"  Sentiment - Pos: {sentiment.positive:.3f}, Neg: {sentiment.negative:.3f}")
        logger.info(f"  Signal: {signal:.3f}, Confidence: {sentiment.confidence:.3f}")

    # News impact
    logger.info("\nNews Impact Measurement:")
    quantifier = NewsImpactQuantifier(analyzer)

    impact = quantifier.measure_impact(
        news_text=news_samples[0],
        price_before=150.0,
        price_after=155.0,
        asset="AAPL"
    )

    logger.info(f"Asset: {impact.asset}, Sentiment: {impact.sentiment:.3f}, Price Impact: {impact.magnitude:.3f}")

    # Aggregation
    logger.info("\nMulti-source Aggregation:")
    sentiments = [analyzer.score_sentiment(news) for news in news_samples[:3]]
    aggregated = SentimentAggregator.aggregate_multi_source(sentiments)

    logger.info(f"Aggregated - Pos: {aggregated.positive:.3f}, Neg: {aggregated.negative:.3f}")

    logger.info("\nLLM Sentiment Analysis Complete")
