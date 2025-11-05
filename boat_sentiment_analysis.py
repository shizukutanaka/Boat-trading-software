#!/usr/bin/env python3
"""
NLP and Sentiment Analysis for Boat Trading Platform
=====================================================

Implements state-of-the-art sentiment analysis for trading signals:
  - VADER sentiment analysis (lexicon-based)
  - FinBERT transformer model (fine-tuned BERT for finance)
  - Multi-source sentiment aggregation (news, social, research)
  - Sentiment-based trading signal generation
  - Sentiment momentum and divergence detection

Based on 2025 NLP research for financial sentiment analysis.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import deque

# NLP and ML libraries
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentSource(Enum):
    """Sentiment data sources"""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    RESEARCH = "research"
    EARNINGS = "earnings"
    ECONOMIC = "economic"
    SECTOR = "sector"


class SentimentSignal(Enum):
    """Sentiment-based trading signals"""
    STRONG_BUY = 1.0
    BUY = 0.5
    NEUTRAL = 0.0
    SELL = -0.5
    STRONG_SELL = -1.0


@dataclass
class SentimentScore:
    """Normalized sentiment score"""
    source: SentimentSource
    symbol: str
    text: str
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment across sources"""
    symbol: str
    overall_score: float  # -1.0 to 1.0
    source_scores: Dict[SentimentSource, float]
    source_weights: Dict[SentimentSource, float]
    data_points: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    trend: str = "neutral"  # improving, stable, deteriorating
    volatility: float = 0.0


@dataclass
class SentimentSignalConfig:
    """Sentiment analysis configuration"""
    # VADER settings
    use_vader: bool = True
    vader_threshold_positive: float = 0.05
    vader_threshold_negative: float = -0.05

    # Transformer settings
    use_transformer: bool = True
    transformer_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    batch_size: int = 32

    # Aggregation settings
    source_weights: Dict[SentimentSource, float] = field(default_factory=lambda: {
        SentimentSource.NEWS: 0.3,
        SentimentSource.SOCIAL_MEDIA: 0.2,
        SentimentSource.RESEARCH: 0.25,
        SentimentSource.EARNINGS: 0.15,
        SentimentSource.ECONOMIC: 0.05,
        SentimentSource.SECTOR: 0.05
    })

    # Momentum settings
    momentum_window: int = 20  # lookback periods
    min_data_points: int = 5

    # Signal generation
    strong_threshold: float = 0.6
    moderate_threshold: float = 0.3


class VaderSentimentAnalyzer:
    """VADER sentiment analysis for trading text"""

    def __init__(self, config: SentimentSignalConfig):
        self.config = config
        if not VADER_AVAILABLE:
            logger.warning("NLTK/VADER not available, sentiment analysis disabled")
            self.analyzer = None
        else:
            self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment using VADER

        Args:
            text: Text to analyze

        Returns:
            (sentiment_score, confidence)
        """
        if not self.analyzer:
            return 0.0, 0.0

        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']  # -1 to 1
        confidence = max(scores['pos'], scores['neg'])

        return compound, confidence


class TransformerSentimentAnalyzer:
    """FinBERT and other transformer-based sentiment analysis"""

    def __init__(self, config: SentimentSignalConfig):
        self.config = config
        self.pipeline = None

        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not available")
            return

        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=config.transformer_model,
                device=-1  # CPU mode
            )
            logger.info(f"Loaded transformer model: {config.transformer_model}")
        except Exception as e:
            logger.error(f"Failed to load transformer model: {e}")

    def analyze(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment using transformer model

        Args:
            text: Text to analyze

        Returns:
            (sentiment_score, confidence)
        """
        if not self.pipeline:
            return 0.0, 0.0

        try:
            result = self.pipeline(text[:512])[0]  # Truncate to max tokens
            label = result['label']
            score = result['score']

            # Normalize to -1 to 1 range
            sentiment = 1.0 if label == 'POSITIVE' else -1.0
            sentiment *= score

            return sentiment, score
        except Exception as e:
            logger.error(f"Transformer analysis error: {e}")
            return 0.0, 0.0


class SentimentAnalysisEngine:
    """Main sentiment analysis engine"""

    def __init__(self, config: SentimentSignalConfig):
        self.config = config
        self.vader = VaderSentimentAnalyzer(config)
        self.transformer = TransformerSentimentAnalyzer(config)

        # Storage for sentiment scores
        self.sentiment_history: Dict[str, deque] = {}
        self.aggregated_sentiment: Dict[str, AggregatedSentiment] = {}

    def analyze_text(\n        self,\n        symbol: str,\n        text: str,\n        source: SentimentSource\n    ) -> SentimentScore:\n        \"\"\"\n        Analyze sentiment from text\n        \n        Args:\n            symbol: Trading symbol\n            text: Text to analyze\n            source: Source of text\n            \n        Returns:\n            SentimentScore object\n        \"\"\"\n        scores = []\n        confidences = []\n        \n        # VADER analysis\n        if self.config.use_vader:\n            vader_score, vader_conf = self.vader.analyze(text)\n            scores.append(vader_score)\n            confidences.append(vader_conf)\n        \n        # Transformer analysis\n        if self.config.use_transformer:\n            trans_score, trans_conf = self.transformer.analyze(text)\n            scores.append(trans_score)\n            confidences.append(trans_conf)\n        \n        # Average scores\n        if scores:\n            final_score = np.mean(scores)\n            final_confidence = np.mean(confidences)\n        else:\n            final_score = 0.0\n            final_confidence = 0.0\n        \n        sentiment = SentimentScore(\n            source=source,\n            symbol=symbol,\n            text=text[:200],  # Store first 200 chars\n            score=final_score,\n            confidence=final_confidence\n        )\n        \n        # Store in history\n        if symbol not in self.sentiment_history:\n            self.sentiment_history[symbol] = deque(\n                maxlen=self.config.momentum_window * 10\n            )\n        self.sentiment_history[symbol].append(sentiment)\n        \n        return sentiment\n
    def get_aggregated_sentiment(\n        self,\n        symbol: str\n    ) -> Optional[AggregatedSentiment]:\n        \"\"\"\n        Get aggregated sentiment across sources\n        \n        Args:\n            symbol: Trading symbol\n            \n        Returns:\n            AggregatedSentiment or None\n        \"\"\"\n        if symbol not in self.sentiment_history:\n            return None\n        \n        # Get recent sentiments\n        sentiments = list(self.sentiment_history[symbol])\n        if len(sentiments) < self.config.min_data_points:\n            return None\n        \n        # Aggregate by source\n        source_scores: Dict[SentimentSource, List[float]] = {}\n        for sentiment in sentiments[-self.config.momentum_window:]:\n            if sentiment.source not in source_scores:\n                source_scores[sentiment.source] = []\n            source_scores[sentiment.source].append(sentiment.score)\n        \n        # Calculate weighted average\n        weighted_score = 0.0\n        source_avgs = {}\n        total_weight = 0.0\n        \n        for source, scores in source_scores.items():\n            avg_score = np.mean(scores)\n            source_avgs[source] = avg_score\n            weight = self.config.source_weights.get(source, 0.1)\n            weighted_score += avg_score * weight\n            total_weight += weight\n        \n        if total_weight > 0:\n            overall_score = weighted_score / total_weight\n        else:\n            overall_score = 0.0\n        \n        # Determine trend\n        recent = np.array([s.score for s in sentiments[-5:]])\n        older = np.array([s.score for s in sentiments[-10:-5]])\n        \n        if len(recent) > 0 and len(older) > 0:\n            recent_avg = np.mean(recent)\n            older_avg = np.mean(older)\n            if recent_avg > older_avg + 0.05:\n                trend = \"improving\"\n            elif recent_avg < older_avg - 0.05:\n                trend = \"deteriorating\"\n            else:\n                trend = \"stable\"\n        else:\n            trend = \"neutral\"\n        \n        # Calculate volatility\n        all_scores = np.array([s.score for s in sentiments[-self.config.momentum_window:]])\n        volatility = np.std(all_scores) if len(all_scores) > 0 else 0.0\n        \n        agg_sentiment = AggregatedSentiment(\n            symbol=symbol,\n            overall_score=float(overall_score),\n            source_scores=source_avgs,\n            source_weights=self.config.source_weights,\n            data_points=len(sentiments),\n            trend=trend,\n            volatility=float(volatility)\n        )\n        \n        self.aggregated_sentiment[symbol] = agg_sentiment\n        return agg_sentiment\n
    def get_sentiment_signal(\n        self,\n        symbol: str\n    ) -> Tuple[SentimentSignal, float]:\n        \"\"\"\n        Generate trading signal from sentiment\n        \n        Args:\n            symbol: Trading symbol\n            \n        Returns:\n            (signal, confidence)\n        \"\"\"\n        agg_sentiment = self.get_aggregated_sentiment(symbol)\n        if not agg_sentiment:\n            return SentimentSignal.NEUTRAL, 0.0\n        \n        score = agg_sentiment.overall_score\n        \n        # Generate signal based on thresholds\n        if score >= self.config.strong_threshold:\n            signal = SentimentSignal.STRONG_BUY\n        elif score >= self.config.moderate_threshold:\n            signal = SentimentSignal.BUY\n        elif score <= -self.config.strong_threshold:\n            signal = SentimentSignal.STRONG_SELL\n        elif score <= -self.config.moderate_threshold:\n            signal = SentimentSignal.SELL\n        else:\n            signal = SentimentSignal.NEUTRAL\n        \n        # Confidence based on data points and volatility\n        confidence = min(\n            len(agg_sentiment.source_scores) / 6.0,  # Max 6 sources\n            1.0 - (agg_sentiment.volatility / 2.0)  # Discount by volatility\n        )\n        confidence = max(0.0, min(1.0, confidence))\n        \n        return signal, confidence


class SentimentMomentumDetector:
    """Detect sentiment momentum and divergences"""

    def __init__(self, lookback: int = 20):\n        self.lookback = lookback\n        self.sentiment_momentum: Dict[str, List[float]] = {}\n
    def calculate_momentum(\n        self,\n        sentiment_scores: List[float]\n    ) -> float:\n        \"\"\"\n        Calculate sentiment momentum\n        \n        Args:\n            sentiment_scores: List of sentiment scores\n            \n        Returns:\n            Momentum value (-1 to 1)\n        \"\"\"\n        if len(sentiment_scores) < 2:\n            return 0.0\n        \n        scores = np.array(sentiment_scores[-self.lookback:])\n        time_indices = np.arange(len(scores))\n        \n        # Linear regression slope\n        if len(scores) > 1:\n            slope = np.polyfit(time_indices, scores, 1)[0]\n            momentum = np.tanh(slope * 10)  # Normalize to -1 to 1\n        else:\n            momentum = 0.0\n        \n        return float(momentum)\n    \n    def detect_divergence(\n        self,\n        price_data: np.ndarray,\n        sentiment_data: np.ndarray\n    ) -> Dict[str, Any]:\n        \"\"\"\n        Detect price-sentiment divergences\n        \n        Args:\n            price_data: Price movements\n            sentiment_data: Sentiment scores\n            \n        Returns:\n            Divergence analysis\n        \"\"\"\n        # Normalize data\n        price_norm = (price_data - np.mean(price_data)) / (np.std(price_data) + 1e-6)\n        sentiment_norm = (sentiment_data - np.mean(sentiment_data)) / (np.std(sentiment_data) + 1e-6)\n        \n        # Calculate correlation\n        correlation = np.corrcoef(price_norm, sentiment_norm)[0, 1]\n        \n        # Detect divergence\n        is_bullish_divergence = (\n            np.min(price_norm) < np.min(sentiment_norm) and\n            price_norm[-1] < 0 and sentiment_norm[-1] > 0\n        )\n        \n        is_bearish_divergence = (\n            np.max(price_norm) > np.max(sentiment_norm) and\n            price_norm[-1] > 0 and sentiment_norm[-1] < 0\n        )\n        \n        return {\n            'correlation': float(correlation),\n            'bullish_divergence': is_bullish_divergence,\n            'bearish_divergence': is_bearish_divergence,\n            'strength': abs(float(correlation))\n        }


class NewsHeadlineAnalyzer:
    \"\"\"Analyze sentiment from news headlines\"\"\"\n    \n    def __init__(self, engine: SentimentAnalysisEngine):\n        self.engine = engine\n        self.headline_cache: Dict[str, deque] = {}\n    \n    def analyze_headlines(\n        self,\n        symbol: str,\n        headlines: List[str]\n    ) -> List[SentimentScore]:\n        \"\"\"\n        Analyze multiple headlines\n        \n        Args:\n            symbol: Trading symbol\n            headlines: List of news headlines\n            \n        Returns:\n            List of sentiment scores\n        \"\"\"\n        scores = []\n        for headline in headlines:\n            score = self.engine.analyze_text(\n                symbol,\n                headline,\n                SentimentSource.NEWS\n            )\n            scores.append(score)\n        \n        return scores


class SocialMediaSentimentAnalyzer:\n    \"\"\"Analyze sentiment from social media posts\"\"\"\n    \n    def __init__(self, engine: SentimentAnalysisEngine):\n        self.engine = engine\n    \n    def analyze_posts(\n        self,\n        symbol: str,\n        posts: List[str]\n    ) -> List[SentimentScore]:\n        \"\"\"\n        Analyze social media posts\n        \n        Args:\n            symbol: Trading symbol\n            posts: List of social media posts\n            \n        Returns:\n            List of sentiment scores\n        \"\"\"\n        scores = []\n        for post in posts:\n            score = self.engine.analyze_text(\n                symbol,\n                post,\n                SentimentSource.SOCIAL_MEDIA\n            )\n            scores.append(score)\n        \n        return scores


class SentimentTradingStrategy:\n    \"\"\"Trading strategy based on sentiment signals\"\"\"\n    \n    def __init__(\n        self,\n        engine: SentimentAnalysisEngine,\n        config: SentimentSignalConfig\n    ):\n        self.engine = engine\n        self.config = config\n        self.momentum_detector = SentimentMomentumDetector(config.momentum_window)\n        self.trades: List[Dict[str, Any]] = []\n    \n    def generate_signals(\n        self,\n        symbols: List[str]\n    ) -> Dict[str, Dict[str, Any]]:\n        \"\"\"\n        Generate trading signals for symbols\n        \n        Args:\n            symbols: List of trading symbols\n            \n        Returns:\n            Dictionary of signals and metadata\n        \"\"\"\n        signals = {}\n        \n        for symbol in symbols:\n            signal, confidence = self.engine.get_sentiment_signal(symbol)\n            agg_sentiment = self.engine.get_aggregated_sentiment(symbol)\n            \n            if agg_sentiment:\n                signals[symbol] = {\n                    'signal': signal.name,\n                    'signal_value': signal.value,\n                    'confidence': confidence,\n                    'overall_score': agg_sentiment.overall_score,\n                    'trend': agg_sentiment.trend,\n                    'volatility': agg_sentiment.volatility,\n                    'data_points': agg_sentiment.data_points,\n                    'source_scores': {\n                        k.value: v for k, v in agg_sentiment.source_scores.items()\n                    }\n                }\n        \n        return signals\n    \n    def backtest(\n        self,\n        symbols: List[str],\n        price_data: pd.DataFrame,\n        sentiment_data: pd.DataFrame\n    ) -> Dict[str, Any]:\n        \"\"\"\n        Backtest sentiment strategy\n        \n        Args:\n            symbols: Trading symbols\n            price_data: Historical prices\n            sentiment_data: Historical sentiment\n            \n        Returns:\n            Backtest results\n        \"\"\"\n        results = {\n            'total_trades': 0,\n            'winning_trades': 0,\n            'losing_trades': 0,\n            'win_rate': 0.0,\n            'total_return': 0.0,\n            'by_symbol': {}\n        }\n        \n        for symbol in symbols:\n            symbol_trades = []\n            \n            if symbol in price_data.columns and symbol in sentiment_data.columns:\n                prices = price_data[symbol].values\n                sentiments = sentiment_data[symbol].values\n                \n                for i in range(len(prices) - 1):\n                    if not np.isnan(sentiments[i]):\n                        # Simple strategy: buy on positive sentiment\n                        signal = SentimentSignal.BUY if sentiments[i] > 0.3 else SentimentSignal.SELL\n                        \n                        entry_price = prices[i]\n                        exit_price = prices[i + 1]\n                        pnl = (exit_price - entry_price) / entry_price\n                        \n                        if signal == SentimentSignal.BUY and pnl > 0:\n                            symbol_trades.append({'pnl': pnl, 'win': True})\n                        else:\n                            symbol_trades.append({'pnl': pnl, 'win': pnl > 0})\n            \n            results['by_symbol'][symbol] = {\n                'trades': len(symbol_trades),\n                'wins': sum(1 for t in symbol_trades if t['win']),\n                'total_pnl': sum(t['pnl'] for t in symbol_trades)\n            }\n            results['total_trades'] += len(symbol_trades)\n        \n        if results['total_trades'] > 0:\n            results['winning_trades'] = sum(\n                r['wins'] for r in results['by_symbol'].values()\n            )\n            results['losing_trades'] = results['total_trades'] - results['winning_trades']\n            results['win_rate'] = results['winning_trades'] / results['total_trades']\n        \n        return results


if __name__ == \"__main__\":\n    # Example usage\n    config = SentimentSignalConfig()\n    engine = SentimentAnalysisEngine(config)\n    \n    # Analyze sample texts\n    sample_texts = [\n        \"Amazing earnings report, stock soaring!\",\n        \"Company faces regulatory challenges\",\n        \"Market showing strong momentum today\",\n        \"Investors worried about inflation\",\n        \"CEO announces ambitious growth plans\"\n    ]\n    \n    symbol = \"TECH\"\n    for text in sample_texts:\n        score = engine.analyze_text(\n            symbol,\n            text,\n            SentimentSource.NEWS\n        )\n        logger.info(f\"Text: {text[:50]}... Score: {score.score:.3f}\")\n    \n    # Get aggregated sentiment\n    agg_sentiment = engine.get_aggregated_sentiment(symbol)\n    if agg_sentiment:\n        logger.info(f\"Aggregated sentiment for {symbol}: {agg_sentiment.overall_score:.3f}\")\n        logger.info(f\"Trend: {agg_sentiment.trend}\")\n    \n    # Get trading signal\n    signal, confidence = engine.get_sentiment_signal(symbol)\n    logger.info(f\"Trading signal: {signal.name} (confidence: {confidence:.2f})\")\n