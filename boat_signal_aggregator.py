"""
BOAT - Multi-Signal Aggregator and Decision System
==================================================

Production-ready signal aggregation system for autonomous trading decisions.

Features:
- Multi-indicator signal fusion
- Confidence-weighted decision making
- Risk-adjusted position sizing
- Real-time signal monitoring
- Conflict resolution between signals
- Performance tracking

Based on 2025 research:
- Multi-factor investment models
- Signal combination techniques
- Ensemble decision systems
- Risk-aware position sizing

Design Philosophy (Carmack/Martin/Pike):
- Simple voting and weighting
- Clear decision logic
- No black boxes
- Practical risk management
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class DecisionType(Enum):
    """Trading decision types"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class SignalSource(Enum):
    """Signal source types"""
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    STRATEGY = "strategy"
    FUNDAMENTAL = "fundamental"


@dataclass
class InputSignal:
    """Input signal from any source"""
    source: SignalSource
    source_name: str
    symbol: str
    direction: str  # buy/sell/hold
    strength: float  # 0-1
    confidence: float  # 0-1
    timestamp: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedDecision:
    """Aggregated trading decision"""
    symbol: str
    decision: DecisionType
    confidence: float  # 0-1
    position_size: float  # Fraction of capital
    buy_signals: int
    sell_signals: int
    hold_signals: int
    contributing_signals: List[InputSignal]
    agreement_score: float
    risk_score: float
    timestamp: int


@dataclass
class DecisionMetrics:
    """Decision quality metrics"""
    total_decisions: int
    strong_buy_count: int
    buy_count: int
    hold_count: int
    sell_count: int
    strong_sell_count: int
    avg_confidence: float
    avg_agreement: float
    high_confidence_decisions: int


class SignalAggregator:
    """
    Multi-signal aggregator for trading decisions.

    Combines signals from multiple sources with confidence weighting
    and conflict resolution.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        min_agreement: float = 0.6,
        max_position_size: float = 0.2
    ):
        """
        Initialize signal aggregator.

        Args:
            min_confidence: Minimum confidence for action
            min_agreement: Minimum agreement ratio for strong signals
            max_position_size: Maximum position size as fraction
        """
        self.min_confidence = min_confidence
        self.min_agreement = min_agreement
        self.max_position_size = max_position_size

        # Signal weights by source
        self.source_weights = {
            SignalSource.TECHNICAL: 1.0,
            SignalSource.SENTIMENT: 0.8,
            SignalSource.STRATEGY: 1.2,
            SignalSource.FUNDAMENTAL: 1.0
        }

        # Decision history
        self.decision_history: List[AggregatedDecision] = []

    def set_source_weight(self, source: SignalSource, weight: float):
        """Set weight for signal source"""
        self.source_weights[source] = weight

    def aggregate_signals(
        self,
        signals: List[InputSignal],
        current_price: float,
        volatility: float = 0.02
    ) -> AggregatedDecision:
        """
        Aggregate multiple signals into trading decision.

        Args:
            signals: List of input signals
            current_price: Current asset price
            volatility: Price volatility estimate

        Returns:
            Aggregated trading decision
        """
        if not signals:
            return self._create_hold_decision(signals[0].symbol if signals else "UNKNOWN", 0)

        symbol = signals[0].symbol

        # Count votes by direction
        buy_signals = []
        sell_signals = []
        hold_signals = []

        for signal in signals:
            direction = signal.direction.lower()
            if direction in ['buy', 'long']:
                buy_signals.append(signal)
            elif direction in ['sell', 'short']:
                sell_signals.append(signal)
            else:
                hold_signals.append(signal)

        # Calculate weighted votes
        buy_weight = self._calculate_weighted_vote(buy_signals)
        sell_weight = self._calculate_weighted_vote(sell_signals)
        hold_weight = self._calculate_weighted_vote(hold_signals)

        total_weight = buy_weight + sell_weight + hold_weight

        # Determine decision
        if total_weight == 0:
            return self._create_hold_decision(symbol, 0)

        buy_ratio = buy_weight / total_weight
        sell_ratio = sell_weight / total_weight

        # Calculate confidence and agreement
        max_ratio = max(buy_ratio, sell_ratio, hold_weight / total_weight)
        confidence = self._calculate_confidence(signals, max_ratio)
        agreement = max_ratio

        # Make decision
        if buy_ratio > sell_ratio and buy_ratio > hold_weight / total_weight:
            # Buy decision
            if buy_ratio >= self.min_agreement and confidence >= self.min_confidence:
                decision = DecisionType.STRONG_BUY
            elif confidence >= self.min_confidence:
                decision = DecisionType.BUY
            else:
                decision = DecisionType.HOLD
        elif sell_ratio > buy_ratio and sell_ratio > hold_weight / total_weight:
            # Sell decision
            if sell_ratio >= self.min_agreement and confidence >= self.min_confidence:
                decision = DecisionType.STRONG_SELL
            elif confidence >= self.min_confidence:
                decision = DecisionType.SELL
            else:
                decision = DecisionType.HOLD
        else:
            decision = DecisionType.HOLD

        # Calculate risk-adjusted position size
        risk_score = self._calculate_risk_score(volatility, agreement, confidence)
        position_size = self._calculate_position_size(
            decision, confidence, risk_score
        )

        aggregated = AggregatedDecision(
            symbol=symbol,
            decision=decision,
            confidence=confidence,
            position_size=position_size,
            buy_signals=len(buy_signals),
            sell_signals=len(sell_signals),
            hold_signals=len(hold_signals),
            contributing_signals=signals,
            agreement_score=agreement,
            risk_score=risk_score,
            timestamp=signals[0].timestamp if signals else 0
        )

        self.decision_history.append(aggregated)
        return aggregated

    def _calculate_weighted_vote(self, signals: List[InputSignal]) -> float:
        """Calculate weighted vote for signal list"""
        weight = 0.0

        for signal in signals:
            source_weight = self.source_weights.get(signal.source, 1.0)
            weight += signal.strength * signal.confidence * source_weight

        return weight

    def _calculate_confidence(
        self,
        signals: List[InputSignal],
        agreement_ratio: float
    ) -> float:
        """Calculate overall confidence"""
        if not signals:
            return 0.0

        # Average confidence of signals
        avg_confidence = np.mean([s.confidence for s in signals])

        # Weight by agreement
        confidence = avg_confidence * (0.5 + 0.5 * agreement_ratio)

        return min(confidence, 1.0)

    def _calculate_risk_score(
        self,
        volatility: float,
        agreement: float,
        confidence: float
    ) -> float:
        """
        Calculate risk score (higher = riskier).

        Args:
            volatility: Price volatility
            agreement: Signal agreement ratio
            confidence: Overall confidence

        Returns:
            Risk score (0-1)
        """
        # High volatility increases risk
        vol_risk = min(volatility / 0.05, 1.0)

        # Low agreement increases risk
        agreement_risk = 1.0 - agreement

        # Low confidence increases risk
        confidence_risk = 1.0 - confidence

        # Combined risk
        risk = (vol_risk + agreement_risk + confidence_risk) / 3

        return risk

    def _calculate_position_size(
        self,
        decision: DecisionType,
        confidence: float,
        risk_score: float
    ) -> float:
        """
        Calculate risk-adjusted position size.

        Args:
            decision: Trading decision
            confidence: Decision confidence
            risk_score: Risk score

        Returns:
            Position size as fraction of capital
        """
        if decision == DecisionType.HOLD:
            return 0.0

        # Base size depends on decision strength
        if decision in [DecisionType.STRONG_BUY, DecisionType.STRONG_SELL]:
            base_size = self.max_position_size
        else:
            base_size = self.max_position_size * 0.5

        # Adjust for confidence
        size = base_size * confidence

        # Adjust for risk (reduce size in high risk)
        size *= (1.0 - risk_score * 0.5)

        return min(size, self.max_position_size)

    def _create_hold_decision(self, symbol: str, timestamp: int) -> AggregatedDecision:
        """Create a HOLD decision"""
        return AggregatedDecision(
            symbol=symbol,
            decision=DecisionType.HOLD,
            confidence=0.0,
            position_size=0.0,
            buy_signals=0,
            sell_signals=0,
            hold_signals=0,
            contributing_signals=[],
            agreement_score=0.0,
            risk_score=0.0,
            timestamp=timestamp
        )

    def get_decision_metrics(self) -> DecisionMetrics:
        """Calculate decision quality metrics"""
        if not self.decision_history:
            return DecisionMetrics(
                total_decisions=0,
                strong_buy_count=0,
                buy_count=0,
                hold_count=0,
                sell_count=0,
                strong_sell_count=0,
                avg_confidence=0.0,
                avg_agreement=0.0,
                high_confidence_decisions=0
            )

        decision_counts = defaultdict(int)
        for decision in self.decision_history:
            decision_counts[decision.decision] += 1

        avg_confidence = np.mean([d.confidence for d in self.decision_history])
        avg_agreement = np.mean([d.agreement_score for d in self.decision_history])
        high_conf = sum(1 for d in self.decision_history if d.confidence >= 0.8)

        return DecisionMetrics(
            total_decisions=len(self.decision_history),
            strong_buy_count=decision_counts[DecisionType.STRONG_BUY],
            buy_count=decision_counts[DecisionType.BUY],
            hold_count=decision_counts[DecisionType.HOLD],
            sell_count=decision_counts[DecisionType.SELL],
            strong_sell_count=decision_counts[DecisionType.STRONG_SELL],
            avg_confidence=avg_confidence,
            avg_agreement=avg_agreement,
            high_confidence_decisions=high_conf
        )

    def filter_conflicting_signals(
        self,
        signals: List[InputSignal]
    ) -> List[InputSignal]:
        """
        Filter out low-quality or conflicting signals.

        Args:
            signals: Input signals

        Returns:
            Filtered signal list
        """
        # Remove low confidence signals
        filtered = [s for s in signals if s.confidence >= 0.3]

        # Group by source
        by_source: Dict[str, List[InputSignal]] = defaultdict(list)
        for signal in filtered:
            by_source[signal.source_name].append(signal)

        # Keep strongest signal per source
        result = []
        for source_signals in by_source.values():
            strongest = max(source_signals, key=lambda s: s.strength * s.confidence)
            result.append(strongest)

        return result


class AutonomousTradingDecision:
    """
    Autonomous trading decision system.

    Integrates technical indicators, sentiment, and strategies for
    fully automated trading decisions.
    """

    def __init__(self):
        """Initialize autonomous decision system"""
        self.aggregator = SignalAggregator(
            min_confidence=0.5,
            min_agreement=0.6,
            max_position_size=0.2
        )

        # Configure source weights
        self.aggregator.set_source_weight(SignalSource.TECHNICAL, 1.0)
        self.aggregator.set_source_weight(SignalSource.SENTIMENT, 0.7)
        self.aggregator.set_source_weight(SignalSource.STRATEGY, 1.3)

    def make_decision(
        self,
        technical_signals: List[Dict],
        sentiment_score: float,
        strategy_signals: List[Dict],
        current_price: float,
        volatility: float
    ) -> AggregatedDecision:
        """
        Make autonomous trading decision.

        Args:
            technical_signals: Technical indicator signals
            sentiment_score: Sentiment score (-1 to 1)
            strategy_signals: Strategy signals
            current_price: Current price
            volatility: Price volatility

        Returns:
            Trading decision
        """
        all_signals = []

        # Convert technical signals
        for tech_sig in technical_signals:
            signal = InputSignal(
                source=SignalSource.TECHNICAL,
                source_name=tech_sig.get('indicator', 'unknown'),
                symbol=tech_sig.get('symbol', 'UNKNOWN'),
                direction=tech_sig.get('signal', 'hold'),
                strength=tech_sig.get('strength', 0.5),
                confidence=tech_sig.get('confidence', 0.7),
                timestamp=tech_sig.get('timestamp', 0)
            )
            all_signals.append(signal)

        # Convert sentiment to signal
        if sentiment_score != 0:
            direction = 'buy' if sentiment_score > 0 else 'sell'
            signal = InputSignal(
                source=SignalSource.SENTIMENT,
                source_name='sentiment_analyzer',
                symbol=technical_signals[0].get('symbol', 'UNKNOWN') if technical_signals else 'UNKNOWN',
                direction=direction,
                strength=abs(sentiment_score),
                confidence=0.6,
                timestamp=0
            )
            all_signals.append(signal)

        # Convert strategy signals
        for strat_sig in strategy_signals:
            signal = InputSignal(
                source=SignalSource.STRATEGY,
                source_name=strat_sig.get('strategy', 'unknown'),
                symbol=strat_sig.get('symbol', 'UNKNOWN'),
                direction=strat_sig.get('signal', 'hold'),
                strength=strat_sig.get('strength', 0.5),
                confidence=strat_sig.get('confidence', 0.8),
                timestamp=strat_sig.get('timestamp', 0)
            )
            all_signals.append(signal)

        # Filter conflicting signals
        filtered_signals = self.aggregator.filter_conflicting_signals(all_signals)

        # Aggregate
        decision = self.aggregator.aggregate_signals(
            filtered_signals, current_price, volatility
        )

        return decision


def test_signal_aggregator():
    """Test Signal Aggregator and Decision System"""
    print("=" * 60)
    print("Testing Multi-Signal Aggregator and Decision System")
    print("=" * 60)

    # Initialize aggregator
    aggregator = SignalAggregator(
        min_confidence=0.5,
        min_agreement=0.6,
        max_position_size=0.2
    )

    print("\n1. Signal Aggregation - Aligned Signals:")
    print("-" * 40)

    # Test with aligned buy signals
    aligned_signals = [
        InputSignal(SignalSource.TECHNICAL, 'RSI', 'AAPL', 'buy', 0.8, 0.9, 100),
        InputSignal(SignalSource.TECHNICAL, 'MACD', 'AAPL', 'buy', 0.7, 0.85, 100),
        InputSignal(SignalSource.SENTIMENT, 'news', 'AAPL', 'buy', 0.6, 0.7, 100),
        InputSignal(SignalSource.STRATEGY, 'momentum', 'AAPL', 'buy', 0.75, 0.8, 100),
    ]

    decision = aggregator.aggregate_signals(aligned_signals, 150.0, 0.02)

    print(f"Symbol: {decision.symbol}")
    print(f"Decision: {decision.decision.value.upper()}")
    print(f"Confidence: {decision.confidence:.2%}")
    print(f"Position Size: {decision.position_size:.2%}")
    print(f"Agreement: {decision.agreement_score:.2%}")
    print(f"Risk Score: {decision.risk_score:.2%}")
    print(f"\nSignal Breakdown:")
    print(f"  Buy: {decision.buy_signals}, Sell: {decision.sell_signals}, Hold: {decision.hold_signals}")

    print("\n2. Signal Aggregation - Conflicting Signals:")
    print("-" * 40)

    # Test with conflicting signals
    conflicting_signals = [
        InputSignal(SignalSource.TECHNICAL, 'RSI', 'MSFT', 'buy', 0.6, 0.8, 200),
        InputSignal(SignalSource.TECHNICAL, 'MACD', 'MSFT', 'sell', 0.7, 0.85, 200),
        InputSignal(SignalSource.SENTIMENT, 'news', 'MSFT', 'sell', 0.5, 0.6, 200),
        InputSignal(SignalSource.STRATEGY, 'trend', 'MSFT', 'buy', 0.4, 0.7, 200),
    ]

    decision = aggregator.aggregate_signals(conflicting_signals, 300.0, 0.03)

    print(f"Symbol: {decision.symbol}")
    print(f"Decision: {decision.decision.value.upper()}")
    print(f"Confidence: {decision.confidence:.2%}")
    print(f"Position Size: {decision.position_size:.2%}")
    print(f"Agreement: {decision.agreement_score:.2%}")
    print(f"Risk Score: {decision.risk_score:.2%}")
    print(f"\nSignal Breakdown:")
    print(f"  Buy: {decision.buy_signals}, Sell: {decision.sell_signals}, Hold: {decision.hold_signals}")

    print("\n3. Signal Filtering:")
    print("-" * 40)

    # Test signal filtering
    noisy_signals = [
        InputSignal(SignalSource.TECHNICAL, 'RSI', 'GOOGL', 'buy', 0.8, 0.9, 300),
        InputSignal(SignalSource.TECHNICAL, 'RSI', 'GOOGL', 'buy', 0.3, 0.4, 300),  # Weak duplicate
        InputSignal(SignalSource.TECHNICAL, 'MACD', 'GOOGL', 'sell', 0.2, 0.3, 300),  # Low confidence
        InputSignal(SignalSource.STRATEGY, 'momentum', 'GOOGL', 'buy', 0.7, 0.8, 300),
    ]

    print(f"Original signals: {len(noisy_signals)}")
    filtered = aggregator.filter_conflicting_signals(noisy_signals)
    print(f"Filtered signals: {len(filtered)}")

    print("\nFiltered signals:")
    for sig in filtered:
        print(f"  {sig.source_name}: {sig.direction} (strength: {sig.strength:.2f}, conf: {sig.confidence:.2f})")

    print("\n4. Position Sizing - Risk Adjustment:")
    print("-" * 40)

    # Test different risk scenarios
    test_scenarios = [
        ("Low Vol, High Agreement", 0.01, 0.9, 0.85),
        ("High Vol, High Agreement", 0.05, 0.9, 0.85),
        ("Low Vol, Low Agreement", 0.01, 0.5, 0.55),
        ("High Vol, Low Agreement", 0.05, 0.5, 0.55),
    ]

    print(f"{'Scenario':<30} {'Risk':<10} {'Position':<10}")
    print("-" * 50)

    for scenario, vol, agreement, confidence in test_scenarios:
        risk = aggregator._calculate_risk_score(vol, agreement, confidence)
        size = aggregator._calculate_position_size(DecisionType.STRONG_BUY, confidence, risk)
        print(f"{scenario:<30} {risk:<10.2%} {size:<10.2%}")

    print("\n5. Autonomous Decision System:")
    print("-" * 40)

    # Test autonomous system
    autonomous = AutonomousTradingDecision()

    tech_sigs = [
        {'indicator': 'RSI', 'symbol': 'TSLA', 'signal': 'buy', 'strength': 0.7, 'confidence': 0.8, 'timestamp': 400},
        {'indicator': 'MACD', 'symbol': 'TSLA', 'signal': 'buy', 'strength': 0.6, 'confidence': 0.75, 'timestamp': 400},
    ]

    sentiment = 0.5  # Positive sentiment

    strat_sigs = [
        {'strategy': 'momentum', 'symbol': 'TSLA', 'signal': 'buy', 'strength': 0.8, 'confidence': 0.85, 'timestamp': 400},
    ]

    decision = autonomous.make_decision(tech_sigs, sentiment, strat_sigs, 200.0, 0.03)

    print(f"Autonomous Decision for TSLA:")
    print(f"  Action: {decision.decision.value.upper()}")
    print(f"  Confidence: {decision.confidence:.2%}")
    print(f"  Recommended Size: {decision.position_size:.2%} of capital")
    print(f"  Risk Assessment: {decision.risk_score:.2%}")
    print(f"  Signals Used: {len(decision.contributing_signals)}")

    print("\n6. Decision History Analysis:")
    print("-" * 40)

    # Make multiple decisions to build history
    test_cases = [
        (aligned_signals, 150.0, 0.02),
        (conflicting_signals, 300.0, 0.03),
        (filtered, 100.0, 0.015),
    ]

    for signals, price, vol in test_cases:
        aggregator.aggregate_signals(signals, price, vol)

    metrics = aggregator.get_decision_metrics()

    print(f"Total Decisions: {metrics.total_decisions}")
    print(f"\nDecision Distribution:")
    print(f"  Strong Buy: {metrics.strong_buy_count}")
    print(f"  Buy: {metrics.buy_count}")
    print(f"  Hold: {metrics.hold_count}")
    print(f"  Sell: {metrics.sell_count}")
    print(f"  Strong Sell: {metrics.strong_sell_count}")
    print(f"\nQuality Metrics:")
    print(f"  Average Confidence: {metrics.avg_confidence:.2%}")
    print(f"  Average Agreement: {metrics.avg_agreement:.2%}")
    print(f"  High Confidence Decisions: {metrics.high_confidence_decisions}")

    print("\n7. Source Weight Impact:")
    print("-" * 40)

    # Test different source weights
    print("Testing weight configurations:")

    weights = [
        ("Equal Weights", {SignalSource.TECHNICAL: 1.0, SignalSource.SENTIMENT: 1.0, SignalSource.STRATEGY: 1.0}),
        ("Tech Heavy", {SignalSource.TECHNICAL: 1.5, SignalSource.SENTIMENT: 0.5, SignalSource.STRATEGY: 1.0}),
        ("Strategy Heavy", {SignalSource.TECHNICAL: 1.0, SignalSource.SENTIMENT: 0.7, SignalSource.STRATEGY: 1.5}),
    ]

    for config_name, weight_dict in weights:
        test_agg = SignalAggregator()
        for source, weight in weight_dict.items():
            test_agg.set_source_weight(source, weight)

        decision = test_agg.aggregate_signals(aligned_signals, 150.0, 0.02)
        print(f"\n{config_name}:")
        print(f"  Decision: {decision.decision.value}, Confidence: {decision.confidence:.2%}, Size: {decision.position_size:.2%}")

    print("\n[SUCCESS] Signal Aggregator test completed successfully!")


if __name__ == "__main__":
    test_signal_aggregator()
