"""
BOAT - Order Book Dynamics Analyzer
===================================

Production-ready limit order book (LOB) analysis system for market microstructure
analysis and high-frequency trading signals.

Features:
- Real-time order book reconstruction
- Microstructure feature extraction
- Order flow imbalance calculation
- Price impact prediction
- Queue position modeling

Based on 2025 research:
- LOBFrame framework for efficient LOB processing
- DeepLOB architecture insights
- Microstructural modeling for HFT
- Practical applicability focus over pure algorithms
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Deque
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import bisect


class OrderType(Enum):
    """Order types in the book"""
    LIMIT = "limit"
    MARKET = "market"
    CANCEL = "cancel"
    MODIFY = "modify"


class Side(Enum):
    """Order side"""
    BID = "bid"
    ASK = "ask"


@dataclass
class Order:
    """Individual order in the book"""
    order_id: int
    side: Side
    price: float
    quantity: int
    timestamp: int
    order_type: OrderType = OrderType.LIMIT


@dataclass
class Level:
    """Price level in the order book"""
    price: float
    quantity: int
    orders: List[Order] = field(default_factory=list)


@dataclass
class OrderBookSnapshot:
    """Point-in-time order book state"""
    timestamp: int
    bid_levels: List[Level]
    ask_levels: List[Level]
    mid_price: float
    spread: float
    depth_imbalance: float


@dataclass
class MicrostructureFeatures:
    """Market microstructure features"""
    bid_ask_spread: float
    mid_price: float
    micro_price: float  # Size-weighted mid
    order_flow_imbalance: float
    book_pressure: float
    volume_imbalance: float
    depth_ratio: float
    spread_derivative: float
    volatility: float


class LimitOrderBook:
    """
    Limit Order Book implementation with FIFO matching.

    Maintains bid and ask sides with price-time priority
    and provides microstructure analytics.
    """

    def __init__(self, max_levels: int = 10):
        """
        Initialize the limit order book.

        Args:
            max_levels: Maximum price levels to track
        """
        self.max_levels = max_levels
        self.bids: Dict[float, Level] = {}  # Price -> Level
        self.asks: Dict[float, Level] = {}
        self.orders: Dict[int, Order] = {}  # Order ID -> Order

        # Sorted price lists for efficient access
        self.bid_prices: List[float] = []
        self.ask_prices: List[float] = []

        # Historical data for analytics
        self.mid_price_history: Deque[float] = deque(maxlen=100)
        self.spread_history: Deque[float] = deque(maxlen=100)
        self.trade_history: List[Tuple[int, float, int, Side]] = []

        self.current_timestamp = 0

    def add_order(self, order: Order) -> bool:
        """
        Add a limit order to the book.

        Args:
            order: Order to add

        Returns:
            True if order was added, False if matched
        """
        self.current_timestamp = order.timestamp

        # Check for immediate execution
        if self._check_immediate_execution(order):
            return False

        # Add to appropriate side
        if order.side == Side.BID:
            self._add_to_bids(order)
        else:
            self._add_to_asks(order)

        self.orders[order.order_id] = order
        return True

    def _add_to_bids(self, order: Order):
        """Add order to bid side"""
        if order.price not in self.bids:
            self.bids[order.price] = Level(order.price, 0, [])
            bisect.insort(self.bid_prices, order.price)
            self.bid_prices = self.bid_prices[-self.max_levels:]

        level = self.bids[order.price]
        level.orders.append(order)
        level.quantity += order.quantity

    def _add_to_asks(self, order: Order):
        """Add order to ask side"""
        if order.price not in self.asks:
            self.asks[order.price] = Level(order.price, 0, [])
            bisect.insort(self.ask_prices, order.price)
            self.ask_prices = self.ask_prices[:self.max_levels]

        level = self.asks[order.price]
        level.orders.append(order)
        level.quantity += order.quantity

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if cancelled, False if not found
        """
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]

        # Remove from appropriate side
        if order.side == Side.BID:
            if order.price in self.bids:
                level = self.bids[order.price]
                level.orders = [o for o in level.orders if o.order_id != order_id]
                level.quantity -= order.quantity

                if level.quantity == 0:
                    del self.bids[order.price]
                    self.bid_prices.remove(order.price)
        else:
            if order.price in self.asks:
                level = self.asks[order.price]
                level.orders = [o for o in level.orders if o.order_id != order_id]
                level.quantity -= order.quantity

                if level.quantity == 0:
                    del self.asks[order.price]
                    self.ask_prices.remove(order.price)

        del self.orders[order_id]
        return True

    def execute_market_order(
        self,
        side: Side,
        quantity: int,
        timestamp: int
    ) -> List[Tuple[float, int]]:
        """
        Execute a market order.

        Args:
            side: Buy or sell
            quantity: Order quantity
            timestamp: Execution timestamp

        Returns:
            List of (price, quantity) fills
        """
        self.current_timestamp = timestamp
        fills = []
        remaining = quantity

        if side == Side.BID:
            # Buy market order - match against asks
            for price in self.ask_prices[:]:
                if remaining == 0:
                    break

                level = self.asks[price]
                fill_qty = min(remaining, level.quantity)

                fills.append((price, fill_qty))
                level.quantity -= fill_qty
                remaining -= fill_qty

                # Record trade
                self.trade_history.append((timestamp, price, fill_qty, side))

                # Remove exhausted orders (FIFO)
                qty_to_remove = fill_qty
                level.orders = [o for o in level.orders
                                if (qty_to_remove := qty_to_remove - o.quantity) >= 0
                                or o]

                if level.quantity == 0:
                    del self.asks[price]
                    self.ask_prices.remove(price)
        else:
            # Sell market order - match against bids
            for price in reversed(self.bid_prices[:]):
                if remaining == 0:
                    break

                level = self.bids[price]
                fill_qty = min(remaining, level.quantity)

                fills.append((price, fill_qty))
                level.quantity -= fill_qty
                remaining -= fill_qty

                # Record trade
                self.trade_history.append((timestamp, price, fill_qty, side))

                # Remove exhausted orders (FIFO)
                qty_to_remove = fill_qty
                level.orders = [o for o in level.orders
                                if (qty_to_remove := qty_to_remove - o.quantity) >= 0
                                or o]

                if level.quantity == 0:
                    del self.bids[price]
                    self.bid_prices.remove(price)

        return fills

    def _check_immediate_execution(self, order: Order) -> bool:
        """Check if order crosses the spread"""
        if order.side == Side.BID and self.ask_prices:
            return order.price >= self.ask_prices[0]
        elif order.side == Side.ASK and self.bid_prices:
            return order.price <= self.bid_prices[-1]
        return False

    def get_best_bid(self) -> Optional[Tuple[float, int]]:
        """Get best bid price and quantity"""
        if not self.bid_prices:
            return None
        price = self.bid_prices[-1]
        return price, self.bids[price].quantity

    def get_best_ask(self) -> Optional[Tuple[float, int]]:
        """Get best ask price and quantity"""
        if not self.ask_prices:
            return None
        price = self.ask_prices[0]
        return price, self.asks[price].quantity

    def get_mid_price(self) -> Optional[float]:
        """Calculate mid price"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            return (best_bid[0] + best_ask[0]) / 2
        return None

    def get_micro_price(self) -> Optional[float]:
        """Calculate size-weighted mid price"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            bid_price, bid_qty = best_bid
            ask_price, ask_qty = best_ask

            total_qty = bid_qty + ask_qty
            if total_qty > 0:
                return (bid_price * ask_qty + ask_price * bid_qty) / total_qty

        return self.get_mid_price()

    def get_spread(self) -> Optional[float]:
        """Calculate bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            return best_ask[0] - best_bid[0]
        return None

    def calculate_order_flow_imbalance(self, levels: int = 5) -> float:
        """
        Calculate order flow imbalance.

        Args:
            levels: Number of levels to consider

        Returns:
            Imbalance ratio (-1 to 1)
        """
        bid_volume = sum(
            self.bids[p].quantity
            for p in self.bid_prices[-levels:]
            if p in self.bids
        )

        ask_volume = sum(
            self.asks[p].quantity
            for p in self.ask_prices[:levels]
            if p in self.asks
        )

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0

        return (bid_volume - ask_volume) / total_volume

    def calculate_book_pressure(self) -> float:
        """
        Calculate book pressure (weighted by distance from mid).

        Returns:
            Pressure value (positive = buy pressure)
        """
        mid = self.get_mid_price()
        if not mid:
            return 0.0

        bid_pressure = 0.0
        ask_pressure = 0.0

        # Weight by inverse distance from mid
        for price in self.bid_prices[-5:]:
            if price in self.bids:
                distance = abs(mid - price) + 0.01  # Avoid division by zero
                bid_pressure += self.bids[price].quantity / distance

        for price in self.ask_prices[:5]:
            if price in self.asks:
                distance = abs(price - mid) + 0.01
                ask_pressure += self.asks[price].quantity / distance

        total_pressure = bid_pressure + ask_pressure
        if total_pressure == 0:
            return 0.0

        return (bid_pressure - ask_pressure) / total_pressure

    def get_depth_profile(self, levels: int = 10) -> Tuple[List[Level], List[Level]]:
        """
        Get depth profile of the book.

        Args:
            levels: Number of levels

        Returns:
            Tuple of (bid_levels, ask_levels)
        """
        bid_levels = [
            self.bids[p] for p in self.bid_prices[-levels:]
            if p in self.bids
        ]

        ask_levels = [
            self.asks[p] for p in self.ask_prices[:levels]
            if p in self.asks
        ]

        return bid_levels, ask_levels

    def extract_features(self) -> MicrostructureFeatures:
        """
        Extract microstructure features for ML models.

        Returns:
            Feature set for prediction
        """
        spread = self.get_spread() or 0.0
        mid_price = self.get_mid_price() or 0.0
        micro_price = self.get_micro_price() or mid_price

        # Order flow imbalance
        ofi = self.calculate_order_flow_imbalance()

        # Book pressure
        pressure = self.calculate_book_pressure()

        # Volume imbalance at best levels
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        volume_imbalance = 0.0
        if best_bid and best_ask:
            total_vol = best_bid[1] + best_ask[1]
            if total_vol > 0:
                volume_imbalance = (best_bid[1] - best_ask[1]) / total_vol

        # Depth ratio (bid depth / ask depth)
        bid_depth = sum(self.bids[p].quantity for p in self.bid_prices[:5] if p in self.bids)
        ask_depth = sum(self.asks[p].quantity for p in self.ask_prices[:5] if p in self.asks)

        depth_ratio = bid_depth / (ask_depth + 1)  # Avoid division by zero

        # Spread derivative
        if len(self.spread_history) > 1:
            spread_derivative = spread - self.spread_history[-1]
        else:
            spread_derivative = 0.0

        # Update history
        self.mid_price_history.append(mid_price)
        self.spread_history.append(spread)

        # Volatility (rolling std of mid price)
        volatility = 0.0
        if len(self.mid_price_history) > 10:
            returns = np.diff(list(self.mid_price_history)[-20:])
            if len(returns) > 0:
                volatility = np.std(returns)

        return MicrostructureFeatures(
            bid_ask_spread=spread,
            mid_price=mid_price,
            micro_price=micro_price,
            order_flow_imbalance=ofi,
            book_pressure=pressure,
            volume_imbalance=volume_imbalance,
            depth_ratio=depth_ratio,
            spread_derivative=spread_derivative,
            volatility=volatility
        )

    def get_snapshot(self) -> OrderBookSnapshot:
        """Get current order book snapshot"""
        bid_levels, ask_levels = self.get_depth_profile()

        return OrderBookSnapshot(
            timestamp=self.current_timestamp,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            mid_price=self.get_mid_price() or 0.0,
            spread=self.get_spread() or 0.0,
            depth_imbalance=self.calculate_order_flow_imbalance()
        )


class OrderBookDynamicsAnalyzer:
    """
    Analyzer for order book dynamics and microstructure patterns.

    Provides advanced analytics for HFT and market making strategies.
    """

    def __init__(self, book: LimitOrderBook):
        """
        Initialize analyzer.

        Args:
            book: Limit order book to analyze
        """
        self.book = book
        self.feature_history: List[MicrostructureFeatures] = []
        self.prediction_horizon = 10  # ticks

    def predict_mid_price_movement(
        self,
        lookback: int = 20
    ) -> Tuple[float, float]:
        """
        Predict mid price movement using microstructure features.

        Args:
            lookback: Historical periods to consider

        Returns:
            Tuple of (direction_probability, expected_move)
        """
        if len(self.feature_history) < lookback:
            return 0.5, 0.0

        # Extract recent features
        recent_features = self.feature_history[-lookback:]

        # Simple model based on order flow imbalance and book pressure
        ofi_signal = np.mean([f.order_flow_imbalance for f in recent_features])
        pressure_signal = np.mean([f.book_pressure for f in recent_features])
        volume_signal = np.mean([f.volume_imbalance for f in recent_features])

        # Combine signals
        composite_signal = (
            ofi_signal * 0.4 +
            pressure_signal * 0.3 +
            volume_signal * 0.3
        )

        # Convert to probability
        prob_up = (composite_signal + 1) / 2  # Map from [-1, 1] to [0, 1]

        # Expected move based on recent volatility
        volatility = recent_features[-1].volatility
        expected_move = composite_signal * volatility * np.sqrt(self.prediction_horizon)

        return prob_up, expected_move

    def detect_momentum(self) -> float:
        """
        Detect momentum in order flow.

        Returns:
            Momentum score (-1 to 1)
        """
        if len(self.book.trade_history) < 10:
            return 0.0

        recent_trades = self.book.trade_history[-20:]

        # Calculate trade imbalance
        buy_volume = sum(t[2] for t in recent_trades if t[3] == Side.BID)
        sell_volume = sum(t[2] for t in recent_trades if t[3] == Side.ASK)

        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.0

        return (buy_volume - sell_volume) / total_volume

    def calculate_price_impact(
        self,
        side: Side,
        quantity: int
    ) -> float:
        """
        Estimate price impact of a hypothetical order.

        Args:
            side: Order side
            quantity: Order size

        Returns:
            Expected price impact (percentage)
        """
        if side == Side.BID:
            # Walking up the ask book
            prices_to_check = self.book.ask_prices[:10]
            levels_dict = self.book.asks
        else:
            # Walking down the bid book
            prices_to_check = list(reversed(self.book.bid_prices[-10:]))
            levels_dict = self.book.bids

        if not prices_to_check:
            return 0.0

        initial_price = prices_to_check[0]
        remaining = quantity
        volume_weighted_price = 0.0
        total_filled = 0

        for price in prices_to_check:
            if price not in levels_dict:
                continue

            level_qty = levels_dict[price].quantity
            fill_qty = min(remaining, level_qty)

            volume_weighted_price += price * fill_qty
            total_filled += fill_qty
            remaining -= fill_qty

            if remaining == 0:
                break

        if total_filled > 0:
            avg_price = volume_weighted_price / total_filled
            impact = abs(avg_price - initial_price) / initial_price
            return impact

        return 0.0

    def update(self):
        """Update feature history"""
        features = self.book.extract_features()
        self.feature_history.append(features)

        # Keep limited history
        if len(self.feature_history) > 1000:
            self.feature_history = self.feature_history[-1000:]


def test_order_book_dynamics():
    """Test the Order Book Dynamics Analyzer"""
    print("=" * 60)
    print("Testing Order Book Dynamics Analyzer")
    print("=" * 60)

    # Create order book
    lob = LimitOrderBook(max_levels=10)
    analyzer = OrderBookDynamicsAnalyzer(lob)

    # Simulate order flow
    np.random.seed(42)
    order_id = 0
    base_price = 100.0

    print("\n1. Building Initial Order Book:")
    print("-" * 40)

    # Add initial orders
    for i in range(5):
        # Bids
        bid_price = base_price - 0.01 * (i + 1)
        bid_qty = np.random.randint(100, 1000)
        order = Order(
            order_id=order_id,
            side=Side.BID,
            price=bid_price,
            quantity=bid_qty,
            timestamp=0
        )
        lob.add_order(order)
        order_id += 1

        # Asks
        ask_price = base_price + 0.01 * (i + 1)
        ask_qty = np.random.randint(100, 1000)
        order = Order(
            order_id=order_id,
            side=Side.ASK,
            price=ask_price,
            quantity=ask_qty,
            timestamp=0
        )
        lob.add_order(order)
        order_id += 1

    # Display book state
    best_bid = lob.get_best_bid()
    best_ask = lob.get_best_ask()
    print(f"Best Bid: ${best_bid[0]:.2f} x {best_bid[1]}")
    print(f"Best Ask: ${best_ask[0]:.2f} x {best_ask[1]}")
    print(f"Spread: ${lob.get_spread():.3f}")
    print(f"Mid Price: ${lob.get_mid_price():.3f}")
    print(f"Micro Price: ${lob.get_micro_price():.3f}")

    # Get depth profile
    bid_levels, ask_levels = lob.get_depth_profile(5)
    print("\nDepth Profile (5 levels):")
    print("BIDS:")
    for level in reversed(bid_levels):
        print(f"  ${level.price:.2f}: {level.quantity:,}")
    print("ASKS:")
    for level in ask_levels:
        print(f"  ${level.price:.2f}: {level.quantity:,}")

    print("\n2. Microstructure Features:")
    print("-" * 40)

    features = lob.extract_features()
    print(f"Order Flow Imbalance: {features.order_flow_imbalance:.3f}")
    print(f"Book Pressure: {features.book_pressure:.3f}")
    print(f"Volume Imbalance: {features.volume_imbalance:.3f}")
    print(f"Depth Ratio: {features.depth_ratio:.3f}")
    print(f"Spread Derivative: {features.spread_derivative:.4f}")

    print("\n3. Market Order Execution:")
    print("-" * 40)

    # Execute buy market order
    fills = lob.execute_market_order(Side.BID, 500, timestamp=1)
    print(f"Buy Market Order (500 shares):")
    total_cost = 0
    for price, qty in fills:
        print(f"  Filled {qty} @ ${price:.2f}")
        total_cost += price * qty
    if fills:
        avg_price = total_cost / sum(q for _, q in fills)
        print(f"  Average Price: ${avg_price:.3f}")

    # Update features
    analyzer.update()

    print("\n4. Order Flow Simulation:")
    print("-" * 40)

    # Simulate dynamic order flow
    for t in range(100):
        # Random orders
        if np.random.random() < 0.7:  # 70% limit orders
            side = Side.BID if np.random.random() < 0.5 else Side.ASK

            if side == Side.BID:
                price = base_price - np.random.exponential(0.02)
            else:
                price = base_price + np.random.exponential(0.02)

            quantity = np.random.randint(50, 500)

            order = Order(
                order_id=order_id,
                side=side,
                price=round(price, 2),
                quantity=quantity,
                timestamp=t
            )
            lob.add_order(order)
            order_id += 1

        # Occasional market orders
        if np.random.random() < 0.1:
            side = Side.BID if np.random.random() < 0.5 else Side.ASK
            quantity = np.random.randint(100, 300)
            lob.execute_market_order(side, quantity, t)

        # Cancel some orders
        if np.random.random() < 0.05 and lob.orders:
            cancel_id = np.random.choice(list(lob.orders.keys()))
            lob.cancel_order(cancel_id)

        # Update analyzer
        analyzer.update()

    print(f"Simulated {t+1} time steps")
    print(f"Active orders: {len(lob.orders)}")
    print(f"Trades executed: {len(lob.trade_history)}")

    # Final state
    print("\nFinal Book State:")
    best_bid = lob.get_best_bid()
    best_ask = lob.get_best_ask()
    if best_bid and best_ask:
        print(f"Best Bid: ${best_bid[0]:.2f} x {best_bid[1]}")
        print(f"Best Ask: ${best_ask[0]:.2f} x {best_ask[1]}")
        print(f"Spread: ${lob.get_spread():.3f}")

    print("\n5. Price Movement Prediction:")
    print("-" * 40)

    prob_up, expected_move = analyzer.predict_mid_price_movement()
    print(f"Probability of upward movement: {prob_up:.2%}")
    print(f"Expected move: {expected_move:.4f}")

    momentum = analyzer.detect_momentum()
    print(f"Order flow momentum: {momentum:.3f}")

    print("\n6. Price Impact Analysis:")
    print("-" * 40)

    test_sizes = [100, 500, 1000, 5000]
    print("Buy order price impact:")
    for size in test_sizes:
        impact = analyzer.calculate_price_impact(Side.BID, size)
        print(f"  {size:,} shares: {impact:.3%}")

    print("\nSell order price impact:")
    for size in test_sizes:
        impact = analyzer.calculate_price_impact(Side.ASK, size)
        print(f"  {size:,} shares: {impact:.3%}")

    print("\n7. Order Book Snapshot:")
    print("-" * 40)

    snapshot = lob.get_snapshot()
    print(f"Timestamp: {snapshot.timestamp}")
    print(f"Mid Price: ${snapshot.mid_price:.3f}")
    print(f"Spread: ${snapshot.spread:.3f}")
    print(f"Depth Imbalance: {snapshot.depth_imbalance:.3f}")
    print(f"Bid Levels: {len(snapshot.bid_levels)}")
    print(f"Ask Levels: {len(snapshot.ask_levels)}")

    print("\n[SUCCESS] Order Book Dynamics Analyzer test completed successfully!")


if __name__ == "__main__":
    test_order_book_dynamics()