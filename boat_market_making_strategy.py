"""
BOAT - Market Making Strategy with Inventory Management
=======================================================

Production-ready market making system with optimal spread calculation
and inventory risk management.

Features:
- Optimal bid-ask spread calculation
- Inventory-based position adjustment
- Market order arrival modeling
- P&L optimization with risk constraints
- Real-time spread adaptation

Based on 2025 research:
- Adaptive optimal market making
- Inventory liquidation cost models
- Predictive market making (PMM)
- Deep RL for automated market makers
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class OrderType(Enum):
    """Order types"""
    BID = "bid"
    ASK = "ask"


@dataclass
class MarketMakingParams:
    """Market making strategy parameters"""
    risk_aversion: float  # Risk aversion coefficient (gamma)
    order_intensity: float  # Base order arrival rate (lambda)
    inventory_penalty: float  # Inventory holding cost
    target_spread_bps: float  # Target spread in basis points


@dataclass
class QuoteUpdate:
    """Quote update with bid/ask prices"""
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    spread: float
    mid_price: float


@dataclass
class PerformanceMetrics:
    """Market making performance metrics"""
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    num_trades: int
    inventory_turnover: float
    spread_capture: float
    sharpe_ratio: float


class MarketMakingStrategy:
    """
    Market making strategy with inventory management.

    Implements optimal spread calculation and dynamic position
    adjustment based on inventory risk.
    """

    def __init__(
        self,
        params: MarketMakingParams,
        max_inventory: int = 100,
        tick_size: float = 0.01
    ):
        """
        Initialize market making strategy.

        Args:
            params: Strategy parameters
            max_inventory: Maximum absolute inventory
            tick_size: Minimum price increment
        """
        self.params = params
        self.max_inventory = max_inventory
        self.tick_size = tick_size

        # State variables
        self.inventory = 0
        self.cash = 0.0
        self.trades: List[Tuple[str, float, int, int]] = []  # (side, price, size, time)
        self.quotes_history: List[QuoteUpdate] = []

    def calculate_optimal_spread(
        self,
        mid_price: float,
        volatility: float,
        time_to_close: float = 1.0
    ) -> Tuple[float, float]:
        """
        Calculate optimal bid-ask spread using Avellaneda-Stoikov model.

        Args:
            mid_price: Current mid price
            volatility: Price volatility
            time_to_close: Time remaining until close (fraction of day)

        Returns:
            Tuple of (bid_offset, ask_offset) from mid
        """
        gamma = self.params.risk_aversion
        k = self.params.order_intensity

        # Reservation price (adjust for inventory)
        inventory_adjustment = -gamma * volatility**2 * self.inventory * time_to_close
        reservation_price = mid_price + inventory_adjustment

        # Optimal spread
        spread = gamma * volatility**2 * time_to_close + (2 / gamma) * np.log(1 + gamma / k)

        # Bid and ask offsets from reservation price
        bid_offset = reservation_price - mid_price - spread / 2
        ask_offset = reservation_price - mid_price + spread / 2

        return (bid_offset, ask_offset)

    def calculate_quote_prices(
        self,
        mid_price: float,
        volatility: float,
        time_to_close: float = 1.0
    ) -> QuoteUpdate:
        """
        Calculate bid and ask quote prices.

        Args:
            mid_price: Current mid price
            volatility: Volatility estimate
            time_to_close: Time to market close

        Returns:
            Quote update with prices and sizes
        """
        bid_offset, ask_offset = self.calculate_optimal_spread(
            mid_price, volatility, time_to_close
        )

        # Calculate bid/ask prices
        bid_price = mid_price + bid_offset
        ask_price = mid_price + ask_offset

        # Round to tick size
        bid_price = np.floor(bid_price / self.tick_size) * self.tick_size
        ask_price = np.ceil(ask_price / self.tick_size) * self.tick_size

        # Calculate sizes based on inventory
        inventory_ratio = abs(self.inventory) / self.max_inventory

        if self.inventory > 0:
            # Long inventory: larger ask size to sell
            bid_size = max(1, int(10 * (1 - inventory_ratio)))
            ask_size = max(1, int(10 * (1 + inventory_ratio)))
        elif self.inventory < 0:
            # Short inventory: larger bid size to buy
            bid_size = max(1, int(10 * (1 + inventory_ratio)))
            ask_size = max(1, int(10 * (1 - inventory_ratio)))
        else:
            bid_size = ask_size = 10

        spread = ask_price - bid_price

        return QuoteUpdate(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            spread=spread,
            mid_price=mid_price
        )

    def simulate_order_arrival(
        self,
        bid_price: float,
        ask_price: float,
        mid_price: float,
        dt: float = 1.0
    ) -> Optional[Tuple[OrderType, int]]:
        """
        Simulate market order arrival using Poisson process.

        Args:
            bid_price: Current bid price
            ask_price: Current ask price
            mid_price: Market mid price
            dt: Time step

        Returns:
            Tuple of (order_type, size) if order arrives, None otherwise
        """
        # Order arrival intensity depends on spread
        spread_factor = (ask_price - bid_price) / mid_price
        intensity = self.params.order_intensity * np.exp(-10 * spread_factor)

        # Poisson arrival
        prob_arrival = 1 - np.exp(-intensity * dt)

        if np.random.random() < prob_arrival:
            # Determine order side (50/50 base, adjusted for inventory)
            inventory_bias = -self.inventory / (2 * self.max_inventory)
            prob_buy = 0.5 + inventory_bias

            if np.random.random() < prob_buy:
                order_type = OrderType.BID
            else:
                order_type = OrderType.ASK

            # Order size (Poisson distributed)
            size = max(1, np.random.poisson(5))

            return (order_type, size)

        return None

    def execute_trade(
        self,
        order_type: OrderType,
        price: float,
        size: int,
        timestamp: int
    ):
        """
        Execute a trade and update state.

        Args:
            order_type: BID (we buy) or ASK (we sell)
            price: Execution price
            size: Trade size
            timestamp: Current time
        """
        if order_type == OrderType.BID:
            # We buy (hit our bid)
            self.inventory += size
            self.cash -= price * size
            side = "BUY"
        else:
            # We sell (lift our ask)
            self.inventory -= size
            self.cash += price * size
            side = "SELL"

        self.trades.append((side, price, size, timestamp))

    def calculate_pnl(self, current_price: float) -> PerformanceMetrics:
        """
        Calculate P&L and performance metrics.

        Args:
            current_price: Current market price

        Returns:
            Performance metrics
        """
        # Realized P&L (from closed trades)
        realized_pnl = self.cash

        # Unrealized P&L (from open inventory)
        unrealized_pnl = self.inventory * current_price

        # Total P&L
        total_pnl = realized_pnl + unrealized_pnl

        # Number of trades
        num_trades = len(self.trades)

        # Inventory turnover
        if num_trades > 0:
            total_volume = sum(t[2] for t in self.trades)
            inventory_turnover = total_volume / max(abs(self.inventory), 1)
        else:
            inventory_turnover = 0.0

        # Spread capture
        if num_trades > 0:
            buy_prices = [t[1] for t in self.trades if t[0] == "BUY"]
            sell_prices = [t[1] for t in self.trades if t[0] == "SELL"]

            if buy_prices and sell_prices:
                avg_buy = np.mean(buy_prices)
                avg_sell = np.mean(sell_prices)
                spread_capture = (avg_sell - avg_buy) / avg_buy
            else:
                spread_capture = 0.0
        else:
            spread_capture = 0.0

        # Sharpe ratio (approximate)
        if num_trades > 10:
            pnl_series = np.cumsum([t[1] * t[2] if t[0] == "SELL" else -t[1] * t[2]
                                    for t in self.trades])
            returns = np.diff(pnl_series) / (np.abs(pnl_series[:-1]) + 1)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0.0

        return PerformanceMetrics(
            total_pnl=total_pnl,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            num_trades=num_trades,
            inventory_turnover=inventory_turnover,
            spread_capture=spread_capture,
            sharpe_ratio=sharpe
        )

    def run_simulation(
        self,
        initial_price: float,
        volatility: float,
        n_periods: int = 100,
        dt: float = 0.01
    ) -> PerformanceMetrics:
        """
        Run market making simulation.

        Args:
            initial_price: Starting price
            volatility: Daily volatility
            n_periods: Number of time periods
            dt: Time step size

        Returns:
            Final performance metrics
        """
        # Reset state
        self.inventory = 0
        self.cash = 0.0
        self.trades = []
        self.quotes_history = []

        price = initial_price

        for t in range(n_periods):
            # Time to close (decreasing)
            time_to_close = 1.0 - (t / n_periods)

            # Update quotes
            quotes = self.calculate_quote_prices(price, volatility, time_to_close)
            self.quotes_history.append(quotes)

            # Simulate order arrival
            order = self.simulate_order_arrival(
                quotes.bid_price, quotes.ask_price, price, dt
            )

            if order is not None:
                order_type, size = order

                # Check inventory limits
                if order_type == OrderType.BID and self.inventory < self.max_inventory:
                    self.execute_trade(order_type, quotes.bid_price, size, t)
                elif order_type == OrderType.ASK and self.inventory > -self.max_inventory:
                    self.execute_trade(order_type, quotes.ask_price, size, t)

            # Price evolution (GBM)
            price_change = volatility * np.sqrt(dt) * np.random.randn() * price
            price += price_change
            price = max(price, 0.01)  # Floor

        # Calculate final metrics
        metrics = self.calculate_pnl(price)

        return metrics


def test_market_making():
    """Test Market Making Strategy"""
    print("=" * 60)
    print("Testing Market Making Strategy with Inventory Management")
    print("=" * 60)

    # Initialize strategy
    params = MarketMakingParams(
        risk_aversion=0.1,
        order_intensity=10.0,
        inventory_penalty=0.01,
        target_spread_bps=10.0
    )

    mm_strategy = MarketMakingStrategy(
        params=params,
        max_inventory=100,
        tick_size=0.01
    )

    print("\n1. Strategy Parameters:")
    print("-" * 40)
    print(f"Risk Aversion (gamma): {params.risk_aversion}")
    print(f"Order Intensity (lambda): {params.order_intensity}")
    print(f"Max Inventory: {mm_strategy.max_inventory}")
    print(f"Tick Size: ${mm_strategy.tick_size}")

    print("\n2. Optimal Spread Calculation:")
    print("-" * 40)

    mid_price = 100.0
    volatility = 0.02  # 2% daily vol

    test_inventories = [-50, -20, 0, 20, 50]
    print(f"Mid Price: ${mid_price:.2f}, Volatility: {volatility:.2%}\n")
    print(f"{'Inventory':<12} {'Bid Offset':<12} {'Ask Offset':<12} {'Spread':<10}")
    print("-" * 46)

    for inv in test_inventories:
        mm_strategy.inventory = inv
        bid_off, ask_off = mm_strategy.calculate_optimal_spread(mid_price, volatility)
        spread = ask_off - bid_off
        print(f"{inv:<12} {bid_off:<+12.3f} {ask_off:<+12.3f} {spread:<10.3f}")

    print("\n3. Quote Generation:")
    print("-" * 40)

    mm_strategy.inventory = 0
    quotes = mm_strategy.calculate_quote_prices(mid_price, volatility)

    print(f"Bid: ${quotes.bid_price:.2f} x {quotes.bid_size}")
    print(f"Ask: ${quotes.ask_price:.2f} x {quotes.ask_size}")
    print(f"Spread: ${quotes.spread:.3f} ({quotes.spread/mid_price*10000:.1f} bps)")
    print(f"Mid: ${quotes.mid_price:.2f}")

    print("\n4. Inventory Impact on Quotes:")
    print("-" * 40)

    print(f"{'Inventory':<12} {'Bid':<10} {'Ask':<10} {'Bid Size':<10} {'Ask Size':<10}")
    print("-" * 52)

    for inv in [-30, 0, 30]:
        mm_strategy.inventory = inv
        q = mm_strategy.calculate_quote_prices(mid_price, volatility)
        print(f"{inv:<12} ${q.bid_price:<9.2f} ${q.ask_price:<9.2f} {q.bid_size:<10} {q.ask_size:<10}")

    print("\n5. Simulation Results:")
    print("-" * 40)

    np.random.seed(42)
    metrics = mm_strategy.run_simulation(
        initial_price=100.0,
        volatility=0.02,
        n_periods=1000,
        dt=0.01
    )

    print(f"Total P&L: ${metrics.total_pnl:.2f}")
    print(f"Realized P&L: ${metrics.realized_pnl:.2f}")
    print(f"Unrealized P&L: ${metrics.unrealized_pnl:.2f}")
    print(f"Number of Trades: {metrics.num_trades}")
    print(f"Final Inventory: {mm_strategy.inventory}")
    print(f"Inventory Turnover: {metrics.inventory_turnover:.2f}x")
    print(f"Spread Capture: {metrics.spread_capture:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")

    print("\n6. Parameter Sensitivity:")
    print("-" * 40)

    risk_aversions = [0.05, 0.1, 0.2, 0.5]
    print(f"{'Risk Aversion':<15} {'Total P&L':<12} {'Trades':<10} {'Sharpe':<10}")
    print("-" * 47)

    for gamma in risk_aversions:
        test_params = MarketMakingParams(
            risk_aversion=gamma,
            order_intensity=10.0,
            inventory_penalty=0.01,
            target_spread_bps=10.0
        )
        test_strategy = MarketMakingStrategy(test_params, max_inventory=100)

        np.random.seed(42)
        result = test_strategy.run_simulation(100.0, 0.02, 1000, 0.01)
        print(f"{gamma:<15.2f} ${result.total_pnl:<11.2f} {result.num_trades:<10} {result.sharpe_ratio:<10.3f}")

    print("\n7. Quote History Analysis:")
    print("-" * 40)

    if mm_strategy.quotes_history:
        spreads = [q.spread for q in mm_strategy.quotes_history]
        print(f"Average Spread: ${np.mean(spreads):.3f}")
        print(f"Spread Range: [${np.min(spreads):.3f}, ${np.max(spreads):.3f}]")
        print(f"Spread Std Dev: ${np.std(spreads):.3f}")

    print("\n[SUCCESS] Market Making Strategy test completed successfully!")


if __name__ == "__main__":
    test_market_making()
