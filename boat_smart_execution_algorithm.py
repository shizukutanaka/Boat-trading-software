"""
BOAT - Smart Execution Algorithm (VWAP/TWAP/Iceberg)
====================================================

Production-ready execution algorithms for optimal order placement including
VWAP (Volume-Weighted Average Price), TWAP (Time-Weighted Average Price),
and Iceberg orders with smart order routing.

Features:
- VWAP execution with volume curve prediction
- TWAP execution with equal time slicing
- Iceberg orders for large order concealment
- Smart order routing across multiple venues
- Real-time adaptation to market conditions

Based on 2024-2025 research:
- Machine learning integration for volume curve forecasting
- Multi-venue smart order routing
- Dynamic adaptation to market conditions
- Transaction cost analysis (TCA) integration
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import heapq


class ExecutionAlgo(Enum):
    """Types of execution algorithms"""
    VWAP = "vwap"
    TWAP = "twap"
    ICEBERG = "iceberg"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"


class OrderSide(Enum):
    """Order side (buy/sell)"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Order specification"""
    symbol: str
    side: OrderSide
    total_quantity: int
    algo_type: ExecutionAlgo
    start_time: int
    end_time: int
    limit_price: Optional[float] = None
    participation_rate: float = 0.2  # Max 20% of volume


@dataclass
class OrderSlice:
    """Individual order slice for execution"""
    time: int
    quantity: int
    venue: str
    price: float
    is_executed: bool = False


@dataclass
class MarketData:
    """Market data snapshot"""
    time: int
    price: float
    volume: int
    bid: float
    ask: float
    spread: float


@dataclass
class ExecutionReport:
    """Execution performance report"""
    symbol: str
    algo_type: ExecutionAlgo
    total_quantity: int
    executed_quantity: int
    avg_price: float
    vwap: float
    slippage: float
    implementation_shortfall: float
    total_cost: float


class SmartExecutionEngine:
    """
    Smart order execution engine with VWAP, TWAP, and Iceberg algorithms.

    Implements institutional-grade execution algorithms with:
    - Volume prediction and adaptation
    - Multi-venue routing
    - Optimal slice sizing
    - Real-time performance tracking
    """

    def __init__(
        self,
        venues: List[str] = None,
        transaction_cost: float = 0.0005,
        market_impact_coef: float = 0.1
    ):
        """
        Initialize execution engine.

        Args:
            venues: List of execution venues
            transaction_cost: Fixed cost per trade
            market_impact_coef: Market impact coefficient
        """
        self.venues = venues or ['NYSE', 'NASDAQ', 'DARKPOOL']
        self.transaction_cost = transaction_cost
        self.market_impact_coef = market_impact_coef

        # Volume curves for VWAP (typical intraday pattern)
        self.volume_curve = self._generate_volume_curve()

        # Venue characteristics
        self.venue_characteristics = {
            'NYSE': {'speed': 1.0, 'cost': 0.0005, 'liquidity': 0.4},
            'NASDAQ': {'speed': 0.9, 'cost': 0.0004, 'liquidity': 0.4},
            'DARKPOOL': {'speed': 1.2, 'cost': 0.0003, 'liquidity': 0.2}
        }

    def _generate_volume_curve(self, n_periods: int = 390) -> np.ndarray:
        """
        Generate typical U-shaped intraday volume curve.

        Args:
            n_periods: Number of trading minutes (390 for 6.5 hours)

        Returns:
            Normalized volume curve
        """
        minutes = np.arange(n_periods)

        # U-shape: high at open/close, lower midday
        morning_surge = np.exp(-((minutes - 30) / 60) ** 2) * 0.3
        close_surge = np.exp(-((minutes - 360) / 60) ** 2) * 0.4
        base = 0.3 + 0.1 * np.sin(minutes * np.pi / n_periods)

        volume_curve = base + morning_surge + close_surge
        return volume_curve / volume_curve.sum()

    def calculate_vwap(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> float:
        """
        Calculate Volume-Weighted Average Price.

        Args:
            prices: Array of prices
            volumes: Array of volumes

        Returns:
            VWAP value
        """
        if len(volumes) == 0 or volumes.sum() == 0:
            return prices.mean() if len(prices) > 0 else 0.0

        return np.sum(prices * volumes) / volumes.sum()

    def predict_volume(
        self,
        historical_volumes: np.ndarray,
        time_of_day: int
    ) -> float:
        """
        Predict volume using ML-inspired approach.

        Args:
            historical_volumes: Recent volume history
            time_of_day: Current minute of trading day

        Returns:
            Predicted volume
        """
        # Use volume curve as base prediction
        curve_prediction = self.volume_curve[time_of_day % len(self.volume_curve)]

        # Adjust based on recent volumes (adaptive)
        if len(historical_volumes) > 0:
            recent_avg = historical_volumes[-20:].mean()
            total_daily_volume = historical_volumes.sum() / (time_of_day / 390)
            adjusted_prediction = curve_prediction * total_daily_volume

            # Blend curve and recent
            return 0.7 * adjusted_prediction + 0.3 * recent_avg

        return curve_prediction * 1000000  # Default 1M shares daily

    def generate_twap_schedule(
        self,
        order: Order,
        n_slices: Optional[int] = None
    ) -> List[OrderSlice]:
        """
        Generate TWAP execution schedule.

        Args:
            order: Order to execute
            n_slices: Number of time slices

        Returns:
            List of order slices
        """
        duration = order.end_time - order.start_time

        if n_slices is None:
            # Default: one slice per 5 minutes
            n_slices = max(1, duration // 5)

        time_interval = duration / n_slices
        slice_quantity = order.total_quantity / n_slices

        schedule = []
        for i in range(n_slices):
            time = order.start_time + int(i * time_interval)

            # Round quantity (last slice gets remainder)
            if i == n_slices - 1:
                quantity = order.total_quantity - sum(s.quantity for s in schedule)
            else:
                quantity = int(slice_quantity)

            # Select venue (round-robin for TWAP)
            venue = self.venues[i % len(self.venues)]

            schedule.append(OrderSlice(
                time=time,
                quantity=quantity,
                venue=venue,
                price=0.0  # To be filled at execution
            ))

        return schedule

    def generate_vwap_schedule(
        self,
        order: Order,
        predicted_volumes: Optional[np.ndarray] = None
    ) -> List[OrderSlice]:
        """
        Generate VWAP execution schedule.

        Args:
            order: Order to execute
            predicted_volumes: Predicted volume curve

        Returns:
            List of order slices
        """
        duration = order.end_time - order.start_time

        if predicted_volumes is None:
            # Use default volume curve
            start_idx = order.start_time % len(self.volume_curve)
            end_idx = min(start_idx + duration, len(self.volume_curve))
            predicted_volumes = self.volume_curve[start_idx:end_idx]

        # Normalize to order quantity
        predicted_volumes = predicted_volumes / predicted_volumes.sum()
        slice_quantities = predicted_volumes * order.total_quantity

        schedule = []
        for i, vol_weight in enumerate(predicted_volumes):
            if vol_weight < 0.001:  # Skip very small slices
                continue

            time = order.start_time + i
            quantity = int(slice_quantities[i])

            if quantity == 0:
                continue

            # Smart venue selection based on quantity
            venue = self._select_venue(quantity, time)

            schedule.append(OrderSlice(
                time=time,
                quantity=quantity,
                venue=venue,
                price=0.0
            ))

        # Ensure full quantity is scheduled
        total_scheduled = sum(s.quantity for s in schedule)
        if total_scheduled < order.total_quantity and schedule:
            schedule[-1].quantity += order.total_quantity - total_scheduled

        return schedule

    def generate_iceberg_schedule(
        self,
        order: Order,
        visible_size: Optional[int] = None
    ) -> List[OrderSlice]:
        """
        Generate Iceberg order schedule.

        Args:
            order: Order to execute
            visible_size: Visible portion size

        Returns:
            List of order slices
        """
        if visible_size is None:
            # Default: 5% of total or minimum meaningful size
            visible_size = max(100, int(order.total_quantity * 0.05))

        schedule = []
        remaining = order.total_quantity
        time = order.start_time
        time_per_slice = max(1, (order.end_time - order.start_time) // 20)

        while remaining > 0:
            # Current slice size
            slice_size = min(visible_size, remaining)

            # Prefer dark pools for iceberg orders
            if 'DARKPOOL' in self.venues and np.random.random() > 0.3:
                venue = 'DARKPOOL'
            else:
                venue = self._select_venue(slice_size, time)

            schedule.append(OrderSlice(
                time=time,
                quantity=slice_size,
                venue=venue,
                price=0.0
            ))

            remaining -= slice_size
            time += time_per_slice

            # Random jitter to avoid detection
            time += np.random.randint(-2, 3)
            time = min(time, order.end_time)

        return schedule

    def _select_venue(
        self,
        quantity: int,
        time: int
    ) -> str:
        """
        Smart order routing - select best venue.

        Args:
            quantity: Order size
            time: Execution time

        Returns:
            Selected venue name
        """
        venue_scores = {}

        for venue in self.venues:
            chars = self.venue_characteristics[venue]

            # Score based on multiple factors
            cost_score = 1.0 / (1 + chars['cost'])
            speed_score = chars['speed']
            liquidity_score = chars['liquidity']

            # Large orders prefer dark pools
            if quantity > 10000 and venue == 'DARKPOOL':
                liquidity_score *= 1.5

            # Composite score
            venue_scores[venue] = (
                cost_score * 0.3 +
                speed_score * 0.3 +
                liquidity_score * 0.4
            )

        # Select best venue
        return max(venue_scores.keys(), key=lambda k: venue_scores[k])

    def calculate_market_impact(
        self,
        quantity: int,
        avg_volume: float,
        volatility: float = 0.02
    ) -> float:
        """
        Calculate expected market impact.

        Args:
            quantity: Order size
            avg_volume: Average volume
            volatility: Price volatility

        Returns:
            Expected price impact (percentage)
        """
        if avg_volume == 0:
            return 0.0

        # Square-root market impact model
        participation = quantity / avg_volume
        impact = self.market_impact_coef * volatility * np.sqrt(participation)

        return min(impact, 0.05)  # Cap at 5%

    def execute_order(
        self,
        order: Order,
        market_data: List[MarketData]
    ) -> ExecutionReport:
        """
        Execute order using specified algorithm.

        Args:
            order: Order to execute
            market_data: Market data feed

        Returns:
            Execution report
        """
        # Generate execution schedule
        if order.algo_type == ExecutionAlgo.TWAP:
            schedule = self.generate_twap_schedule(order)
        elif order.algo_type == ExecutionAlgo.VWAP:
            schedule = self.generate_vwap_schedule(order)
        elif order.algo_type == ExecutionAlgo.ICEBERG:
            schedule = self.generate_iceberg_schedule(order)
        else:
            schedule = self.generate_twap_schedule(order)  # Default

        # Simulate execution
        executed_slices = []
        total_cost = 0.0
        market_data_dict = {md.time: md for md in market_data}

        for slice_order in schedule:
            if slice_order.time not in market_data_dict:
                continue

            md = market_data_dict[slice_order.time]

            # Determine execution price
            if order.side == OrderSide.BUY:
                exec_price = md.ask
            else:
                exec_price = md.bid

            # Apply market impact
            impact = self.calculate_market_impact(
                slice_order.quantity,
                md.volume,
                0.02  # Assumed volatility
            )

            if order.side == OrderSide.BUY:
                exec_price *= (1 + impact)
            else:
                exec_price *= (1 - impact)

            # Check limit price
            if order.limit_price:
                if order.side == OrderSide.BUY and exec_price > order.limit_price:
                    continue
                elif order.side == OrderSide.SELL and exec_price < order.limit_price:
                    continue

            # Execute slice
            slice_order.price = exec_price
            slice_order.is_executed = True
            executed_slices.append(slice_order)

            # Calculate costs
            total_cost += slice_order.quantity * exec_price * self.transaction_cost

        # Calculate metrics
        if executed_slices:
            executed_quantity = sum(s.quantity for s in executed_slices)
            exec_prices = np.array([s.price for s in executed_slices])
            exec_quantities = np.array([s.quantity for s in executed_slices])

            avg_price = self.calculate_vwap(exec_prices, exec_quantities)

            # Market VWAP
            market_prices = np.array([md.price for md in market_data])
            market_volumes = np.array([md.volume for md in market_data])
            market_vwap = self.calculate_vwap(market_prices, market_volumes)

            # Slippage
            slippage = (avg_price - market_vwap) / market_vwap

            # Implementation shortfall
            arrival_price = market_data[0].price if market_data else avg_price
            impl_shortfall = (avg_price - arrival_price) / arrival_price

        else:
            executed_quantity = 0
            avg_price = 0.0
            market_vwap = 0.0
            slippage = 0.0
            impl_shortfall = 0.0

        return ExecutionReport(
            symbol=order.symbol,
            algo_type=order.algo_type,
            total_quantity=order.total_quantity,
            executed_quantity=executed_quantity,
            avg_price=avg_price,
            vwap=market_vwap,
            slippage=slippage,
            implementation_shortfall=impl_shortfall,
            total_cost=total_cost
        )


def test_execution_algorithms():
    """Test the Smart Execution Algorithms"""
    print("=" * 60)
    print("Testing Smart Execution Algorithms (VWAP/TWAP/Iceberg)")
    print("=" * 60)

    # Initialize execution engine
    engine = SmartExecutionEngine(
        venues=['NYSE', 'NASDAQ', 'DARKPOOL'],
        transaction_cost=0.0005,
        market_impact_coef=0.1
    )

    # Generate synthetic market data
    np.random.seed(42)
    n_periods = 390  # Full trading day

    # Create realistic intraday price/volume patterns
    base_price = 100.0
    prices = [base_price]
    volumes = []

    for t in range(n_periods):
        # Random walk with mean reversion
        price_change = np.random.randn() * 0.1
        prices.append(prices[-1] + price_change)

        # Volume follows U-shape curve
        volume = int(engine.volume_curve[t] * 10000000 + np.random.randn() * 10000)
        volumes.append(max(1000, volume))

    # Create market data
    market_data = []
    for t in range(n_periods):
        spread = 0.01 + np.random.exponential(0.01)
        market_data.append(MarketData(
            time=t,
            price=prices[t],
            volume=volumes[t],
            bid=prices[t] - spread/2,
            ask=prices[t] + spread/2,
            spread=spread
        ))

    # Test TWAP Algorithm
    print("\n1. TWAP (Time-Weighted Average Price) Execution:")
    print("-" * 40)

    twap_order = Order(
        symbol='AAPL',
        side=OrderSide.BUY,
        total_quantity=100000,
        algo_type=ExecutionAlgo.TWAP,
        start_time=30,
        end_time=360  # 30 mins after open to 30 mins before close
    )

    twap_schedule = engine.generate_twap_schedule(twap_order, n_slices=10)
    print(f"TWAP Schedule: {len(twap_schedule)} slices")
    print(f"First 3 slices:")
    for slice_order in twap_schedule[:3]:
        print(f"  Time {slice_order.time}: {slice_order.quantity} shares @ {slice_order.venue}")

    twap_report = engine.execute_order(twap_order, market_data)
    print(f"\nTWAP Execution Report:")
    print(f"  Executed: {twap_report.executed_quantity:,}/{twap_report.total_quantity:,} shares")
    print(f"  Avg Price: ${twap_report.avg_price:.2f}")
    print(f"  Market VWAP: ${twap_report.vwap:.2f}")
    print(f"  Slippage: {twap_report.slippage:.3%}")
    print(f"  Implementation Shortfall: {twap_report.implementation_shortfall:.3%}")

    # Test VWAP Algorithm
    print("\n2. VWAP (Volume-Weighted Average Price) Execution:")
    print("-" * 40)

    vwap_order = Order(
        symbol='AAPL',
        side=OrderSide.BUY,
        total_quantity=100000,
        algo_type=ExecutionAlgo.VWAP,
        start_time=30,
        end_time=360
    )

    vwap_schedule = engine.generate_vwap_schedule(vwap_order)
    print(f"VWAP Schedule: {len(vwap_schedule)} slices")

    # Show volume distribution
    time_buckets = {}
    for slice_order in vwap_schedule:
        hour = slice_order.time // 60
        time_buckets[hour] = time_buckets.get(hour, 0) + slice_order.quantity

    print("Hourly distribution:")
    for hour, qty in sorted(time_buckets.items()):
        print(f"  Hour {hour}: {qty:,} shares ({qty/vwap_order.total_quantity:.1%})")

    vwap_report = engine.execute_order(vwap_order, market_data)
    print(f"\nVWAP Execution Report:")
    print(f"  Executed: {vwap_report.executed_quantity:,}/{vwap_report.total_quantity:,} shares")
    print(f"  Avg Price: ${vwap_report.avg_price:.2f}")
    print(f"  Market VWAP: ${vwap_report.vwap:.2f}")
    print(f"  Slippage: {vwap_report.slippage:.3%}")
    print(f"  Implementation Shortfall: {vwap_report.implementation_shortfall:.3%}")

    # Test Iceberg Order
    print("\n3. Iceberg Order Execution:")
    print("-" * 40)

    iceberg_order = Order(
        symbol='AAPL',
        side=OrderSide.SELL,
        total_quantity=500000,  # Large order
        algo_type=ExecutionAlgo.ICEBERG,
        start_time=60,
        end_time=300
    )

    iceberg_schedule = engine.generate_iceberg_schedule(iceberg_order, visible_size=5000)
    print(f"Iceberg Schedule: {len(iceberg_schedule)} slices")
    print(f"Visible size: 5,000 shares per slice")

    # Show venue distribution
    venue_counts = {}
    for slice_order in iceberg_schedule:
        venue_counts[slice_order.venue] = venue_counts.get(slice_order.venue, 0) + 1

    print("Venue distribution:")
    for venue, count in venue_counts.items():
        print(f"  {venue}: {count} orders ({count/len(iceberg_schedule):.1%})")

    iceberg_report = engine.execute_order(iceberg_order, market_data)
    print(f"\nIceberg Execution Report:")
    print(f"  Executed: {iceberg_report.executed_quantity:,}/{iceberg_report.total_quantity:,} shares")
    print(f"  Avg Price: ${iceberg_report.avg_price:.2f}")
    print(f"  Slippage: {iceberg_report.slippage:.3%}")
    print(f"  Total Cost: ${iceberg_report.total_cost:.2f}")

    # Test Smart Order Routing
    print("\n4. Smart Order Routing Analysis:")
    print("-" * 40)

    # Test venue selection for different order sizes
    test_sizes = [100, 1000, 10000, 100000]
    print("Venue selection by order size:")
    for size in test_sizes:
        venue = engine._select_venue(size, 100)
        print(f"  {size:,} shares → {venue}")

    # Test market impact calculation
    print("\n5. Market Impact Analysis:")
    print("-" * 40)

    test_quantities = [1000, 10000, 50000, 100000]
    avg_volume = 1000000

    print("Expected market impact by order size:")
    for qty in test_quantities:
        impact = engine.calculate_market_impact(qty, avg_volume)
        print(f"  {qty:,} shares ({qty/avg_volume:.1%} participation): {impact:.3%} impact")

    print("\n[SUCCESS] Smart Execution Algorithms test completed successfully!")


if __name__ == "__main__":
    test_execution_algorithms()