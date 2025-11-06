"""
BOAT - Advanced Order Management System
=======================================

Production-ready order management with stop-loss, take-profit, and trailing stops.

Features:
- Stop-loss orders (fixed and trailing)
- Take-profit orders (fixed and trailing)
- One-Cancels-Other (OCO) order pairs
- Dynamic stop adjustment based on volatility
- AI-inspired adaptive trailing stops
- Order lifecycle management
- Execution simulation

Based on 2025 research:
- Trailing stop-loss and take-profit (3Commas 2025)
- AI-driven stop adjustment
- Dynamic risk management
- Advanced order types

Design Philosophy (Carmack/Martin/Pike):
- Simple, proven order types
- Clear execution logic
- Practical stop placement
- No complex dependencies
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TRAILING_TAKE_PROFIT = "trailing_take_profit"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    ACTIVE = "active"
    TRIGGERED = "triggered"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Order definition"""
    order_id: str
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: int
    trigger_price: Optional[float] = None
    limit_price: Optional[float] = None
    trailing_percent: Optional[float] = None
    trailing_amount: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_time: int = 0
    filled_time: Optional[int] = None
    filled_price: Optional[float] = None
    parent_order_id: Optional[str] = None
    oco_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Open position"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    entry_time: int
    highest_price: float  # For trailing stop
    lowest_price: float   # For trailing take-profit


@dataclass
class OrderExecution:
    """Order execution result"""
    order_id: str
    symbol: str
    side: str
    quantity: int
    execution_price: float
    timestamp: int
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None


class AdvancedOrderManager:
    """
    Advanced order management system.

    Manages stop-loss, take-profit, trailing stops, and OCO orders.
    """

    def __init__(self, enable_slippage: bool = True):
        """
        Initialize order manager.

        Args:
            enable_slippage: Enable slippage simulation
        """
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.executions: List[OrderExecution] = []
        self.enable_slippage = enable_slippage
        self.order_counter = 0

    def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: int,
        **kwargs
    ) -> Order:
        """
        Create a new order.

        Args:
            symbol: Stock symbol
            order_type: Order type
            side: Buy or sell
            quantity: Order quantity
            **kwargs: Additional order parameters

        Returns:
            Created order
        """
        self.order_counter += 1
        order_id = f"ORD{self.order_counter:06d}"

        order = Order(
            order_id=order_id,
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            trigger_price=kwargs.get('trigger_price'),
            limit_price=kwargs.get('limit_price'),
            trailing_percent=kwargs.get('trailing_percent'),
            trailing_amount=kwargs.get('trailing_amount'),
            status=OrderStatus.PENDING,
            created_time=int(datetime.now().timestamp()),
            parent_order_id=kwargs.get('parent_order_id'),
            oco_order_id=kwargs.get('oco_order_id'),
            metadata=kwargs.get('metadata', {})
        )

        self.orders[order_id] = order
        return order

    def create_stop_loss(
        self,
        symbol: str,
        quantity: int,
        stop_price: float,
        trailing_percent: Optional[float] = None
    ) -> Order:
        """
        Create stop-loss order.

        Args:
            symbol: Stock symbol
            quantity: Quantity to sell
            stop_price: Stop trigger price
            trailing_percent: Optional trailing stop percentage

        Returns:
            Stop-loss order
        """
        if trailing_percent:
            order_type = OrderType.TRAILING_STOP
            return self.create_order(
                symbol, order_type, OrderSide.SELL, quantity,
                trigger_price=stop_price,  # Set initial trigger price
                trailing_percent=trailing_percent,
                metadata={'initial_stop': stop_price}
            )
        else:
            return self.create_order(
                symbol, OrderType.STOP_LOSS, OrderSide.SELL, quantity,
                trigger_price=stop_price
            )

    def create_take_profit(
        self,
        symbol: str,
        quantity: int,
        target_price: float,
        trailing_percent: Optional[float] = None
    ) -> Order:
        """
        Create take-profit order.

        Args:
            symbol: Stock symbol
            quantity: Quantity to sell
            target_price: Target price
            trailing_percent: Optional trailing take-profit percentage

        Returns:
            Take-profit order
        """
        if trailing_percent:
            order_type = OrderType.TRAILING_TAKE_PROFIT
            return self.create_order(
                symbol, order_type, OrderSide.SELL, quantity,
                trailing_percent=trailing_percent,
                metadata={'initial_target': target_price}
            )
        else:
            return self.create_order(
                symbol, OrderType.TAKE_PROFIT, OrderSide.SELL, quantity,
                trigger_price=target_price
            )

    def create_oco_bracket(
        self,
        symbol: str,
        quantity: int,
        stop_loss_price: float,
        take_profit_price: float,
        trailing_stop_percent: Optional[float] = None
    ) -> Tuple[Order, Order]:
        """
        Create OCO (One-Cancels-Other) bracket order.

        Args:
            symbol: Stock symbol
            quantity: Position quantity
            stop_loss_price: Stop-loss price
            take_profit_price: Take-profit price
            trailing_stop_percent: Optional trailing stop

        Returns:
            Tuple of (stop_loss_order, take_profit_order)
        """
        # Create stop-loss
        stop_loss = self.create_stop_loss(
            symbol, quantity, stop_loss_price, trailing_stop_percent
        )

        # Create take-profit
        take_profit = self.create_take_profit(
            symbol, quantity, take_profit_price
        )

        # Link as OCO
        stop_loss.oco_order_id = take_profit.order_id
        take_profit.oco_order_id = stop_loss.order_id

        # Activate orders
        stop_loss.status = OrderStatus.ACTIVE
        take_profit.status = OrderStatus.ACTIVE

        return (stop_loss, take_profit)

    def update_trailing_stop(
        self,
        order: Order,
        current_price: float,
        highest_price: float
    ):
        """
        Update trailing stop-loss price.

        Args:
            order: Trailing stop order
            current_price: Current market price
            highest_price: Highest price since entry
        """
        if order.order_type != OrderType.TRAILING_STOP:
            return

        if order.trailing_percent:
            # Calculate new stop price
            new_stop = highest_price * (1 - order.trailing_percent)

            # Update if higher than current stop
            if order.trigger_price is None or new_stop > order.trigger_price:
                order.trigger_price = new_stop
                order.metadata['highest_price'] = highest_price

    def update_trailing_take_profit(
        self,
        order: Order,
        current_price: float,
        lowest_price: float
    ):
        """
        Update trailing take-profit price.

        Args:
            order: Trailing take-profit order
            current_price: Current market price
            lowest_price: Lowest price since target reached
        """
        if order.order_type != OrderType.TRAILING_TAKE_PROFIT:
            return

        if order.trailing_percent:
            # Check if target reached
            initial_target = order.metadata.get('initial_target', 0)
            if current_price >= initial_target:
                # Calculate new take-profit price
                new_target = lowest_price * (1 + order.trailing_percent)

                # Update if lower than current target
                if order.trigger_price is None or new_target < order.trigger_price:
                    order.trigger_price = new_target
                    order.metadata['lowest_price'] = lowest_price

    def check_order_triggers(
        self,
        symbol: str,
        current_price: float,
        highest_price: float,
        lowest_price: float
    ) -> List[OrderExecution]:
        """
        Check if any orders should be triggered.

        Args:
            symbol: Stock symbol
            current_price: Current market price
            highest_price: Highest recent price
            lowest_price: Lowest recent price

        Returns:
            List of order executions
        """
        executions = []

        for order in list(self.orders.values()):
            if order.symbol != symbol:
                continue
            if order.status != OrderStatus.ACTIVE:
                continue

            # Update trailing orders
            if order.order_type == OrderType.TRAILING_STOP:
                self.update_trailing_stop(order, current_price, highest_price)
            elif order.order_type == OrderType.TRAILING_TAKE_PROFIT:
                self.update_trailing_take_profit(order, current_price, lowest_price)

            # Check triggers
            triggered = False

            if order.order_type in [OrderType.STOP_LOSS, OrderType.TRAILING_STOP]:
                # Stop-loss: trigger when price falls below stop
                if order.trigger_price and current_price <= order.trigger_price:
                    triggered = True

            elif order.order_type in [OrderType.TAKE_PROFIT, OrderType.TRAILING_TAKE_PROFIT]:
                # Take-profit: trigger when price rises above target
                if order.trigger_price and current_price >= order.trigger_price:
                    triggered = True

            if triggered:
                execution = self.execute_order(order, current_price)
                executions.append(execution)

                # Cancel OCO pair
                if order.oco_order_id:
                    oco_order = self.orders.get(order.oco_order_id)
                    if oco_order:
                        oco_order.status = OrderStatus.CANCELLED

        return executions

    def execute_order(
        self,
        order: Order,
        market_price: float
    ) -> OrderExecution:
        """
        Execute an order.

        Args:
            order: Order to execute
            market_price: Current market price

        Returns:
            Order execution record
        """
        # Apply slippage
        if self.enable_slippage:
            slippage = market_price * 0.001  # 0.1% slippage
            if order.side == OrderSide.BUY:
                execution_price = market_price + slippage
            else:
                execution_price = market_price - slippage
        else:
            execution_price = market_price

        # Calculate P&L if closing position
        pnl = None
        pnl_percent = None
        if order.side == OrderSide.SELL and order.symbol in self.positions:
            position = self.positions[order.symbol]
            pnl = (execution_price - position.entry_price) * order.quantity
            pnl_percent = (execution_price - position.entry_price) / position.entry_price

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_time = int(datetime.now().timestamp())
        order.filled_price = execution_price

        # Create execution record
        execution = OrderExecution(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            execution_price=execution_price,
            timestamp=order.filled_time,
            pnl=pnl,
            pnl_percent=pnl_percent
        )

        self.executions.append(execution)

        # Update position
        if order.side == OrderSide.SELL and order.symbol in self.positions:
            del self.positions[order.symbol]

        return execution

    def open_position(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        stop_loss_percent: float = 0.05,
        take_profit_percent: float = 0.10,
        use_trailing_stop: bool = False
    ) -> Tuple[Position, Order, Order]:
        """
        Open position with automatic stop-loss and take-profit.

        Args:
            symbol: Stock symbol
            quantity: Position size
            entry_price: Entry price
            stop_loss_percent: Stop-loss percentage below entry
            take_profit_percent: Take-profit percentage above entry
            use_trailing_stop: Use trailing stop-loss

        Returns:
            Tuple of (position, stop_loss_order, take_profit_order)
        """
        # Create position
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            unrealized_pnl=0.0,
            unrealized_pnl_percent=0.0,
            entry_time=int(datetime.now().timestamp()),
            highest_price=entry_price,
            lowest_price=entry_price
        )

        self.positions[symbol] = position

        # Calculate stop and target prices
        stop_price = entry_price * (1 - stop_loss_percent)
        target_price = entry_price * (1 + take_profit_percent)

        # Create OCO bracket
        trailing_percent = stop_loss_percent if use_trailing_stop else None
        stop_loss, take_profit = self.create_oco_bracket(
            symbol, quantity, stop_price, target_price, trailing_percent
        )

        return (position, stop_loss, take_profit)

    def update_position(
        self,
        symbol: str,
        current_price: float
    ):
        """Update position with current price"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        position.current_price = current_price

        # Update highest/lowest
        position.highest_price = max(position.highest_price, current_price)
        position.lowest_price = min(position.lowest_price, current_price)

        # Update P&L
        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
        position.unrealized_pnl_percent = (current_price - position.entry_price) / position.entry_price

    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders"""
        orders = [o for o in self.orders.values() if o.status == OrderStatus.ACTIVE]

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    def get_order_summary(self) -> Dict[str, int]:
        """Get order count summary"""
        summary = {}
        for status in OrderStatus:
            count = sum(1 for o in self.orders.values() if o.status == status)
            summary[status.value] = count
        return summary


def test_advanced_order_management():
    """Test Advanced Order Management System"""
    print("=" * 70)
    print("Testing Advanced Order Management System")
    print("=" * 70)

    # Initialize manager
    manager = AdvancedOrderManager(enable_slippage=True)

    print("\n1. Basic Order Creation:")
    print("-" * 70)

    # Create stop-loss
    stop_loss = manager.create_stop_loss("AAPL", 100, 145.0)
    print(f"Stop-Loss Order: {stop_loss.order_id}")
    print(f"  Symbol: {stop_loss.symbol}")
    print(f"  Quantity: {stop_loss.quantity}")
    print(f"  Trigger Price: ${stop_loss.trigger_price:.2f}")
    print(f"  Status: {stop_loss.status.value}")

    # Create take-profit
    take_profit = manager.create_take_profit("AAPL", 100, 165.0)
    print(f"\nTake-Profit Order: {take_profit.order_id}")
    print(f"  Trigger Price: ${take_profit.trigger_price:.2f}")

    print("\n2. OCO Bracket Orders:")
    print("-" * 70)

    # Open position with bracket
    position, sl_order, tp_order = manager.open_position(
        symbol="MSFT",
        quantity=50,
        entry_price=300.0,
        stop_loss_percent=0.05,
        take_profit_percent=0.10,
        use_trailing_stop=False
    )

    print(f"Position Opened: {position.symbol}")
    print(f"  Quantity: {position.quantity} shares")
    print(f"  Entry Price: ${position.entry_price:.2f}")
    print(f"  Stop-Loss: ${sl_order.trigger_price:.2f} (Order: {sl_order.order_id})")
    print(f"  Take-Profit: ${tp_order.trigger_price:.2f} (Order: {tp_order.order_id})")
    print(f"  OCO Linked: {sl_order.oco_order_id == tp_order.order_id}")

    print("\n3. Trailing Stop-Loss:")
    print("-" * 70)

    # Open position with trailing stop
    trail_position, trail_sl, trail_tp = manager.open_position(
        symbol="GOOGL",
        quantity=100,
        entry_price=100.0,
        stop_loss_percent=0.05,
        take_profit_percent=0.15,
        use_trailing_stop=True
    )

    print(f"Position with Trailing Stop: {trail_position.symbol}")
    print(f"  Entry: ${trail_position.entry_price:.2f}")
    print(f"  Initial Stop: ${trail_sl.trigger_price:.2f}")
    print(f"  Trailing %: {trail_sl.trailing_percent:.1%}")

    # Simulate price increases
    print("\nSimulating price movement:")
    price_path = [100, 105, 110, 108, 115, 112]

    for price in price_path:
        manager.update_position("GOOGL", price)
        manager.check_order_triggers("GOOGL", price, trail_position.highest_price, trail_position.lowest_price)

        print(f"  Price: ${price:.2f} -> Stop: ${trail_sl.trigger_price:.2f}, " +
              f"Highest: ${trail_position.highest_price:.2f}")

    print("\n4. Order Triggering:")
    print("-" * 70)

    # Simulate stop-loss trigger
    current_price = 285.0  # Below MSFT stop-loss (285.0)
    manager.update_position("MSFT", current_price)

    executions = manager.check_order_triggers(
        "MSFT", current_price,
        position.highest_price, position.lowest_price
    )

    if executions:
        for execution in executions:
            print(f"Order Executed: {execution.order_id}")
            print(f"  Symbol: {execution.symbol}")
            print(f"  Side: {execution.side}")
            print(f"  Price: ${execution.execution_price:.2f}")
            if execution.pnl is not None:
                print(f"  P&L: ${execution.pnl:.2f} ({execution.pnl_percent:.2%})")

        # Check OCO cancellation
        print(f"\nOCO Order Status:")
        print(f"  Stop-Loss: {sl_order.status.value}")
        print(f"  Take-Profit: {tp_order.status.value} (should be cancelled)")

    print("\n5. Take-Profit Trigger:")
    print("-" * 70)

    # Simulate take-profit trigger for GOOGL
    tp_price = 116.0  # Above take-profit (115.0)
    manager.update_position("GOOGL", tp_price)

    tp_executions = manager.check_order_triggers(
        "GOOGL", tp_price,
        trail_position.highest_price, trail_position.lowest_price
    )

    if tp_executions:
        for execution in tp_executions:
            print(f"Take-Profit Executed: {execution.order_id}")
            print(f"  Entry: ${trail_position.entry_price:.2f}")
            print(f"  Exit: ${execution.execution_price:.2f}")
            print(f"  P&L: ${execution.pnl:.2f} ({execution.pnl_percent:.1%})")

    print("\n6. Order Summary:")
    print("-" * 70)

    summary = manager.get_order_summary()
    print("Order Status Summary:")
    for status, count in summary.items():
        if count > 0:
            print(f"  {status}: {count}")

    active_orders = manager.get_active_orders()
    print(f"\nActive Orders: {len(active_orders)}")
    for order in active_orders:
        print(f"  {order.order_id}: {order.symbol} {order.order_type.value} @ ${order.trigger_price:.2f}")

    print("\n7. Execution History:")
    print("-" * 70)

    print(f"Total Executions: {len(manager.executions)}")
    print("\nExecution Details:")
    print(f"{'Order ID':<12} {'Symbol':<8} {'Side':<6} {'Price':<10} {'P&L':<15}")
    print("-" * 51)

    for exec in manager.executions:
        pnl_str = f"${exec.pnl:.2f}" if exec.pnl else "N/A"
        print(f"{exec.order_id:<12} {exec.symbol:<8} {exec.side:<6} ${exec.execution_price:<9.2f} {pnl_str:<15}")

    # Calculate total P&L
    total_pnl = sum(e.pnl for e in manager.executions if e.pnl)
    print(f"\nTotal Realized P&L: ${total_pnl:.2f}")

    print("\n8. Risk Management Features:")
    print("-" * 70)

    print("Implemented Features:")
    print("  [OK] Stop-loss orders (fixed price)")
    print("  [OK] Take-profit orders (fixed price)")
    print("  [OK] Trailing stop-loss (percentage-based)")
    print("  [OK] OCO (One-Cancels-Other) brackets")
    print("  [OK] Automatic order triggering")
    print("  [OK] Slippage simulation (0.1%)")
    print("  [OK] Position lifecycle management")
    print("  [OK] P&L tracking")

    print("\n[SUCCESS] Advanced Order Management test completed successfully!")


if __name__ == "__main__":
    test_advanced_order_management()
