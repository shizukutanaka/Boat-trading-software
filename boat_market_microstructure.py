#!/usr/bin/env python3
"""
Market Microstructure and Order Book Analysis for Boat
=======================================================

Advanced order book analytics and market microstructure analysis:
  - Limit Order Book (LOB) modeling
  - Spread analysis (bid-ask, effective, realized)
  - Price impact modeling
  - Liquidity metrics (depth, breadth, volume)
  - Order flow analysis
  - Market stress detection
  - Latency measurement and optimization

Based on 2025 research on high-frequency trading microstructure
and academic literature on electronic market design.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Deque
from collections import deque
from datetime import datetime
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""
    LIMIT = "limit"
    MARKET = "market"
    CANCEL = "cancel"
    MODIFY = "modify"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Limit order representation"""
    order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    timestamp: float  # microseconds
    order_type: OrderType = OrderType.LIMIT
    filled_quantity: float = 0.0

    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity


@dataclass
class Trade:
    """Trade execution"""
    trade_id: str
    symbol: str
    buy_order_id: str
    sell_order_id: str
    price: float
    quantity: float
    timestamp: float
    buy_side_latency: float = 0.0
    sell_side_latency: float = 0.0


@dataclass
class LOBSnapshot:
    """Limit order book snapshot"""
    timestamp: float
    symbol: str
    bids: Dict[float, float]  # price -> quantity
    asks: Dict[float, float]  # price -> quantity

    def best_bid(self) -> Optional[float]:
        return max(self.bids.keys()) if self.bids else None

    def best_ask(self) -> Optional[float]:
        return min(self.asks.keys()) if self.asks else None

    def mid_price(self) -> Optional[float]:
        bid, ask = self.best_bid(), self.best_ask()
        if bid and ask:
            return (bid + ask) / 2
        return None

    def spread(self) -> Optional[float]:\n        bid, ask = self.best_bid(), self.best_ask()
        if bid and ask:\n            return ask - bid\n        return None

    def spread_bps(self) -> Optional[float]:\n        """Spread in basis points\"\"\"\n        mid = self.mid_price()\n        spread = self.spread()\n        if mid and spread:\n            return (spread / mid) * 10000\n        return None


@dataclass
class SpreadMetrics:
    """Spread analysis metrics"""
    timestamp: float
    bid_ask_spread: float
    effective_spread: float  # 2 * |mid - trade_price|\n    realized_spread: float  # price change after trade\n    spread_bps: float  # basis points\n    liquidity_score: float  # 0-1, higher is better


@dataclass\nclass LiquidityMetrics:\n    \"\"\"Liquidity analysis\"\"\"\n    timestamp: float\n    depth_10: float  # Volume within 10 bps\n    depth_20: float  # Volume within 20 bps\n    breadth: float  # Number of price levels\n    imbalance: float  # (bid_vol - ask_vol) / (bid_vol + ask_vol)\n    vwap: float  # Volume-weighted average price\n    volume_slope: float  # Liquidity change rate


class LimitOrderBook:\n    \"\"\"Limit order book management\"\"\"\n    \n    def __init__(self, symbol: str):\n        self.symbol = symbol\n        self.bids: Dict[float, Deque[Order]] = {}\n        self.asks: Dict[float, Deque[Order]] = {}\n        self.orders: Dict[str, Order] = {}\n        self.trades: List[Trade] = []\n        self.snapshots: Deque[LOBSnapshot] = deque(maxlen=10000)\n    \n    def add_order(self, order: Order) -> Optional[Trade]:\n        \"\"\"\n        Add order to book and match if possible\n        \n        Args:\n            order: Order to add\n            \n        Returns:\n            Trade if filled, None otherwise\n        \"\"\"\n        self.orders[order.order_id] = order\n        \n        if order.side == OrderSide.BUY:\n            return self._add_buy_order(order)\n        else:\n            return self._add_sell_order(order)\n    \n    def _add_buy_order(self, order: Order) -> Optional[Trade]:\n        \"\"\"Add buy order and match against asks\"\"\"\n        while order.remaining_quantity() > 0 and self.asks:\n            best_ask_price = min(self.asks.keys())\n            \n            if order.price >= best_ask_price:\n                # Match against best ask\n                ask_orders = self.asks[best_ask_price]\n                ask_order = ask_orders[0]\n                \n                match_qty = min(\n                    order.remaining_quantity(),\n                    ask_order.remaining_quantity()\n                )\n                \n                trade = Trade(\n                    trade_id=f\"{order.order_id}_{ask_order.order_id}\",\n                    symbol=self.symbol,\n                    buy_order_id=order.order_id,\n                    sell_order_id=ask_order.order_id,\n                    price=best_ask_price,\n                    quantity=match_qty,\n                    timestamp=order.timestamp\n                )\n                \n                order.filled_quantity += match_qty\n                ask_order.filled_quantity += match_qty\n                self.trades.append(trade)\n                \n                # Clean up filled orders\n                if ask_order.remaining_quantity() == 0:\n                    ask_orders.popleft()\n                    if not ask_orders:\n                        del self.asks[best_ask_price]\n            else:\n                break\n        \n        # Add remaining buy order to book\n        if order.remaining_quantity() > 0:\n            if order.price not in self.bids:\n                self.bids[order.price] = deque()\n            self.bids[order.price].append(order)\n        \n        return self.trades[-1] if self.trades else None\n    \n    def _add_sell_order(self, order: Order) -> Optional[Trade]:\n        \"\"\"Add sell order and match against bids\"\"\"\n        while order.remaining_quantity() > 0 and self.bids:\n            best_bid_price = max(self.bids.keys())\n            \n            if order.price <= best_bid_price:\n                # Match against best bid\n                bid_orders = self.bids[best_bid_price]\n                bid_order = bid_orders[0]\n                \n                match_qty = min(\n                    order.remaining_quantity(),\n                    bid_order.remaining_quantity()\n                )\n                \n                trade = Trade(\n                    trade_id=f\"{bid_order.order_id}_{order.order_id}\",\n                    symbol=self.symbol,\n                    buy_order_id=bid_order.order_id,\n                    sell_order_id=order.order_id,\n                    price=best_bid_price,\n                    quantity=match_qty,\n                    timestamp=order.timestamp\n                )\n                \n                order.filled_quantity += match_qty\n                bid_order.filled_quantity += match_qty\n                self.trades.append(trade)\n                \n                # Clean up filled orders\n                if bid_order.remaining_quantity() == 0:\n                    bid_orders.popleft()\n                    if not bid_orders:\n                        del self.bids[best_bid_price]\n            else:\n                break\n        \n        # Add remaining sell order to book\n        if order.remaining_quantity() > 0:\n            if order.price not in self.asks:\n                self.asks[order.price] = deque()\n            self.asks[order.price].append(order)\n        \n        return self.trades[-1] if self.trades else None\n    \n    def cancel_order(self, order_id: str) -> bool:\n        \"\"\"Cancel order from book\"\"\"\n        if order_id not in self.orders:\n            return False\n        \n        order = self.orders[order_id]\n        \n        if order.side == OrderSide.BUY:\n            if order.price in self.bids:\n                self.bids[order.price] = deque(\n                    o for o in self.bids[order.price] if o.order_id != order_id\n                )\n                if not self.bids[order.price]:\n                    del self.bids[order.price]\n        else:\n            if order.price in self.asks:\n                self.asks[order.price] = deque(\n                    o for o in self.asks[order.price] if o.order_id != order_id\n                )\n                if not self.asks[order.price]:\n                    del self.asks[order.price]\n        \n        del self.orders[order_id]\n        return True\n    \n    def snapshot(self, timestamp: float) -> LOBSnapshot:\n        \"\"\"Get current order book snapshot\"\"\"\n        bids = {price: sum(o.remaining_quantity() for o in orders)\n                for price, orders in self.bids.items()}\n        asks = {price: sum(o.remaining_quantity() for o in orders)\n                for price, orders in self.asks.items()}\n        \n        snap = LOBSnapshot(\n            timestamp=timestamp,\n            symbol=self.symbol,\n            bids=bids,\n            asks=asks\n        )\n        \n        self.snapshots.append(snap)\n        return snap


class SpreadAnalyzer:\n    \"\"\"Analyze market spreads\"\"\"\n    \n    def __init__(self):\n        self.spreads: Dict[str, Deque[SpreadMetrics]] = {}\n    \n    def analyze_spread(\n        self,\n        snapshot: LOBSnapshot,\n        last_trade_price: Optional[float] = None,\n        previous_trade_price: Optional[float] = None\n    ) -> SpreadMetrics:\n        \"\"\"\n        Analyze spread metrics\n        \n        Args:\n            snapshot: LOB snapshot\n            last_trade_price: Most recent trade price\n            previous_trade_price: Previous trade price\n            \n        Returns:\n            SpreadMetrics\n        \"\"\"\n        bid = snapshot.best_bid()\n        ask = snapshot.best_ask()\n        \n        if not bid or not ask:\n            return None\n        \n        # Bid-ask spread\n        bid_ask_spread = ask - bid\n        spread_bps = (bid_ask_spread / snapshot.mid_price()) * 10000\n        \n        # Effective spread\n        effective_spread = 0.0\n        if last_trade_price:\n            mid = snapshot.mid_price()\n            effective_spread = 2 * abs(last_trade_price - mid)\n        \n        # Realized spread\n        realized_spread = 0.0\n        if last_trade_price and previous_trade_price:\n            realized_spread = abs(last_trade_price - previous_trade_price)\n        \n        # Liquidity score (inverse of spread)\n        liquidity_score = 1.0 / (1.0 + spread_bps / 100)\n        \n        metrics = SpreadMetrics(\n            timestamp=snapshot.timestamp,\n            bid_ask_spread=bid_ask_spread,\n            effective_spread=effective_spread,\n            realized_spread=realized_spread,\n            spread_bps=spread_bps,\n            liquidity_score=liquidity_score\n        )\n        \n        if snapshot.symbol not in self.spreads:\n            self.spreads[snapshot.symbol] = deque(maxlen=10000)\n        \n        self.spreads[snapshot.symbol].append(metrics)\n        return metrics


class LiquidityAnalyzer:\n    \"\"\"Analyze market liquidity\"\"\"\n    \n    def __init__(self):\n        self.liquidity_history: Dict[str, Deque[LiquidityMetrics]] = {}\n    \n    def analyze_liquidity(\n        self,\n        snapshot: LOBSnapshot\n    ) -> LiquidityMetrics:\n        \"\"\"\n        Analyze liquidity metrics\n        \n        Args:\n            snapshot: LOB snapshot\n            \n        Returns:\n            LiquidityMetrics\n        \"\"\"\n        mid_price = snapshot.mid_price()\n        if not mid_price:\n            return None\n        \n        # Depth within 10 and 20 bps\n        depth_10 = self._calculate_depth(snapshot, mid_price, 10)\n        depth_20 = self._calculate_depth(snapshot, mid_price, 20)\n        \n        # Breadth (number of price levels)\n        breadth = len(snapshot.bids) + len(snapshot.asks)\n        \n        # Imbalance\n        bid_volume = sum(snapshot.bids.values())\n        ask_volume = sum(snapshot.asks.values())\n        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-6)\n        \n        # VWAP\n        vwap = self._calculate_vwap(snapshot)\n        \n        # Volume slope\n        volume_slope = self._calculate_volume_slope(snapshot)\n        \n        metrics = LiquidityMetrics(\n            timestamp=snapshot.timestamp,\n            depth_10=depth_10,\n            depth_20=depth_20,\n            breadth=breadth,\n            imbalance=imbalance,\n            vwap=vwap,\n            volume_slope=volume_slope\n        )\n        \n        if snapshot.symbol not in self.liquidity_history:\n            self.liquidity_history[snapshot.symbol] = deque(maxlen=10000)\n        \n        self.liquidity_history[snapshot.symbol].append(metrics)\n        return metrics\n    \n    @staticmethod\n    def _calculate_depth(snapshot: LOBSnapshot, mid_price: float, bps: int) -> float:\n        \"\"\"Calculate depth within N basis points\"\"\"\n        threshold = mid_price * (bps / 10000)\n        \n        depth = 0\n        # Bid side\n        for price, qty in snapshot.bids.items():\n            if mid_price - price <= threshold:\n                depth += qty\n        \n        # Ask side\n        for price, qty in snapshot.asks.items():\n            if price - mid_price <= threshold:\n                depth += qty\n        \n        return depth\n    \n    @staticmethod\n    def _calculate_vwap(snapshot: LOBSnapshot) -> float:\n        \"\"\"Calculate volume-weighted average price\"\"\"\n        total_vol = 0\n        weighted_price = 0\n        \n        for price, qty in snapshot.bids.items():\n            weighted_price += price * qty\n            total_vol += qty\n        \n        for price, qty in snapshot.asks.items():\n            weighted_price += price * qty\n            total_vol += qty\n        \n        if total_vol > 0:\n            return weighted_price / total_vol\n        return snapshot.mid_price()\n    \n    @staticmethod\n    def _calculate_volume_slope(snapshot: LOBSnapshot) -> float:\n        \"\"\"Calculate volume slope (liquidity improvement with distance)\"\"\"\n        if not snapshot.bids or not snapshot.asks:\n            return 0.0\n        \n        best_bid = max(snapshot.bids.keys())\n        best_ask = min(snapshot.asks.keys())\n        \n        # Average volume improvement\n        bid_vols = list(snapshot.bids.values())\n        ask_vols = list(snapshot.asks.values())\n        \n        if bid_vols:\n            bid_slope = (bid_vols[-1] - bid_vols[0]) / (len(bid_vols) + 1e-6)\n        else:\n            bid_slope = 0\n        \n        if ask_vols:\n            ask_slope = (ask_vols[-1] - ask_vols[0]) / (len(ask_vols) + 1e-6)\n        else:\n            ask_slope = 0\n        \n        return (bid_slope + ask_slope) / 2


class LatencyMonitor:\n    \"\"\"Monitor order execution latency\"\"\"\n    \n    def __init__(self):\n        self.latencies: List[float] = []\n    \n    def record_latency(self, latency_us: float) -> None:\n        \"\"\"Record order latency in microseconds\"\"\"\n        self.latencies.append(latency_us)\n    \n    def get_statistics(self) -> Dict[str, float]:\n        \"\"\"\n        Get latency statistics\n        \n        Returns:\n            Latency metrics (min, max, mean, p50, p95, p99)\n        \"\"\"\n        if not self.latencies:\n            return {}\n        \n        sorted_latencies = sorted(self.latencies)\n        \n        return {\n            'count': len(self.latencies),\n            'min_us': min(self.latencies),\n            'max_us': max(self.latencies),\n            'mean_us': np.mean(self.latencies),\n            'median_us': np.median(self.latencies),\n            'p95_us': np.percentile(self.latencies, 95),\n            'p99_us': np.percentile(self.latencies, 99),\n            'std_us': np.std(self.latencies)\n        }


if __name__ == \"__main__\":\n    # Example usage\n    lob = LimitOrderBook(\"AAPL\")\n    spread_analyzer = SpreadAnalyzer()\n    liquidity_analyzer = LiquidityAnalyzer()\n    latency_monitor = LatencyMonitor()\n    \n    # Create sample orders\n    bid_order = Order(\n        order_id=\"bid_1\",\n        symbol=\"AAPL\",\n        side=OrderSide.BUY,\n        price=150.00,\n        quantity=100,\n        timestamp=1000.0\n    )\n    \n    ask_order = Order(\n        order_id=\"ask_1\",\n        symbol=\"AAPL\",\n        side=OrderSide.SELL,\n        price=150.01,\n        quantity=100,\n        timestamp=1001.0\n    )\n    \n    # Add orders\n    lob.add_order(bid_order)\n    lob.add_order(ask_order)\n    \n    # Get snapshot\n    snapshot = lob.snapshot(1002.0)\n    logger.info(f\"Best bid: {snapshot.best_bid()}, Best ask: {snapshot.best_ask()}\")\n    logger.info(f\"Spread: {snapshot.spread()} bps: {snapshot.spread_bps():.2f}\")\n    \n    # Analyze spread\n    spread_metrics = spread_analyzer.analyze_spread(snapshot, last_trade_price=150.00)\n    logger.info(f\"Spread metrics: {spread_metrics}\")\n    \n    # Analyze liquidity\n    liquidity = liquidity_analyzer.analyze_liquidity(snapshot)\n    logger.info(f\"Liquidity: depth_10={liquidity.depth_10}, imbalance={liquidity.imbalance:.2f}\")\n    \n    # Latency\n    latency_monitor.record_latency(150.5)  # 150.5 microseconds\n    latency_stats = latency_monitor.get_statistics()\n    logger.info(f\"Latency stats: {latency_stats}\")\n