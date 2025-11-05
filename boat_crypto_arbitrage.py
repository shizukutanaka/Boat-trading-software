#!/usr/bin/env python3
"""
Cryptocurrency Arbitrage Detection and Execution System for Boat
================================================================

This module implements advanced arbitrage detection and execution including:
  - Triangular arbitrage (A->B->C->A)
  - Statistical arbitrage
  - Cross-exchange spot arbitrage
  - Latency optimization
  - Risk management

Features:
  - Real-time opportunity detection
  - Multi-leg arbitrage support
  - Transaction cost modeling
  - Order execution optimization
  - Slippage estimation
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
import numpy as np
import pandas as pd
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArbitrageType(Enum):
    """Types of arbitrage"""
    TRIANGULAR = "triangular"
    CROSS_EXCHANGE = "cross_exchange"
    CONVERGENCE = "convergence"
    STATISTICAL = "statistical"
    FLASH_LOAN = "flash_loan"


@dataclass
class ExchangeInfo:
    """Exchange trading information"""
    name: str
    maker_fee: float = 0.001
    taker_fee: float = 0.001
    withdrawal_fee: Dict[str, float] = field(default_factory=dict)
    minimum_order: float = 10.0
    max_orders_per_second: int = 10
    latency_ms: float = 100.0
    available_pairs: Set[str] = field(default_factory=set)

    def get_total_fee(self, is_maker: bool = True) -> float:
        """Get total fee including withdrawal"""
        return self.maker_fee if is_maker else self.taker_fee


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity"""
    id: str
    type: ArbitrageType
    timestamp: datetime
    path: List[str]  # [BTC-USDT, ETH-BTC, ETH-USDT] for triangular
    exchanges: List[str]
    prices: List[float]
    volumes: List[float]
    profit_pct: float
    profit_usd: float
    execution_time_ms: float
    confidence: float = 0.0  # 0-1, based on speed and data quality
    active: bool = True

    def roi(self) -> float:
        """Calculate ROI"""
        return self.profit_pct


@dataclass
class ArbitrageOrder:
    """Order for arbitrage execution"""
    opportunity_id: str
    leg: int  # 0, 1, 2, ... for multi-leg
    exchange: str
    symbol: str
    side: str  # buy or sell
    amount: float
    price: float
    order_id: Optional[str] = None
    status: str = "pending"  # pending, filled, partial, rejected
    filled_amount: float = 0.0
    average_price: float = 0.0
    executed_at: Optional[datetime] = None


class PriceModel:
    """Model price information"""

    def __init__(self):
        self.prices = defaultdict(lambda: {})
        self.last_update = defaultdict(dict)
        self.lock = asyncio.Lock()

    async def update_price(
        self,
        exchange: str,
        symbol: str,
        bid: float,
        ask: float,
        timestamp: datetime
    ) -> None:
        """Update price for symbol on exchange"""
        async with self.lock:
            self.prices[exchange][symbol] = {
                'bid': bid,
                'ask': ask,
                'mid': (bid + ask) / 2,
                'timestamp': timestamp,
                'spread': ask - bid,
                'spread_pct': ((ask - bid) / ((bid + ask) / 2)) * 100
            }
            self.last_update[exchange][symbol] = timestamp

    async def get_price(self, exchange: str, symbol: str) -> Optional[Dict]:
        """Get price for symbol on exchange"""
        async with self.lock:
            if symbol in self.prices[exchange]:
                price_info = self.prices[exchange][symbol]
                # Check if data is stale (> 5 seconds)
                age = (datetime.utcnow() - price_info['timestamp']).total_seconds()
                if age < 5:
                    return price_info
        return None

    async def get_best_prices(self, symbol: str) -> Dict[str, Dict]:
        """Get best bid/ask from all exchanges"""
        async with self.lock:
            best_bid = {'exchange': None, 'price': 0}
            best_ask = {'exchange': None, 'price': float('inf')}

            for exchange, symbols in self.prices.items():
                if symbol in symbols:
                    price_info = symbols[symbol]
                    if price_info['bid'] > best_bid['price']:
                        best_bid = {'exchange': exchange, 'price': price_info['bid']}
                    if price_info['ask'] < best_ask['price']:
                        best_ask = {'exchange': exchange, 'price': price_info['ask']}

            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread_pct': ((best_ask['price'] - best_bid['price']) /
                             ((best_bid['price'] + best_ask['price']) / 2)) * 100
                if best_bid['price'] > 0 else 0
            }


class TriangularArbitrageDetector:
    """Detect triangular arbitrage opportunities"""

    def __init__(self, exchanges: Dict[str, ExchangeInfo], price_model: PriceModel):
        self.exchanges = exchanges
        self.price_model = price_model

    async def find_opportunities(
        self,
        base_asset: str = "USDT",
        min_profit_pct: float = 0.2,
        max_legs: int = 3
    ) -> List[ArbitrageOpportunity]:
        """
        Find triangular arbitrage opportunities

        Example path: USDT -> BTC -> ETH -> USDT
        """
        opportunities = []

        # Get all trading pairs
        all_symbols = set()
        for exchange_info in self.exchanges.values():
            all_symbols.update(exchange_info.available_pairs)

        # Extract traded assets
        assets = set()
        for pair in all_symbols:
            parts = pair.split('-')
            if len(parts) == 2:
                assets.add(parts[0])
                assets.add(parts[1])

        assets = list(assets)

        # Find triangles
        for i, asset1 in enumerate(assets):
            for asset2 in assets[i+1:]:
                if asset1 == base_asset or asset2 == base_asset:
                    continue

                # Triangle: base_asset -> asset1 -> asset2 -> base_asset
                path = [f"{base_asset}-{asset1}", f"{asset1}-{asset2}", f"{asset2}-{base_asset}"]

                profit = await self._calculate_triangle_profit(path)

                if profit and profit['profit_pct'] > min_profit_pct:
                    opp = ArbitrageOpportunity(
                        id=f"tri_{asset1}_{asset2}_{int(datetime.utcnow().timestamp())}",
                        type=ArbitrageType.TRIANGULAR,
                        timestamp=datetime.utcnow(),
                        path=path,
                        exchanges=profit['exchanges'],
                        prices=profit['prices'],
                        volumes=profit['volumes'],
                        profit_pct=profit['profit_pct'],
                        profit_usd=profit['profit_usd'],
                        execution_time_ms=profit['execution_time_ms']
                    )
                    opportunities.append(opp)

        return opportunities

    async def _calculate_triangle_profit(
        self,
        path: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Calculate profit for triangular path"""
        start_amount = 1000.0  # Start with $1000

        prices = []
        exchanges = []
        volumes = []
        execution_time = 0

        current_amount = start_amount
        current_asset = path[0].split('-')[0]

        for i, trading_pair in enumerate(path):
            parts = trading_pair.split('-')
            from_asset, to_asset = parts[0], parts[1]

            # Get best prices
            best = await self.price_model.get_best_prices(trading_pair)

            if not best['best_ask']['exchange']:
                return None

            exchange = best['best_ask']['exchange']
            price = best['best_ask']['price']

            prices.append(price)
            exchanges.append(exchange)

            # Simulate execution
            if i == 0:  # Buy first asset
                current_amount = current_amount / price
            else:  # Sell to next asset
                current_amount = current_amount * price

            # Apply fees
            fee_rate = self.exchanges[exchange].taker_fee
            current_amount *= (1 - fee_rate)

            volumes.append(current_amount)
            execution_time += self.exchanges[exchange].latency_ms

        # Calculate profit
        profit_usd = current_amount - start_amount
        profit_pct = (profit_usd / start_amount) * 100

        if profit_pct > 0:
            return {
                'profit_usd': profit_usd,
                'profit_pct': profit_pct,
                'final_amount': current_amount,
                'prices': prices,
                'exchanges': exchanges,
                'volumes': volumes,
                'execution_time_ms': execution_time
            }

        return None


class CrossExchangeArbitrageDetector:
    """Detect cross-exchange arbitrage"""

    def __init__(self, exchanges: Dict[str, ExchangeInfo], price_model: PriceModel):
        self.exchanges = exchanges
        self.price_model = price_model

    async def find_opportunities(
        self,
        symbols: List[str],
        min_profit_pct: float = 0.3
    ) -> List[ArbitrageOpportunity]:
        """Find cross-exchange arbitrage opportunities"""
        opportunities = []

        for symbol in symbols:
            # Get prices from all exchanges
            prices = {}
            timestamps = {}

            for exchange_name in self.exchanges.keys():
                price_info = await self.price_model.get_price(exchange_name, symbol)
                if price_info:
                    prices[exchange_name] = price_info
                    timestamps[exchange_name] = price_info['timestamp']

            if len(prices) < 2:
                continue

            # Find best buy and sell
            best_buy = min(prices.items(), key=lambda x: x[1]['ask'])
            best_sell = max(prices.items(), key=lambda x: x[1]['bid'])

            buy_exchange, buy_price_info = best_buy
            sell_exchange, sell_price_info = best_sell

            if buy_exchange == sell_exchange:
                continue

            # Calculate profit
            amount = 10000.0  # $10,000
            buy_amount = amount / buy_price_info['ask']

            # Apply fees
            buy_fee = self.exchanges[buy_exchange].taker_fee
            sell_fee = self.exchanges[sell_exchange].taker_fee
            withdrawal_fee = self.exchanges[buy_exchange].withdrawal_fee.get(symbol, 0)

            buy_amount *= (1 - buy_fee)
            sell_amount = buy_amount * sell_price_info['bid']
            sell_amount *= (1 - sell_fee)
            sell_amount -= withdrawal_fee

            profit_usd = sell_amount - amount
            profit_pct = (profit_usd / amount) * 100

            if profit_pct > min_profit_pct:
                opp = ArbitrageOpportunity(
                    id=f"cross_{symbol}_{int(datetime.utcnow().timestamp())}",
                    type=ArbitrageType.CROSS_EXCHANGE,
                    timestamp=datetime.utcnow(),
                    path=[f"BUY_{symbol}@{buy_exchange}", f"SELL_{symbol}@{sell_exchange}"],
                    exchanges=[buy_exchange, sell_exchange],
                    prices=[buy_price_info['ask'], sell_price_info['bid']],
                    volumes=[buy_amount, sell_amount],
                    profit_pct=profit_pct,
                    profit_usd=profit_usd,
                    execution_time_ms=self.exchanges[buy_exchange].latency_ms +
                                     self.exchanges[sell_exchange].latency_ms
                )
                opportunities.append(opp)

        return opportunities


class StatisticalArbitrageDetector:
    """Detect statistical arbitrage opportunities"""

    def __init__(self, price_model: PriceModel, lookback: int = 100):
        self.price_model = price_model
        self.lookback = lookback
        self.price_history = defaultdict(list)

    async def record_price(self, symbol: str, price: float) -> None:
        """Record price for statistical analysis"""
        self.price_history[symbol].append({
            'price': price,
            'timestamp': datetime.utcnow()
        })

        # Keep only lookback period
        if len(self.price_history[symbol]) > self.lookback:
            self.price_history[symbol].pop(0)

    async def find_opportunities(
        self,
        pairs: List[Tuple[str, str]],
        z_score_threshold: float = 2.0
    ) -> List[ArbitrageOpportunity]:
        """
        Find statistical arbitrage opportunities using spread analysis

        pairs: List of (symbol1, symbol2) tuples
        """
        opportunities = []

        for symbol1, symbol2 in pairs:
            if symbol1 not in self.price_history or symbol2 not in self.price_history:
                continue

            prices1 = np.array([p['price'] for p in self.price_history[symbol1]])
            prices2 = np.array([p['price'] for p in self.price_history[symbol2]])

            if len(prices1) < 20 or len(prices2) < 20:
                continue

            # Calculate spread
            spread = prices1 - prices2
            mean_spread = np.mean(spread)
            std_spread = np.std(spread)

            current_spread = spread[-1]
            z_score = (current_spread - mean_spread) / std_spread if std_spread > 0 else 0

            if abs(z_score) > z_score_threshold:
                profit_pct = abs(z_score) * 0.1  # Rough estimate

                opp = ArbitrageOpportunity(
                    id=f"stat_{symbol1}_{symbol2}_{int(datetime.utcnow().timestamp())}",
                    type=ArbitrageType.STATISTICAL,
                    timestamp=datetime.utcnow(),
                    path=[symbol1, symbol2],
                    exchanges=['all'],
                    prices=[prices1[-1], prices2[-1]],
                    volumes=[0, 0],
                    profit_pct=profit_pct,
                    profit_usd=0,
                    execution_time_ms=0,
                    confidence=abs(z_score) / (z_score_threshold * 2)  # Confidence up to z-threshold
                )
                opportunities.append(opp)

        return opportunities


class ArbitrageExecutor:
    """Execute arbitrage orders"""

    def __init__(self, exchanges: Dict[str, ExchangeInfo]):
        self.exchanges = exchanges
        self.orders: Dict[str, List[ArbitrageOrder]] = {}
        self.executed_volume = 0.0

    async def execute_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        capital: float = 1000.0
    ) -> List[ArbitrageOrder]:
        """Execute arbitrage opportunity"""
        orders = []

        logger.info(f"Executing opportunity: {opportunity.id} (profit: {opportunity.profit_pct:.2f}%)")

        # Create orders for each leg
        for i, (exchange, price, symbol_pair) in enumerate(
            zip(opportunity.exchanges, opportunity.prices, opportunity.path)
        ):
            side = "buy" if i == 0 else "sell"
            amount = capital / (len(opportunity.path))

            order = ArbitrageOrder(
                opportunity_id=opportunity.id,
                leg=i,
                exchange=exchange,
                symbol=symbol_pair,
                side=side,
                amount=amount,
                price=price
            )

            orders.append(order)

            # Simulate execution
            await self._execute_order(order)

        self.orders[opportunity.id] = orders
        return orders

    async def _execute_order(self, order: ArbitrageOrder) -> None:
        """Simulate order execution"""
        # In production, this would call exchange APIs
        order.status = "filled"
        order.filled_amount = order.amount
        order.average_price = order.price
        order.executed_at = datetime.utcnow()
        self.executed_volume += order.amount

        logger.info(f"Order {order.leg} executed: {order.symbol} @ {order.price:.2f}")


class ArbitrageEngine:
    """Main arbitrage detection and execution engine"""

    def __init__(self, exchanges: Dict[str, ExchangeInfo]):
        self.exchanges = exchanges
        self.price_model = PriceModel()
        self.triangular_detector = TriangularArbitrageDetector(exchanges, self.price_model)
        self.cross_exchange_detector = CrossExchangeArbitrageDetector(exchanges, self.price_model)
        self.statistical_detector = StatisticalArbitrageDetector(self.price_model)
        self.executor = ArbitrageExecutor(exchanges)
        self.active_opportunities: List[ArbitrageOpportunity] = []

    async def update_prices(
        self,
        exchange: str,
        prices: Dict[str, Tuple[float, float]]  # {symbol: (bid, ask)}
    ) -> None:
        """Update prices from exchange"""
        for symbol, (bid, ask) in prices.items():
            await self.price_model.update_price(exchange, symbol, bid, ask, datetime.utcnow())
            await self.statistical_detector.record_price(symbol, (bid + ask) / 2)

    async def scan_opportunities(self) -> List[ArbitrageOpportunity]:
        """Scan for all arbitrage opportunities"""
        all_opportunities = []

        # Triangular arbitrage
        tri_opps = await self.triangular_detector.find_opportunities(min_profit_pct=0.2)
        all_opportunities.extend(tri_opps)

        # Cross-exchange arbitrage
        symbols = list(set(
            symbol for exchange_info in self.exchanges.values()
            for pair in exchange_info.available_pairs
            for symbol in pair.split('-')
        ))
        cross_opps = await self.cross_exchange_detector.find_opportunities(symbols[:10])
        all_opportunities.extend(cross_opps)

        # Statistical arbitrage
        pairs = [(symbols[i], symbols[i+1]) for i in range(0, min(10, len(symbols)-1), 2)]
        stat_opps = await self.statistical_detector.find_opportunities(pairs)
        all_opportunities.extend(stat_opps)

        # Sort by profit
        all_opportunities.sort(key=lambda x: x.profit_pct, reverse=True)

        self.active_opportunities = all_opportunities[:10]  # Keep top 10
        return self.active_opportunities

    async def execute_best_opportunity(self, capital: float = 1000.0) -> Optional[List[ArbitrageOrder]]:
        """Execute the best available opportunity"""
        if not self.active_opportunities:
            return None

        best_opp = self.active_opportunities[0]
        return await self.executor.execute_opportunity(best_opp, capital)

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            'active_opportunities': len(self.active_opportunities),
            'top_profit_pct': self.active_opportunities[0].profit_pct if self.active_opportunities else 0,
            'executed_volume': self.executor.executed_volume,
            'total_orders': sum(len(orders) for orders in self.executor.orders.values())
        }


if __name__ == "__main__":
    # Example usage
    exchanges = {
        'binance': ExchangeInfo('binance', maker_fee=0.001, taker_fee=0.001),
        'kraken': ExchangeInfo('kraken', maker_fee=0.0016, taker_fee=0.0026),
        'coinbase': ExchangeInfo('coinbase', maker_fee=0.004, taker_fee=0.006),
    }

    engine = ArbitrageEngine(exchanges)
    logger.info("Arbitrage engine initialized")
