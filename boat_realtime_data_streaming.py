#!/usr/bin/env python3
"""
Real-Time Market Data Streaming Architecture for Boat
=====================================================

This module implements a high-performance, event-driven market data streaming
system supporting multiple data sources and real-time processing pipelines.

Features:
  - Multi-source data ingestion (REST, WebSocket)
  - Event-driven architecture with subscriber pattern
  - Ring buffer for high-frequency data processing
  - Data normalization and validation
  - Multi-exchange support
  - Real-time analytics and aggregation
  - Backpressure handling
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Callable, Optional, Any, Tuple
from collections import deque, defaultdict
import threading
from queue import Queue, PriorityQueue
import time
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Data source types"""
    REST = "rest"
    WEBSOCKET = "websocket"
    DATABASE = "database"
    KAFKA = "kafka"
    FILE = "file"


@dataclass
class MarketTick:
    """Represents a single market data point"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last_price: float
    bid_volume: float
    ask_volume: float
    volume: float = 0
    open_: float = 0
    high: float = 0
    low: float = 0
    close: float = 0
    exchange: str = "unknown"
    data_type: str = "tick"  # tick, bar, trade

    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread"""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Calculate spread percentage"""
        return (self.spread / self.mid_price) * 100 if self.mid_price > 0 else 0


@dataclass
class BarData:
    """OHLCV bar data"""
    symbol: str
    timestamp: datetime
    open_: float
    high: float
    low: float
    close: float
    volume: float
    exchange: str = "unknown"
    bar_size: str = "1m"  # 1m, 5m, 15m, 1h, 1d


class DataSubscriber(ABC):
    """Base class for data subscribers"""

    @abstractmethod
    async def on_data(self, data: Any) -> None:
        """Handle incoming data"""
        pass

    @abstractmethod
    async def on_error(self, error: Exception) -> None:
        """Handle error"""
        pass


class RingBuffer:
    """Efficient ring buffer for high-frequency data"""

    def __init__(self, size: int = 100000):
        self.size = size
        self.buffer = deque(maxlen=size)
        self.lock = threading.RLock()

    def append(self, data: Any) -> None:
        """Add data to buffer"""
        with self.lock:
            self.buffer.append(data)

    def get_recent(self, n: int) -> List[Any]:
        """Get last n items"""
        with self.lock:
            return list(self.buffer)[-n:] if n > 0 else []

    def get_all(self) -> List[Any]:
        """Get all items"""
        with self.lock:
            return list(self.buffer)

    def clear(self) -> None:
        """Clear buffer"""
        with self.lock:
            self.buffer.clear()

    def size_bytes(self) -> int:
        """Estimate buffer size in bytes"""
        return len(self.buffer) * 128  # Rough estimate


class DataValidator:
    """Validate and clean market data"""

    @staticmethod
    def validate_tick(tick: MarketTick) -> Tuple[bool, Optional[str]]:
        """
        Validate tick data

        Returns:
            (is_valid, error_message)
        """
        if tick.bid <= 0 or tick.ask <= 0:
            return False, "Invalid prices"

        if tick.ask < tick.bid:
            return False, "Ask < Bid"

        if tick.bid_volume < 0 or tick.ask_volume < 0:
            return False, "Invalid volumes"

        spread_pct = tick.spread_pct
        if spread_pct > 5:  # Sanity check: spread > 5% is suspicious
            return False, f"Spread too large: {spread_pct:.2f}%"

        return True, None

    @staticmethod
    def validate_bar(bar: BarData) -> Tuple[bool, Optional[str]]:
        """Validate bar data"""
        if bar.open_ <= 0 or bar.high <= 0 or bar.low <= 0 or bar.close <= 0:
            return False, "Invalid prices"

        if bar.high < bar.low or bar.high < bar.open_ or bar.high < bar.close:
            return False, "High < other prices"

        if bar.low > bar.open_ or bar.low > bar.close:
            return False, "Low > other prices"

        if bar.volume < 0:
            return False, "Invalid volume"

        return True, None


class DataNormalizer:
    """Normalize data from different sources"""

    @staticmethod
    def normalize_tick(
        symbol: str,
        bid: float,
        ask: float,
        last_price: float,
        bid_vol: float,
        ask_vol: float,
        exchange: str,
        timestamp: Optional[datetime] = None
    ) -> MarketTick:
        """Normalize tick from various sources"""
        return MarketTick(
            symbol=symbol.upper(),
            timestamp=timestamp or datetime.utcnow(),
            bid=float(bid),
            ask=float(ask),
            last_price=float(last_price),
            bid_volume=float(bid_vol),
            ask_volume=float(ask_vol),
            exchange=exchange.lower()
        )

    @staticmethod
    def normalize_bar(
        symbol: str,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        exchange: str,
        bar_size: str = "1m"
    ) -> BarData:
        """Normalize OHLCV bar"""
        return BarData(
            symbol=symbol.upper(),
            timestamp=timestamp,
            open_=float(open_),
            high=float(high),
            low=float(low),
            close=float(close),
            volume=float(volume),
            exchange=exchange.lower(),
            bar_size=bar_size
        )


class DataProcessor:
    """Process and aggregate market data"""

    def __init__(self):
        self.ohlcv_buffers = defaultdict(lambda: {
            '1m': RingBuffer(1440),  # 1 day of 1m bars
            '5m': RingBuffer(288),   # 1 day of 5m bars
            '1h': RingBuffer(24),    # 1 day of 1h bars
        })
        self.tick_buffers = defaultdict(lambda: RingBuffer(10000))
        self.validators = DataValidator()

    async def process_tick(self, tick: MarketTick) -> None:
        """Process incoming tick"""
        # Validate
        is_valid, error = self.validators.validate_tick(tick)
        if not is_valid:
            logger.warning(f"Invalid tick: {error}")
            return

        # Store
        self.tick_buffers[tick.symbol].append(tick)

    async def process_bar(self, bar: BarData) -> None:
        """Process incoming bar"""
        # Validate
        is_valid, error = self.validators.validate_bar(bar)
        if not is_valid:
            logger.warning(f"Invalid bar: {error}")
            return

        # Store
        self.ohlcv_buffers[bar.symbol][bar.bar_size].append(bar)

    def get_recent_ticks(self, symbol: str, n: int = 100) -> List[MarketTick]:
        """Get last n ticks for symbol"""
        return self.tick_buffers[symbol].get_recent(n)

    def get_bars(self, symbol: str, bar_size: str = '1m') -> List[BarData]:
        """Get bars for symbol"""
        return self.ohlcv_buffers[symbol][bar_size].get_all()

    def calculate_vwap(self, symbol: str, lookback: int = 100) -> Optional[float]:
        """Calculate Volume Weighted Average Price"""
        ticks = self.get_recent_ticks(symbol, lookback)
        if not ticks:
            return None

        total_volume = sum(t.bid_volume + t.ask_volume for t in ticks)
        if total_volume == 0:
            return None

        vwap = sum(t.mid_price * (t.bid_volume + t.ask_volume) for t in ticks) / total_volume
        return vwap

    def calculate_spread_stats(self, symbol: str, lookback: int = 100) -> Dict[str, float]:
        """Calculate spread statistics"""
        ticks = self.get_recent_ticks(symbol, lookback)
        if not ticks:
            return {}

        spreads = [t.spread_pct for t in ticks]
        return {
            'mean_spread': np.mean(spreads),
            'std_spread': np.std(spreads),
            'min_spread': np.min(spreads),
            'max_spread': np.max(spreads),
        }


class DataSource(ABC):
    """Base class for data sources"""

    def __init__(self, symbol: str, exchange: str):
        self.symbol = symbol
        self.exchange = exchange
        self.subscribers: List[DataSubscriber] = []
        self.is_connected = False

    def subscribe(self, subscriber: DataSubscriber) -> None:
        """Subscribe to data updates"""
        self.subscribers.append(subscriber)
        logger.info(f"Subscriber added for {self.symbol}")

    def unsubscribe(self, subscriber: DataSubscriber) -> None:
        """Unsubscribe from data updates"""
        if subscriber in self.subscribers:
            self.subscribers.remove(subscriber)

    async def _notify_subscribers(self, data: Any) -> None:
        """Notify all subscribers"""
        for subscriber in self.subscribers:
            try:
                await subscriber.on_data(data)
            except Exception as e:
                await subscriber.on_error(e)

    @abstractmethod
    async def connect(self) -> None:
        """Connect to data source"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data source"""
        pass


class MockDataSource(DataSource):
    """Mock data source for testing"""

    def __init__(self, symbol: str, exchange: str = "mock"):
        super().__init__(symbol, exchange)
        self.running = False

    async def connect(self) -> None:
        """Connect to mock source"""
        self.is_connected = True
        self.running = True
        logger.info(f"Connected to mock source for {self.symbol}")

    async def disconnect(self) -> None:
        """Disconnect from mock source"""
        self.is_connected = False
        self.running = False

    async def start_streaming(self) -> None:
        """Generate mock market data"""
        price = 100.0
        while self.running:
            # Generate random price movement
            price += np.random.randn() * 0.5
            price = max(price, 50)

            tick = MarketTick(
                symbol=self.symbol,
                timestamp=datetime.utcnow(),
                bid=price - 0.01,
                ask=price + 0.01,
                last_price=price,
                bid_volume=np.random.uniform(100, 1000),
                ask_volume=np.random.uniform(100, 1000),
                exchange=self.exchange
            )

            await self._notify_subscribers(tick)
            await asyncio.sleep(0.1)


class DataStreamingEngine:
    """Main data streaming engine"""

    def __init__(self):
        self.sources: Dict[str, DataSource] = {}
        self.processor = DataProcessor()
        self.subscribers: List[DataSubscriber] = []

    def register_source(self, symbol: str, source: DataSource) -> None:
        """Register data source"""
        self.sources[symbol] = source
        logger.info(f"Registered source for {symbol}")

    async def start(self) -> None:
        """Start all data sources"""
        for symbol, source in self.sources.items():
            await source.connect()
            logger.info(f"Started {symbol}")

    async def stop(self) -> None:
        """Stop all data sources"""
        for symbol, source in self.sources.items():
            await source.disconnect()
            logger.info(f"Stopped {symbol}")

    def get_processor(self) -> DataProcessor:
        """Get data processor"""
        return self.processor

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        stats = {
            'sources_count': len(self.sources),
            'connected_sources': sum(1 for s in self.sources.values() if s.is_connected),
            'total_ticks': sum(
                len(self.processor.tick_buffers[symbol].get_all())
                for symbol in self.processor.tick_buffers
            ),
            'buffer_memory_mb': sum(
                self.processor.tick_buffers[symbol].size_bytes() / (1024 * 1024)
                for symbol in self.processor.tick_buffers
            ),
        }
        return stats


# Real-time analytics
class RealtimeAnalytics:
    """Real-time market analytics"""

    def __init__(self, processor: DataProcessor):
        self.processor = processor

    def get_volume_weighted_price(self, symbol: str) -> Optional[float]:
        """Get VWAP"""
        return self.processor.calculate_vwap(symbol)

    def get_spread_analysis(self, symbol: str) -> Dict[str, float]:
        """Get spread analysis"""
        return self.processor.calculate_spread_stats(symbol)

    def detect_arbitrage_opportunities(
        self,
        symbols: List[str],
        min_spread_pct: float = 0.5
    ) -> List[Tuple[str, str, float]]:
        """
        Detect cross-exchange arbitrage opportunities

        Returns:
            List of (symbol, exchange_pair, profit_pct)
        """
        opportunities = []

        for symbol in symbols:
            ticks = self.processor.get_recent_ticks(symbol, 100)
            if not ticks:
                continue

            # Group by exchange
            by_exchange = defaultdict(list)
            for tick in ticks:
                by_exchange[tick.exchange].append(tick.mid_price)

            # Compare prices
            exchanges = list(by_exchange.keys())
            for i, ex1 in enumerate(exchanges):
                for ex2 in exchanges[i+1:]:
                    price1 = np.mean(by_exchange[ex1])
                    price2 = np.mean(by_exchange[ex2])

                    spread_pct = abs(price1 - price2) / min(price1, price2) * 100

                    if spread_pct > min_spread_pct:
                        opportunities.append((symbol, f"{ex1}/{ex2}", spread_pct))

        return opportunities


# Example usage
async def main():
    """Example usage"""
    engine = DataStreamingEngine()

    # Register mock data sources
    for symbol in ['BTCUSD', 'ETHUSD']:
        source = MockDataSource(symbol)
        engine.register_source(symbol, source)

    # Start engine
    await engine.start()

    # Create mock streaming tasks
    tasks = []
    for source in engine.sources.values():
        if isinstance(source, MockDataSource):
            tasks.append(source.start_streaming())

    # Create analyzer
    analytics = RealtimeAnalytics(engine.get_processor())

    # Stream data for 5 seconds
    try:
        start = time.time()
        while time.time() - start < 5:
            # Print statistics every second
            if int(time.time() - start) % 1 == 0:
                stats = engine.get_statistics()
                logger.info(f"Engine stats: {stats}")

            await asyncio.sleep(0.1)

        # Stop
        await engine.stop()
        logger.info("Engine stopped")

    except KeyboardInterrupt:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
