"""
Market data models for the Algo-Trading system.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass
class OHLCV:
    """OHLCV (Open, High, Low, Close, Volume) data model."""
    timestamp: datetime
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    source: str = "binance"
    load_id: str | None = None

    def __post_init__(self):
        """Validate OHLCV data integrity."""
        if self.high < self.low:
            raise ValueError("High cannot be less than low")
        if self.high < self.open or self.high < self.close:
            raise ValueError("High must be >= open and close")
        if self.low > self.open or self.low > self.close:
            raise ValueError("Low must be <= open and close")


@dataclass
class Tick:
    """Tick data model for real-time price updates."""
    timestamp: datetime
    symbol: str
    price: Decimal
    volume: Decimal
    side: str  # 'buy' or 'sell'
    source: str = "binance"


@dataclass
class OrderBook:
    """Order book data model."""
    timestamp: datetime
    symbol: str
    bids: list[tuple[Decimal, Decimal]]  # (price, quantity)
    asks: list[tuple[Decimal, Decimal]]  # (price, quantity)
    source: str = "binance"

    def get_best_bid(self) -> Decimal | None:
        """Get best bid price."""
        return self.bids[0][0] if self.bids else None

    def get_best_ask(self) -> Decimal | None:
        """Get best ask price."""
        return self.asks[0][0] if self.asks else None

    def get_spread(self) -> Decimal | None:
        """Get bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return best_ask - best_bid
        return None





