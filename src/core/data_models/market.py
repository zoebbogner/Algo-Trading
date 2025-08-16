"""Market data models."""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import Field

from .base import TimeSeriesEntity


class Bar(TimeSeriesEntity):
    """OHLCV bar data."""
    
    open: Decimal = Field(..., description="Opening price")
    high: Decimal = Field(..., description="Highest price")
    low: Decimal = Field(..., description="Lowest price")
    close: Decimal = Field(..., description="Closing price")
    volume: Decimal = Field(..., description="Trading volume")
    spread: Optional[Decimal] = Field(None, description="Bid-ask spread")
    interval: str = Field(..., description="Bar interval (e.g., 1m, 5m, 1h)")
    
    @property
    def range(self) -> Decimal:
        """Price range (high - low)."""
        return self.high - self.low
    
    @property
    def body(self) -> Decimal:
        """Body size (close - open)."""
        return self.close - self.open
    
    @property
    def is_bullish(self) -> bool:
        """Whether the bar is bullish (close > open)."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Whether the bar is bearish (close < open)."""
        return self.close < self.open


class Feature(TimeSeriesEntity):
    """Engineered feature data."""
    
    feature_name: str = Field(..., description="Feature name")
    feature_value: float = Field(..., description="Feature value")
    lookback_period: Optional[int] = Field(None, description="Lookback period for calculation")
    source: str = Field(..., description="Source of the feature (e.g., 'technical', 'price', 'volume')")

class MarketData(TimeSeriesEntity):
    """Aggregated market data including bar and features."""
    
    bar: Bar = Field(..., description="OHLCV bar data")
    features: list[Feature] = Field(default_factory=list, description="Engineered features")
    bid: Optional[Decimal] = Field(None, description="Best bid price")
    ask: Optional[Decimal] = Field(None, description="Best ask price")
    last_price: Optional[Decimal] = Field(None, description="Last traded price")
    timestamp_received: datetime = Field(..., description="When data was received")
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Mid price between bid and ask."""
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return None
    
    @property
    def spread_absolute(self) -> Optional[Decimal]:
        """Absolute spread (ask - bid)."""
        if self.bid and self.ask:
            return self.ask - self.bid
        return None
    
    @property
    def spread_relative(self) -> Optional[float]:
        """Relative spread as percentage of mid price."""
        if self.spread_absolute and self.mid_price:
            return float(self.spread_absolute / self.mid_price)
        return None
