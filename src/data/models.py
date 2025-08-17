"""Data models for market data and features."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional


@dataclass
class Bar:
    """OHLCV bar data."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    symbol: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        """Validate bar data."""
        if self.high < max(self.open, self.close):
            raise ValueError("High must be >= max(open, close)")
        if self.low > min(self.open, self.close):
            raise ValueError("Low must be <= min(open, close)")
        if self.volume < 0:
            raise ValueError("Volume must be >= 0")


@dataclass
class Feature:
    """Technical indicator or feature value."""

    name: str
    value: float
    timestamp: datetime
    symbol: str
    parameters: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class MarketData:
    """Market data for a specific timestamp."""

    timestamp: datetime
    bars: dict[str, Bar]
    features: dict[str, list[Feature]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class DataRequest:
    """Request for data collection."""

    symbol: str
    start_date: datetime
    end_date: datetime
    interval: str = "1d"
    features: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate request parameters."""
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        if self.interval not in ["1m", "5m", "15m", "1h", "4h", "1d"]:
            raise ValueError("Invalid interval")


@dataclass
class DataResponse:
    """Response from data collection."""

    request: DataRequest
    data: MarketData
    success: bool
    error_message: Optional[str] = None
    collection_timestamp: datetime = field(default_factory=datetime.now)
