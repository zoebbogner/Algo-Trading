"""Base data adapter for market data sources."""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Optional

from ...core.data_models.market import Bar
from ...utils.logging import logger


class DataAdapter(ABC):
    """Abstract base class for data adapters."""

    def __init__(self, config: dict[str, Any]):
        """Initialize data adapter.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.name = config.get("name", "UnknownAdapter")
        self.enabled = config.get("enabled", True)
        self.rate_limit_delay = config.get("rate_limit_delay", 0.1)

        logger.logger.info(f"Initialized data adapter: {self.name}")

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data source.

        Returns:
            True if connection successful
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data source."""

    @abstractmethod
    async def get_historical_data(
        self, symbol: str, timeframe: str, limit: int, since: Optional[datetime] = None
    ) -> list[Bar]:
        """Get historical market data.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USD')
            timeframe: Time interval (e.g., '1m', '5m', '1h')
            limit: Number of bars to retrieve
            since: Start time for data retrieval

        Returns:
            List of Bar objects
        """

    @abstractmethod
    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price or None if unavailable
        """

    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 20) -> Optional[dict]:
        """Get order book for a symbol.

        Args:
            symbol: Trading symbol
            depth: Number of orders to retrieve

        Returns:
            Order book dictionary or None if unavailable
        """

    def is_connected(self) -> bool:
        """Check if adapter is connected.

        Returns:
            True if connected
        """
        return hasattr(self, "_connected") and self._connected

    def is_enabled(self) -> bool:
        """Check if adapter is enabled.

        Returns:
            True if enabled
        """
        return self.enabled

    def get_config(self) -> dict[str, Any]:
        """Get adapter configuration.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def update_config(self, new_config: dict[str, Any]) -> None:
        """Update adapter configuration.

        Args:
            new_config: New configuration values
        """
        self.config.update(new_config)
        logger.logger.info(f"Updated configuration for adapter: {self.name}")

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to exchange-specific format.

        Args:
            timeframe: Standard timeframe (e.g., '1m', '5m', '1h')

        Returns:
            Exchange-specific timeframe string
        """
        # Default conversion (can be overridden by subclasses)
        timeframe_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }
        return timeframe_map.get(timeframe, "1m")

    def _validate_symbol(self, symbol: str) -> bool:
        """Validate trading symbol format.

        Args:
            symbol: Trading symbol to validate

        Returns:
            True if symbol is valid
        """
        # Basic validation - can be overridden by subclasses
        if not symbol or "/" not in symbol:
            return False

        base, quote = symbol.split("/")
        return len(base) > 0 and len(quote) > 0

    def _create_bar_from_data(
        self, data: dict[str, Any], symbol: str, interval: str
    ) -> Bar:
        """Create Bar object from raw data.

        Args:
            data: Raw market data
            symbol: Trading symbol
            interval: Time interval

        Returns:
            Bar object
        """
        # Default implementation - can be overridden by subclasses
        timestamp = datetime.fromtimestamp(
            data.get("timestamp", 0) / 1000, tz=UTC
        )

        return Bar(
            timestamp=timestamp,
            symbol=symbol,
            open=Decimal(str(data.get("open", 0))),
            high=Decimal(str(data.get("high", 0))),
            low=Decimal(str(data.get("low", 0))),
            close=Decimal(str(data.get("close", 0))),
            volume=Decimal(str(data.get("volume", 0))),
            interval=interval,
        )
