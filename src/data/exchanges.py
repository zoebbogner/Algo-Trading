"""Exchange-specific data collectors."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

import aiohttp

from .models import Bar, DataRequest, MarketData

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Base class for all data collectors."""

    def __init__(self, config: dict[str, Any]):
        """Initialize base collector."""
        self.config = config
        self.rate_limit_delay = config.get("rate_limit_delay", 1.0)
        self.max_retries = config.get("max_retries", 3)
        self.last_request_time = 0.0

    @abstractmethod
    async def collect_data(self, request: DataRequest, session: aiohttp.ClientSession) -> MarketData:
        """Collect data from the exchange."""

    @abstractmethod
    def get_available_symbols(self) -> list[str]:
        """Get list of available symbols."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if collector is available."""

    async def _rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None
    ) -> dict[str, Any]:
        """Make HTTP request with retry logic."""
        for attempt in range(self.max_retries):
            try:
                await self._rate_limit()

                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    if response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited, waiting {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        continue
                    logger.error(f"HTTP {response.status}: {response.reason}")
                    response.raise_for_status()

            except Exception as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        raise Exception("Max retries exceeded")


class BinanceCollector(BaseCollector):
    """Binance exchange data collector."""

    def __init__(self, config: dict[str, Any]):
        """Initialize Binance collector."""
        super().__init__(config)
        self.base_url = "https://api.binance.com/api/v3"
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")

    async def collect_data(self, request: DataRequest, session: aiohttp.ClientSession) -> MarketData:
        """Collect historical kline data from Binance."""
        try:
            # Convert symbol to Binance format
            symbol = request.symbol.upper() + "USDT"

            # Convert interval to Binance format
            interval_map = {
                "1m": "1m", "5m": "5m", "15m": "15m",
                "1h": "1h", "4h": "4h", "1d": "1d"
            }
            interval = interval_map.get(request.interval, "1d")

            # Calculate start and end times in milliseconds
            start_time = int(request.start_date.timestamp() * 1000)
            end_time = int(request.end_date.timestamp() * 1000)

            url = f"{self.base_url}/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time,
                "limit": 1000
            }

            data = await self._make_request(session, url, params)

            # Parse kline data
            bars = {}
            for kline in data:
                bar = Bar(
                    timestamp=datetime.fromtimestamp(kline[0] / 1000),
                    open=Decimal(str(kline[1])),
                    high=Decimal(str(kline[2])),
                    low=Decimal(str(kline[3])),
                    close=Decimal(str(kline[4])),
                    volume=Decimal(str(kline[5])),
                    symbol=request.symbol
                )
                bars[bar.timestamp] = bar

            return MarketData(
                timestamp=datetime.now(),
                bars=bars
            )

        except Exception as e:
            logger.error(f"Error collecting Binance data: {e}")
            raise

    def get_available_symbols(self) -> list[str]:
        """Get available symbols from Binance."""
        # Common major cryptocurrencies
        return ["BTC", "ETH", "ADA", "DOT", "LINK", "LTC", "BCH", "XLM", "XRP", "DOGE"]

    def is_available(self) -> bool:
        """Check if Binance collector is available."""
        return True  # Public API is always available


class CoinGeckoCollector(BaseCollector):
    """CoinGecko API data collector."""

    def __init__(self, config: dict[str, Any]):
        """Initialize CoinGecko collector."""
        super().__init__(config)
        self.base_url = "https://api.coingecko.com/api/v3"
        self.api_key = config.get("api_key")

    async def collect_data(self, request: DataRequest, session: aiohttp.ClientSession) -> MarketData:
        """Collect historical data from CoinGecko."""
        try:
            # Get coin ID from symbol
            coin_id = self._get_coin_id(request.symbol)
            if not coin_id:
                raise ValueError(f"Unknown symbol: {request.symbol}")

            url = f"{self.base_url}/coins/{coin_id}/market_chart/range"
            params = {
                "vs_currency": "usd",
                "from": int(request.start_date.timestamp()),
                "to": int(request.end_date.timestamp())
            }

            if self.api_key:
                params["x_cg_demo_api_key"] = self.api_key

            data = await self._make_request(session, url, params)

            # Parse price data
            bars = {}
            for price_data in data.get("prices", []):
                timestamp = datetime.fromtimestamp(price_data[0] / 1000)
                price = Decimal(str(price_data[1]))

                # Create synthetic OHLCV data (CoinGecko only provides price)
                bar = Bar(
                    timestamp=timestamp,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=Decimal("0"),  # Not available
                    symbol=request.symbol
                )
                bars[timestamp] = bar

            return MarketData(
                timestamp=datetime.now(),
                bars=bars
            )

        except Exception as e:
            logger.error(f"Error collecting CoinGecko data: {e}")
            raise

    def _get_coin_id(self, symbol: str) -> Optional[str]:
        """Get CoinGecko coin ID from symbol."""
        coin_map = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "ADA": "cardano",
            "DOT": "polkadot",
            "LINK": "chainlink",
            "LTC": "litecoin",
            "BCH": "bitcoin-cash",
            "XLM": "stellar",
            "XRP": "ripple",
            "DOGE": "dogecoin"
        }
        return coin_map.get(symbol.upper())

    def get_available_symbols(self) -> list[str]:
        """Get available symbols from CoinGecko."""
        return ["BTC", "ETH", "ADA", "DOT", "LINK", "LTC", "BCH", "XLM", "XRP", "DOGE"]

    def is_available(self) -> bool:
        """Check if CoinGecko collector is available."""
        return True


class CCXTCollector(BaseCollector):
    """CCXT library data collector for multiple exchanges."""

    def __init__(self, config: dict[str, Any]):
        """Initialize CCXT collector."""
        super().__init__(config)
        self.exchanges = config.get("exchanges", ["kucoin", "okx", "gate", "huobi", "bybit"])

    async def collect_data(self, _request: DataRequest, _session: aiohttp.ClientSession) -> MarketData:
        """Collect data using CCXT (placeholder implementation)."""
        # This would require CCXT library integration
        # For now, return empty data
        logger.warning("CCXT collector not fully implemented")
        return MarketData(
            timestamp=datetime.now(),
            bars={}
        )

    def get_available_symbols(self) -> list[str]:
        """Get available symbols from CCXT exchanges."""
        return ["BTC", "ETH", "ADA", "DOT", "LINK", "LTC", "BCH", "XLM", "XRP", "DOGE"]

    def is_available(self) -> bool:
        """Check if CCXT collector is available."""
        return False  # Not fully implemented yet
