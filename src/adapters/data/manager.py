"""Data manager for coordinating multiple data sources."""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Optional

from ...core.data_models.market import Bar, MarketData
from ...core.features.technical import FeatureEngine
from ...utils.logging import logger
from .base import DataAdapter
from .binance import BinanceAdapter


class DataManager:
    """Manages multiple data sources and provides unified data access."""

    def __init__(self, config: dict[str, Any]):
        """Initialize data manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.adapters: dict[str, DataAdapter] = {}
        self.feature_engine = FeatureEngine()
        self.data_cache: dict[str, list[Bar]] = {}
        self.cache_size = config.get("cache_size", 1000)

        self._setup_adapters()
        logger.logger.info("Data manager initialized")

    def _setup_adapters(self) -> None:
        """Setup data adapters from configuration."""
        adapters_config = self.config.get("adapters", {})

        # Setup Binance adapter
        if adapters_config.get("binance", {}).get("enabled", True):
            binance_config = adapters_config["binance"]
            binance_config.setdefault("sandbox", True)  # Use testnet by default
            binance_config.setdefault("name", "Binance")

            self.adapters["binance"] = BinanceAdapter(binance_config)
            logger.logger.info("Binance adapter configured")

        # Add more adapters here as needed
        # self.adapters["coinbase"] = CoinbaseAdapter(adapters_config.get("coinbase", {}))

        logger.logger.info(f"Configured {len(self.adapters)} data adapters")

    async def connect_all(self) -> bool:
        """Connect to all enabled data adapters.

        Returns:
            True if all connections successful
        """
        success_count = 0
        total_count = len(self.adapters)

        for name, adapter in self.adapters.items():
            if adapter.is_enabled():
                try:
                    if await adapter.connect():
                        success_count += 1
                        logger.logger.info(f"Connected to {name}")
                    else:
                        logger.logger.error(f"Failed to connect to {name}")
                except Exception as e:
                    logger.logger.error(f"Error connecting to {name}: {e}")

        logger.logger.info(f"Connected to {success_count}/{total_count} data adapters")
        return success_count > 0

    async def disconnect_all(self) -> None:
        """Disconnect from all data adapters."""
        for name, adapter in self.adapters.items():
            try:
                await adapter.disconnect()
                logger.logger.info(f"Disconnected from {name}")
            except Exception as e:
                logger.logger.error(f"Error disconnecting from {name}: {e}")

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        since: Optional[datetime] = None,
        adapter_name: Optional[str] = None,
    ) -> list[Bar]:
        """Get historical data from the best available adapter.

        Args:
            symbol: Trading symbol
            timeframe: Time interval
            limit: Number of bars to retrieve
            since: Start time for data retrieval
            adapter_name: Specific adapter to use (optional)

        Returns:
            List of Bar objects
        """
        # Use specific adapter if requested
        if adapter_name and adapter_name in self.adapters:
            adapter = self.adapters[adapter_name]
            if adapter.is_enabled():
                return await adapter.get_historical_data(
                    symbol, timeframe, limit, since
                )

        # Try all adapters in order of preference
        for name, adapter in self.adapters.items():
            if adapter.is_enabled() and adapter.is_connected():
                try:
                    data = await adapter.get_historical_data(
                        symbol, timeframe, limit, since
                    )
                    if data:
                        # Cache the data
                        cache_key = f"{symbol}_{timeframe}"
                        if cache_key not in self.data_cache:
                            self.data_cache[cache_key] = []

                        self.data_cache[cache_key].extend(data)

                        # Maintain cache size
                        if len(self.data_cache[cache_key]) > self.cache_size:
                            self.data_cache[cache_key] = self.data_cache[cache_key][
                                -self.cache_size :
                            ]

                        logger.logger.info(f"Retrieved {len(data)} bars from {name}")
                        return data

                except Exception as e:
                    logger.logger.warning(f"Failed to get data from {name}: {e}")
                    continue

        logger.logger.error(
            f"Failed to get historical data from any adapter for {symbol}"
        )
        return []

    async def get_current_price(
        self, symbol: str, adapter_name: Optional[str] = None
    ) -> Optional[Decimal]:
        """Get current price from the best available adapter.

        Args:
            symbol: Trading symbol
            adapter_name: Specific adapter to use (optional)

        Returns:
            Current price or None if unavailable
        """
        # Use specific adapter if requested
        if adapter_name and adapter_name in self.adapters:
            adapter = self.adapters[adapter_name]
            if adapter.is_enabled():
                return await adapter.get_current_price(symbol)

        # Try all adapters
        for name, adapter in self.adapters.items():
            if adapter.is_enabled() and adapter.is_connected():
                try:
                    price = await adapter.get_current_price(symbol)
                    if price:
                        return price
                except Exception as e:
                    logger.logger.debug(f"Failed to get price from {name}: {e}")
                    continue

        return None

    async def get_historical_data_range(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        adapter_name: Optional[str] = None,
    ) -> list[Bar]:
        """Get historical data for a specific date range.

        Args:
            symbol: Trading symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Time interval
            adapter_name: Specific adapter to use (optional)

        Returns:
            List of Bar objects
        """
        # Calculate the number of bars needed based on timeframe and date range
        if timeframe == "1h":
            hours_diff = (end_date - start_date).total_seconds() / 3600
            limit = int(hours_diff) + 1
        elif timeframe == "1d":
            days_diff = (end_date - start_date).days
            limit = days_diff + 1
        else:
            # Default to 1000 bars for other timeframes
            limit = 1000

        # Use specific adapter if requested
        if adapter_name and adapter_name in self.adapters:
            adapter = self.adapters[adapter_name]
            if adapter.is_enabled():
                return await adapter.get_historical_data(
                    symbol, timeframe, limit, start_date
                )

        # Try all adapters in order of preference
        for name, adapter in self.adapters.items():
            if adapter.is_enabled() and adapter.is_connected():
                try:
                    data = await adapter.get_historical_data(
                        symbol, timeframe, limit, start_date
                    )
                    if data:
                        # Filter data by date range
                        filtered_data = [
                            bar
                            for bar in data
                            if start_date <= bar.timestamp <= end_date
                        ]

                        if filtered_data:
                            # Cache the data
                            cache_key = f"{symbol}_{timeframe}"
                            if cache_key not in self.data_cache:
                                self.data_cache[cache_key] = []

                            self.data_cache[cache_key].extend(filtered_data)

                            # Maintain cache size
                            if len(self.data_cache[cache_key]) > self.cache_size:
                                self.data_cache[cache_key] = self.data_cache[cache_key][
                                    -self.cache_size :
                                ]

                            logger.logger.info(
                                f"Retrieved {len(filtered_data)} bars from {name}"
                            )
                            return filtered_data

                except Exception as e:
                    logger.logger.warning(f"Failed to get data from {name}: {e}")
                    continue

        logger.logger.error(
            f"Failed to get historical data from any adapter for {symbol}"
        )
        return []

    async def get_market_data(
        self, symbol: str, timeframe: str, include_features: bool = True
    ) -> Optional[MarketData]:
        """Get market data with optional features.

        Args:
            symbol: Trading symbol
            timeframe: Time interval
            include_features: Whether to calculate technical features

        Returns:
            MarketData object or None if unavailable
        """
        # Get the latest bar
        bars = await self.get_historical_data(symbol, timeframe, 1)
        if not bars:
            return None

        latest_bar = bars[-1]

        # Get order book for bid/ask prices
        order_book = None
        for name, adapter in self.adapters.items():
            if adapter.is_enabled() and adapter.is_connected():
                try:
                    order_book = await adapter.get_order_book(symbol, depth=5)
                    if order_book:
                        break
                except Exception:
                    continue

        # Calculate features if requested
        features = []
        if include_features:
            # Get more bars for feature calculation
            feature_bars = await self.get_historical_data(symbol, timeframe, 50)
            if feature_bars:
                features_dict = self.feature_engine.calculate_features(feature_bars)

                # Convert to Feature objects
                for feature_name, feature_value in features_dict.items():
                    if isinstance(feature_value, (int, float)):
                        try:
                            # Check for NaN
                            if feature_value == feature_value:  # Not NaN
                                from ...core.data_models.market import Feature

                                feature = Feature(
                                    timestamp=latest_bar.timestamp,
                                    symbol=symbol,
                                    feature_name=feature_name,
                                    feature_value=float(feature_value),
                                    source="technical",
                                )
                                features.append(feature)
                        except Exception:
                            continue

        # Create MarketData object
        market_data = MarketData(
            timestamp=latest_bar.timestamp,
            symbol=symbol,
            bar=latest_bar,
            features=features,
            timestamp_received=datetime.now(UTC),
        )

        # Add bid/ask prices if available
        if order_book and order_book["bids"] and order_book["asks"]:
            market_data.bid = order_book["bids"][0][0]  # Best bid
            market_data.ask = order_book["asks"][0][0]  # Best ask
            market_data.last_price = latest_bar.close

        return market_data

    async def get_multiple_symbols_data(
        self, symbols: list[str], timeframe: str, include_features: bool = True
    ) -> dict[str, MarketData]:
        """Get market data for multiple symbols.

        Args:
            symbols: List of trading symbols
            timeframe: Time interval
            include_features: Whether to calculate technical features

        Returns:
            Dictionary mapping symbols to MarketData objects
        """
        results = {}

        # Process symbols concurrently
        tasks = []
        for symbol in symbols:
            task = self.get_market_data(symbol, timeframe, include_features)
            tasks.append((symbol, task))

        # Execute tasks with rate limiting
        for symbol, task in tasks:
            try:
                market_data = await task
                if market_data:
                    results[symbol] = market_data

                # Rate limiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.logger.error(f"Error getting data for {symbol}: {e}")

        logger.logger.info(f"Retrieved data for {len(results)}/{len(symbols)} symbols")
        return results

    def get_cached_data(self, symbol: str, timeframe: str) -> list[Bar]:
        """Get cached data for a symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Time interval

        Returns:
            List of cached Bar objects
        """
        cache_key = f"{symbol}_{timeframe}"
        return self.data_cache.get(cache_key, [])

    def clear_cache(
        self, symbol: Optional[str] = None, timeframe: Optional[str] = None
    ) -> None:
        """Clear data cache.

        Args:
            symbol: Specific symbol to clear (optional)
            timeframe: Specific timeframe to clear (optional)
        """
        if symbol and timeframe:
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.data_cache:
                del self.data_cache[cache_key]
                logger.logger.info(f"Cleared cache for {cache_key}")
        else:
            self.data_cache.clear()
            logger.logger.info("Cleared all data cache")

    def get_adapter_status(self) -> dict[str, dict]:
        """Get status of all adapters.

        Returns:
            Dictionary mapping adapter names to their status
        """
        status = {}
        for name, adapter in self.adapters.items():
            status[name] = {
                "enabled": adapter.is_enabled(),
                "connected": adapter.is_connected(),
                "name": adapter.name,
            }
        return status

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect_all()
