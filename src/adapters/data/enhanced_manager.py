"""
Enhanced Data Manager for Multiple Cryptocurrencies

This module provides access to multiple cryptocurrency data sources
without requiring API keys, supporting a wide range of coins and tokens.
"""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Optional

import aiohttp
import ccxt

from ..data_models.market import Bar, MarketData


class EnhancedDataManager:
    """
    Enhanced data manager supporting multiple cryptocurrencies and data sources
    """

    def __init__(self, config: dict):
        self.config = config
        self.adapters = {}
        self.symbols = config.get("symbols", self._get_default_symbols())
        self.timeframe = config.get("timeframe", "1h")
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1)

        # Initialize free data sources
        self._initialize_free_sources()

        print("🚀 Enhanced Data Manager initialized")
        print(f"   Supported symbols: {len(self.symbols)} cryptocurrencies")
        print(f"   Timeframe: {self.timeframe}")
        print(f"   Free sources: {len(self.adapters)}")

    def _get_default_symbols(self) -> list[str]:
        """Get comprehensive list of cryptocurrency symbols"""
        return [
            # Major cryptocurrencies
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "ADA/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "DOT/USDT",
            "DOGE/USDT",
            "AVAX/USDT",
            "MATIC/USDT",
            # DeFi tokens
            "UNI/USDT",
            "LINK/USDT",
            "AAVE/USDT",
            "COMP/USDT",
            "SUSHI/USDT",
            "CRV/USDT",
            "YFI/USDT",
            "SNX/USDT",
            "BAL/USDT",
            "REN/USDT",
            # Layer 1 alternatives
            "ATOM/USDT",
            "NEAR/USDT",
            "FTM/USDT",
            "ALGO/USDT",
            "ICP/USDT",
            "FIL/USDT",
            "VET/USDT",
            "TRX/USDT",
            "EOS/USDT",
            "XLM/USDT",
            # Gaming and Metaverse
            "AXS/USDT",
            "MANA/USDT",
            "SAND/USDT",
            "ENJ/USDT",
            "CHZ/USDT",
            "GALA/USDT",
            "ROBLOX/USDT",
            "THETA/USDT",
            "BAT/USDT",
            "HOT/USDT",
            # AI and Big Data
            "OCEAN/USDT",
            "FET/USDT",
            "AGIX/USDT",
            "RLC/USDT",
            "GRT/USDT",
            "BAND/USDT",
            "API3/USDT",
            "UMA/USDT",
            "ZRX/USDT",
            "KNC/USDT",
            # Privacy and Security
            "XMR/USDT",
            "ZEC/USDT",
            "DASH/USDT",
            "LTC/USDT",
            "BCH/USDT",
            "XNO/USDT",
            "RVN/USDT",
            "BEAM/USDT",
            "PIVX/USDT",
            "ARRR/USDT",
            # Exchange tokens
            "OKB/USDT",
            "HT/USDT",
            "KCS/USDT",
            "CRO/USDT",
            "BTT/USDT",
            "LEO/USDT",
            "FTT/USDT",
            "GT/USDT",
            "MX/USDT",
            "WOO/USDT",
            # Stablecoins and yield farming
            "USDC/USDT",
            "DAI/USDT",
            "BUSD/USDT",
            "TUSD/USDT",
            "FRAX/USDT",
            "USDP/USDT",
            "GUSD/USDT",
            "HUSD/USDT",
            "USDK/USDT",
            "USDN/USDT",
            # Meme coins and trending
            "SHIB/USDT",
            "PEPE/USDT",
            "FLOKI/USDT",
            "BONK/USDT",
            "WIF/USDT",
            "MYRO/USDT",
            "POPCAT/USDT",
            "BOOK/USDT",
            "TURBO/USDT",
            "SPONGE/USDT",
            # Emerging sectors
            "RNDR/USDT",
            "HIVE/USDT",
            "STEEM/USDT",
            "WAXP/USDT",
            "TLM/USDT",
            "ALICE/USDT",
            "ALPHA/USDT",
            "AUDIO/USDT",
            "CLV/USDT",
            "CTSI/USDT",
        ]

    def _initialize_free_sources(self):
        """Initialize free data sources that don't require API keys"""

        # 1. CoinGecko (free tier - no API key required)
        try:
            self.adapters["coingecko"] = CoinGeckoAdapter(
                {
                    "base_url": "https://api.coingecko.com/api/v3",
                    "rate_limit": 50,  # requests per minute
                    "timeout": 30,
                }
            )
            print("✅ CoinGecko adapter initialized (free)")
        except Exception as e:
            print(f"⚠️ CoinGecko adapter failed: {e}")

        # 2. CoinMarketCap (free tier - no API key required for basic data)
        try:
            self.adapters["coinmarketcap"] = CoinMarketCapAdapter(
                {
                    "base_url": "https://pro-api.coinmarketcap.com/v1",
                    "rate_limit": 30,  # requests per minute
                    "timeout": 30,
                }
            )
            print("✅ CoinMarketCap adapter initialized (free)")
        except Exception as e:
            print(f"⚠️ CoinMarketCap adapter failed: {e}")

        # 3. Binance (public endpoints - no API key required)
        try:
            self.adapters["binance"] = BinancePublicAdapter(
                {
                    "base_url": "https://api.binance.com",
                    "rate_limit": 1200,  # requests per minute
                    "timeout": 30,
                }
            )
            print("✅ Binance public adapter initialized (free)")
        except Exception as e:
            print(f"⚠️ Binance public adapter failed: {e}")

        # 4. Alternative exchanges (public endpoints)
        exchanges = ["kucoin", "okx", "gate", "huobi", "bybit"]
        for exchange in exchanges:
            try:
                exchange_class = getattr(ccxt, exchange)
                exchange_instance = exchange_class(
                    {
                        "enableRateLimit": True,
                        "timeout": 30000,
                        "rateLimit": 1000,  # milliseconds between requests
                    }
                )

                self.adapters[exchange] = CCXTAdapter(
                    exchange_instance,
                    {"rate_limit": 60, "timeout": 30},  # requests per minute
                )
                print(f"✅ {exchange.capitalize()} adapter initialized (free)")
            except Exception as e:
                print(f"⚠️ {exchange.capitalize()} adapter failed: {e}")

        # 5. Web scraping fallback (for very specific data)
        try:
            self.adapters["scraper"] = WebScraperAdapter(
                {"rate_limit": 10, "timeout": 60}  # requests per minute
            )
            print("✅ Web scraper adapter initialized (fallback)")
        except Exception as e:
            print(f"⚠️ Web scraper adapter failed: {e}")

    async def get_historical_data(
        self, symbol: str, limit: int = 500, timeframe: str = None
    ) -> Optional[list[Bar]]:
        """
        Get historical data for a symbol from multiple sources
        """
        if timeframe is None:
            timeframe = self.timeframe

        # Try multiple sources in order of preference
        sources = [
            "binance",
            "kucoin",
            "okx",
            "gate",
            "huobi",
            "bybit",
            "coingecko",
            "coinmarketcap",
        ]

        for source_name in sources:
            if source_name in self.adapters:
                try:
                    print(f"📊 Fetching {symbol} data from {source_name}...")
                    data = await self.adapters[source_name].get_historical_data(
                        symbol, limit, timeframe
                    )

                    if (
                        data and len(data) >= limit * 0.8
                    ):  # At least 80% of requested data
                        print(
                            f"✅ Successfully fetched {len(data)} bars from {source_name}"
                        )
                        return data
                    else:
                        print(
                            f"⚠️ {source_name} returned insufficient data: {len(data) if data else 0} bars"
                        )

                except Exception as e:
                    print(f"❌ {source_name} failed for {symbol}: {e}")
                    continue

        print(f"❌ All sources failed for {symbol}")
        return None

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        # Try multiple sources
        sources = ["binance", "kucoin", "okx", "coingecko"]

        for source_name in sources:
            if source_name in self.adapters:
                try:
                    price = await self.adapters[source_name].get_current_price(symbol)
                    if price and price > 0:
                        return price
                except Exception:
                    continue

        return None

    async def get_market_data(self, symbols: list[str] = None) -> MarketData:
        """Get market data for multiple symbols"""
        if symbols is None:
            symbols = self.symbols[:20]  # Limit to first 20 symbols for performance

        bars_data = {}
        features_data = {}

        # Fetch data for each symbol
        for symbol in symbols:
            try:
                bars = await self.get_historical_data(symbol, limit=100)
                if bars:
                    bars_data[symbol] = bars

                    # Calculate basic features
                    features = self._calculate_basic_features(bars)
                    features_data[symbol] = features

            except Exception as e:
                print(f"⚠️ Failed to get data for {symbol}: {e}")
                continue

        return MarketData(
            bars=bars_data,
            features=features_data,
            timestamp_received=datetime.now(UTC),
        )

    def _calculate_basic_features(self, bars: list[Bar]) -> list[dict]:
        """Calculate basic technical features from bars"""
        if len(bars) < 20:
            return []

        features = []

        # Calculate moving averages
        closes = [float(bar.close) for bar in bars]

        # Simple moving averages
        sma_5 = sum(closes[-5:]) / 5 if len(closes) >= 5 else closes[-1]
        sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else closes[-1]

        # RSI calculation
        if len(closes) >= 14:
            gains = []
            losses = []
            for i in range(1, len(closes)):
                change = closes[i] - closes[i - 1]
                gains.append(max(change, 0))
                losses.append(max(-change, 0))

            avg_gain = sum(gains[-14:]) / 14
            avg_loss = sum(losses[-14:]) / 14

            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100
        else:
            rsi = 50

        # Volatility
        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))
        ]
        volatility = (
            (sum(r * r for r in returns[-20:]) / len(returns[-20:])) ** 0.5
            if returns
            else 0
        )

        features.append(
            {
                "timestamp": bars[-1].timestamp,
                "symbol": bars[-1].symbol,
                "feature_name": "sma_5",
                "feature_value": sma_5,
                "lookback_period": 5,
            }
        )

        features.append(
            {
                "timestamp": bars[-1].timestamp,
                "symbol": bars[-1].symbol,
                "feature_name": "sma_20",
                "feature_value": sma_20,
                "lookback_period": 20,
            }
        )

        features.append(
            {
                "timestamp": bars[-1].timestamp,
                "symbol": bars[-1].symbol,
                "feature_name": "rsi",
                "feature_value": rsi,
                "lookback_period": 14,
            }
        )

        features.append(
            {
                "timestamp": bars[-1].timestamp,
                "symbol": bars[-1].symbol,
                "feature_name": "volatility",
                "feature_value": volatility,
                "lookback_period": 20,
            }
        )

        return features

    async def get_symbols_info(self) -> dict[str, dict]:
        """Get information about all supported symbols"""
        symbols_info = {}

        # Try to get info from CoinGecko (most comprehensive)
        if "coingecko" in self.adapters:
            try:
                symbols_info = await self.adapters["coingecko"].get_symbols_info()
            except Exception as e:
                print(f"⚠️ Failed to get symbols info from CoinGecko: {e}")

        # If CoinGecko fails, create basic info
        if not symbols_info:
            for symbol in self.symbols:
                symbols_info[symbol] = {
                    "name": symbol.split("/")[0],
                    "base_asset": symbol.split("/")[0],
                    "quote_asset": symbol.split("/")[1] if "/" in symbol else "USDT",
                    "status": "trading",
                    "type": "spot",
                }

        return symbols_info

    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is available for trading"""
        try:
            # Try to get current price
            price = await self.get_current_price(symbol)
            return price is not None and price > 0
        except:
            return False

    async def get_top_gainers_losers(self, limit: int = 10) -> dict[str, list]:
        """Get top gainers and losers in the last 24h"""
        if "coingecko" in self.adapters:
            try:
                return await self.adapters["coingecko"].get_top_gainers_losers(limit)
            except Exception as e:
                print(f"⚠️ Failed to get top gainers/losers: {e}")

        return {"gainers": [], "losers": []}

    async def get_market_cap_ranking(self, limit: int = 100) -> list[dict]:
        """Get market cap ranking of cryptocurrencies"""
        if "coingecko" in self.adapters:
            try:
                return await self.adapters["coingecko"].get_market_cap_ranking(limit)
            except Exception as e:
                print(f"⚠️ Failed to get market cap ranking: {e}")

        return []

    async def close(self):
        """Close all adapters"""
        for adapter in self.adapters.values():
            if hasattr(adapter, "close"):
                await adapter.close()


class CoinGeckoAdapter:
    """CoinGecko API adapter (free tier)"""

    def __init__(self, config: dict):
        self.base_url = config["base_url"]
        self.rate_limit = config["rate_limit"]
        self.timeout = config["timeout"]
        self.session = None

    async def get_historical_data(
        self, symbol: str, limit: int, timeframe: str
    ) -> Optional[list[Bar]]:
        """Get historical data from CoinGecko"""
        try:
            # Convert symbol format (e.g., BTC/USDT -> bitcoin)
            coin_id = self._symbol_to_coin_id(symbol)
            if not coin_id:
                return None

            # Map timeframe
            days = self._timeframe_to_days(timeframe)

            url = f"{self.base_url}/coins/{coin_id}/ohlc"
            params = {"vs_currency": "usd", "days": days}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        bars = []
                        for item in data[-limit:]:  # Get last 'limit' bars
                            timestamp = datetime.fromtimestamp(
                                item[0] / 1000, tz=UTC
                            )
                            bar = Bar(
                                timestamp=timestamp,
                                symbol=symbol,
                                open=Decimal(str(item[1])),
                                high=Decimal(str(item[2])),
                                low=Decimal(str(item[3])),
                                close=Decimal(str(item[4])),
                                volume=Decimal(str(item[5] if len(item) > 5 else 0)),
                            )
                            bars.append(bar)

                        return bars
                    else:
                        print(f"CoinGecko API error: {response.status}")
                        return None

        except Exception as e:
            print(f"CoinGecko error: {e}")
            return None

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from CoinGecko"""
        try:
            coin_id = self._symbol_to_coin_id(symbol)
            if not coin_id:
                return None

            url = f"{self.base_url}/simple/price"
            params = {"ids": coin_id, "vs_currencies": "usd"}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get(coin_id, {}).get("usd")
                    else:
                        return None

        except Exception as e:
            print(f"CoinGecko price error: {e}")
            return None

    def _symbol_to_coin_id(self, symbol: str) -> Optional[str]:
        """Convert symbol to CoinGecko coin ID"""
        # This is a simplified mapping - in practice, you'd use the CoinGecko API
        # to get the full list of supported coins
        symbol_mapping = {
            "BTC/USDT": "bitcoin",
            "ETH/USDT": "ethereum",
            "BNB/USDT": "binancecoin",
            "ADA/USDT": "cardano",
            "SOL/USDT": "solana",
            "XRP/USDT": "ripple",
            "DOT/USDT": "polkadot",
            "DOGE/USDT": "dogecoin",
            "AVAX/USDT": "avalanche-2",
            "MATIC/USDT": "matic-network",
        }
        return symbol_mapping.get(symbol)

    def _timeframe_to_days(self, timeframe: str) -> int:
        """Convert timeframe to days for CoinGecko API"""
        timeframe_mapping = {"1m": 1, "5m": 1, "15m": 1, "1h": 1, "4h": 7, "1d": 30}
        return timeframe_mapping.get(timeframe, 1)


class BinancePublicAdapter:
    """Binance public API adapter (no API key required)"""

    def __init__(self, config: dict):
        self.base_url = config["base_url"]
        self.rate_limit = config["rate_limit"]
        self.timeout = config["timeout"]

    async def get_historical_data(
        self, symbol: str, limit: int, timeframe: str
    ) -> Optional[list[Bar]]:
        """Get historical data from Binance public API"""
        try:
            # Convert timeframe format
            interval = self._timeframe_to_interval(timeframe)

            url = f"{self.base_url}/api/v3/klines"
            params = {
                "symbol": symbol.replace("/", ""),
                "interval": interval,
                "limit": limit,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        bars = []
                        for item in data:
                            timestamp = datetime.fromtimestamp(
                                item[0] / 1000, tz=UTC
                            )
                            bar = Bar(
                                timestamp=timestamp,
                                symbol=symbol,
                                open=Decimal(str(item[1])),
                                high=Decimal(str(item[2])),
                                low=Decimal(str(item[3])),
                                close=Decimal(str(item[4])),
                                volume=Decimal(str(item[5])),
                            )
                            bars.append(bar)

                        return bars
                    else:
                        print(f"Binance API error: {response.status}")
                        return None

        except Exception as e:
            print(f"Binance error: {e}")
            return None

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from Binance"""
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {"symbol": symbol.replace("/", "")}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return float(data["price"])
                    else:
                        return None

        except Exception as e:
            print(f"Binance price error: {e}")
            return None

    def _timeframe_to_interval(self, timeframe: str) -> str:
        """Convert timeframe to Binance interval format"""
        interval_mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }
        return interval_mapping.get(timeframe, "1h")


class CCXTAdapter:
    """CCXT exchange adapter wrapper"""

    def __init__(self, exchange, config: dict):
        self.exchange = exchange
        self.rate_limit = config["rate_limit"]
        self.timeout = config["timeout"]

    async def get_historical_data(
        self, symbol: str, limit: int, timeframe: str
    ) -> Optional[list[Bar]]:
        """Get historical data using CCXT"""
        try:
            # Convert timeframe format
            ccxt_timeframe = self._timeframe_to_ccxt(timeframe)

            # Fetch OHLCV data
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.exchange.fetch_ohlcv(symbol, ccxt_timeframe, limit=limit),
            )

            if not ohlcv:
                return None

            bars = []
            for item in ohlcv:
                timestamp = datetime.fromtimestamp(item[0] / 1000, tz=UTC)
                bar = Bar(
                    timestamp=timestamp,
                    symbol=symbol,
                    open=Decimal(str(item[1])),
                    high=Decimal(str(item[2])),
                    low=Decimal(str(item[3])),
                    close=Decimal(str(item[4])),
                    volume=Decimal(str(item[5])),
                )
                bars.append(bar)

            return bars

        except Exception as e:
            print(f"CCXT error for {symbol}: {e}")
            return None

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price using CCXT"""
        try:
            ticker = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.exchange.fetch_ticker(symbol)
            )

            if ticker and "last" in ticker:
                return float(ticker["last"])
            else:
                return None

        except Exception as e:
            print(f"CCXT price error for {symbol}: {e}")
            return None

    def _timeframe_to_ccxt(self, timeframe: str) -> str:
        """Convert timeframe to CCXT format"""
        ccxt_mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }
        return ccxt_mapping.get(timeframe, "1h")

    async def close(self):
        """Close the exchange connection"""
        if hasattr(self.exchange, "close"):
            await asyncio.get_event_loop().run_in_executor(None, self.exchange.close)


class WebScraperAdapter:
    """Web scraping fallback adapter"""

    def __init__(self, config: dict):
        self.rate_limit = config["rate_limit"]
        self.timeout = config["timeout"]

    async def get_historical_data(
        self, symbol: str, limit: int, timeframe: str
    ) -> Optional[list[Bar]]:
        """Web scraping fallback - not implemented for now"""
        print(f"⚠️ Web scraping not implemented for {symbol}")
        return None

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Web scraping fallback - not implemented for now"""
        print(f"⚠️ Web scraping not implemented for {symbol}")
        return None


class CoinMarketCapAdapter:
    """CoinMarketCap API adapter (free tier)"""

    def __init__(self, config: dict):
        self.base_url = config["base_url"]
        self.rate_limit = config["rate_limit"]
        self.timeout = config["timeout"]

    async def get_historical_data(
        self, symbol: str, limit: int, timeframe: str
    ) -> Optional[list[Bar]]:
        """CoinMarketCap doesn't provide free historical data"""
        print(
            f"⚠️ CoinMarketCap free tier doesn't support historical data for {symbol}"
        )
        return None

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from CoinMarketCap (limited free tier)"""
        print(f"⚠️ CoinMarketCap free tier has limited access for {symbol}")
        return None
