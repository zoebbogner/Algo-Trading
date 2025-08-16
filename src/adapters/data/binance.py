"""Binance data adapter using CCXT library."""

import asyncio
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any

import ccxt.async_support as ccxt

from .base import DataAdapter
from ...core.data_models.market import Bar
from ...utils.logging import logger


class BinanceAdapter(DataAdapter):
    """Binance exchange data adapter."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Binance adapter.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Binance-specific configuration
        self.api_key = config.get("api_key")
        self.secret = config.get("secret")
        self.sandbox = config.get("sandbox", True)  # Use testnet by default
        self.rate_limit_delay = config.get("rate_limit_delay", 0.1)
        
        # Initialize CCXT exchange
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.secret,
            'sandbox': self.sandbox,
            'enableRateLimit': True,
            'rateLimit': int(self.rate_limit_delay * 1000)
        })
        
        self._connected = False
        logger.logger.info(f"Initialized Binance adapter (sandbox: {self.sandbox})")
    
    async def connect(self) -> bool:
        """Connect to Binance.
        
        Returns:
            True if connection successful
        """
        try:
            # Test connection by loading markets
            await self.exchange.load_markets()
            self._connected = True
            logger.logger.info("Connected to Binance successfully")
            return True
            
        except Exception as e:
            logger.logger.error(f"Failed to connect to Binance: {e}")
            self._connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Binance."""
        try:
            await self.exchange.close()
            self._connected = False
            logger.logger.info("Disconnected from Binance")
        except Exception as e:
            logger.logger.error(f"Error disconnecting from Binance: {e}")
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        limit: int,
        since: Optional[datetime] = None
    ) -> List[Bar]:
        """Get historical OHLCV data from Binance.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Time interval (e.g., '1m', '5m', '1h')
            limit: Number of bars to retrieve
            since: Start time for data retrieval
            
        Returns:
            List of Bar objects
        """
        if not self._connected:
            await self.connect()
        
        try:
            # Convert symbol format (CCXT uses 'BTC/USDT', Binance uses 'BTCUSDT')
            ccxt_symbol = symbol.replace('/', '')
            
            # Convert timeframe
            ccxt_timeframe = self._convert_timeframe(timeframe)
            
            # Get OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(
                ccxt_symbol, 
                ccxt_timeframe, 
                limit=limit,
                since=int(since.timestamp() * 1000) if since else None
            )
            
            # Convert to Bar objects
            bars = []
            for data in ohlcv:
                bar = self._create_bar_from_data({
                    'timestamp': data[0],
                    'open': data[1],
                    'high': data[2],
                    'low': data[3],
                    'close': data[4],
                    'volume': data[5]
                }, symbol, timeframe)
                bars.append(bar)
            
            logger.logger.info(f"Retrieved {len(bars)} historical bars for {symbol}")
            return bars
            
        except Exception as e:
            logger.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []
    
    async def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None if unavailable
        """
        if not self._connected:
            await self.connect()
        
        try:
            ccxt_symbol = symbol.replace('/', '')
            ticker = await self.exchange.fetch_ticker(ccxt_symbol)
            
            if ticker and 'last' in ticker:
                price = Decimal(str(ticker['last']))
                logger.logger.debug(f"Current price for {symbol}: {price}")
                return price
            
            return None
            
        except Exception as e:
            logger.logger.error(f"Error fetching current price for {symbol}: {e}")
            return None
    
    async def get_order_book(self, symbol: str, depth: int = 20) -> Optional[Dict]:
        """Get order book for a symbol.
        
        Args:
            symbol: Trading symbol
            depth: Number of orders to retrieve
            
        Returns:
            Order book dictionary or None if unavailable
        """
        if not self._connected:
            await self.connect()
        
        try:
            ccxt_symbol = symbol.replace('/', '')
            order_book = await self.exchange.fetch_order_book(ccxt_symbol, depth)
            
            if order_book:
                # Convert to our format
                return {
                    'bids': [[Decimal(str(price)), Decimal(str(amount))] for price, amount in order_book['bids']],
                    'asks': [[Decimal(str(price)), Decimal(str(amount))] for price, amount in order_book['asks']],
                    'timestamp': datetime.fromtimestamp(order_book['timestamp'] / 1000, tz=timezone.utc),
                    'symbol': symbol
                }
            
            return None
            
        except Exception as e:
            logger.logger.error(f"Error fetching order book for {symbol}: {e}")
            return None
    
    async def get_24h_ticker(self, symbol: str) -> Optional[Dict]:
        """Get 24-hour ticker statistics.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Ticker statistics or None if unavailable
        """
        if not self._connected:
            await self.connect()
        
        try:
            ccxt_symbol = symbol.replace('/', '')
            ticker = await self.exchange.fetch_ticker(ccxt_symbol)
            
            if ticker:
                return {
                    'symbol': symbol,
                    'open': Decimal(str(ticker['open'])),
                    'high': Decimal(str(ticker['high'])),
                    'low': Decimal(str(ticker['low'])),
                    'close': Decimal(str(ticker['close'])),
                    'volume': Decimal(str(ticker['baseVolume'])),
                    'change': Decimal(str(ticker['change'])),
                    'change_percent': Decimal(str(ticker['percentage'])),
                    'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000, tz=timezone.utc)
                }
            
            return None
            
        except Exception as e:
            logger.logger.error(f"Error fetching 24h ticker for {symbol}: {e}")
            return None
    
    async def get_exchange_info(self) -> Optional[Dict]:
        """Get exchange information and supported symbols.
        
        Returns:
            Exchange information or None if unavailable
        """
        if not self._connected:
            await self.connect()
        
        try:
            markets = await self.exchange.load_markets()
            
            # Filter for USDT pairs (most common)
            usdt_pairs = [symbol for symbol in markets.keys() if symbol.endswith('/USDT')]
            
            return {
                'exchange': 'Binance',
                'sandbox': self.sandbox,
                'total_symbols': len(markets),
                'usdt_pairs': len(usdt_pairs),
                'sample_symbols': usdt_pairs[:10],  # First 10 USDT pairs
                'timestamp': datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.logger.error(f"Error fetching exchange info: {e}")
            return None
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Binance format.
        
        Args:
            timeframe: Standard timeframe (e.g., '1m', '5m', '1h')
            
        Returns:
            Binance-specific timeframe string
        """
        # Binance uses the same format as standard
        return super()._convert_timeframe(timeframe)
    
    def _create_bar_from_data(self, data: Dict[str, Any], symbol: str, interval: str) -> Bar:
        """Create Bar object from Binance OHLCV data.
        
        Args:
            data: Raw OHLCV data
            symbol: Trading symbol
            interval: Time interval
            
        Returns:
            Bar object
        """
        # Binance timestamp is in milliseconds
        timestamp = datetime.fromtimestamp(data.get('timestamp', 0) / 1000, tz=timezone.utc)
        
        return Bar(
            timestamp=timestamp,
            symbol=symbol,
            open=Decimal(str(data.get('open', 0))),
            high=Decimal(str(data.get('high', 0))),
            low=Decimal(str(data.get('low', 0))),
            close=Decimal(str(data.get('close', 0))),
            volume=Decimal(str(data.get('volume', 0))),
            interval=interval
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
