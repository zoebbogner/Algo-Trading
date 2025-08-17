"""Data collection from various cryptocurrency exchanges and APIs."""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

from .models import Bar, MarketData, DataRequest, DataResponse
from .exchanges import BinanceCollector, CoinGeckoCollector, CCXTCollector


logger = logging.getLogger(__name__)


class DataCollector:
    """Main data collector that coordinates multiple data sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data collector with configuration."""
        self.config = config
        self.collectors = self._initialize_collectors()
        self.session: Optional[aiohttp.ClientSession] = None
        
    def _initialize_collectors(self) -> Dict[str, Any]:
        """Initialize data collectors for different sources."""
        collectors = {}
        
        if self.config.get("binance", {}).get("enabled", False):
            collectors["binance"] = BinanceCollector(self.config["binance"])
            
        if self.config.get("coingecko", {}).get("enabled", False):
            collectors["coingecko"] = CoinGeckoCollector(self.config["coingecko"])
            
        if self.config.get("ccxt", {}).get("enabled", False):
            collectors["ccxt"] = CCXTCollector(self.config["ccxt"])
            
        return collectors
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def collect_historical_data(
        self, 
        request: DataRequest
    ) -> DataResponse:
        """Collect historical data for the given request."""
        try:
            logger.info(f"Collecting data for {request.symbol} from {request.start_date} to {request.end_date}")
            
            # Try multiple sources in order of preference
            for source_name, collector in self.collectors.items():
                try:
                    logger.info(f"Trying {source_name} collector...")
                    data = await collector.collect_data(request, self.session)
                    if data and len(data.bars) > 0:
                        logger.info(f"Successfully collected {len(data.bars)} bars from {source_name}")
                        return DataResponse(
                            request=request,
                            data=data,
                            success=True
                        )
                except Exception as e:
                    logger.warning(f"Failed to collect from {source_name}: {e}")
                    continue
            
            # If all collectors fail
            error_msg = f"All data collectors failed for {request.symbol}"
            logger.error(error_msg)
            return DataResponse(
                request=request,
                data=MarketData(
                    timestamp=datetime.now(),
                    bars={}
                ),
                success=False,
                error_message=error_msg
            )
            
        except Exception as e:
            logger.error(f"Error collecting data: {e}")
            return DataResponse(
                request=request,
                data=MarketData(
                    timestamp=datetime.now(),
                    bars={}
                ),
                success=False,
                error_message=str(e)
            )
    
    async def collect_multiple_symbols(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime,
        interval: str = "1d"
    ) -> Dict[str, DataResponse]:
        """Collect data for multiple symbols concurrently."""
        tasks = []
        
        for symbol in symbols:
            request = DataRequest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            task = self.collect_historical_data(request)
            tasks.append((symbol, task))
        
        # Execute all tasks concurrently
        results = {}
        for symbol, task in tasks:
            try:
                result = await task
                results[symbol] = result
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
                results[symbol] = DataResponse(
                    request=DataRequest(symbol=symbol, start_date=start_date, end_date=end_date, interval=interval),
                    data=MarketData(timestamp=datetime.now(), bars={}),
                    success=False,
                    error_message=str(e)
                )
        
        return results
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from all collectors."""
        symbols = set()
        for collector in self.collectors.values():
            symbols.update(collector.get_available_symbols())
        return sorted(list(symbols))
    
    def get_collector_status(self) -> Dict[str, bool]:
        """Get status of all collectors."""
        return {
            name: collector.is_available() 
            for name, collector in self.collectors.items()
        }
