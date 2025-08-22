#!/usr/bin/env python3
"""
Binance REST API Data Fetcher

Fetches recent 1-minute OHLCV data from Binance REST API to top up CSV data
and fill any gaps. This script handles the secondary data source for recent data.

Usage:
    python scripts/fetch_binance_rest.py --symbols BTCUSDT ETHUSDT --days 7
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import aiohttp
import yaml
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.base import load_config as load_base_config
from configs.data_history import load_config as load_data_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rest_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BinanceRESTFetcher:
    """Fetches recent data from Binance REST API."""
    
    def __init__(self, config: dict):
        """Initialize the REST fetcher with configuration."""
        self.config = config
        self.base_url = "https://api.binance.com/api/v3"
        self.raw_dir = Path(config['raw_dir'])
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting settings
        self.rate_limit_ms = config.get('collection', {}).get('rest_rate_limit_ms', 100)
        self.batch_size = config.get('collection', {}).get('rest_batch_size', 1000)
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for symbol in self.config['symbols']:
            symbol_dir = self.raw_dir / symbol / self.config['timeframe']
            symbol_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {symbol_dir}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _get_last_csv_timestamp(self, symbol: str) -> Optional[datetime]:
        """Get the last timestamp from existing CSV data."""
        symbol_dir = self.raw_dir / symbol / self.config['timeframe']
        
        if not symbol_dir.exists():
            return None
        
        # Find the most recent CSV file
        csv_files = list(symbol_dir.glob("*.csv"))
        if not csv_files:
            return None
        
        # Sort by modification time and get the most recent
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        
        # Extract date from filename (e.g., BTCUSDT-1m-2025-08.csv)
        try:
            date_str = latest_file.stem.split('-')[-1]  # Get YYYY-MM part
            return datetime.strptime(date_str, '%Y-%m')
        except (ValueError, IndexError):
            logger.warning(f"Could not parse date from filename: {latest_file.name}")
            return None
    
    async def _fetch_klines(
        self, 
        symbol: str, 
        start_time: int, 
        end_time: int
    ) -> List[List[Any]]:
        """Fetch klines data from Binance REST API."""
        url = f"{self.base_url}/klines"
        params = {
            'symbol': symbol,
            'interval': '1m',
            'startTime': start_time,
            'endTime': end_time,
            'limit': self.batch_size
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Fetched {len(data)} klines for {symbol}")
                    return data
                else:
                    logger.error(f"HTTP {response.status}: {response.reason}")
                    response.raise_for_status()
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            raise
    
    def _convert_klines_to_csv_format(self, klines: List[List[Any]], symbol: str) -> str:
        """Convert klines data to CSV format matching the raw schema."""
        csv_lines = [
            "open_time,open,high,low,close,volume,close_time,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume,ignore"
        ]
        
        for kline in klines:
            # Binance klines format:
            # [open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
            csv_line = ','.join(str(field) for field in kline)
            csv_lines.append(csv_line)
        
        return '\n'.join(csv_lines)
    
    async def _save_rest_data(
        self, 
        symbol: str, 
        data: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Path:
        """Save REST API data to a CSV file."""
        # Generate filename with timestamp range
        start_str = start_time.strftime('%Y%m%d_%H%M')
        end_str = end_time.strftime('%Y%m%d_%H%M')
        filename = f"{symbol}-{self.config['timeframe']}-rest_{start_str}_to_{end_str}.csv"
        
        file_path = self.raw_dir / symbol / self.config['timeframe'] / filename
        
        with open(file_path, 'w') as f:
            f.write(data)
        
        logger.info(f"Saved REST data to: {file_path}")
        return file_path
    
    async def fetch_recent_data(
        self, 
        symbol: str, 
        days_back: int = 7
    ) -> dict:
        """Fetch recent data for a symbol."""
        logger.info(f"Fetching recent data for {symbol} (last {days_back} days)")
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)
        
        # Convert to milliseconds for API
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        fetch_results = {
            'symbol': symbol,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'klines_fetched': 0,
            'files_saved': 0,
            'errors': []
        }
        
        try:
            # Fetch data in batches
            current_start = start_ms
            batch_count = 0
            
            while current_start < end_ms:
                current_end = min(current_start + (self.batch_size * 60 * 1000), end_ms)
                
                # Fetch batch
                klines = await self._fetch_klines(symbol, current_start, current_end)
                
                if klines:
                    # Convert to CSV format
                    csv_data = self._convert_klines_to_csv_format(klines, symbol)
                    
                    # Save to file
                    batch_start = datetime.fromtimestamp(current_start / 1000)
                    batch_end = datetime.fromtimestamp(current_end / 1000)
                    
                    file_path = await self._save_rest_data(symbol, csv_data, batch_start, batch_end)
                    
                    fetch_results['klines_fetched'] += len(klines)
                    fetch_results['files_saved'] += 1
                    batch_count += 1
                
                # Move to next batch
                current_start = current_end
                
                # Rate limiting between batches
                if current_start < end_ms:
                    await asyncio.sleep(self.rate_limit_ms / 1000)
            
            logger.info(f"Fetched {fetch_results['klines_fetched']} klines for {symbol}")
            
        except Exception as e:
            error_msg = f"Error fetching data for {symbol}: {e}"
            logger.error(error_msg)
            fetch_results['errors'].append(error_msg)
        
        return fetch_results
    
    async def fill_gaps(
        self, 
        symbol: str, 
        gap_threshold_hours: int = 24
    ) -> dict:
        """Fill gaps in historical data for a symbol."""
        logger.info(f"Checking for gaps in {symbol} data")
        
        gap_results = {
            'symbol': symbol,
            'gaps_found': 0,
            'gaps_filled': 0,
            'errors': []
        }
        
        try:
            # Get last CSV timestamp
            last_csv_time = self._get_last_csv_timestamp(symbol)
            if not last_csv_time:
                logger.info(f"No existing CSV data for {symbol}, skipping gap check")
                return gap_results
            
            # Calculate time since last CSV
            now = datetime.utcnow()
            time_since_csv = now - last_csv_time
            
            # Check if gap is large enough to fill
            if time_since_csv.total_seconds() / 3600 < gap_threshold_hours:
                logger.info(f"Gap for {symbol} is {time_since_csv.total_seconds() / 3600:.1f} hours, below threshold")
                return gap_results
            
            logger.info(f"Filling gap for {symbol}: {time_since_csv.total_seconds() / 3600:.1f} hours")
            
            # Fetch data to fill the gap
            gap_days = int(time_since_csv.total_seconds() / 86400) + 1
            gap_data = await self.fetch_recent_data(symbol, gap_days)
            
            gap_results['gaps_found'] = 1
            if not gap_data['errors']:
                gap_results['gaps_filled'] = 1
            
        except Exception as e:
            error_msg = f"Error filling gaps for {symbol}: {e}"
            logger.error(error_msg)
            gap_results['errors'].append(error_msg)
        
        return gap_results
    
    async def fetch_all_symbols(
        self, 
        days_back: int = 7, 
        fill_gaps: bool = True
    ) -> dict:
        """Fetch recent data for all configured symbols."""
        logger.info(f"Starting REST fetch for {len(self.config['symbols'])} symbols")
        
        all_results = {
            'fetch_time': datetime.utcnow().isoformat(),
            'days_back': days_back,
            'fill_gaps': fill_gaps,
            'symbols': {},
            'summary': {
                'total_klines': 0,
                'total_files': 0,
                'total_errors': 0
            }
        }
        
        # Fetch data for each symbol
        for symbol in self.config['symbols']:
            symbol_results = await self.fetch_recent_data(symbol, days_back)
            all_results['symbols'][symbol] = symbol_results
            
            # Update summary
            all_results['summary']['total_klines'] += symbol_results['klines_fetched']
            all_results['summary']['total_files'] += symbol_results['files_saved']
            all_results['summary']['total_errors'] += len(symbol_results['errors'])
            
            # Fill gaps if requested
            if fill_gaps:
                gap_results = await self.fill_gaps(symbol)
                symbol_results['gap_filling'] = gap_results
        
        # Log summary
        logger.info(f"REST fetch completed:")
        logger.info(f"  Total klines: {all_results['summary']['total_klines']}")
        logger.info(f"  Total files: {all_results['summary']['total_files']}")
        logger.info(f"  Total errors: {all_results['summary']['total_errors']}")
        
        return all_results

async def main():
    """Main function to run the REST fetcher."""
    parser = argparse.ArgumentParser(description='Fetch recent Binance data via REST API')
    parser.add_argument('--symbols', nargs='+', help='Trading symbols to fetch')
    parser.add_argument('--days', type=int, default=7, help='Number of days to fetch back')
    parser.add_argument('--fill-gaps', action='store_true', help='Fill gaps in historical data')
    parser.add_argument('--config', default='configs/data.history.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        base_config = load_base_config()
        data_config = load_data_config()
        config = {**base_config, **data_config}
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Override config with command line arguments
    if args.symbols:
        config['symbols'] = args.symbols
    
    # Create fetcher and run
    async with BinanceRESTFetcher(config) as fetcher:
        results = await fetcher.fetch_all_symbols(args.days, args.fill_gaps)
        
        # Save results to reports directory
        reports_dir = Path('reports/runs')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = reports_dir / f'rest_fetch_{run_id}.yaml'
        
        with open(results_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False, indent=2)
        
        logger.info(f"REST fetch results saved to: {results_file}")
        
        # Return exit code based on success
        if results['summary']['total_errors'] == 0:
            logger.info("All REST fetches completed successfully!")
            return 0
        else:
            logger.warning(f"Some REST fetches had errors: {results['summary']['total_errors']}")
            return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
