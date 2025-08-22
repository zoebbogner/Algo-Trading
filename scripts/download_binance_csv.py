#!/usr/bin/env python3
"""
Binance CSV Data Downloader

Downloads historical 1-minute OHLCV data from Binance's public data dump.
This script handles bulk downloads for fast historical data collection.

Usage:
    python scripts/download_binance_csv.py --symbols BTCUSDT ETHUSDT --start-date 2024-07-01 --end-date 2025-08-20
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import aiohttp
import yaml

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
        logging.FileHandler('logs/data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BinanceCSVDownloader:
    """Downloads historical CSV data from Binance's public data dump."""
    
    def __init__(self, config: dict):
        """Initialize the downloader with configuration."""
        self.config = config
        self.base_url = "https://data.binance.vision/data/spot/monthly/klines"
        self.raw_dir = Path(config['raw_dir'])
        self.session: Optional[aiohttp.ClientSession] = None
        
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
    
    def _generate_monthly_dates(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Generate list of monthly dates between start and end."""
        dates = []
        current = start_date.replace(day=1)  # Start of month
        
        while current <= end_date:
            dates.append(current)
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return dates
    
    def _get_csv_url(self, symbol: str, interval: str, date: datetime) -> str:
        """Generate CSV download URL for a specific month."""
        year_month = date.strftime("%Y-%m")
        filename = f"{symbol}-{interval}-{year_month}.zip"
        return f"{self.base_url}/{symbol}/{interval}/{filename}"
    
    async def _download_file(self, url: str, local_path: Path) -> bool:
        """Download a single file from URL to local path."""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    # Create parent directory if it doesn't exist
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download and save file
                    with open(local_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    
                    logger.info(f"Downloaded: {local_path.name}")
                    return True
                else:
                    logger.warning(f"Failed to download {url}: HTTP {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False
    
    async def download_symbol_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> dict:
        """Download all available CSV data for a symbol in the date range."""
        logger.info(f"Starting download for {symbol} from {start_date} to {end_date}")
        
        monthly_dates = self._generate_monthly_dates(start_date, end_date)
        download_results = {
            'symbol': symbol,
            'total_files': len(monthly_dates),
            'downloaded': 0,
            'failed': 0,
            'errors': []
        }
        
        # Download files with rate limiting
        for i, date in enumerate(monthly_dates):
            url = self._get_csv_url(symbol, self.config['timeframe'], date)
            filename = f"{symbol}-{self.config['timeframe']}-{date.strftime('%Y-%m')}.zip"
            local_path = self.raw_dir / symbol / self.config['timeframe'] / filename
            
            # Skip if file already exists
            if local_path.exists():
                logger.info(f"File already exists, skipping: {filename}")
                download_results['downloaded'] += 1
                continue
            
            # Download file
            success = await self._download_file(url, local_path)
            if success:
                download_results['downloaded'] += 1
            else:
                download_results['failed'] += 1
                download_results['errors'].append(f"Failed to download {filename}")
            
            # Rate limiting between downloads
            if i < len(monthly_dates) - 1:  # Don't sleep after last file
                await asyncio.sleep(0.1)  # 100ms delay between downloads
        
        logger.info(f"Download completed for {symbol}: {download_results['downloaded']}/{download_results['total_files']} files")
        return download_results
    
    async def download_all_symbols(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> dict:
        """Download data for all configured symbols."""
        logger.info(f"Starting bulk download for {len(self.config['symbols'])} symbols")
        
        all_results = {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'symbols': {},
            'summary': {
                'total_files': 0,
                'total_downloaded': 0,
                'total_failed': 0
            }
        }
        
        # Download data for each symbol
        for symbol in self.config['symbols']:
            symbol_results = await self.download_symbol_data(symbol, start_date, end_date)
            all_results['symbols'][symbol] = symbol_results
            
            # Update summary
            all_results['summary']['total_files'] += symbol_results['total_files']
            all_results['summary']['total_downloaded'] += symbol_results['downloaded']
            all_results['summary']['total_failed'] += symbol_results['failed']
        
        # Log summary
        logger.info(f"Bulk download completed:")
        logger.info(f"  Total files: {all_results['summary']['total_files']}")
        logger.info(f"  Downloaded: {all_results['summary']['total_downloaded']}")
        logger.info(f"  Failed: {all_results['summary']['total_failed']}")
        
        return all_results

async def main():
    """Main function to run the CSV downloader."""
    parser = argparse.ArgumentParser(description='Download Binance historical CSV data')
    parser.add_argument('--symbols', nargs='+', help='Trading symbols to download')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
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
    
    # Parse dates
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = datetime.strptime(config['history']['start_ts_utc'][:10], '%Y-%m-%d')
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.strptime(config['history']['end_ts_utc'][:10], '%Y-%m-%d')
    
    # Validate dates
    if start_date >= end_date:
        logger.error("Start date must be before end date")
        return 1
    
    # Create downloader and run
    async with BinanceCSVDownloader(config) as downloader:
        results = await downloader.download_all_symbols(start_date, end_date)
        
        # Save results to reports directory
        reports_dir = Path('reports/runs')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = reports_dir / f'csv_download_{run_id}.yaml'
        
        with open(results_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False, indent=2)
        
        logger.info(f"Download results saved to: {results_file}")
        
        # Return exit code based on success
        if results['summary']['total_failed'] == 0:
            logger.info("All downloads completed successfully!")
            return 0
        else:
            logger.warning(f"Some downloads failed: {results['summary']['total_failed']} files")
            return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
