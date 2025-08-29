#!/usr/bin/env python3
"""
Binance CSV data fetcher for bulk historical data.

This module handles downloading and processing bulk CSV data from Binance public data dumps.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import aiohttp

from src.data.fetch.base import BaseDataFetcher
from src.config.base import load_config as load_base_config
from src.config.data_history import load_config as load_data_config
from src.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


class BinanceCSVFetcher(BaseDataFetcher):
    """Fetches historical CSV data from Binance's public data dump."""

    def __init__(self, config: dict[str, str]):
        """Initialize the CSV fetcher."""
        super().__init__(config)
        self.base_url = "https://data.binance.vision/data/spot/monthly/klines"
        self.session = None

    async def __aenter__(self):
        """Async context manager entry."""
        # Create session with SSL context that handles certificate verification
        connector = aiohttp.TCPConnector(ssl=False)  # Disable SSL verification for now
        self.session = aiohttp.ClientSession(connector=connector)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def _generate_monthly_urls(self, symbol: str, start_date: str, end_date: str) -> list[str]:
        """Generate monthly download URLs for the given date range."""
        urls = []

        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Generate monthly URLs
        current = start.replace(day=1)  # Start of month

        while current <= end:
            month_str = current.strftime("%Y-%m")
            url = f"{self.base_url}/{symbol}/1m/{symbol}-1m-{month_str}.zip"
            urls.append(url)

            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return urls

    async def _download_file(self, url: str, symbol: str, month: str) -> dict[str, str]:
        """Download a single CSV file."""
        try:
            # Create symbol directory
            symbol_dir = Path(f"data/raw/binance/{symbol}/1m")
            symbol_dir.mkdir(parents=True, exist_ok=True)

            # Download file
            async with self.session.get(url) as response:
                if response.status == 200:
                    # Save ZIP file
                    zip_filename = f"{symbol}-1m-{month}.zip"
                    zip_path = symbol_dir / zip_filename

                    with open(zip_path, 'wb') as f:
                        f.write(await response.read())

                    file_size = zip_path.stat().st_size
                    logger.info(f"Downloaded {zip_filename} ({file_size:,} bytes)")

                    return {
                        'status': 'success',
                        'url': url,
                        'filename': zip_filename,
                        'size_bytes': file_size,
                        'symbol': symbol,
                        'month': month
                    }
                else:
                    logger.warning(f"Failed to download {url}: HTTP {response.status}")
                    return {
                        'status': 'failed',
                        'url': url,
                        'error': f"HTTP {response.status}",
                        'symbol': symbol,
                        'month': month
                    }

        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return {
                'status': 'error',
                'url': url,
                'error': str(e),
                'symbol': symbol,
                'month': month
            }

    async def fetch_data(self, symbols: list[str], start_date: str, end_date: str) -> dict[str, str]:
        """Fetch CSV data for the given symbols and date range."""
        logger.info(f"Starting CSV download for {len(symbols)} symbols")
        logger.info(f"Date range: {start_date} to {end_date}")

        start_time = datetime.now()
        results = {
            'run_id': self.run_id,
            'start_time': start_time.isoformat(),
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'downloads': [],
            'summary': {}
        }

        # Download for each symbol
        for symbol in symbols:
            logger.info(f"Processing {symbol}")

            # Generate URLs for this symbol
            urls = self._generate_monthly_urls(symbol, start_date, end_date)
            logger.info(f"Generated {len(urls)} URLs for {symbol}")

            # Download files concurrently
            tasks = []
            for url in urls:
                month = url.split('-')[-1].replace('.zip', '')
                task = self._download_file(url, symbol, month)
                tasks.append(task)

            # Wait for all downloads to complete
            symbol_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in symbol_results:
                if isinstance(result, Exception):
                    logger.error(f"Exception in download: {result}")
                    results['downloads'].append({
                        'status': 'exception',
                        'error': str(result),
                        'symbol': symbol
                    })
                else:
                    results['downloads'].append(result)

        # Generate summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        successful = [r for r in results['downloads'] if r['status'] == 'success']
        failed = [r for r in results['downloads'] if r['status'] in ['failed', 'error']]

        results['summary'] = {
            'total_downloads': len(results['downloads']),
            'successful': len(successful),
            'failed': len(failed),
            'duration_seconds': duration,
            'end_time': end_time.isoformat()
        }

        # Save report
        report_filename = f"csv_download_report_{self.run_id}.yaml"
        self.save_report(results, report_filename)

        # Log summary
        self.log_summary("CSV Download", results['summary'])

        return results


async def main():
    """Main function for CSV downloading."""
    import argparse

    parser = argparse.ArgumentParser(description='Download Binance CSV data')
    parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to download')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')

    args = parser.parse_args()

    # Load configuration
    base_config = load_base_config()
    data_config = load_data_config()
    config = {**base_config, **data_config}

    # Create fetcher and download data
    async with BinanceCSVFetcher(config) as fetcher:
        results = await fetcher.fetch_data(args.symbols, args.start_date, args.end_date)

        if results['summary']['successful'] > 0:
            logger.info("CSV download completed successfully!")
            return 0
        else:
            logger.error("CSV download failed")
            return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
