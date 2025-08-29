#!/usr/bin/env python3
"""
Binance REST API data fetcher for recent data.

This module handles fetching recent data via Binance's REST API.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

from src.data.fetch.base import BaseDataFetcher
from src.utils.logging import get_logger
from src.config.base import load_config as load_base_config
from src.config.data_history import load_config as load_data_config

# Get logger for this module
logger = get_logger(__name__)


class BinanceRESTFetcher(BaseDataFetcher):
    """Fetches recent data from Binance REST API."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the REST fetcher."""
        super().__init__(config)
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.session = None
        self.rate_limit_ms = config.get('collection', {}).get('rest_rate_limit_ms', 100)

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

    def _get_timestamp_range(self, days: int) -> tuple:
        """Get timestamp range for the specified number of days."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Convert to milliseconds since epoch
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        return start_ms, end_ms

    async def _fetch_klines(self, symbol: str, start_time: int, end_time: int, limit: int = 1000) -> list[list]:
        """Fetch klines data for a specific time range."""
        params = {
            'symbol': symbol,
            'interval': '1m',
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }

        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Failed to fetch klines for {symbol}: HTTP {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return []

    def _process_klines(self, klines_data: list[list], symbol: str) -> pd.DataFrame:
        """Process raw klines data into a DataFrame."""
        if not klines_data:
            return pd.DataFrame()

        # Binance klines format: [open_time, open, high, low, close, volume, close_time, ...]
        df = pd.DataFrame(klines_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'n_trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convert types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'n_trades', 'taker_buy_base', 'taker_buy_quote']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert timestamps
        df['ts'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df['ts_raw'] = df['open_time']

        # Add metadata
        df['symbol'] = symbol
        df['source'] = 'binance:rest'
        df['load_id'] = self.run_id

        # Select and reorder columns
        columns = ['ts', 'ts_raw', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'source', 'load_id']
        df = df[columns]

        return df

    async def fetch_data(self, symbols: list[str], days: int = 7) -> dict[str, Any]:
        """Fetch recent data for the given symbols."""
        logger.info(f"Starting REST API fetch for {len(symbols)} symbols")
        logger.info(f"Fetching {days} days of data")

        start_time = datetime.now()
        results = {
            'run_id': self.run_id,
            'start_time': start_time.isoformat(),
            'symbols': symbols,
            'days': days,
            'fetched_data': {},
            'summary': {}
        }

        # Get timestamp range
        start_ms, end_ms = self._get_timestamp_range(days)

        # Fetch data for each symbol
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")

            try:
                # Fetch klines data
                klines_data = await self._fetch_klines(symbol, start_ms, end_ms)

                if klines_data:
                    # Process data
                    df = self._process_klines(klines_data, symbol)

                    # Save to CSV
                    symbol_dir = Path(f"data/raw/binance/{symbol}/1m")
                    symbol_dir.mkdir(parents=True, exist_ok=True)

                    filename = f"{symbol.lower()}_rest_{self.run_id}.csv"
                    filepath = symbol_dir / filename
                    df.to_csv(filepath, index=False)

                    results['fetched_data'][symbol] = {
                        'rows': len(df),
                        'filename': filename,
                        'filepath': str(filepath),
                        'start_ts': df['ts'].min().isoformat() if not df.empty else None,
                        'end_ts': df['ts'].max().isoformat() if not df.empty else None
                    }

                    logger.info(f"Saved {len(df)} rows for {symbol} to {filename}")

                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_ms / 1000)
                else:
                    logger.warning(f"No data fetched for {symbol}")
                    results['fetched_data'][symbol] = {
                        'rows': 0,
                        'error': 'No data returned from API'
                    }

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results['fetched_data'][symbol] = {
                    'rows': 0,
                    'error': str(e)
                }

        # Generate summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        total_rows = sum(data.get('rows', 0) for data in results['fetched_data'].values())
        successful_symbols = sum(1 for data in results['fetched_data'].values() if data.get('rows', 0) > 0)

        results['summary'] = {
            'total_symbols': len(symbols),
            'successful_symbols': successful_symbols,
            'total_rows': total_rows,
            'duration_seconds': duration,
            'end_time': end_time.isoformat()
        }

        # Save report
        report_filename = f"rest_fetch_report_{self.run_id}.yaml"
        self.save_report(results, report_filename)

        # Log summary
        self.log_summary("REST API Fetch", results['summary'])

        return results


async def main():
    """Main function for REST API fetching."""
    import argparse

    parser = argparse.ArgumentParser(description='Fetch Binance REST API data')
    parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to fetch')
    parser.add_argument('--days', type=int, default=7, help='Number of days to fetch (default: 7)')

    args = parser.parse_args()

    # Load configuration
    base_config = load_base_config()
    data_config = load_data_config()
    config = {**base_config, **data_config}

    # Create fetcher and fetch data
    async with BinanceRESTFetcher(config) as fetcher:
        results = await fetcher.fetch_data(args.symbols, args.days)

        if results['summary']['successful_symbols'] > 0:
            logger.info("REST API fetch completed successfully!")
            return 0
        else:
            logger.error("REST API fetch failed")
            return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
