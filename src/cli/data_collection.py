#!/usr/bin/env python3
"""
Data Collection CLI for the Algo-Trading system.

Provides command-line interface for:
- Historical data collection from Binance
- Real-time data streaming
- Data validation and quality control
- Data consolidation and partitioning
"""

import asyncio
import sys
from pathlib import Path

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.base import load_config as load_base_config
from src.config.data_history import load_config as load_data_config
from src.data.fetch.binance_csv import BinanceCSVFetcher
from src.data.fetch.binance_rest import BinanceRESTFetcher
from src.data.processing.consolidator import DataConsolidator
from src.utils.logging import get_logger, setup_logging


@click.group()
@click.option('--config', '-c', default='configs/data.history.yaml',
              help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def data_collection(ctx, config: str, verbose: bool):
    """Data Collection CLI for Algo-Trading system."""
    # Setup logging
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(log_level=log_level)

    # Load configuration
    base_config = load_base_config()
    data_config = load_data_config()
    ctx.obj = {**base_config, **data_config}

    # Ensure data directories exist
    Path(ctx.obj['data_root']).mkdir(parents=True, exist_ok=True)
    Path(ctx.obj['data_root'] / 'raw').mkdir(parents=True, exist_ok=True)
    Path(ctx.obj['data_root'] / 'processed').mkdir(parents=True, exist_ok=True)
    Path(ctx.obj['data_root'] / 'features').mkdir(parents=True, exist_ok=True)


@data_collection.command()
@click.option('--symbols', '-s', multiple=True,
              default=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT'],
              help='Trading symbols to collect')
@click.option('--days', '-d', default=30, help='Number of days of historical data')
@click.option('--output-dir', '-o', default='data/raw', help='Output directory')
@click.pass_context
def collect_historical(ctx, symbols: list[str], days: int, output_dir: str):
    """Collect historical data from Binance CSV dumps."""
    logger = get_logger(__name__)
    config = ctx.obj

    logger.info(f"Starting historical data collection for {len(symbols)} symbols")
    logger.info(f"Collecting {days} days of data")

    try:
        # Initialize CSV fetcher
        fetcher = BinanceCSVFetcher(config)

        # Collect data for each symbol
        for symbol in symbols:
            logger.info(f"Collecting data for {symbol}")

            success = fetcher.download_historical_data(
                symbol=symbol,
                days=days,
                output_dir=output_dir
            )

            if success:
                logger.info(f"✅ Successfully collected data for {symbol}")
            else:
                logger.error(f"❌ Failed to collect data for {symbol}")

        logger.info("Historical data collection completed")

    except Exception as e:
        logger.error(f"Error during historical data collection: {e}")
        sys.exit(1)


@data_collection.command()
@click.option('--symbols', '-s', multiple=True,
              default=['BTCUSDT', 'ETHUSDT'],
              help='Trading symbols to collect')
@click.option('--hours', '-h', default=24, help='Number of hours of recent data')
@click.option('--output-dir', '-o', default='data/raw', help='Output directory')
@click.pass_context
def collect_recent(ctx, symbols: list[str], hours: int, output_dir: str):
    """Collect recent data from Binance REST API."""
    logger = get_logger(__name__)
    config = ctx.obj

    logger.info(f"Starting recent data collection for {len(symbols)} symbols")
    logger.info(f"Collecting {hours} hours of data")

    async def collect_data():
        try:
            # Initialize REST fetcher
            async with BinanceRESTFetcher(config) as fetcher:
                # Collect data for each symbol
                for symbol in symbols:
                    logger.info(f"Collecting recent data for {symbol}")

                    success = await fetcher.fetch_recent_data(
                        symbol=symbol,
                        hours=hours,
                        output_dir=output_dir
                    )

                    if success:
                        logger.info(f"✅ Successfully collected recent data for {symbol}")
                    else:
                        logger.error(f"❌ Failed to collect recent data for {symbol}")

                logger.info("Recent data collection completed")

        except Exception as e:
            logger.error(f"Error during recent data collection: {e}")
            sys.exit(1)

    # Run async collection
    asyncio.run(collect_data())


@data_collection.command()
@click.option('--input-dir', '-i', default='data/raw', help='Input directory with raw data')
@click.option('--output-dir', '-o', default='data/processed', help='Output directory for processed data')
@click.option('--symbols', '-s', multiple=True, help='Specific symbols to process (all if not specified)')
@click.pass_context
def consolidate_data(ctx, input_dir: str, output_dir: str, symbols: list[str] | None):
    """Consolidate raw data files into partitioned datasets."""
    logger = get_logger(__name__)

    logger.info("Starting data consolidation")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    try:
        # Initialize consolidator
        consolidator = DataConsolidator(input_dir, output_dir)

        # Consolidate 1-minute bars
        success = consolidator.consolidate_bars_1m(symbols=list(symbols) if symbols else None)

        if success:
            logger.info("✅ Data consolidation completed successfully")

            # Validate partitions
            if consolidator.validate_partitions():
                logger.info("✅ Partition validation passed")
            else:
                logger.warning("⚠️ Partition validation failed")
        else:
            logger.error("❌ Data consolidation failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error during data consolidation: {e}")
        sys.exit(1)


@data_collection.command()
@click.option('--data-dir', '-d', default='data/raw', help='Data directory to validate')
@click.option('--symbols', '-s', multiple=True, help='Specific symbols to validate (all if not specified)')
@click.pass_context
def validate_data(ctx, data_dir: str, symbols: list[str] | None):
    """Validate data quality and integrity."""
    logger = get_logger(__name__)

    logger.info("Starting data validation")
    logger.info(f"Data directory: {data_dir}")

    try:
        from src.utils.validation import DataValidator

        validator = DataValidator()

        # Find data files
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error(f"Data directory does not exist: {data_dir}")
            sys.exit(1)

        # Validate each symbol file
        symbol_files = []
        if symbols:
            symbol_files = [data_path / f"{symbol.lower()}_bars_1m.parquet" for symbol in symbols]
        else:
            symbol_files = list(data_path.glob("*_bars_1m.parquet"))

        logger.info(f"Found {len(symbol_files)} data files to validate")

        validation_results = []
        for file_path in symbol_files:
            if file_path.exists():
                logger.info(f"Validating {file_path.name}")

                try:
                    import pandas as pd
                    df = pd.read_parquet(file_path)

                    # Validate structure
                    structure_result = validator.validate_dataframe(
                        df, ['ts', 'open', 'high', 'low', 'close', 'volume']
                    )

                    # Validate timestamps
                    timestamp_result = validator.validate_timestamps(df, 'ts')

                    # Validate OHLCV
                    ohlcv_result = validator.validate_ohlcv_data(
                        df, ['open', 'high', 'low', 'close']
                    )

                    validation_results.append({
                        'file': file_path.name,
                        'structure': structure_result['valid'],
                        'timestamps': timestamp_result['valid'],
                        'ohlcv': ohlcv_result['valid'],
                        'rows': len(df)
                    })

                    if all([structure_result['valid'], timestamp_result['valid'], ohlcv_result['valid']]):
                        logger.info(f"✅ {file_path.name} - Valid")
                    else:
                        logger.warning(f"⚠️ {file_path.name} - Validation issues found")

                except Exception as e:
                    logger.error(f"❌ Error validating {file_path.name}: {e}")
                    validation_results.append({
                        'file': file_path.name,
                        'error': str(e)
                    })
            else:
                logger.warning(f"File not found: {file_path}")

        # Summary
        valid_files = sum(1 for r in validation_results if r.get('structure') and r.get('timestamps') and r.get('ohlcv'))
        total_files = len(validation_results)

        logger.info(f"Validation completed: {valid_files}/{total_files} files valid")

    except Exception as e:
        logger.error(f"Error during data validation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    data_collection()

