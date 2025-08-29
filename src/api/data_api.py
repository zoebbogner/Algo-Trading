#!/usr/bin/env python3
"""
Data API endpoints for the Algo-Trading system.

Provides REST API for:
- Data collection operations
- Data validation and quality control
- Data consolidation and management
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

from flask import Blueprint, jsonify, request

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.base import load_config as load_base_config
from src.config.data_history import load_config as load_data_config
from src.data.fetch.binance_csv import BinanceCSVFetcher
from src.data.fetch.binance_rest import BinanceRESTFetcher
from src.data.processing.consolidator import DataConsolidator
from src.utils.logging import get_logger

# Create blueprint
data_router = Blueprint('data', __name__, url_prefix='/api/data')
logger = get_logger(__name__)

# Load configuration
base_config = load_base_config()
data_config = load_data_config()
config = {**base_config, **data_config}


@data_router.route('/health', methods=['GET'])
def health_check():
    """Check data API health."""
    return jsonify({
        'status': 'healthy',
        'service': 'data-api',
        'config_loaded': bool(config)
    })


@data_router.route('/collect/historical', methods=['POST'])
def collect_historical_data():
    """Collect historical data from Binance CSV dumps."""
    try:
        data = request.get_json()
        symbols = data.get('symbols', ['BTCUSDT', 'ETHUSDT'])
        days = data.get('days', 30)

        logger.info(f"Starting historical data collection for {len(symbols)} symbols")

        # Initialize CSV fetcher
        fetcher = BinanceCSVFetcher(config)

        # Run async collection
        async def collect_historical():
            async with fetcher:
                results = []
                for symbol in symbols:
                    try:
                        success = await fetcher.fetch_data(
                            symbols=[symbol],
                            start_date=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                            end_date=datetime.now().strftime('%Y-%m-%d')
                        )

                        results.append({
                            'symbol': symbol,
                            'success': bool(success.get('downloaded_files', [])),
                            'status': 'completed' if success.get('downloaded_files') else 'failed'
                        })

                    except Exception as e:
                        logger.error(f"Error collecting data for {symbol}: {e}")
                        results.append({
                            'symbol': symbol,
                            'success': False,
                            'status': 'error',
                            'error': str(e)
                        })
                return results

        results = asyncio.run(collect_historical())

        # Summary
        successful = sum(1 for r in results if r['success'])
        total = len(results)

        return jsonify({
            'success': True,
            'message': f'Historical data collection completed: {successful}/{total} successful',
            'results': results,
            'summary': {
                'total_symbols': total,
                'successful': successful,
                'failed': total - successful
            }
        })

    except Exception as e:
        logger.error(f"Error in historical data collection: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_router.route('/collect/recent', methods=['POST'])
def collect_recent_data():
    """Collect recent data from Binance REST API."""
    try:
        data = request.get_json()
        symbols = data.get('symbols', ['BTCUSDT', 'ETHUSDT'])
        hours = data.get('hours', 24)
        output_dir = data.get('output_dir', 'data/raw')

        logger.info(f"Starting recent data collection for {len(symbols)} symbols")

        async def collect_data():
            try:
                async with BinanceRESTFetcher(config) as fetcher:
                    results = []

                    for symbol in symbols:
                        try:
                            success = await fetcher.fetch_recent_data(
                                symbol=symbol,
                                hours=hours,
                                output_dir=output_dir
                            )

                            results.append({
                                'symbol': symbol,
                                'success': success,
                                'status': 'completed' if success else 'failed'
                            })

                        except Exception as e:
                            logger.error(f"Error collecting recent data for {symbol}: {e}")
                            results.append({
                                'symbol': symbol,
                                'success': False,
                                'status': 'error',
                                'error': str(e)
                            })

                    return results

            except Exception as e:
                logger.error(f"Error in recent data collection: {e}")
                raise

        # Run async collection
        results = asyncio.run(collect_data())

        # Summary
        successful = sum(1 for r in results if r['success'])
        total = len(results)

        return jsonify({
            'success': True,
            'message': f'Recent data collection completed: {successful}/{total} successful',
            'results': results,
            'summary': {
                'total_symbols': total,
                'successful': successful,
                'failed': total - successful
            }
        })

    except Exception as e:
        logger.error(f"Error in recent data collection: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_router.route('/consolidate', methods=['POST'])
def consolidate_data():
    """Consolidate raw data files into partitioned datasets."""
    try:
        data = request.get_json()
        input_dir = data.get('input_dir', 'data/raw')
        output_dir = data.get('output_dir', 'data/processed')
        symbols = data.get('symbols', [])

        logger.info("Starting data consolidation")

        # Initialize consolidator
        consolidator = DataConsolidator(input_dir, output_dir)

        # Consolidate 1-minute bars
        success = consolidator.consolidate_bars_1m(symbols=list(symbols) if symbols else None)

        if success:
            # Validate partitions
            validation_success = consolidator.validate_partitions()

            return jsonify({
                'success': True,
                'message': 'Data consolidation completed successfully',
                'validation': {
                    'passed': validation_success,
                    'status': 'valid' if validation_success else 'failed'
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Data consolidation failed'
            }), 500

    except Exception as e:
        logger.error(f"Error in data consolidation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_router.route('/validate', methods=['POST'])
def validate_data():
    """Validate data quality and integrity."""
    try:
        data = request.get_json()
        data_dir = data.get('data_dir', 'data/raw')
        symbols = data.get('symbols', [])

        logger.info("Starting data validation")

        from src.utils.validation import DataValidator

        validator = DataValidator()

        # Find data files
        data_path = Path(data_dir)
        if not data_path.exists():
            return jsonify({
                'success': False,
                'error': f'Data directory does not exist: {data_dir}'
            }), 400

        # Find files to validate
        if symbols:
            symbol_files = [data_path / f"{symbol.lower()}_bars_1m.parquet" for symbol in symbols]
        else:
            symbol_files = list(data_path.glob("*_bars_1m.parquet"))

        logger.info(f"Found {len(symbol_files)} data files to validate")

        validation_results = []
        for file_path in symbol_files:
            if file_path.exists():
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
                        'rows': len(df),
                        'status': 'valid' if all([structure_result['valid'], timestamp_result['valid'], ohlcv_result['valid']]) else 'invalid'
                    })

                except Exception as e:
                    validation_results.append({
                        'file': file_path.name,
                        'error': str(e),
                        'status': 'error'
                    })
            else:
                validation_results.append({
                    'file': file_path.name,
                    'error': 'File not found',
                    'status': 'missing'
                })

        # Summary
        valid_files = sum(1 for r in validation_results if r.get('status') == 'valid')
        total_files = len(validation_results)

        return jsonify({
            'success': True,
            'message': f'Data validation completed: {valid_files}/{total_files} files valid',
            'results': validation_results,
            'summary': {
                'total_files': total_files,
                'valid_files': valid_files,
                'invalid_files': total_files - valid_files
            }
        })

    except Exception as e:
        logger.error(f"Error in data validation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_router.route('/status', methods=['GET'])
def get_data_status():
    """Get current data status and statistics."""
    try:
        data_root = Path(config['data_root'])

        status = {
            'data_root': str(data_root),
            'directories': {},
            'file_counts': {},
            'total_size_mb': 0
        }

        # Check each data directory
        for subdir in ['raw', 'processed', 'features']:
            dir_path = data_root / subdir
            if dir_path.exists():
                # Count files
                files = list(dir_path.glob("*.parquet"))
                file_count = len(files)

                # Calculate total size
                total_size = sum(f.stat().st_size for f in files)
                size_mb = total_size / (1024 * 1024)

                status['directories'][subdir] = {
                    'exists': True,
                    'file_count': file_count,
                    'size_mb': round(size_mb, 2)
                }

                status['file_counts'][subdir] = file_count
                status['total_size_mb'] += size_mb
            else:
                status['directories'][subdir] = {
                    'exists': False,
                    'file_count': 0,
                    'size_mb': 0
                }

        status['total_size_mb'] = round(status['total_size_mb'], 2)

        return jsonify({
            'success': True,
            'data': status
        })

    except Exception as e:
        logger.error(f"Error getting data status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500




