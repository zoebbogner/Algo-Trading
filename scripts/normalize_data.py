#!/usr/bin/env python3
"""
Data Normalization Script

Converts raw CSV data from Binance data dump and REST API into a canonical schema.
This script handles data cleaning, validation, and conversion to processed format.

Usage:
    python scripts/normalize_data.py --symbols BTCUSDT ETHUSDT
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
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
        logging.FileHandler('logs/data_normalization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataNormalizer:
    """Normalizes raw CSV data to canonical schema."""
    
    def __init__(self, config: dict):
        """Initialize the normalizer with configuration."""
        self.config = config
        self.raw_dir = Path(config['raw_dir'])
        self.processed_dir = Path(config['processed_dir'])
        self.symbols = config['symbols']
        self.timeframe = config['timeframe']
        
        # Ensure processed directory exists
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load quality control settings
        self.qc_config = config.get('quality', {})
    
    def _load_raw_csv_data(self, symbol: str) -> pd.DataFrame:
        """Load all raw CSV data for a symbol."""
        symbol_dir = self.raw_dir / symbol / self.timeframe
        
        if not symbol_dir.exists():
            logger.warning(f"No raw data directory found for {symbol}")
            return pd.DataFrame()
        
        # Find all CSV files
        csv_files = list(symbol_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found for {symbol}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(csv_files)} CSV files for {symbol}")
        
        # Load and combine all CSV files
        all_data = []
        for csv_file in csv_files:
            try:
                # Determine source from filename
                if 'rest' in csv_file.name:
                    source = 'binance:rest'
                else:
                    source = 'binance:dump'
                
                # Load CSV data
                df = pd.read_csv(csv_file)
                
                # Add source and load_id columns
                df['source'] = source
                df['load_id'] = csv_file.stem
                
                all_data.append(df)
                logger.info(f"Loaded {len(df)} rows from {csv_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined {len(combined_df)} total rows for {symbol}")
        
        return combined_df
    
    def _normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert timestamps to ISO8601 UTC format."""
        # Convert open_time from milliseconds to datetime
        df['ts'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        
        # Round to minute close (hh:mm:00Z)
        df['ts'] = df['ts'].dt.floor('min')
        
        # Convert to ISO8601 string
        df['ts'] = df['ts'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        return df
    
    def _clean_and_validate_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and validate the data according to quality control rules."""
        initial_rows = len(df)
        
        # Remove rows with missing values
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        # Validate price ranges
        if self.qc_config.get('validate_price_ranges', True):
            # All prices must be positive
            price_mask = (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)
            df = df[price_mask]
            
            # High must be >= low
            high_low_mask = df['high'] >= df['low']
            df = df[high_low_mask]
            
            # Open and close must be between high and low
            open_mask = (df['open'] >= df['low']) & (df['open'] <= df['high'])
            close_mask = (df['close'] >= df['low']) & (df['close'] <= df['high'])
            df = df[open_mask & close_mask]
        
        # Validate volume
        if self.qc_config.get('validate_volume_ranges', True):
            volume_mask = df['volume'] >= self.qc_config.get('min_volume_threshold', 0.0)
            df = df[volume_mask]
        
        # Remove future timestamps
        now = datetime.now(timezone.utc)  # Use timezone-aware datetime
        df['ts_dt'] = pd.to_datetime(df['ts'], utc=True)
        future_mask = df['ts_dt'] <= now
        df = df[future_mask]
        df = df.drop('ts_dt', axis=1)
        
        # Sort by timestamp
        if self.qc_config.get('require_monotonic_ts', True):
            df = df.sort_values('ts')
        
        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} invalid rows for {symbol}")
        
        return df
    
    def _create_canonical_schema(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Convert data to canonical schema."""
        # Select and rename columns
        canonical_df = df[[
            'ts', 'open', 'high', 'low', 'close', 'volume', 
            'source', 'load_id'
        ]].copy()
        
        # Add symbol column
        canonical_df['symbol'] = symbol
        
        # Add spread_est column (placeholder for future SIM mode)
        canonical_df['spread_est'] = None
        
        # Add ingestion timestamp
        canonical_df['ingestion_ts'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Remove duplicates after all columns are present
        if self.qc_config.get('max_duplicates', 0) == 0:
            canonical_df = canonical_df.drop_duplicates(subset=['ts', 'symbol'])
        
        # Ensure proper column order
        column_order = [
            'ts', 'symbol', 'open', 'high', 'low', 'close', 'volume',
            'spread_est', 'source', 'load_id', 'ingestion_ts'
        ]
        
        return canonical_df[column_order]
    
    def _generate_quality_report(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Generate quality control report for the data."""
        if df.empty:
            return {
                'symbol': symbol,
                'status': 'error',
                'message': 'No data available after normalization'
            }
        
        # Calculate basic statistics
        report = {
            'symbol': symbol,
            'status': 'success',
            'total_rows': len(df),
            'date_range': {
                'start': df['ts'].min(),
                'end': df['ts'].max()
            },
            'data_quality': {
                'missing_values': df.isnull().sum().to_dict(),
                'duplicates': df.duplicated(subset=['ts']).sum(),
                'price_validation': {
                    'negative_prices': (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum(),
                    'high_low_violations': (df['high'] < df['low']).sum(),
                    'open_close_range_violations': (
                        (df['open'] < df['low']) | (df['open'] > df['high']) |
                        (df['close'] < df['low']) | (df['close'] > df['high'])
                    ).sum()
                },
                'volume_validation': {
                    'negative_volumes': (df['volume'] < 0).sum()
                }
            },
            'source_distribution': df['source'].value_counts().to_dict(),
            'load_ids': df['load_id'].nunique()
        }
        
        # Check coverage
        if len(df) > 1:
            # Calculate expected minutes in date range
            start_dt = pd.to_datetime(df['ts'].min(), utc=True)
            end_dt = pd.to_datetime(df['ts'].max(), utc=True)
            expected_minutes = int((end_dt - start_dt).total_seconds() / 60) + 1
            actual_minutes = len(df)
            coverage_pct = (actual_minutes / expected_minutes) * 100
            
            report['coverage'] = {
                'expected_minutes': expected_minutes,
                'actual_minutes': actual_minutes,
                'coverage_pct': round(coverage_pct, 2),
                'meets_threshold': coverage_pct >= self.qc_config.get('min_coverage_pct', 99.9)
            }
        
        return report
    
    def normalize_symbol_data(self, symbol: str) -> Dict[str, Any]:
        """Normalize data for a single symbol."""
        logger.info(f"Starting normalization for {symbol}")
        
        # Load raw data
        raw_df = self._load_raw_csv_data(symbol)
        if raw_df.empty:
            return {
                'symbol': symbol,
                'status': 'error',
                'message': 'No raw data available'
            }
        
        # Normalize timestamps
        raw_df = self._normalize_timestamps(raw_df)
        
        # Clean and validate data
        cleaned_df = self._clean_and_validate_data(raw_df, symbol)
        
        # Create canonical schema
        canonical_df = self._create_canonical_schema(cleaned_df, symbol)
        
        # Generate quality report
        quality_report = self._generate_quality_report(canonical_df, symbol)
        
        # Save processed data
        if not canonical_df.empty:
            output_file = self.processed_dir / f"{symbol.lower()}_bars_{self.timeframe}.parquet"
            canonical_df.to_parquet(output_file, index=False)
            logger.info(f"Saved processed data to: {output_file}")
            
            quality_report['output_file'] = str(output_file)
            quality_report['rows_saved'] = len(canonical_df)
        
        return quality_report
    
    def normalize_all_symbols(self) -> Dict[str, Any]:
        """Normalize data for all configured symbols."""
        logger.info(f"Starting normalization for {len(self.symbols)} symbols")
        
        all_results = {
            'normalization_time': datetime.utcnow().isoformat(),
            'symbols': {},
            'summary': {
                'total_symbols': len(self.symbols),
                'successful': 0,
                'failed': 0,
                'total_rows': 0
            }
        }
        
        # Normalize each symbol
        for symbol in self.symbols:
            symbol_result = self.normalize_symbol_data(symbol)
            all_results['symbols'][symbol] = symbol_result
            
            # Update summary
            if symbol_result['status'] == 'success':
                all_results['summary']['successful'] += 1
                all_results['summary']['total_rows'] += symbol_result.get('rows_saved', 0)
            else:
                all_results['summary']['failed'] += 1
        
        # Log summary
        logger.info(f"Normalization completed:")
        logger.info(f"  Successful: {all_results['summary']['successful']}")
        logger.info(f"  Failed: {all_results['summary']['failed']}")
        logger.info(f"  Total rows: {all_results['summary']['total_rows']}")
        
        return all_results

def main():
    """Main function to run the data normalizer."""
    parser = argparse.ArgumentParser(description='Normalize raw CSV data to canonical schema')
    parser.add_argument('--symbols', nargs='+', help='Trading symbols to normalize')
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
    
    # Create normalizer and run
    normalizer = DataNormalizer(config)
    results = normalizer.normalize_all_symbols()
    
    # Save results to reports directory
    reports_dir = Path('reports/runs')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = reports_dir / f'data_normalization_{run_id}.yaml'
    
    with open(results_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, indent=2)
    
    logger.info(f"Normalization results saved to: {results_file}")
    
    # Return exit code based on success
    if results['summary']['failed'] == 0:
        logger.info("All normalizations completed successfully!")
        return 0
    else:
        logger.warning(f"Some normalizations failed: {results['summary']['failed']} symbols")
        return 1

if __name__ == "__main__":
    sys.exit(main())
