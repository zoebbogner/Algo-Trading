#!/usr/bin/env python3
"""
Top 10 Crypto Data Collection Script

Specialized script for collecting data from the top 10 Binance USDT pairs:
- BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT
- ADAUSDT, DOGEUSDT, AVAXUSDT, TRXUSDT, DOTUSDT

This script optimizes the collection process for multiple symbols.

Usage:
    python3 scripts/collect_top10_crypto.py --days 7 --fill-gaps
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.base import load_config as load_base_config
from configs.data_history import load_config as load_data_config
from scripts.fetch_binance_rest import BinanceRESTFetcher
from scripts.normalize_data import DataNormalizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/top10_crypto_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Top10CryptoCollector:
    """Specialized collector for top 10 crypto pairs."""
    
    def __init__(self, config: dict):
        """Initialize the collector with configuration."""
        self.config = config
        self.symbols = config['symbols']
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ensure logs directory exists
        Path('logs').mkdir(exist_ok=True)
        
        # Ensure reports directory exists
        Path('reports/runs').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Top 10 Crypto Collector with run ID: {self.run_id}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
    
    async def collect_recent_data(self, days_back: int = 7, fill_gaps: bool = True) -> Dict[str, Any]:
        """Collect recent data for all symbols via REST API."""
        logger.info("=" * 60)
        logger.info("COLLECTING RECENT DATA FOR TOP 10 CRYPTO PAIRS")
        logger.info("=" * 60)
        
        try:
            async with BinanceRESTFetcher(self.config) as fetcher:
                results = await fetcher.fetch_all_symbols(days_back, fill_gaps)
                
                # Save results
                results_file = Path('reports/runs') / f'top10_rest_fetch_{self.run_id}.yaml'
                with open(results_file, 'w') as f:
                    yaml.dump(results, f, default_flow_style=False, indent=2)
                
                logger.info(f"REST fetch results saved to: {results_file}")
                return results
                
        except Exception as e:
            logger.error(f"REST fetch failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'phase': 'rest_fetch'
            }
    
    def normalize_all_data(self) -> Dict[str, Any]:
        """Normalize data for all symbols."""
        logger.info("=" * 60)
        logger.info("NORMALIZING DATA FOR ALL SYMBOLS")
        logger.info("=" * 60)
        
        try:
            normalizer = DataNormalizer(self.config)
            results = normalizer.normalize_all_symbols()
            
            # Save results
            results_file = Path('reports/runs') / f'top10_normalization_{self.run_id}.yaml'
            with open(results_file, 'w') as f:
                yaml.dump(results, f, default_flow_style=False, indent=2)
            
            logger.info(f"Normalization results saved to: {results_file}")
            return results
            
        except Exception as e:
            logger.error(f"Data normalization failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'phase': 'data_normalization'
            }
    
    def generate_collection_report(
        self, 
        rest_results: Dict[str, Any], 
        normalization_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive collection report."""
        collection_report = {
            'run_id': self.run_id,
            'collection_time': datetime.now().isoformat(),
            'symbols': self.symbols,
            'configuration': {
                'timeframe': self.config['timeframe'],
                'data_sources': self.config.get('sources', {}),
                'quality_thresholds': self.config.get('quality', {})
            },
            'phase_results': {
                'rest_fetch': rest_results,
                'data_normalization': normalization_results
            },
            'summary': {
                'overall_status': 'success',
                'phases_completed': 0,
                'total_symbols': len(self.symbols),
                'total_data_rows': 0,
                'total_errors': 0
            }
        }
        
        # Analyze results and update summary
        phases = [rest_results, normalization_results]
        for phase_result in phases:
            if phase_result.get('status') != 'error':
                collection_report['summary']['phases_completed'] += 1
                
                # Count data rows if available
                if 'summary' in phase_result:
                    if 'total_rows' in phase_result['summary']:
                        collection_report['summary']['total_data_rows'] += phase_result['summary']['total_rows']
                    if 'total_klines' in phase_result['summary']:
                        collection_report['summary']['total_data_rows'] += phase_result['summary']['total_klines']
            else:
                collection_report['summary']['total_errors'] += 1
                collection_report['summary']['overall_status'] = 'partial_failure'
        
        # Determine overall status
        if collection_report['summary']['total_errors'] == 0:
            collection_report['summary']['overall_status'] = 'success'
        elif collection_report['summary']['phases_completed'] == 0:
            collection_report['summary']['overall_status'] = 'complete_failure'
        
        return collection_report
    
    async def run_collection(
        self, 
        days_back: int = 7, 
        fill_gaps: bool = True
    ) -> Dict[str, Any]:
        """Run the complete top 10 crypto collection process."""
        logger.info("=" * 80)
        logger.info("STARTING TOP 10 CRYPTO DATA COLLECTION")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Days Back: {days_back}")
        logger.info(f"Fill Gaps: {fill_gaps}")
        logger.info("=" * 80)
        
        collection_start_time = datetime.now()
        
        # Phase 1: REST API Collection
        rest_results = await self.collect_recent_data(days_back, fill_gaps)
        
        # Phase 2: Data Normalization
        normalization_results = self.normalize_all_data()
        
        # Generate comprehensive report
        collection_report = self.generate_collection_report(rest_results, normalization_results)
        
        # Add timing information
        collection_end_time = datetime.now()
        collection_duration = (collection_end_time - collection_start_time).total_seconds()
        collection_report['collection_duration_seconds'] = collection_duration
        collection_report['collection_end_time'] = collection_end_time.isoformat()
        
        # Save collection report
        report_file = Path('reports/runs') / f'top10_collection_report_{self.run_id}.yaml'
        with open(report_file, 'w') as f:
            yaml.dump(collection_report, f, default_flow_style=False, indent=2)
        
        # Log final summary
        logger.info("=" * 80)
        logger.info("TOP 10 CRYPTO COLLECTION COMPLETED")
        logger.info(f"Overall Status: {collection_report['summary']['overall_status']}")
        logger.info(f"Phases Completed: {collection_report['summary']['phases_completed']}/2")
        logger.info(f"Total Symbols: {collection_report['summary']['total_symbols']}")
        logger.info(f"Total Data Rows: {collection_report['summary']['total_data_rows']}")
        logger.info(f"Total Errors: {collection_report['summary']['total_errors']}")
        logger.info(f"Duration: {collection_duration:.1f} seconds")
        logger.info(f"Report saved to: {report_file}")
        logger.info("=" * 80)
        
        return collection_report

async def main():
    """Main function to run the top 10 crypto collection."""
    parser = argparse.ArgumentParser(description='Collect data for top 10 Binance crypto pairs')
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
    
    # Create collector and run
    collector = Top10CryptoCollector(config)
    results = await collector.run_collection(args.days, args.fill_gaps)
    
    # Return exit code based on success
    if results['summary']['overall_status'] == 'success':
        logger.info("Top 10 crypto collection completed successfully!")
        return 0
    elif results['summary']['overall_status'] == 'partial_failure':
        logger.warning("Top 10 crypto collection completed with some failures")
        return 1
    else:
        logger.error("Top 10 crypto collection failed completely")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
