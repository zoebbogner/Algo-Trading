#!/usr/bin/env python3
"""
Bulk Historical Data Collection Script

Downloads and processes large amounts of historical data for all top 10 crypto pairs.
This script handles bulk CSV downloads and then normalizes all the data.

Usage:
    python3 scripts/bulk_historical_collection.py --start-date 2024-01-01 --end-date 2025-07-31
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
from scripts.download_binance_csv import BinanceCSVDownloader
from scripts.fetch_binance_rest import BinanceRESTFetcher
from scripts.normalize_data import DataNormalizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bulk_historical_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BulkHistoricalCollector:
    """Comprehensive collector for large-scale historical data."""
    
    def __init__(self, config: dict):
        """Initialize the bulk collector with configuration."""
        self.config = config
        self.symbols = config['symbols']
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ensure logs directory exists
        Path('logs').mkdir(exist_ok=True)
        
        # Ensure reports directory exists
        Path('reports/runs').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Bulk Historical Collector with run ID: {self.run_id}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
    
    async def download_bulk_csv_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Phase 1: Download bulk CSV data from Binance."""
        logger.info("=" * 60)
        logger.info("PHASE 1: BULK CSV DATA DOWNLOAD")
        logger.info("=" * 60)
        
        try:
            async with BinanceCSVDownloader(self.config) as downloader:
                results = await downloader.download_all_symbols(start_date, end_date)
                
                # Save results
                results_file = Path('reports/runs') / f'bulk_csv_download_{self.run_id}.yaml'
                with open(results_file, 'w') as f:
                    yaml.dump(results, f, default_flow_style=False, indent=2)
                
                logger.info(f"Bulk CSV download results saved to: {results_file}")
                return results
                
        except Exception as e:
            logger.error(f"Bulk CSV download failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'phase': 'bulk_csv_download'
            }
    
    async def fetch_recent_data(self, days_back: int = 7, fill_gaps: bool = True) -> Dict[str, Any]:
        """Phase 2: Fetch recent data via REST API."""
        logger.info("=" * 60)
        logger.info("PHASE 2: REST API RECENT DATA FETCH")
        logger.info("=" * 60)
        
        try:
            async with BinanceRESTFetcher(self.config) as fetcher:
                results = await fetcher.fetch_all_symbols(days_back, fill_gaps)
                
                # Save results
                results_file = Path('reports/runs') / f'bulk_rest_fetch_{self.run_id}.yaml'
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
        """Phase 3: Normalize and validate all data."""
        logger.info("=" * 60)
        logger.info("PHASE 3: COMPREHENSIVE DATA NORMALIZATION")
        logger.info("=" * 60)
        
        try:
            normalizer = DataNormalizer(self.config)
            results = normalizer.normalize_all_symbols()
            
            # Save results
            results_file = Path('reports/runs') / f'bulk_normalization_{self.run_id}.yaml'
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
    
    def generate_comprehensive_report(
        self, 
        csv_results: Dict[str, Any], 
        rest_results: Dict[str, Any], 
        normalization_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive bulk collection report."""
        comprehensive_report = {
            'run_id': self.run_id,
            'collection_start_time': datetime.now().isoformat(),
            'symbols': self.symbols,
            'configuration': {
                'timeframe': self.config['timeframe'],
                'data_sources': self.config.get('sources', {}),
                'quality_thresholds': self.config.get('quality', {})
            },
            'phase_results': {
                'bulk_csv_download': csv_results,
                'rest_fetch': rest_results,
                'data_normalization': normalization_results
            },
            'summary': {
                'overall_status': 'success',
                'phases_completed': 0,
                'total_symbols': len(self.symbols),
                'total_csv_files': 0,
                'total_rest_klines': 0,
                'total_processed_rows': 0,
                'total_errors': 0
            }
        }
        
        # Analyze results and update summary
        phases = [csv_results, rest_results, normalization_results]
        for phase_result in phases:
            if phase_result.get('status') != 'error':
                comprehensive_report['summary']['phases_completed'] += 1
                
                # Count various metrics
                if 'summary' in phase_result:
                    if 'total_downloaded' in phase_result['summary']:
                        comprehensive_report['summary']['total_csv_files'] += phase_result['summary']['total_downloaded']
                    if 'total_klines' in phase_result['summary']:
                        comprehensive_report['summary']['total_rest_klines'] += phase_result['summary']['total_klines']
                    if 'total_rows' in phase_result['summary']:
                        comprehensive_report['summary']['total_processed_rows'] += phase_result['summary']['total_rows']
            else:
                comprehensive_report['summary']['total_errors'] += 1
                comprehensive_report['summary']['overall_status'] = 'partial_failure'
        
        # Determine overall status
        if comprehensive_report['summary']['total_errors'] == 0:
            comprehensive_report['summary']['overall_status'] = 'success'
        elif comprehensive_report['summary']['phases_completed'] == 0:
            comprehensive_report['summary']['overall_status'] = 'complete_failure'
        
        return comprehensive_report
    
    async def run_bulk_collection(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        rest_days: int = 7,
        fill_gaps: bool = True
    ) -> Dict[str, Any]:
        """Run the complete bulk historical collection process."""
        logger.info("=" * 80)
        logger.info("STARTING BULK HISTORICAL DATA COLLECTION")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Date Range: {start_date.date()} to {end_date.date()}")
        logger.info(f"Rest Days: {rest_days}")
        logger.info(f"Fill Gaps: {fill_gaps}")
        logger.info("=" * 80)
        
        collection_start_time = datetime.now()
        
        # Phase 1: Bulk CSV Download
        csv_results = await self.download_bulk_csv_data(start_date, end_date)
        
        # Phase 2: REST API Recent Data
        rest_results = await self.fetch_recent_data(rest_days, fill_gaps)
        
        # Phase 3: Comprehensive Data Normalization
        normalization_results = self.normalize_all_data()
        
        # Generate comprehensive report
        comprehensive_report = self.generate_comprehensive_report(csv_results, rest_results, normalization_results)
        
        # Add timing information
        collection_end_time = datetime.now()
        collection_duration = (collection_end_time - collection_start_time).total_seconds()
        comprehensive_report['collection_duration_seconds'] = collection_duration
        comprehensive_report['collection_end_time'] = collection_end_time.isoformat()
        
        # Save comprehensive report
        report_file = Path('reports/runs') / f'bulk_historical_report_{self.run_id}.yaml'
        with open(report_file, 'w') as f:
            yaml.dump(comprehensive_report, f, default_flow_style=False, indent=2)
        
        # Log final summary
        logger.info("=" * 80)
        logger.info("BULK HISTORICAL COLLECTION COMPLETED")
        logger.info(f"Overall Status: {comprehensive_report['summary']['overall_status']}")
        logger.info(f"Phases Completed: {comprehensive_report['summary']['phases_completed']}/3")
        logger.info(f"Total Symbols: {comprehensive_report['summary']['total_symbols']}")
        logger.info(f"Total CSV Files: {comprehensive_report['summary']['total_csv_files']}")
        logger.info(f"Total REST Klines: {comprehensive_report['summary']['total_rest_klines']}")
        logger.info(f"Total Processed Rows: {comprehensive_report['summary']['total_processed_rows']}")
        logger.info(f"Total Errors: {comprehensive_report['summary']['total_errors']}")
        logger.info(f"Duration: {collection_duration:.1f} seconds")
        logger.info(f"Report saved to: {report_file}")
        logger.info("=" * 80)
        
        return comprehensive_report

async def main():
    """Main function to run the bulk historical collection."""
    parser = argparse.ArgumentParser(description='Collect bulk historical data for top 10 crypto pairs')
    parser.add_argument('--start-date', required=True, help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date for historical data (YYYY-MM-DD)')
    parser.add_argument('--rest-days', type=int, default=7, help='Days of recent data to fetch via REST')
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
    
    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError as e:
        logger.error(f"Invalid date format: {e}. Use YYYY-MM-DD format.")
        return 1
    
    # Validate dates
    if start_date >= end_date:
        logger.error("Start date must be before end date")
        return 1
    
    # Create collector and run
    collector = BulkHistoricalCollector(config)
    results = await collector.run_bulk_collection(
        start_date=start_date,
        end_date=end_date,
        rest_days=args.rest_days,
        fill_gaps=args.fill_gaps
    )
    
    # Return exit code based on success
    if results['summary']['overall_status'] == 'success':
        logger.info("Bulk historical collection completed successfully!")
        return 0
    elif results['summary']['overall_status'] == 'partial_failure':
        logger.warning("Bulk historical collection completed with some failures")
        return 1
    else:
        logger.error("Bulk historical collection failed completely")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
