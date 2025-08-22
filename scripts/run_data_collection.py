#!/usr/bin/env python3
"""
Main Data Collection Pipeline

Orchestrates the complete data collection process:
1. Download historical CSV data from Binance
2. Fetch recent data via REST API
3. Normalize and validate data
4. Generate quality reports

Usage:
    python scripts/run_data_collection.py --symbols BTCUSDT ETHUSDT --start-date 2024-07-01 --end-date 2025-08-20
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.base import load_config as load_base_config
from configs.data_history import load_config as load_data_config

# Import the data collection modules
from scripts.download_binance_csv import BinanceCSVDownloader
from scripts.fetch_binance_rest import BinanceRESTFetcher
from scripts.normalize_data import DataNormalizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_collection_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataCollectionPipeline:
    """Main pipeline for collecting and processing crypto historical data."""
    
    def __init__(self, config: dict):
        """Initialize the pipeline with configuration."""
        self.config = config
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ensure logs directory exists
        Path('logs').mkdir(exist_ok=True)
        
        # Ensure reports directory exists
        Path('reports/runs').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized data collection pipeline with run ID: {self.run_id}")
    
    async def run_csv_download(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Phase 1: Download historical CSV data from Binance."""
        logger.info("=" * 60)
        logger.info("PHASE 1: CSV Data Download")
        logger.info("=" * 60)
        
        try:
            async with BinanceCSVDownloader(self.config) as downloader:
                results = await downloader.download_all_symbols(start_date, end_date)
                
                # Save results
                results_file = Path('reports/runs') / f'csv_download_{self.run_id}.yaml'
                with open(results_file, 'w') as f:
                    yaml.dump(results, f, default_flow_style=False, indent=2)
                
                logger.info(f"CSV download results saved to: {results_file}")
                return results
                
        except Exception as e:
            logger.error(f"CSV download failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'phase': 'csv_download'
            }
    
    async def run_rest_fetch(self, days_back: int = 7, fill_gaps: bool = True) -> Dict[str, Any]:
        """Phase 2: Fetch recent data via REST API."""
        logger.info("=" * 60)
        logger.info("PHASE 2: REST API Data Fetch")
        logger.info("=" * 60)
        
        try:
            async with BinanceRESTFetcher(self.config) as fetcher:
                results = await fetcher.fetch_all_symbols(days_back, fill_gaps)
                
                # Save results
                results_file = Path('reports/runs') / f'rest_fetch_{self.run_id}.yaml'
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
    
    def run_data_normalization(self) -> Dict[str, Any]:
        """Phase 3: Normalize and validate data."""
        logger.info("=" * 60)
        logger.info("PHASE 3: Data Normalization")
        logger.info("=" * 60)
        
        try:
            normalizer = DataNormalizer(self.config)
            results = normalizer.normalize_all_symbols()
            
            # Save results
            results_file = Path('reports/runs') / f'data_normalization_{self.run_id}.yaml'
            with open(results_file, 'w') as f:
                yaml.dump(results, f, default_flow_style=False, indent=2)
            
            logger.info(f"Data normalization results saved to: {results_file}")
            return results
            
        except Exception as e:
            logger.error(f"Data normalization failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'phase': 'data_normalization'
            }
    
    def generate_pipeline_report(
        self, 
        csv_results: Dict[str, Any], 
        rest_results: Dict[str, Any], 
        normalization_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive pipeline report."""
        pipeline_report = {
            'run_id': self.run_id,
            'pipeline_start_time': datetime.now().isoformat(),
            'configuration': {
                'symbols': self.config['symbols'],
                'timeframe': self.config['timeframe'],
                'data_sources': self.config.get('sources', {}),
                'quality_thresholds': self.config.get('quality', {})
            },
            'phase_results': {
                'csv_download': csv_results,
                'rest_fetch': rest_results,
                'data_normalization': normalization_results
            },
            'summary': {
                'overall_status': 'success',
                'phases_completed': 0,
                'total_errors': 0,
                'total_data_rows': 0
            }
        }
        
        # Analyze results and update summary
        phases = [csv_results, rest_results, normalization_results]
        for phase_result in phases:
            if phase_result.get('status') != 'error':
                pipeline_report['summary']['phases_completed'] += 1
                
                # Count data rows if available
                if 'summary' in phase_result:
                    if 'total_rows' in phase_result['summary']:
                        pipeline_report['summary']['total_data_rows'] += phase_result['summary']['total_rows']
                    if 'total_klines' in phase_result['summary']:
                        pipeline_report['summary']['total_data_rows'] += phase_result['summary']['total_klines']
            else:
                pipeline_report['summary']['total_errors'] += 1
                pipeline_report['summary']['overall_status'] = 'partial_failure'
        
        # Determine overall status
        if pipeline_report['summary']['total_errors'] == 0:
            pipeline_report['summary']['overall_status'] = 'success'
        elif pipeline_report['summary']['phases_completed'] == 0:
            pipeline_report['summary']['overall_status'] = 'complete_failure'
        
        return pipeline_report
    
    async def run_pipeline(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        rest_days: int = 7,
        fill_gaps: bool = True
    ) -> Dict[str, Any]:
        """Run the complete data collection pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE DATA COLLECTION PIPELINE")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Symbols: {', '.join(self.config['symbols'])}")
        logger.info(f"Date Range: {start_date.date()} to {end_date.date()}")
        logger.info("=" * 80)
        
        pipeline_start_time = datetime.now()
        
        # Phase 1: CSV Download
        csv_results = await self.run_csv_download(start_date, end_date)
        
        # Phase 2: REST Fetch
        rest_results = await self.run_rest_fetch(rest_days, fill_gaps)
        
        # Phase 3: Data Normalization
        normalization_results = self.run_data_normalization()
        
        # Generate comprehensive report
        pipeline_report = self.generate_pipeline_report(csv_results, rest_results, normalization_results)
        
        # Add timing information
        pipeline_end_time = datetime.now()
        pipeline_duration = (pipeline_end_time - pipeline_start_time).total_seconds()
        pipeline_report['pipeline_duration_seconds'] = pipeline_duration
        pipeline_report['pipeline_end_time'] = pipeline_end_time.isoformat()
        
        # Save pipeline report
        report_file = Path('reports/runs') / f'pipeline_report_{self.run_id}.yaml'
        with open(report_file, 'w') as f:
            yaml.dump(pipeline_report, f, default_flow_style=False, indent=2)
        
        # Log final summary
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED")
        logger.info(f"Overall Status: {pipeline_report['summary']['overall_status']}")
        logger.info(f"Phases Completed: {pipeline_report['summary']['phases_completed']}/3")
        logger.info(f"Total Errors: {pipeline_report['summary']['total_errors']}")
        logger.info(f"Total Data Rows: {pipeline_report['summary']['total_data_rows']}")
        logger.info(f"Duration: {pipeline_duration:.1f} seconds")
        logger.info(f"Report saved to: {report_file}")
        logger.info("=" * 80)
        
        return pipeline_report

async def main():
    """Main function to run the data collection pipeline."""
    parser = argparse.ArgumentParser(description='Run complete crypto data collection pipeline')
    parser.add_argument('--symbols', nargs='+', help='Trading symbols to collect data for')
    parser.add_argument('--start-date', help='Start date for historical data (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for historical data (YYYY-MM-DD)')
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
    
    # Create pipeline and run
    pipeline = DataCollectionPipeline(config)
    results = await pipeline.run_pipeline(
        start_date=start_date,
        end_date=end_date,
        rest_days=args.rest_days,
        fill_gaps=args.fill_gaps
    )
    
    # Return exit code based on success
    if results['summary']['overall_status'] == 'success':
        logger.info("Pipeline completed successfully!")
        return 0
    elif results['summary']['overall_status'] == 'partial_failure':
        logger.warning("Pipeline completed with some failures")
        return 1
    else:
        logger.error("Pipeline failed completely")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
