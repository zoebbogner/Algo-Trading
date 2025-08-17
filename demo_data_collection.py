#!/usr/bin/env python3
"""Demo script to show the data collection system working."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data import DataCollector, DataStorage
from src.data.models import DataRequest


async def demo_data_collection():
    """Demonstrate the data collection system."""
    print("🚀 Data Collection Demo")
    print("=" * 50)
    
    # Configuration for data collection
    config = {
        "binance": {
            "enabled": True,
            "api_key": "",
            "secret_key": "",
            "rate_limit_delay": 1.0,
            "max_retries": 3
        },
        "coingecko": {
            "enabled": True,
            "api_key": "",
            "rate_limit_delay": 2.0,
            "max_retries": 3
        },
        "ccxt": {
            "enabled": False
        }
    }
    
    # Initialize components
    print("📊 Initializing data collection system...")
    data_storage = DataStorage("data")
    data_collector = DataCollector(config)
    
    # Clear previous data
    print("🧹 Clearing previous data...")
    data_storage.clear_previous_run()
    
    # Define symbols and date range
    symbols = ["BTC", "ETH", "ADA"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days for demo
    
    print(f"📈 Collecting data for {len(symbols)} symbols")
    print(f"📅 Period: {start_date.date()} to {end_date.date()}")
    print("=" * 50)
    
    try:
        # Collect data
        async with data_collector as collector:
            print("🔄 Collecting data from exchanges...")
            results = await collector.collect_multiple_symbols(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
        
        # Process results
        successful_collections = 0
        total_bars = 0
        
        for symbol, response in results.items():
            if response.success and response.data.bars:
                # Save data
                file_path = data_storage.save_market_data(symbol, response.data)
                bars_count = len(response.data.bars)
                total_bars += bars_count
                successful_collections += 1
                
                print(f"✅ {symbol}: {bars_count} bars saved to {file_path}")
            else:
                print(f"❌ {symbol}: Failed - {response.error_message}")
        
        print("=" * 50)
        print(f"📊 Collection Summary:")
        print(f"   Successful: {successful_collections}/{len(symbols)} symbols")
        print(f"   Total bars: {total_bars}")
        
        # Show data storage info
        print("\n💾 Data Storage Info:")
        for symbol in symbols:
            info = data_storage.get_data_info(symbol)
            if info["available"]:
                print(f"   {symbol}: {info['total_bars']} bars, {len(info['files'])} files")
            else:
                print(f"   {symbol}: No data available")
        
        # Export summary
        summary = data_storage.export_data_summary()
        print(f"\n📋 Overall Summary:")
        print(f"   Total symbols: {summary['total_symbols']}")
        print(f"   Total files: {summary['total_files']}")
        print(f"   Total bars: {summary['total_bars']}")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(demo_data_collection())
