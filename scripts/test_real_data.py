#!/usr/bin/env python3
"""Test script for real market data integration."""

import asyncio
import sys
from datetime import datetime, timezone, timedelta

# Add src to path
sys.path.insert(0, 'src')

from src.adapters.data.manager import DataManager


async def test_binance_connection():
    """Test connection to Binance."""
    print("🔌 Testing Binance Connection")
    print("-" * 40)
    
    config = {
        "adapters": {
            "binance": {
                "enabled": True,
                "sandbox": True,
                "name": "Binance Testnet"
            }
        },
        "cache_size": 100
    }
    
    async with DataManager(config) as data_manager:
        # Test connection
        status = data_manager.get_adapter_status()
        print(f"Adapter status: {status}")
        
        # Test exchange info
        print("\n📊 Getting exchange information...")
        exchange_info = await data_manager.adapters["binance"].get_exchange_info()
        if exchange_info:
            print(f"Exchange: {exchange_info['exchange']}")
            print(f"Sandbox: {exchange_info['sandbox']}")
            print(f"Total symbols: {exchange_info['total_symbols']}")
            print(f"USDT pairs: {exchange_info['usdt_pairs']}")
            print(f"Sample symbols: {exchange_info['sample_symbols'][:5]}")
        else:
            print("❌ Failed to get exchange info")
            return False
        
        return True


async def test_historical_data():
    """Test historical data retrieval."""
    print("\n📈 Testing Historical Data Retrieval")
    print("-" * 40)
    
    config = {
        "adapters": {
            "binance": {
                "enabled": True,
                "sandbox": True
            }
        },
        "cache_size": 100
    }
    
    async with DataManager(config) as data_manager:
        symbol = "BTC/USDT"
        timeframe = "1h"
        limit = 24  # Last 24 hours
        
        print(f"Fetching {limit} {timeframe} bars for {symbol}...")
        
        # Get historical data
        bars = await data_manager.get_historical_data(symbol, timeframe, limit)
        
        if bars:
            print(f"✅ Retrieved {len(bars)} bars")
            
            # Show sample data
            print(f"First bar: {bars[0].timestamp} - O:{bars[0].open} H:{bars[0].high} L:{bars[0].low} C:{bars[0].close} V:{bars[0].volume}")
            print(f"Last bar:  {bars[-1].timestamp} - O:{bars[-1].open} H:{bars[-1].high} L:{bars[-1].low} C:{bars[-1].close} V:{bars[-1].volume}")
            
            # Calculate price change
            price_change = (bars[-1].close - bars[0].open) / bars[0].open * 100
            print(f"Price change: {price_change:.2f}%")
            
            return True
        else:
            print("❌ Failed to retrieve historical data")
            return False


async def test_current_prices():
    """Test current price retrieval."""
    print("\n💰 Testing Current Price Retrieval")
    print("-" * 40)
    
    config = {
        "adapters": {
            "binance": {
                "enabled": True,
                "sandbox": True
            }
        }
    }
    
    async with DataManager(config) as data_manager:
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        
        print(f"Getting current prices for {len(symbols)} symbols...")
        
        for symbol in symbols:
            price = await data_manager.get_current_price(symbol)
            if price:
                print(f"✅ {symbol}: ${price:,.2f}")
            else:
                print(f"❌ {symbol}: Failed to get price")
        
        return True


async def test_market_data_with_features():
    """Test market data with technical features."""
    print("\n🔧 Testing Market Data with Features")
    print("-" * 40)
    
    config = {
        "adapters": {
            "binance": {
                "enabled": True,
                "sandbox": True
            }
        },
        "cache_size": 100
    }
    
    async with DataManager(config) as data_manager:
        symbol = "ETH/USDT"
        timeframe = "15m"
        
        print(f"Getting market data with features for {symbol} ({timeframe})...")
        
        # Get market data with features
        market_data = await data_manager.get_market_data(symbol, timeframe, include_features=True)
        
        if market_data:
            print(f"✅ Market data retrieved successfully")
            print(f"Timestamp: {market_data.timestamp}")
            print(f"Price: ${market_data.bar.close:,.2f}")
            print(f"Volume: {market_data.bar.volume:,.0f}")
            
            if market_data.features:
                print(f"Features generated: {len(market_data.features)}")
                
                # Show some key features
                feature_names = [f.feature_name for f in market_data.features[:10]]
                print(f"Sample features: {', '.join(feature_names)}")
                
                # Show specific feature values
                for feature in market_data.features[:5]:
                    if hasattr(feature, 'feature_value'):
                        print(f"  {feature.feature_name}: {feature.feature_value:.4f}")
            else:
                print("⚠️ No features generated (insufficient data)")
            
            # Show bid/ask if available
            if market_data.bid and market_data.ask:
                spread = market_data.ask - market_data.bid
                spread_pct = (spread / market_data.bid) * 100
                print(f"Bid: ${market_data.bid:,.2f}, Ask: ${market_data.ask:,.2f}")
                print(f"Spread: ${spread:,.2f} ({spread_pct:.3f}%)")
            
            return True
        else:
            print("❌ Failed to retrieve market data")
            return False


async def test_multiple_symbols():
    """Test retrieving data for multiple symbols."""
    print("\n🔄 Testing Multiple Symbols Data")
    print("-" * 40)
    
    config = {
        "adapters": {
            "binance": {
                "enabled": True,
                "sandbox": True
            }
        }
    }
    
    async with DataManager(config) as data_manager:
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        timeframe = "5m"
        
        print(f"Getting {timeframe} data for {len(symbols)} symbols...")
        
        # Get data for multiple symbols
        market_data_dict = await data_manager.get_multiple_symbols_data(symbols, timeframe, include_features=False)
        
        if market_data_dict:
            print(f"✅ Retrieved data for {len(market_data_dict)} symbols")
            
            for symbol, market_data in market_data_dict.items():
                print(f"  {symbol}: ${market_data.bar.close:,.2f} (V: {market_data.bar.volume:,.0f})")
            
            return True
        else:
            print("❌ Failed to retrieve multiple symbols data")
            return False


async def test_order_book():
    """Test order book retrieval."""
    print("\n📚 Testing Order Book Retrieval")
    print("-" * 40)
    
    config = {
        "adapters": {
            "binance": {
                "enabled": True,
                "sandbox": True
            }
        }
    }
    
    async with DataManager(config) as data_manager:
        symbol = "BTC/USDT"
        depth = 10
        
        print(f"Getting order book for {symbol} (depth: {depth})...")
        
        order_book = await data_manager.adapters["binance"].get_order_book(symbol, depth)
        
        if order_book:
            print(f"✅ Order book retrieved successfully")
            print(f"Timestamp: {order_book['timestamp']}")
            
            # Show top bids and asks
            print(f"Top 3 bids:")
            for i, (price, amount) in enumerate(order_book['bids'][:3]):
                print(f"  {i+1}. ${price:,.2f} - {amount:.4f}")
            
            print(f"Top 3 asks:")
            for i, (price, amount) in enumerate(order_book['asks'][:3]):
                print(f"  {i+1}. ${price:,.2f} - {amount:.4f}")
            
            # Calculate spread
            best_bid = order_book['bids'][0][0]
            best_ask = order_book['asks'][0][0]
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100
            
            print(f"Spread: ${spread:,.2f} ({spread_pct:.3f}%)")
            
            return True
        else:
            print("❌ Failed to retrieve order book")
            return False


async def main():
    """Run all tests."""
    print("🚀 Real Market Data Integration Test")
    print("=" * 60)
    print("Testing Binance integration (no API key required)")
    print()
    
    tests = [
        ("Binance Connection", test_binance_connection),
        ("Historical Data", test_historical_data),
        ("Current Prices", test_current_prices),
        ("Market Data with Features", test_market_data_with_features),
        ("Multiple Symbols", test_multiple_symbols),
        ("Order Book", test_order_book)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"🧪 Running: {test_name}")
            result = await test_func()
            results.append((test_name, result))
            print()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, False))
            print()
    
    # Summary
    print("📊 Test Results Summary")
    print("=" * 40)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Real market data integration is working!")
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
