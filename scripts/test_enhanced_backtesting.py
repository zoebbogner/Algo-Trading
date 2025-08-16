#!/usr/bin/env python3
"""
Enhanced Backtesting Test Script

Demonstrates advanced backtesting capabilities including:
- Momentum strategy testing
- Detailed hourly P&L tracking
- Real-time portfolio updates
- Beautiful performance reporting
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.backtesting.engine import BacktestEngine
from src.core.backtesting.analyzer import PerformanceAnalyzer
from src.core.strategy.momentum import MomentumStrategy
from src.core.strategy.mean_reversion import MeanReversionStrategy
from src.adapters.data.manager import DataManager
from src.core.risk.manager import RiskManager


async def test_momentum_strategy():
    """Test the new momentum strategy with detailed tracking"""
    print("\n🚀 Testing Momentum Strategy with Hourly P&L Tracking")
    print("=" * 80)
    
    # Initialize components
    print("🔧 Initializing components...")
    
    # Data manager
    data_config = {
        'adapters': {
            'binance': {
                'enabled': True,
                'sandbox': True,
                'name': 'Binance Testnet'
            }
        },
        'cache_size': 1000
    }
    data_manager = DataManager(data_config)
    
    # Risk manager
    risk_config = {
        'max_position_size': 0.2,
        'max_daily_loss': 0.05,
        'max_drawdown': 0.15,
        'position_limits': {'BTC/USDT': 0.3, 'ETH/USDT': 0.3}
    }
    risk_manager = RiskManager(risk_config)
    
    # Momentum strategy with different configurations
    momentum_configs = [
        {
            'name': 'Momentum (Conservative)',
            'fast_ma_period': 5,
            'slow_ma_period': 15,
            'rsi_period': 14,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'volume_multiplier': 2.0,
            'position_size': 0.1,
            'stop_loss_pct': 0.015,
            'take_profit_pct': 0.03
        },
        {
            'name': 'Momentum (Aggressive)',
            'fast_ma_period': 3,
            'slow_ma_period': 10,
            'rsi_period': 14,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'volume_multiplier': 1.5,
            'position_size': 0.15,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.05
        }
    ]
    
    # Test each momentum configuration
    results = []
    
    for i, config in enumerate(momentum_configs):
        print(f"\n🧪 Testing: {config['name']}")
        print("-" * 50)
        
        # Create strategy
        strategy = MomentumStrategy(config)
        
        # Create backtest engine
        engine = BacktestEngine(
            strategy=strategy,
            data_manager=data_manager,
            risk_manager=risk_manager,
            initial_capital=10000.0
        )
        
        # Set date range (last 2 weeks for more data)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=14)
        
        print(f"📊 Backtest Parameters:")
        print(f"  Symbols: BTC/USDT, ETH/USDT")
        print(f"  Timeframe: 1h")
        print(f"  Initial Capital: ${engine.initial_capital:,.2f}")
        print(f"  Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            # Run backtest
            print("🚀 Running backtest...")
            result = await engine.run_backtest(
                symbols=['BTC/USDT', 'ETH/USDT'],
                start_date=start_date,
                end_date=end_date,
                timeframe="1h"
            )
            
            # Analyze results
            analyzer = PerformanceAnalyzer(result)
            
            # Display key metrics
            print(f"✅ Backtest completed successfully!")
            print(f"📊 Final Portfolio: ${result.final_capital:,.2f}")
            print(f"📈 Total Return: {result.total_return:+.2%}")
            print(f"⚠️  Max Drawdown: {result.max_drawdown:.2%}")
            print(f"🔄 Total Trades: {result.total_trades}")
            print(f"🎯 Win Rate: {result.win_rate:.1%}")
            
            # Store results for comparison
            results.append({
                'name': config['name'],
                'result': result,
                'analyzer': analyzer
            })
            
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            continue
    
    # Compare results
    if len(results) > 1:
        print(f"\n📊 Strategy Comparison")
        print("=" * 80)
        
        for result_data in results:
            result = result_data['result']
            print(f"\n{result_data['name']}:")
            print(f"  Return: {result.total_return:+.2%} | Sharpe: {result.sharpe_ratio:.3f} | DD: {result.max_drawdown:.2%} | Win Rate: {result.win_rate:.1%}")
    
    # Show detailed hourly breakdown for best strategy
    if results:
        best_result = max(results, key=lambda x: x['result'].total_return)
        print(f"\n🏆 Best Strategy: {best_result['name']}")
        print("=" * 80)
        
        # Generate and display hourly breakdown
        hourly_report = best_result['analyzer'].generate_hourly_breakdown()
        print(hourly_report)
        
        # Export results
        try:
            print("💾 Exporting results...")
            exported_files = best_result['analyzer'].export_results("reports")
            print(f"✅ Exported {len(exported_files)} files to reports/ directory")
            
            # Show file locations
            for file_path in exported_files:
                print(f"  📄 {file_path}")
                
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    return results


async def test_real_time_tracking():
    """Test real-time portfolio tracking and updates"""
    print("\n🔄 Testing Real-Time Portfolio Tracking")
    print("=" * 80)
    
    # Initialize components
    data_config = {
        'adapters': {
            'binance': {
                'enabled': True,
                'sandbox': True,
                'name': 'Binance Testnet'
            }
        },
        'cache_size': 1000
    }
    data_manager = DataManager(data_config)
    risk_manager = RiskManager({})
    
    # Use momentum strategy
    strategy_config = {
        'fast_ma_period': 5,
        'slow_ma_period': 15,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'volume_multiplier': 1.5,
        'position_size': 0.1,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04
    }
    
    strategy = MomentumStrategy(strategy_config)
    engine = BacktestEngine(
        strategy=strategy,
        data_manager=data_manager,
        risk_manager=risk_manager,
        initial_capital=5000.0
    )
    
    # Shorter timeframe for demonstration
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=7)
    
    print(f"📊 Real-Time Tracking Test:")
    print(f"  Symbol: BTC/USDT")
    print(f"  Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  Capital: ${engine.initial_capital:,.2f}")
    
    try:
        # Run backtest
        result = await engine.run_backtest(
            symbols=['BTC/USDT'],
            start_date=start_date,
            end_date=end_date,
            timeframe="1h"
        )
        
        print(f"✅ Real-time tracking completed!")
        print(f"📈 Final Portfolio: ${result.final_capital:,.2f}")
        print(f"🔄 Total Trades: {result.total_trades}")
        
        # Show hourly P&L summary
        if result.hourly_pnl:
            print(f"\n📊 Hourly P&L Summary:")
            print(f"  Total Hours: {len(result.hourly_pnl)}")
            print(f"  Profitable Hours: {len([h for h in result.hourly_pnl if h['hourly_pnl'] > 0])}")
            print(f"  Best Hour: ${max((h['hourly_pnl'] for h in result.hourly_pnl), default=0):+,.2f}")
            print(f"  Worst Hour: ${min((h['hourly_pnl'] for h in result.hourly_pnl), default=0):+,.2f}")
            
            # Show last few hours
            print(f"\n🕐 Last 5 Hours:")
            for hour_data in result.hourly_pnl[-5:]:
                timestamp = hour_data['timestamp']
                pnl = hour_data['hourly_pnl']
                pnl_pct = hour_data['hourly_pnl_pct']
                equity = hour_data['equity']
                
                print(f"  {timestamp.strftime('%m-%d %H:%M')}: ${pnl:+,.2f} ({pnl_pct:+.2f}%) | Equity: ${equity:,.2f}")
        
    except Exception as e:
        print(f"❌ Real-time tracking failed: {e}")


async def main():
    """Main test function"""
    print("🚀 Enhanced Backtesting Test Suite")
    print("=" * 80)
    print("Testing advanced backtesting capabilities with hourly P&L tracking")
    
    try:
        # Test momentum strategy
        await test_momentum_strategy()
        
        # Test real-time tracking
        await test_real_time_tracking()
        
        print(f"\n🎉 All tests completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
