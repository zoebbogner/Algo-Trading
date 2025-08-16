#!/usr/bin/env python3
"""Test script for the backtesting engine."""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from decimal import Decimal

# Add src to path
sys.path.insert(0, 'src')

from src.core.backtesting.engine import BacktestEngine
from src.core.backtesting.analyzer import PerformanceAnalyzer
from src.core.strategies.mean_reversion import MeanReversionStrategy
from src.core.risk.manager import RiskManager
from src.adapters.data.manager import DataManager


async def test_backtest_engine():
    """Test the backtesting engine with real data."""
    print("🧪 Testing Backtesting Engine")
    print("=" * 60)
    
    # Configuration
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
    
    # Initialize components
    print("🔧 Initializing components...")
    data_manager = DataManager(config)
    risk_manager = RiskManager()
    strategy = MeanReversionStrategy()
    
    # Test parameters
    symbols = ["BTC/USDT", "ETH/USDT"]
    timeframe = "1h"
    initial_capital = Decimal("10000")
    start_date = datetime.now(timezone.utc) - timedelta(days=7)  # Last 7 days
    end_date = datetime.now(timezone.utc)
    
    print(f"📊 Backtest Parameters:")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print(f"  Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print()
    
    try:
        # Initialize backtest engine
        print("🚀 Initializing backtest engine...")
        backtest_engine = BacktestEngine(
            strategy=strategy,
            data_manager=data_manager,
            risk_manager=risk_manager,
            initial_capital=initial_capital,
            commission_rate=Decimal("0.001"),  # 0.1%
            slippage_rate=Decimal("0.0005"),   # 0.05%
            start_date=start_date,
            end_date=end_date
        )
        
        # Run backtest
        print("📈 Running backtest...")
        result = await backtest_engine.run_backtest(
            symbols=symbols,
            timeframe=timeframe,
            include_features=True
        )
        
        print("✅ Backtest completed successfully!")
        print()
        
        # Analyze results
        print("📊 Analyzing results...")
        analyzer = PerformanceAnalyzer(result)
        
        # Print summary
        analyzer.print_summary()
        
        # Export results
        print("\n💾 Exporting results...")
        exported_files = analyzer.export_results("reports")
        
        print("📁 Exported files:")
        for report_type, file_path in exported_files.items():
            print(f"  {report_type}: {file_path}")
        
        # Additional analysis
        print("\n📈 Additional Analysis:")
        
        # Monthly returns
        monthly_returns = analyzer.generate_monthly_returns()
        if monthly_returns:
            print("  Monthly Returns:")
            for month, return_pct in monthly_returns.items():
                print(f"    {month}: {return_pct:+.2f}%")
        
        # Drawdown analysis
        drawdown_analysis = analyzer.generate_drawdown_analysis()
        if drawdown_analysis:
            print(f"  Drawdown Analysis:")
            print(f"    Total periods: {drawdown_analysis['total_periods']}")
            print(f"    Average duration: {drawdown_analysis['avg_duration_days']:.1f} days")
            
            max_dd_period = drawdown_analysis['max_drawdown_period']
            print(f"    Max drawdown period:")
            print(f"      Start: {max_dd_period['start'].strftime('%Y-%m-%d %H:%M')}")
            print(f"      End: {max_dd_period['end'].strftime('%Y-%m-%d %H:%M')}")
            print(f"      Duration: {max_dd_period['duration_days']} days")
        
        # Key metrics
        key_metrics = analyzer.get_key_metrics()
        print(f"\n🎯 Key Metrics Summary:")
        print(f"  Total Return: {key_metrics['total_return_pct']:+.2f}%")
        print(f"  Annualized Return: {key_metrics['annualized_return']:+.2f}%")
        print(f"  Sharpe Ratio: {key_metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {key_metrics['max_drawdown_pct']:.2f}%")
        print(f"  Win Rate: {key_metrics['win_rate']:.1%}")
        print(f"  Profit Factor: {key_metrics['profit_factor']:.3f}")
        print(f"  Total Trades: {key_metrics['total_trades']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backtest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        await data_manager.disconnect_all()


async def test_multiple_strategies():
    """Test multiple strategies for comparison."""
    print("\n🔄 Testing Multiple Strategies")
    print("=" * 60)
    
    # Configuration
    config = {
        "adapters": {
            "binance": {
                "enabled": True,
                "sandbox": True
            }
        }
    }
    
    # Test parameters
    symbols = ["BTC/USDT"]
    timeframe = "4h"
    initial_capital = Decimal("10000")
    start_date = datetime.now(timezone.utc) - timedelta(days=14)  # Last 14 days
    end_date = datetime.now(timezone.utc)
    
    # Strategy configurations to test
    strategy_configs = [
        {
            "name": "Mean Reversion (Conservative)",
            "strategy": MeanReversionStrategy(),
            "params": {"rsi_oversold": 30, "rsi_overbought": 70, "position_size": 0.05}
        },
        {
            "name": "Mean Reversion (Aggressive)",
            "strategy": MeanReversionStrategy(),
            "params": {"rsi_oversold": 25, "rsi_overbought": 75, "position_size": 0.10}
        }
    ]
    
    results = []
    
    try:
        data_manager = DataManager(config)
        risk_manager = RiskManager()
        
        for config_item in strategy_configs:
            print(f"\n🧪 Testing: {config_item['name']}")
            print("-" * 40)
            
            # Configure strategy
            strategy = config_item['strategy']
            for param, value in config_item['params'].items():
                if hasattr(strategy, param):
                    setattr(strategy, param, value)
            
            # Run backtest
            backtest_engine = BacktestEngine(
                strategy=strategy,
                data_manager=data_manager,
                risk_manager=risk_manager,
                initial_capital=initial_capital,
                start_date=start_date,
                end_date=end_date
            )
            
            result = await backtest_engine.run_backtest(
                symbols=symbols,
                timeframe=timeframe,
                include_features=True
            )
            
            # Analyze results
            analyzer = PerformanceAnalyzer(result)
            key_metrics = analyzer.get_key_metrics()
            
            results.append({
                'name': config_item['name'],
                'metrics': key_metrics,
                'result': result
            })
            
            print(f"  Total Return: {key_metrics['total_return_pct']:+.2f}%")
            print(f"  Sharpe Ratio: {key_metrics['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {key_metrics['max_drawdown_pct']:.2f}%")
            print(f"  Win Rate: {key_metrics['win_rate']:.1%}")
        
        # Compare results
        print(f"\n📊 Strategy Comparison")
        print("=" * 60)
        
        for result_item in results:
            metrics = result_item['metrics']
            print(f"\n{result_item['name']}:")
            print(f"  Return: {metrics['total_return_pct']:+.2f}% | "
                  f"Sharpe: {metrics['sharpe_ratio']:.3f} | "
                  f"DD: {metrics['max_drawdown_pct']:.2f}% | "
                  f"Win Rate: {metrics['win_rate']:.1%}")
        
        # Find best strategy
        best_strategy = max(results, key=lambda x: x['metrics']['sharpe_ratio'])
        print(f"\n🏆 Best Strategy (by Sharpe Ratio): {best_strategy['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multiple strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await data_manager.disconnect_all()


async def test_parameter_optimization():
    """Test parameter optimization capabilities."""
    print("\n🔧 Testing Parameter Optimization")
    print("=" * 60)
    
    # Configuration
    config = {
        "adapters": {
            "binance": {
                "enabled": True,
                "sandbox": True
            }
        }
    }
    
    # Test parameters
    symbols = ["ETH/USDT"]
    timeframe = "1h"
    initial_capital = Decimal("10000")
    start_date = datetime.now(timezone.utc) - timedelta(days=10)
    end_date = datetime.now(timezone.utc)
    
    # Parameter combinations to test
    param_combinations = [
        {"rsi_oversold": 25, "rsi_overbought": 75},
        {"rsi_oversold": 30, "rsi_overbought": 70},
        {"rsi_oversold": 35, "rsi_overbought": 65},
        {"rsi_oversold": 20, "rsi_overbought": 80}
    ]
    
    optimization_results = []
    
    try:
        data_manager = DataManager(config)
        risk_manager = RiskManager()
        
        for i, params in enumerate(param_combinations):
            print(f"\n🧪 Testing Parameter Set {i+1}: {params}")
            print("-" * 40)
            
            # Create strategy with parameters
            strategy = MeanReversionStrategy()
            for param, value in params.items():
                if hasattr(strategy, param):
                    setattr(strategy, param, value)
            
            # Run backtest
            backtest_engine = BacktestEngine(
                strategy=strategy,
                data_manager=data_manager,
                risk_manager=risk_manager,
                initial_capital=initial_capital,
                start_date=start_date,
                end_date=end_date
            )
            
            result = await backtest_engine.run_backtest(
                symbols=symbols,
                timeframe=timeframe,
                include_features=True
            )
            
            # Analyze results
            analyzer = PerformanceAnalyzer(result)
            key_metrics = analyzer.get_key_metrics()
            
            optimization_results.append({
                'parameters': params,
                'metrics': key_metrics,
                'result': result
            })
            
            print(f"  Return: {key_metrics['total_return_pct']:+.2f}%")
            print(f"  Sharpe: {key_metrics['sharpe_ratio']:.3f}")
            print(f"  Max DD: {key_metrics['max_drawdown_pct']:.2f}%")
        
        # Find optimal parameters
        print(f"\n📊 Parameter Optimization Results")
        print("=" * 60)
        
        # Sort by Sharpe ratio
        optimization_results.sort(key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)
        
        for i, result_item in enumerate(optimization_results):
            params = result_item['parameters']
            metrics = result_item['metrics']
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            
            print(f"\n{rank} Parameters: {params}")
            print(f"  Return: {metrics['total_return_pct']:+.2f}% | "
                  f"Sharpe: {metrics['sharpe_ratio']:.3f} | "
                  f"DD: {metrics['max_drawdown_pct']:.2f}%")
        
        # Best parameters
        best_params = optimization_results[0]['parameters']
        print(f"\n🏆 Optimal Parameters: {best_params}")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await data_manager.disconnect_all()


async def main():
    """Run all backtesting tests."""
    print("🚀 Backtesting Engine Test Suite")
    print("=" * 80)
    print("Testing comprehensive backtesting capabilities with real market data")
    print()
    
    tests = [
        ("Basic Backtesting", test_backtest_engine),
        ("Multiple Strategies", test_multiple_strategies),
        ("Parameter Optimization", test_parameter_optimization)
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
        print("🎉 All tests passed! Backtesting engine is working perfectly!")
        print("\n🚀 What you can do now:")
        print("  • Test different strategies with real historical data")
        print("  • Optimize strategy parameters")
        print("  • Compare multiple strategies")
        print("  • Export detailed performance reports")
        print("  • Use results to improve your trading strategies")
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
