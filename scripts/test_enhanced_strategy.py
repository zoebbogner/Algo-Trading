#!/usr/bin/env python3
"""
Test Enhanced Momentum Strategy

This script tests the improved momentum strategy with:
- Multiple confirmation signals
- Better risk management
- Trailing stops
- Volatility-based position sizing
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any

# Add src to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.backtesting.engine import BacktestEngine
from src.core.strategy.momentum import MomentumStrategy
from src.core.risk.manager import RiskManager
from src.adapters.data.manager import DataManager


async def test_enhanced_strategy():
    """Test the enhanced momentum strategy"""
    print("🚀 Testing Enhanced Momentum Strategy")
    print("=" * 50)
    
    # Enhanced strategy configuration with advanced features
    strategy_config = {
        'name': 'Advanced Enhanced Momentum Strategy',
        'fast_ma_period': 8,           # Faster EMA for quick signals
        'slow_ma_period': 21,          # Medium EMA for trend confirmation
        'trend_ma_period': 50,         # Long-term trend
        'rsi_period': 14,
        'rsi_oversold': 35,            # Less strict oversold
        'rsi_overbought': 65,          # Less strict overbought
        'adx_period': 14,              # Trend strength
        'adx_threshold': 25,           # Minimum trend strength
        'volume_multiplier': 1.2,      # Lower volume requirement
        'position_size': 0.08,         # Smaller positions for risk management
        'stop_loss_pct': 0.015,        # Tighter stop loss (1.5%)
        'take_profit_pct': 0.03,       # Better risk-reward (2:1)
        'trailing_stop_pct': 0.01,     # 1% trailing stop
        
        # Advanced parameters
        'atr_period': 14,              # Volatility measurement
        'volatility_threshold': 0.02,  # 2% volatility threshold
        'correlation_threshold': 0.7,  # Correlation limit
        'max_correlation_exposure': 0.3,  # Max 30% in correlated assets
        
        # Multi-timeframe parameters
        'short_ma_period': 5,          # Very short-term momentum
        'medium_ma_period': 13,        # Short-term trend
        'long_ma_period': 34,          # Medium-term trend
    }
    
    # Data manager configuration
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
    
    # Risk manager configuration
    risk_config = {
        'max_position_size': 0.2,
        'max_daily_loss': 0.05,
        'max_drawdown': 0.15
    }
    
    print(f"📊 Strategy Configuration:")
    for key, value in strategy_config.items():
        print(f"   {key}: {value}")
    print()
    
    # Initialize components
    strategy = MomentumStrategy(strategy_config)
    risk_manager = RiskManager(risk_config)
    data_manager = DataManager(data_config)
    
    # Initialize backtest engine
    initial_capital = Decimal("10000")
    engine = BacktestEngine(
        strategy=strategy,
        data_manager=data_manager,
        risk_manager=risk_manager,
        initial_capital=initial_capital
    )
    
    # Test parameters
    symbols = ['BTC/USDT', 'ETH/USDT']
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=14)  # 2 weeks for better testing
    timeframe = "1h"
    
    print(f"📈 Backtest Parameters:")
    print(f"   Symbols: {symbols}")
    print(f"   Start Date: {start_date}")
    print(f"   End Date: {end_date}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print()
    
    try:
        # Run enhanced backtest
        print("🚀 Starting Enhanced Strategy Backtest...")
        results = await engine.run_backtest(symbols, start_date, end_date, timeframe)
        
        # Display results
        print("\n" + "=" * 60)
        print("🎯 ENHANCED STRATEGY RESULTS")
        print("=" * 60)
        
        print(f"📊 Strategy Name: {results.strategy_name}")
        print(f"📅 Test Period: {results.start_date} → {results.end_date}")
        print(f"💰 Initial Capital: ${results.initial_capital:,.2f}")
        print(f"💰 Final Capital: ${results.final_capital:,.2f}")
        print(f"📈 Total Return: {results.total_return:+.2%}")
        print(f"📊 Annualized Return: {results.annualized_return:+.2%}")
        print(f"📉 Max Drawdown: {results.max_drawdown:.2%}")
        print(f"📊 Sharpe Ratio: {results.sharpe_ratio:.3f}")
        print()
        
        print(f"🔄 Trading Statistics:")
        print(f"   Total Trades: {results.total_trades}")
        print(f"   Win Rate: {results.win_rate:.1%}")
        print(f"   Profit Factor: {results.profit_factor:.2f}")
        print()
        
        # Analyze trade quality
        if results.trades:
            winning_trades = [t for t in results.trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in results.trades if t.get('pnl', 0) < 0]
            
            if winning_trades:
                avg_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades)
                print(f"   Average Win: ${avg_win:+.2f}")
            
            if losing_trades:
                avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades)
                print(f"   Average Loss: ${avg_loss:+.2f}")
                
            if winning_trades and losing_trades:
                risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                print(f"   Risk-Reward Ratio: {risk_reward:.2f}")
        
        # Performance assessment
        print(f"\n🏆 PERFORMANCE ASSESSMENT:")
        if results.total_return > 0.05:  # >5%
            print("   🟢 EXCELLENT: Strong positive returns")
        elif results.total_return > 0.02:  # >2%
            print("   🟡 GOOD: Positive returns")
        elif results.total_return > -0.02:  # >-2%
            print("   🟠 ACCEPTABLE: Small loss, good risk management")
        else:
            print("   🔴 NEEDS IMPROVEMENT: Significant losses")
        
        if results.win_rate > 0.5:  # >50%
            print("   🟢 EXCELLENT: High win rate")
        elif results.win_rate > 0.4:  # >40%
            print("   🟡 GOOD: Above average win rate")
        elif results.win_rate > 0.3:  # >30%
            print("   🟠 ACCEPTABLE: Moderate win rate")
        else:
            print("   🔴 NEEDS IMPROVEMENT: Low win rate")
        
        if results.max_drawdown < 0.05:  # <5%
            print("   🟢 EXCELLENT: Very low drawdown")
        elif results.max_drawdown < 0.1:  # <10%
            print("   🟡 GOOD: Low drawdown")
        elif results.max_drawdown < 0.15:  # <15%
            print("   🟠 ACCEPTABLE: Moderate drawdown")
        else:
            print("   🔴 NEEDS IMPROVEMENT: High drawdown")
        
        print(f"\n💾 Reports exported to reports/ directory")
        print(f"🎉 Enhanced Strategy Test Completed!")
        
        return results
        
    except Exception as e:
        print(f"❌ Enhanced strategy test failed: {e}")
        logging.error(f"Enhanced strategy test error: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run enhanced strategy test
    asyncio.run(test_enhanced_strategy())
