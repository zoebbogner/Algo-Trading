#!/usr/bin/env python3
"""
Ultra-Aggressive Backtesting Test

This strategy is designed to generate LOTS of trades for demonstration purposes.
It uses very short moving averages and simple conditions to ensure we see action!
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.backtesting.engine import BacktestEngine
from src.core.backtesting.analyzer import PerformanceAnalyzer
from src.core.strategy.momentum import MomentumStrategy
from src.adapters.data.manager import DataManager
from src.core.risk.manager import RiskManager
from src.core.data_models.market import MarketData
from src.core.data_models.trading import Portfolio


class UltraAggressiveStrategy(MomentumStrategy):
    """Ultra-aggressive strategy that generates many trades"""
    
    def __init__(self, config: Dict[str, Any]):
        # Override with ultra-aggressive settings
        ultra_config = {
            'fast_ma_period': 2,      # Very short - 2 periods
            'slow_ma_period': 5,      # Very short - 5 periods
            'rsi_period': 7,          # Short RSI
            'rsi_oversold': 40,       # Relaxed oversold
            'rsi_overbought': 60,     # Relaxed overbought
            'volume_multiplier': 1.0, # Any volume
            'position_size': 0.2,     # Larger positions
            'stop_loss_pct': 0.01,    # Tight stop loss
            'take_profit_pct': 0.02,  # Tight take profit
            'name': 'Ultra-Aggressive'
        }
        ultra_config.update(config)
        super().__init__(ultra_config)
    
    def generate_signals(self, market_data: MarketData, portfolio: Portfolio) -> Dict[str, Any]:
        """Generate ultra-aggressive trading signals"""
        print(f"🔍 ULTRA AGGRESSIVE: Starting signal generation...")
        print(f"🔍 ULTRA AGGRESSIVE: Market data has {len(market_data.bars)} symbols")
        print(f"🔍 ULTRA AGGRESSIVE: Bars keys: {list(market_data.bars.keys())}")
        
        if not market_data.bars:
            print(f"❌ ULTRA AGGRESSIVE: No bars data")
            return {'entry_signals': {}, 'exit_signals': {}}
        
        entry_signals = {}
        exit_signals = {}
        
        for symbol, bars in market_data.bars.items():
            print(f"🔍 ULTRA AGGRESSIVE: Processing {symbol} with {len(bars)} bars")
            
            if len(bars) < self.slow_ma_period:
                print(f"❌ ULTRA AGGRESSIVE: {symbol} has only {len(bars)} bars, need {self.slow_ma_period}")
                continue
                
            # Extract price data
            prices = [float(bar.close) for bar in bars]
            
            print(f"🔍 ULTRA AGGRESSIVE: {symbol} prices: {prices[:5]}... (last: {prices[-1]:.2f})")
            
            # Calculate very short moving averages
            fast_ma = self._calculate_sma(prices, self.fast_ma_period)
            slow_ma = self._calculate_sma(prices, self.slow_ma_period)
            
            print(f"🔍 ULTRA AGGRESSIVE: {symbol} indicators - Fast MA: {fast_ma}, Slow MA: {slow_ma}")
            
            if fast_ma is None or slow_ma is None:
                print(f"❌ ULTRA AGGRESSIVE: {symbol} has None indicators")
                continue
            
            current_price = prices[-1]
            current_position = self._get_position(symbol, portfolio)
            
            print(f"🔍 ULTRA AGGRESSIVE: {symbol}: Fast MA: {fast_ma:.2f}, Slow MA: {slow_ma:.2f}")
            print(f"🔍 ULTRA AGGRESSIVE: {symbol}: Current position: {current_position}")
            
            # ULTRA AGGRESSIVE: Trade on ANY moving average crossover
            if not current_position:
                print(f"🔍 ULTRA AGGRESSIVE: {symbol} has no position, checking entry conditions...")
                
                # Buy when fast MA crosses above slow MA
                if fast_ma > slow_ma:
                    print(f"🚀 ULTRA AGGRESSIVE: {symbol} BUY condition met: Fast MA({fast_ma:.2f}) > Slow MA({slow_ma:.2f})")
                    entry_signals[symbol] = {
                        'side': 'buy',
                        'price': current_price,
                        'quantity': self._calculate_position_size(current_price, portfolio),
                        'reason': f'ULTRA AGGRESSIVE BUY: Fast MA({fast_ma:.2f}) > Slow MA({slow_ma:.2f})',
                        'timestamp': datetime.now(timezone.utc)
                    }
                    print(f"🚀 ULTRA AGGRESSIVE BUY for {symbol}!")
                
                # Only BUY when no position (no short selling for simplicity)
                # SELL signals will be generated in the exit logic when we have positions
            
            # Exit positions quickly
            elif current_position:
                position = current_position
                entry_price = float(position.average_cost)
                
                if position.quantity > 0:  # Long position
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    # Quick exit on any reversal
                    if fast_ma < slow_ma:
                        exit_signals[symbol] = {
                            'side': 'sell',
                            'price': current_price,
                            'quantity': abs(position.quantity),
                            'reason': f'Quick exit: Fast MA({fast_ma:.2f}) < Slow MA({slow_ma:.2f})',
                            'timestamp': datetime.now(timezone.utc)
                        }
                        print(f"🔄 QUICK EXIT for {symbol}")
                
                elif position.quantity < 0:  # Short position
                    pnl_pct = (entry_price - current_price) / entry_price
                    
                    # Quick exit on any reversal
                    if fast_ma > slow_ma:
                        exit_signals[symbol] = {
                            'side': 'buy',
                            'price': current_price,
                            'quantity': abs(position.quantity),
                            'reason': f'Quick exit: Fast MA({fast_ma:.2f}) > Slow MA({slow_ma:.2f})',
                            'timestamp': datetime.now(timezone.utc)
                        }
                        print(f"🔄 QUICK EXIT for {symbol}")
        
        print(f"📊 ULTRA AGGRESSIVE: Generated {len(entry_signals)} entry signals and {len(exit_signals)} exit signals")
        return {
            'entry_signals': entry_signals,
            'exit_signals': exit_signals
        }


async def test_ultra_aggressive():
    """Test the ultra-aggressive strategy"""
    print("\n🚀 Testing Ultra-Aggressive Strategy")
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
    
    risk_config = {
        'max_position_size': 0.3,
        'max_daily_loss': 0.1,
        'max_drawdown': 0.2
    }
    risk_manager = RiskManager(risk_config)
    
    # Create ultra-aggressive strategy
    strategy = UltraAggressiveStrategy({
        'name': 'Ultra-Aggressive Momentum',
        'fast_ma_period': 2,
        'slow_ma_period': 5
    })
    
    # Create backtest engine
    engine = BacktestEngine(
        strategy=strategy,
        data_manager=data_manager,
        risk_manager=risk_manager,
        initial_capital=10000.0
    )
    
    # Set date range (shorter for faster testing)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=7)  # Just 1 week
    
    print(f"📊 Ultra-Aggressive Test Parameters:")
    print(f"  Symbols: BTC/USDT, ETH/USDT")
    print(f"  Timeframe: 1h")
    print(f"  Initial Capital: ${engine.initial_capital:,.2f}")
    print(f"  Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  Strategy: {strategy.name}")
    print(f"  Fast MA: {strategy.fast_ma_period} periods")
    print(f"  Slow MA: {strategy.slow_ma_period} periods")
    
    try:
        # Run backtest
        print("🚀 Running ultra-aggressive backtest...")
        result = await engine.run_backtest(
            symbols=['BTC/USDT', 'ETH/USDT'],
            start_date=start_date,
            end_date=end_date,
            timeframe="1h"
        )
        
        # Analyze results
        analyzer = PerformanceAnalyzer(result)
        
        # Display results
        print(f"\n✅ Ultra-Aggressive Backtest Completed!")
        print(f"📊 Final Portfolio: ${result.final_capital:,.2f}")
        print(f"📈 Total Return: {result.total_return:+.2%}")
        print(f"🔄 Total Trades: {result.total_trades}")
        print(f"🎯 Win Rate: {result.win_rate:.1%}")
        
        if result.total_trades > 0:
            print(f"🎉 SUCCESS! Generated {result.total_trades} trades!")
            
            # Show hourly breakdown
            if result.hourly_pnl:
                print(f"\n📊 Hourly P&L Summary:")
                print(f"  Total Hours: {len(result.hourly_pnl)}")
                print(f"  Profitable Hours: {len([h for h in result.hourly_pnl if h['hourly_pnl'] > 0])}")
                print(f"  Best Hour: ${max((h['hourly_pnl'] for h in result.hourly_pnl), default=0):+,.2f}")
                print(f"  Worst Hour: ${min((h['hourly_pnl'] for h in result.hourly_pnl), default=0):+,.2f}")
            
            # Export results
            try:
                exported_files = analyzer.export_results("reports")
                print(f"💾 Exported {len(exported_files)} files to reports/ directory")
            except Exception as e:
                print(f"❌ Export failed: {e}")
        else:
            print(f"❌ Still no trades generated. Let me investigate further...")
            
    except Exception as e:
        print(f"❌ Ultra-aggressive backtest failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function"""
    print("🚀 Ultra-Aggressive Backtesting Test")
    print("=" * 80)
    print("Testing strategy designed to generate LOTS of trades!")
    
    try:
        await test_ultra_aggressive()
        print(f"\n🎉 Ultra-aggressive test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
