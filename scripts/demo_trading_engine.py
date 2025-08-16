#!/usr/bin/env python3
"""Demo script to test the trading engine interactively."""

import sys
from decimal import Decimal
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, 'src')

from src.core.engine.trading_engine import TradingEngine
from src.core.data_models.market import Bar
from src.core.data_models.trading import Portfolio


def demo_trading_engine():
    """Demonstrate the trading engine functionality."""
    print("🚀 Algo-Trading Engine Demo")
    print("=" * 50)
    
    # Initialize trading engine
    config = {
        "risk": {
            "circuit_breakers": {
                "daily_loss_threshold": 0.03,
                "drawdown_threshold": 0.10,
                "volatility_threshold": 0.05,
                "spread_stress_threshold": 0.02,
                "cooldown_minutes": 60
            }
        },
        "data": {
            "feature_lookback_bars": 50
        }
    }
    
    print("1. Initializing trading engine...")
    engine = TradingEngine(config)
    
    # Initialize portfolio
    print("2. Initializing portfolio with $10,000...")
    engine.initialize_portfolio(Decimal("10000"))
    
    print(f"   Portfolio equity: ${engine.portfolio.equity:,.2f}")
    print(f"   Portfolio cash: ${engine.portfolio.cash:,.2f}")
    
    # Add market data
    print("\n3. Adding sample market data...")
    symbol = "BTC/USD"
    
    # Create sample bars with price movement
    base_price = 50000
    for i in range(30):  # Need at least 20 bars for features
        # Simulate some price movement
        price_change = (i % 10 - 5) * 100  # Oscillate around base price
        current_price = base_price + price_change
        
        bar = Bar(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            open=Decimal(str(current_price)),
            high=Decimal(str(current_price + 200)),
            low=Decimal(str(current_price - 200)),
            close=Decimal(str(current_price)),
            volume=Decimal("1000"),
            interval="1m"
        )
        engine.add_market_data(symbol, bar)
    
    print(f"   Added {len(engine.market_data_cache[symbol])} bars for {symbol}")
    
    # Process market data
    print("\n4. Processing market data and generating features...")
    market_data = engine.process_market_data(symbol)
    
    if market_data:
        print(f"   ✓ Market data processed successfully")
        print(f"   ✓ Generated {len(market_data.features)} features")
        
        # Show some features
        feature_names = [f.feature_name for f in market_data.features[:5]]
        print(f"   Sample features: {', '.join(feature_names)}")
    else:
        print("   ✗ Failed to process market data")
        return
    
    # Generate signals
    print("\n5. Generating trading signals...")
    signals = engine.generate_signals(symbol)
    
    if signals:
        print(f"   ✓ Generated signals from {len(signals)} strategies")
        for strategy_name, strategy_signals in signals.items():
            entry_count = len(strategy_signals.get("entry_signals", {}))
            exit_count = len(strategy_signals.get("exit_signals", {}))
            print(f"   Strategy '{strategy_name}': {entry_count} entry, {exit_count} exit signals")
    else:
        print("   No signals generated (this is normal for demo data)")
    
    # Check risk status
    print("\n6. Checking risk management...")
    risk_metrics = engine.risk_manager.get_risk_metrics(engine.portfolio)
    
    print(f"   Circuit breakers: {len(risk_metrics.get('circuit_breakers', {}))}")
    can_trade, reason = engine.risk_manager.can_trade(engine.portfolio)
    print(f"   Can trade: {can_trade}")
    if not can_trade:
        print(f"   Reason: {reason}")
    
    # Performance summary
    print("\n7. Performance summary...")
    summary = engine.get_performance_summary()
    
    print(f"   Portfolio equity: ${summary['portfolio']['equity']:,.2f}")
    print(f"   Portfolio exposure: ${summary['portfolio']['exposure']:,.2f}")
    print(f"   Portfolio drawdown: {summary['portfolio']['drawdown']:.2%}")
    
    # Strategy performance
    for strategy_name, strategy_summary in summary['strategies'].items():
        print(f"   Strategy '{strategy_name}': {strategy_summary.get('total_trades', 0)} trades")
    
    print("\n🎉 Demo completed successfully!")
    print("The trading engine is working correctly!")


if __name__ == "__main__":
    try:
        demo_trading_engine()
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
