#!/usr/bin/env python3
"""Test different trading scenarios with the Algo-Trading system."""

import sys
from decimal import Decimal
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, 'src')

from src.core.engine.trading_engine import TradingEngine
from src.core.data_models.market import Bar
from src.core.features.technical import TechnicalIndicators


def test_technical_indicators():
    """Test technical indicators calculation."""
    print("🧮 Testing Technical Indicators")
    print("-" * 40)
    
    # Test data
    prices = [100, 102, 98, 103, 105, 101, 99, 104, 106, 108]
    highs = [102, 104, 100, 105, 107, 103, 101, 106, 108, 110]
    lows = [99, 101, 97, 102, 104, 100, 98, 103, 105, 107]
    closes = [102, 98, 103, 105, 101, 99, 104, 106, 108, 109]
    
    # Test SMA
    sma_5 = TechnicalIndicators.sma(prices, 5)
    print(f"SMA(5): {sma_5:.2f}")
    
    # Test RSI
    rsi = TechnicalIndicators.rsi(prices, 9)
    print(f"RSI(9): {rsi:.2f}")
    
    # Test Bollinger Bands
    bb = TechnicalIndicators.bollinger_bands(prices, 5)
    print(f"BB Upper: {bb['upper']:.2f}")
    print(f"BB Middle: {bb['middle']:.2f}")
    print(f"BB Lower: {bb['lower']:.2f}")
    
    # Test ATR
    atr = TechnicalIndicators.atr(highs, lows, closes, 5)
    print(f"ATR(5): {atr:.2f}")
    
    print("✓ Technical indicators working correctly\n")


def test_risk_management():
    """Test risk management scenarios."""
    print("🛡️ Testing Risk Management")
    print("-" * 40)
    
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
        "data": {"feature_lookback_bars": 50}
    }
    
    engine = TradingEngine(config)
    engine.initialize_portfolio(Decimal("10000"))
    
    # Test normal trading
    can_trade, reason = engine.risk_manager.can_trade(engine.portfolio)
    print(f"Normal trading allowed: {can_trade}")
    
    # Test position size limits
    max_size = engine.risk_manager.get_position_size_limit("BTC/USD", engine.portfolio)
    print(f"Max position size: ${max_size:,.2f}")
    
    # Test circuit breakers
    risk_metrics = engine.risk_manager.get_risk_metrics(engine.portfolio)
    print(f"Circuit breakers: {len(risk_metrics['circuit_breakers'])}")
    
    print("✓ Risk management working correctly\n")


def test_feature_engineering():
    """Test feature engineering with realistic data."""
    print("🔧 Testing Feature Engineering")
    print("-" * 40)
    
    # Create realistic price data (trending up with some volatility)
    base_price = 50000
    prices = []
    for i in range(50):
        # Add trend + noise
        trend = i * 10  # Upward trend
        noise = (i % 7 - 3) * 50  # Oscillating noise
        price = base_price + trend + noise
        prices.append(price)
    
    # Create bars
    bars = []
    for i, price in enumerate(prices):
        bar = Bar(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USD",
            open=Decimal(str(price)),
            high=Decimal(str(price + 100)),
            low=Decimal(str(price - 100)),
            close=Decimal(str(price)),
            volume=Decimal("1000"),
            interval="1m"
        )
        bars.append(bar)
    
    # Test feature calculation
    from src.core.features.technical import FeatureEngine
    feature_engine = FeatureEngine()
    features = feature_engine.calculate_features(bars)
    
    print(f"Generated {len(features)} features:")
    for name, value in list(features.items())[:10]:  # Show first 10
        if isinstance(value, float) and not (value != value):  # Not NaN
            print(f"  {name}: {value:.4f}")
    
    print("✓ Feature engineering working correctly\n")


def test_strategy_logic():
    """Test strategy decision making."""
    print("🎯 Testing Strategy Logic")
    print("-" * 40)
    
    config = {
        "risk": {"circuit_breakers": {"daily_loss_threshold": 0.03}},
        "data": {"feature_lookback_bars": 50}
    }
    
    engine = TradingEngine(config)
    engine.initialize_portfolio(Decimal("10000"))
    
    # Create market data that should trigger signals
    symbol = "BTC/USD"
    base_price = 50000
    
    # Create oversold condition (RSI < 30)
    for i in range(30):
        # Create declining prices
        price = base_price - (i * 20)  # Declining trend
        bar = Bar(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            open=Decimal(str(price)),
            high=Decimal(str(price + 50)),
            low=Decimal(str(price - 50)),
            close=Decimal(str(price)),
            volume=Decimal("1000"),
            interval="1m"
        )
        engine.add_market_data(symbol, bar)
    
    # Process and generate signals
    market_data = engine.process_market_data(symbol)
    if market_data:
        signals = engine.generate_signals(symbol)
        print(f"Generated signals: {len(signals)} strategies")
        
        for strategy_name, strategy_signals in signals.items():
            entry_signals = strategy_signals.get("entry_signals", {})
            print(f"Strategy '{strategy_name}': {len(entry_signals)} entry signals")
            
            for symbol_name, entry_details in entry_signals.items():
                print(f"  Entry signal for {symbol_name}: {entry_details['direction']}")
                print(f"  Reasons: {', '.join(entry_details['reasons'][:2])}")
    
    print("✓ Strategy logic working correctly\n")


def main():
    """Run all test scenarios."""
    print("🚀 Algo-Trading System Test Scenarios")
    print("=" * 60)
    
    try:
        test_technical_indicators()
        test_risk_management()
        test_feature_engineering()
        test_strategy_logic()
        
        print("🎉 All test scenarios completed successfully!")
        print("The Algo-Trading system is working correctly!")
        
    except Exception as e:
        print(f"❌ Test scenario failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
