#!/usr/bin/env python3
"""Demo script to show the trading system working."""

import sys
from pathlib import Path
from decimal import Decimal

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.trading.portfolio import Portfolio
from src.trading.risk import RiskManager
from src.trading.strategies import MomentumStrategy, MeanReversionStrategy
from src.trading.engine import TradingEngine


def demo_trading_system():
    """Demonstrate the trading system."""
    print("🚀 Trading System Demo")
    print("=" * 50)
    
    # Initialize portfolio
    print("💰 Initializing portfolio...")
    portfolio = Portfolio(
        initial_capital=100000.0,
        symbols=["BTC", "ETH", "ADA"]
    )
    
    print(f"   Initial capital: ${portfolio.initial_capital:,.2f}")
    print(f"   Available symbols: {', '.join(portfolio.symbols)}")
    print(f"   Current cash: ${portfolio.cash:,.2f}")
    
    # Initialize risk manager
    print("\n🛡️ Initializing risk manager...")
    risk_manager = RiskManager(
        max_position_size=0.1,      # Max 10% in single position
        max_portfolio_risk=0.02,    # Max 2% portfolio risk
        stop_loss_pct=0.05          # 5% stop loss
    )
    
    print(f"   Max position size: {risk_manager.max_position_size:.1%}")
    print(f"   Max portfolio risk: {risk_manager.max_portfolio_risk:.1%}")
    print(f"   Stop loss: {risk_manager.stop_loss_pct:.1%}")
    
    # Initialize strategies
    print("\n📊 Initializing trading strategies...")
    momentum_config = {
        "name": "BTC_Momentum",
        "parameters": {
            "fast_period": 10,
            "slow_period": 30,
            "rsi_period": 14
        }
    }
    
    mean_reversion_config = {
        "name": "ETH_MeanReversion",
        "parameters": {
            "lookback_period": 20,
            "std_dev_threshold": 2.0,
            "rsi_period": 14
        }
    }
    
    momentum_strategy = MomentumStrategy(momentum_config)
    mean_reversion_strategy = MeanReversionStrategy(mean_reversion_config)
    
    strategies = {
        "momentum": momentum_strategy,
        "mean_reversion": mean_reversion_strategy
    }
    
    print(f"   Loaded strategies: {', '.join(strategies.keys())}")
    
    # Initialize trading engine
    print("\n⚙️ Initializing trading engine...")
    trading_engine = TradingEngine(portfolio, risk_manager, strategies)
    
    # Simulate some trading activity
    print("\n📈 Simulating trading activity...")
    
    # Add BTC position
    print("   Buying BTC...")
    portfolio.add_position("BTC", Decimal("0.5"), Decimal("50000.0"), "LONG")
    
    # Add ETH position
    print("   Buying ETH...")
    portfolio.add_position("ETH", Decimal("2.0"), Decimal("3000.0"), "LONG")
    
    # Show portfolio status
    print("\n📊 Portfolio Status:")
    current_prices = {"BTC": 52000.0, "ETH": 3200.0, "ADA": 0.5}
    
    summary = portfolio.get_summary(current_prices)
    print(f"   Total value: ${summary['current_value']:,.2f}")
    print(f"   Cash: ${summary['cash']:,.2f}")
    print(f"   Positions: {summary['positions_count']}")
    print(f"   Unrealized P&L: ${summary['unrealized_pnl']:,.2f}")
    print(f"   Total return: {summary['return_pct']:.2f}%")
    print(f"   Drawdown: {summary['drawdown_pct']:.2f}%")
    
    # Show individual positions
    print("\n🔍 Position Details:")
    for symbol, position in portfolio.positions.items():
        current_price = current_prices.get(symbol, 0)
        position_value = portfolio.get_position_value(symbol, current_price)
        print(f"   {symbol}: {float(position.quantity)} @ ${float(position.average_cost):,.2f}")
        print(f"      Current price: ${current_price:,.2f}")
        print(f"      Position value: ${position_value:,.2f}")
        print(f"      Side: {position.side}")
    
    # Generate trading signals
    print("\n🎯 Generating trading signals...")
    market_data = {"BTC": {"price": 52000}, "ETH": {"price": 3200}}  # Placeholder data
    signals = trading_engine.generate_signals(market_data)
    
    print(f"   Generated {len(signals)} signals")
    for signal in signals:
        print(f"      {signal['action']} {signal['symbol']} {float(signal['quantity'])} @ ${float(signal['price']):,.2f}")
    
    # Risk management check
    print("\n⚠️ Risk management check...")
    violations = risk_manager.check_risk_limits(portfolio)
    
    if violations:
        print(f"   Found {len(violations)} risk violations:")
        for violation in violations:
            print(f"      {violation['type']} for {violation['symbol']}: {violation['current_ratio']:.1%} > {violation['limit']:.1%}")
    else:
        print("   No risk violations detected")
    
    # Trading engine summary
    print("\n📋 Trading Engine Summary:")
    engine_summary = trading_engine.get_trading_summary()
    print(f"   Total signals: {engine_summary['total_signals']}")
    print(f"   Strategies: {engine_summary['strategies_count']}")
    print(f"   Portfolio value: ${engine_summary['portfolio_value']:,.2f}")
    print(f"   Positions: {engine_summary['positions_count']}")
    print(f"   Total trades: {engine_summary['trades_count']}")
    
    # Risk metrics
    print("\n📊 Risk Metrics:")
    risk_metrics = risk_manager.get_risk_metrics(portfolio, {})
    print(f"   95% VaR: {risk_metrics.var_95:.1%}")
    print(f"   99% VaR: {risk_metrics.var_99:.1%}")
    print(f"   Max drawdown: {risk_metrics.max_drawdown:.1%}")
    print(f"   Volatility: {risk_metrics.volatility:.1%}")
    print(f"   Sharpe ratio: {risk_metrics.sharpe_ratio:.2f}")
    print(f"   Sortino ratio: {risk_metrics.sortino_ratio:.2f}")
    
    print("\n🎉 Trading system demo completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    demo_trading_system()
