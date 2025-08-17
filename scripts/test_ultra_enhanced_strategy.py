#!/usr/bin/env python3
"""
Test Ultra-Enhanced Momentum Strategy

This script tests the ultra-enhanced momentum strategy with:
- Machine Learning signal confirmation
- Advanced risk modeling (VaR, Kelly Criterion)
- Portfolio optimization algorithms
- Out-of-sample validation for unseen data
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
from src.core.strategy.momentum import UltraEnhancedMomentumStrategy
from src.core.risk.manager import RiskManager
from src.adapters.data.manager import DataManager


async def test_ultra_enhanced_strategy():
    """Test the ultra-enhanced momentum strategy"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("🚀 Testing Ultra-Enhanced Momentum Strategy with ML & Advanced Risk Management")
    print("=" * 80)
    
    # Data configuration
    data_config = {
        'adapters': {
            'binance': {
                'enabled': True,
                'api_key': '',
                'api_secret': '',
                'sandbox': True
            }
        },
        'symbols': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'],
        'timeframe': '1h',
        'limit': 500
    }
    
    # Risk configuration
    risk_config = {
        'max_position_size': 0.20,
        'max_portfolio_exposure': 0.80,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04,
        'max_drawdown_pct': 0.15,
        'circuit_breaker_threshold': 0.10
    }
    
    # ML configuration
    ml_config = {
        'confidence_threshold': 0.7,
        'lookback_periods': [5, 10, 20],
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
    }
    
    # Advanced risk modeling configuration
    advanced_risk_config = {
        'var_confidence_level': 0.95,
        'var_time_horizon': 1,
        'kelly_max_fraction': 0.25,
        'kelly_risk_free_rate': 0.02,
        'stress_scenarios': {
            'market_crash': -0.20,
            'volatility_spike': 0.50,
            'correlation_breakdown': 0.8,
            'liquidity_crisis': -0.15
        }
    }
    
    # Portfolio optimization configuration
    optimization_config = {
        'optimization_method': 'efficient_frontier',
        'risk_free_rate': 0.02,
        'max_iterations': 1000,
        'min_weight': 0.01,
        'max_weight': 0.40,
        'rebalancing_threshold': 0.05,
        'rebalancing_frequency': 'monthly'
    }
    
    # Ultra-enhanced strategy configuration
    strategy_config = {
        'name': 'Ultra-Enhanced Momentum Strategy',
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
        'take_profit_pct': 0.03,       # Take profit at 3%
        'trailing_stop_pct': 0.01,     # Trailing stop at 1%
        
        # Multi-timeframe parameters
        'short_ma_period': 5,
        'medium_ma_period': 13,
        'long_ma_period': 34,
        
        # Advanced parameters
        'atr_period': 14,
        'volatility_threshold': 0.02,
        'correlation_threshold': 0.7,
        'max_correlation_exposure': 0.3,
        
        # Feature flags
        'use_ml_confirmation': True,
        'use_advanced_risk': True,
        'use_portfolio_optimization': True,
        
        # Advanced module configurations
        'ml_config': ml_config,
        'risk_config': advanced_risk_config,
        'optimization_config': optimization_config
    }
    
    try:
        # Initialize components
        print("🔧 Initializing components...")
        
        data_manager = DataManager(data_config)
        await data_manager.connect()
        
        risk_manager = RiskManager(risk_config)
        strategy = UltraEnhancedMomentumStrategy(strategy_config)
        
        # Initialize backtesting engine
        backtest_engine = BacktestEngine(
            data_manager=data_manager,
            strategy=strategy,
            risk_manager=risk_manager,
            initial_capital=Decimal("10000")
        )
        
        print("✅ Components initialized successfully!")
        
        # Train ML models if enough data is available
        print("\n🤖 Training ML models...")
        if hasattr(strategy, 'train_ml_models'):
            # We need to run a short backtest first to gather historical data
            print("   Running initial backtest to gather training data...")
            
            # Run a short backtest for training data
            training_results = await backtest_engine.run_backtest(
                start_date=datetime.now(timezone.utc) - timedelta(days=30),
                end_date=datetime.now(timezone.utc) - timedelta(days=7)
            )
            
            if training_results:
                print("   Training data gathered, training ML models...")
                ml_trained = strategy.train_ml_models()
                if ml_trained:
                    print("   ✅ ML models trained successfully!")
                else:
                    print("   ⚠️ ML model training failed, continuing without ML")
            else:
                print("   ⚠️ No training data available, continuing without ML")
        
        # Run main backtest
        print("\n🚀 Running main backtest with ultra-enhanced strategy...")
        
        main_results = await backtest_engine.run_backtest(
            start_date=datetime.now(timezone.utc) - timedelta(days=7),
            end_date=datetime.now(timezone.utc)
        )
        
        if main_results:
            print("\n🎉 Ultra-Enhanced Strategy Backtest Completed!")
            print("=" * 60)
            
            # Display key results
            print(f"📊 Total Return: {main_results.get('total_return', 0):.2%}")
            print(f"📈 Annualized Return: {main_results.get('annualized_return', 0):.2%}")
            print(f"📉 Maximum Drawdown: {main_results.get('max_drawdown', 0):.2%}")
            print(f"🎯 Total Trades: {main_results.get('total_trades', 0)}")
            print(f"✅ Winning Trades: {main_results.get('winning_trades', 0)}")
            print(f"❌ Losing Trades: {main_results.get('losing_trades', 0)}")
            print(f"📊 Win Rate: {main_results.get('win_rate', 0):.1%}")
            print(f"💰 Risk-Reward Ratio: {main_results.get('risk_reward_ratio', 0):.2f}")
            print(f"📈 Sharpe Ratio: {main_results.get('sharpe_ratio', 0):.2f}")
            
            # Display advanced metrics if available
            if 'advanced_metrics' in main_results:
                advanced = main_results['advanced_metrics']
                print(f"\n🤖 Advanced Features Status:")
                print(f"   ML Confirmation: {'✅' if advanced.get('ml_models_available') else '❌'}")
                print(f"   Advanced Risk: {'✅' if advanced.get('risk_modeling_available') else '❌'}")
                print(f"   Portfolio Optimization: {'✅' if advanced.get('portfolio_optimization_available') else '❌'}")
            
            # Display portfolio optimization results if available
            if 'portfolio_optimization' in main_results:
                opt = main_results['portfolio_optimization']
                print(f"\n🎯 Portfolio Optimization Results:")
                print(f"   Method: {opt.get('method', 'Unknown')}")
                if 'optimal_portfolio' in opt:
                    optimal = opt['optimal_portfolio']
                    print(f"   Expected Return: {optimal.get('expected_return', 0):.4f}")
                    print(f"   Volatility: {optimal.get('volatility', 0):.4f}")
                    print(f"   Sharpe Ratio: {optimal.get('sharpe_ratio', 0):.3f}")
            
            # Display risk metrics if available
            if 'risk_metrics' in main_results:
                risk = main_results['risk_metrics']
                print(f"\n📊 Advanced Risk Metrics:")
                if 'portfolio_var' in risk:
                    var = risk['portfolio_var']
                    print(f"   Portfolio VaR: {var.get('portfolio_var_percentage', 0):.2%}")
                if 'stress_test' in risk:
                    stress = risk['stress_test']
                    print(f"   Portfolio Robustness: {stress.get('portfolio_robustness', {}).get('robustness_score', 0):.3f}")
            
            print("\n" + "=" * 60)
            print("🎯 Strategy Performance Summary:")
            
            # Performance evaluation
            total_return = main_results.get('total_return', 0)
            max_drawdown = main_results.get('max_drawdown', 0)
            win_rate = main_results.get('win_rate', 0)
            sharpe_ratio = main_results.get('sharpe_ratio', 0)
            
            if total_return > 0.05 and max_drawdown < 0.10 and win_rate > 0.5:
                print("   🏆 EXCELLENT: High returns, low drawdown, good win rate!")
            elif total_return > 0.02 and max_drawdown < 0.15:
                print("   ✅ GOOD: Positive returns with acceptable risk")
            elif total_return > 0:
                print("   ⚠️ MODERATE: Small positive returns")
            else:
                print("   ❌ NEEDS IMPROVEMENT: Negative returns")
            
            if sharpe_ratio > 1.0:
                print("   📈 Strong risk-adjusted returns")
            elif sharpe_ratio > 0.5:
                print("   📊 Acceptable risk-adjusted returns")
            else:
                print("   📉 Poor risk-adjusted returns")
            
        else:
            print("❌ Backtest failed to complete")
        
        # Cleanup
        await data_manager.disconnect()
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_ultra_enhanced_strategy())
