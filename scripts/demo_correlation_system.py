#!/usr/bin/env python3
"""
Quick Demo: Enhanced Cryptocurrency Correlation Analysis

This script demonstrates the key features of the correlation analysis system
in a quick, visual way.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config.crypto_symbols import (
    get_all_symbols, 
    get_symbols_by_category, 
    CryptoCategory,
    get_diversification_recommendations,
    get_correlation_trading_pairs
)


def demo_crypto_coverage():
    """Demonstrate the extensive cryptocurrency coverage"""
    print("🚀 CRYPTOCURRENCY COVERAGE DEMONSTRATION")
    print("=" * 60)
    
    # Show total symbols
    all_symbols = get_all_symbols()
    print(f"📊 Total Cryptocurrencies: {len(all_symbols)}")
    print()
    
    # Show categories
    categories = [
        CryptoCategory.MAJOR,
        CryptoCategory.DEFI,
        CryptoCategory.LAYER1,
        CryptoCategory.GAMING,
        CryptoCategory.AI_BIGDATA,
        CryptoCategory.PRIVACY,
        CryptoCategory.EXCHANGE,
        CryptoCategory.STABLECOIN,
        CryptoCategory.MEME,
        CryptoCategory.EMERGING
    ]
    
    print("🏷️ CATEGORY BREAKDOWN:")
    print("-" * 30)
    
    for category in categories:
        category_symbols = get_symbols_by_category(category)
        print(f"   {category.value.upper()}: {len(category_symbols)} coins")
        
        # Show first 3 symbols in each category
        if category_symbols:
            symbols_str = ", ".join(category_symbols[:3])
            if len(category_symbols) > 3:
                symbols_str += f" (+{len(category_symbols)-3} more)"
            print(f"      Examples: {symbols_str}")
        print()
    
    # Show some specific examples
    print("💎 SAMPLE CRYPTOCURRENCIES:")
    print("-" * 30)
    
    sample_symbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
        'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT'
    ]
    
    for symbol in sample_symbols:
        print(f"   • {symbol}")
    
    print()


def demo_correlation_insights():
    """Demonstrate correlation insights and analysis"""
    print("🔗 CORRELATION ANALYSIS INSIGHTS")
    print("=" * 50)
    
    # Bitcoin correlation examples
    print("📈 BITCOIN CORRELATION EXAMPLES:")
    print("-" * 35)
    
    high_correlation = [
        ('ETH/USDT', '0.85', 'Strong Bitcoin follower'),
        ('MATIC/USDT', '0.80', 'Scaling solution, high correlation'),
        ('ADA/USDT', '0.70', 'Layer 1, market sentiment driven'),
        ('BNB/USDT', '0.75', 'Exchange token, follows market')
    ]
    
    medium_correlation = [
        ('SOL/USDT', '0.65', 'Independent but influenced'),
        ('XRP/USDT', '0.60', 'Banking focus, moderate correlation'),
        ('DOT/USDT', '0.75', 'Infrastructure, follows trends')
    ]
    
    low_correlation = [
        ('XMR/USDT', '0.45', 'Privacy focus, independent'),
        ('USDC/USDT', '0.05', 'Stablecoin, no correlation'),
        ('DAI/USDT', '0.10', 'Algorithmic stablecoin')
    ]
    
    print("   🔴 HIGH CORRELATION (≥0.7):")
    for symbol, corr, desc in high_correlation:
        print(f"      {symbol}: {corr} - {desc}")
    
    print("\n   🟡 MEDIUM CORRELATION (0.4-0.7):")
    for symbol, corr, desc in medium_correlation:
        print(f"      {symbol}: {corr} - {desc}")
    
    print("\n   🟢 LOW CORRELATION (≤0.4):")
    for symbol, corr, desc in low_correlation:
        print(f"      {symbol}: {corr} - {desc}")
    
    print()


def demo_trading_applications():
    """Demonstrate trading applications"""
    print("🎯 TRADING APPLICATIONS")
    print("=" * 35)
    
    # Portfolio diversification
    print("🔀 PORTFOLIO DIVERSIFICATION:")
    print("-" * 30)
    
    diversification_symbols = get_diversification_recommendations()
    print(f"   Diversification candidates: {len(diversification_symbols)} symbols")
    
    for i, symbol in enumerate(diversification_symbols[:5]):
        print(f"      {i+1}. {symbol}")
    
    print("\n   💡 These symbols have low Bitcoin correlation and can help diversify your portfolio")
    
    # Pairs trading
    print("\n🔄 PAIRS TRADING OPPORTUNITIES:")
    print("-" * 35)
    
    trading_pairs = get_correlation_trading_pairs()
    high_corr_pairs = [p for p in trading_pairs if p[2] == "high_correlation"]
    low_corr_pairs = [p for p in trading_pairs if p[2] == "low_correlation"]
    
    print(f"   High correlation pairs: {len(high_corr_pairs)}")
    for i, (symbol1, symbol2, pair_type) in enumerate(high_corr_pairs[:3]):
        print(f"      {i+1}. {symbol1} ↔ {symbol2}")
    
    print(f"\n   Low correlation pairs: {len(low_corr_pairs)}")
    for i, (symbol1, symbol2, pair_type) in enumerate(low_corr_pairs[:3]):
        print(f"      {i+1}. {symbol1} ↔ {symbol2}")
    
    print()


def demo_market_regimes():
    """Demonstrate market regime analysis"""
    print("🌊 MARKET REGIME ANALYSIS")
    print("=" * 35)
    
    print("📊 How correlations change in different market conditions:")
    print()
    
    regimes = [
        ("🐂 BULL MARKET", "Bitcoin > +10% daily", "Correlations increase, most coins follow Bitcoin"),
        ("🐻 BEAR MARKET", "Bitcoin < -10% daily", "Correlations increase, panic selling affects all"),
        ("➡️ SIDEWAYS", "Bitcoin ±5% daily", "Correlations decrease, individual coin fundamentals matter")
    ]
    
    for regime, condition, description in regimes:
        print(f"   {regime}")
        print(f"      Condition: {condition}")
        print(f"      Impact: {description}")
        print()
    
    print("💡 This helps adjust trading strategies based on market conditions")


def demo_usage_examples():
    """Demonstrate usage examples"""
    print("🔧 USAGE EXAMPLES")
    print("=" * 25)
    
    print("📝 BASIC USAGE:")
    print("-" * 20)
    
    basic_code = '''
from src.analytics.correlation_analyzer import CryptocurrencyCorrelationAnalyzer
from src.adapters.data.enhanced_manager import EnhancedDataManager

# Initialize system
data_manager = EnhancedDataManager({'symbols': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']})
analyzer = CryptocurrencyCorrelationAnalyzer({'correlation_window': 30})

# Analyze correlations
correlations = await analyzer.analyze_all_correlations()
bitcoin_impacts = await analyzer.analyze_bitcoin_impact(['ETH/USDT', 'ADA/USDT'])

# Get trading signals
signals = analyzer.get_correlation_trading_signals()
'''
    
    print(basic_code)
    
    print("📊 ADVANCED USAGE:")
    print("-" * 20)
    
    advanced_code = '''
# Get diversification recommendations
diversification_symbols = get_diversification_recommendations()

# Analyze specific categories
defi_symbols = get_symbols_by_category(CryptoCategory.DEFI)
gaming_symbols = get_symbols_by_category(CryptoCategory.GAMING)

# Build low-correlation portfolio
portfolio_symbols = ['BTC/USDT'] + diversification_symbols[:4]
'''
    
    print(advanced_code)


def main():
    """Main demonstration function"""
    print("🎉 ENHANCED CRYPTOCURRENCY CORRELATION ANALYSIS SYSTEM")
    print("=" * 70)
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run demonstrations
    demo_crypto_coverage()
    demo_correlation_insights()
    demo_trading_applications()
    demo_market_regimes()
    demo_usage_examples()
    
    print("🎯 NEXT STEPS:")
    print("=" * 20)
    print("1. Run the full test: python scripts/test_correlation_analysis.py")
    print("2. Customize configuration in src/config/strategy_config.py")
    print("3. Add your own cryptocurrency symbols")
    print("4. Integrate with your trading strategy")
    print()
    
    print("🚀 READY TO ANALYZE CRYPTOCURRENCY CORRELATIONS!")
    print("=" * 60)


if __name__ == "__main__":
    main()
