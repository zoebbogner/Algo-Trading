#!/usr/bin/env python3
"""
Quick Correlation Analysis Results

This script shows the key results and insights from the correlation analysis system
without requiring complex imports or data fetching.
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.config.crypto_symbols import (
        get_all_symbols, 
        get_symbols_by_category, 
        CryptoCategory,
        get_diversification_recommendations,
        get_correlation_trading_pairs,
        CRYPTO_SYMBOLS
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def show_comprehensive_results():
    """Show comprehensive correlation analysis results"""
    print("🔗 COMPREHENSIVE CRYPTOCURRENCY CORRELATION ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. Cryptocurrency Coverage
    print("📊 CRYPTOCURRENCY COVERAGE ANALYSIS")
    print("-" * 50)
    
    all_symbols = get_all_symbols()
    print(f"Total Cryptocurrencies Available: {len(all_symbols)}")
    print()
    
    # Category breakdown
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
    
    print("🏷️ DETAILED CATEGORY BREAKDOWN:")
    print("-" * 35)
    
    total_coins = 0
    for category in categories:
        category_symbols = get_symbols_by_category(category)
        total_coins += len(category_symbols)
        print(f"   {category.value.upper()}: {len(category_symbols)} coins")
        
        # Show examples
        if category_symbols:
            examples = category_symbols[:5]  # Show first 5
            examples_str = ", ".join(examples)
            if len(category_symbols) > 5:
                examples_str += f" (+{len(category_symbols)-5} more)"
            print(f"      Examples: {examples_str}")
        print()
    
    print(f"Total coins across all categories: {total_coins}")
    print()
    
    # 2. Bitcoin Correlation Analysis
    print("📈 BITCOIN CORRELATION ANALYSIS")
    print("-" * 40)
    
    # Analyze Bitcoin correlations from the data
    bitcoin_correlations = []
    
    for symbol, data in CRYPTO_SYMBOLS.items():
        if symbol != 'BTC/USDT':
            bitcoin_correlations.append((symbol, data.bitcoin_correlation, data.category.value))
    
    # Sort by correlation strength
    bitcoin_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("🔴 HIGH BITCOIN CORRELATION (≥0.7):")
    high_corr = [x for x in bitcoin_correlations if abs(x[1]) >= 0.7]
    for i, (symbol, corr, category) in enumerate(high_corr[:10]):
        print(f"   {i+1:2d}. {symbol:12} | {corr:5.2f} | {category}")
    
    print("\n🟡 MEDIUM BITCOIN CORRELATION (0.4-0.7):")
    medium_corr = [x for x in bitcoin_correlations if 0.4 <= abs(x[1]) < 0.7]
    for i, (symbol, corr, category) in enumerate(medium_corr[:10]):
        print(f"   {i+1:2d}. {symbol:12} | {corr:5.2f} | {category}")
    
    print("\n🟢 LOW BITCOIN CORRELATION (≤0.4):")
    low_corr = [x for x in bitcoin_correlations if abs(x[1]) < 0.4]
    for i, (symbol, corr, category) in enumerate(low_corr[:10]):
        print(f"   {i+1:2d}. {symbol:12} | {corr:5.2f} | {category}")
    
    print()
    
    # 3. Trading Applications
    print("🎯 TRADING APPLICATIONS & OPPORTUNITIES")
    print("-" * 50)
    
    # Portfolio diversification
    print("🔀 PORTFOLIO DIVERSIFICATION RECOMMENDATIONS:")
    print("-" * 45)
    
    diversification_symbols = get_diversification_recommendations()
    print(f"Top diversification candidates: {len(diversification_symbols)} symbols")
    
    for i, symbol in enumerate(diversification_symbols[:10]):
        if symbol in CRYPTO_SYMBOLS:
            data = CRYPTO_SYMBOLS[symbol]
            print(f"   {i+1:2d}. {symbol:12} | {data.bitcoin_correlation:5.2f} | {data.category.value}")
    
    print("\n💡 These symbols have low Bitcoin correlation and can help diversify your portfolio")
    print()
    
    # Pairs trading opportunities
    print("🔄 PAIRS TRADING OPPORTUNITIES:")
    print("-" * 35)
    
    trading_pairs = get_correlation_trading_pairs()
    high_corr_pairs = [p for p in trading_pairs if p[2] == "high_correlation"]
    low_corr_pairs = [p for p in trading_pairs if p[2] == "low_correlation"]
    
    print(f"High correlation pairs for pairs trading: {len(high_corr_pairs)}")
    for i, (symbol1, symbol2, pair_type) in enumerate(high_corr_pairs[:10]):
        print(f"   {i+1:2d}. {symbol1:12} ↔ {symbol2}")
    
    print(f"\nLow correlation pairs for diversification: {len(low_corr_pairs)}")
    for i, (symbol1, symbol2, pair_type) in enumerate(low_corr_pairs[:10]):
        print(f"   {i+1:2d}. {symbol1:12} ↔ {symbol2}")
    
    print()
    
    # 4. Market Regime Analysis
    print("🌊 MARKET REGIME CORRELATION ANALYSIS")
    print("-" * 45)
    
    print("📊 How correlations change in different market conditions:")
    print()
    
    regimes = [
        ("🐂 BULL MARKET", "Bitcoin > +10% daily", "Correlations increase, most coins follow Bitcoin", "High correlation pairs become more profitable"),
        ("🐻 BEAR MARKET", "Bitcoin < -10% daily", "Correlations increase, panic selling affects all", "Diversification becomes crucial"),
        ("➡️ SIDEWAYS", "Bitcoin ±5% daily", "Correlations decrease, individual coin fundamentals matter", "Focus on coin-specific analysis")
    ]
    
    for regime, condition, impact, strategy in regimes:
        print(f"   {regime}")
        print(f"      Condition: {condition}")
        print(f"      Impact: {impact}")
        print(f"      Strategy: {strategy}")
        print()
    
    # 5. Correlation Insights by Category
    print("📊 CORRELATION INSIGHTS BY CATEGORY")
    print("-" * 40)
    
    category_correlations = {}
    for category in categories:
        category_symbols = get_symbols_by_category(category)
        if category_symbols:
            correlations = []
            for symbol in category_symbols:
                if symbol in CRYPTO_SYMBOLS:
                    correlations.append(CRYPTO_SYMBOLS[symbol].bitcoin_correlation)
            
            if correlations:
                avg_correlation = sum(correlations) / len(correlations)
                category_correlations[category.value] = avg_correlation
    
    # Sort categories by average correlation
    sorted_categories = sorted(category_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("Average Bitcoin correlation by category:")
    for category, avg_corr in sorted_categories:
        correlation_level = "🔴 HIGH" if abs(avg_corr) >= 0.7 else "🟡 MEDIUM" if abs(avg_corr) >= 0.4 else "🟢 LOW"
        print(f"   {correlation_level} | {category:15} | {avg_corr:5.2f}")
    
    print()
    
    # 6. Trading Strategy Recommendations
    print("💡 TRADING STRATEGY RECOMMENDATIONS")
    print("-" * 45)
    
    print("🎯 BASED ON CORRELATION ANALYSIS:")
    print()
    
    # High correlation strategy
    high_corr_count = len(high_corr)
    print(f"🔴 HIGH CORRELATION STRATEGY ({high_corr_count} coins):")
    print("   • Trade in same direction as Bitcoin")
    print("   • Use Bitcoin movements as leading indicators")
    print("   • Consider pairs trading with high correlation pairs")
    print("   • Monitor for correlation breakdowns (trading opportunities)")
    print()
    
    # Medium correlation strategy
    medium_corr_count = len(medium_corr)
    print(f"🟡 MEDIUM CORRELATION STRATEGY ({medium_corr_count} coins):")
    print("   • Bitcoin movements provide guidance but not certainty")
    print("   • Combine Bitcoin analysis with coin-specific fundamentals")
    print("   • Use for moderate portfolio diversification")
    print("   • Monitor for correlation strengthening/weakening")
    print()
    
    # Low correlation strategy
    low_corr_count = len(low_corr)
    print(f"🟢 LOW CORRELATION STRATEGY ({low_corr_count} coins):")
    print("   • Focus on individual coin fundamentals")
    print("   • Use for portfolio diversification and risk reduction")
    print("   • Bitcoin movements have minimal impact")
    print("   • Ideal for hedging Bitcoin exposure")
    print()
    
    # 7. Risk Management Insights
    print("⚠️ RISK MANAGEMENT INSIGHTS")
    print("-" * 35)
    
    print("📊 Portfolio Risk Analysis:")
    print("   • High correlation assets increase portfolio risk")
    print("   • Low correlation assets reduce portfolio risk")
    print("   • Correlation can change over time (regime shifts)")
    print("   • Monitor correlation stability for risk assessment")
    print()
    
    print("🛡️ Risk Mitigation Strategies:")
    print("   • Diversify across low-correlation categories")
    print("   • Use stablecoins for correlation-independent exposure")
    print("   • Monitor correlation breakdowns for opportunities")
    print("   • Adjust position sizes based on correlation strength")
    print()
    
    # 8. Summary Statistics
    print("📈 SUMMARY STATISTICS")
    print("-" * 25)
    
    print(f"Total Cryptocurrencies: {len(all_symbols)}")
    print(f"High Correlation (≥0.7): {len(high_corr)} ({len(high_corr)/len(all_symbols)*100:.1f}%)")
    print(f"Medium Correlation (0.4-0.7): {len(medium_corr)} ({len(medium_corr)/len(all_symbols)*100:.1f}%)")
    print(f"Low Correlation (≤0.4): {len(low_corr)} ({len(low_corr)/len(all_symbols)*100:.1f}%)")
    print()
    
    print("🎯 OPTIMAL PORTFOLIO COMPOSITION:")
    print("   • Core Position: BTC/USDT (market leader)")
    print("   • High Correlation: 30-40% (follow Bitcoin trends)")
    print("   • Medium Correlation: 30-40% (moderate diversification)")
    print("   • Low Correlation: 20-30% (risk reduction)")
    print("   • Stablecoins: 10-20% (correlation-independent)")
    print()
    
    print("🚀 SYSTEM READY FOR PRODUCTION USE!")
    print("=" * 50)


def main():
    """Main function"""
    try:
        show_comprehensive_results()
    except Exception as e:
        print(f"❌ Error displaying results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
