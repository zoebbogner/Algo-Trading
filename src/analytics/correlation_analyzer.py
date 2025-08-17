"""
Advanced Cryptocurrency Correlation Analysis System

This module provides sophisticated correlation analysis between cryptocurrencies:
- Bitcoin dominance impact analysis
- Cross-asset correlation matrices
- Real-time correlation tracking
- Correlation-based trading signals
- Market regime detection through correlations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timezone, timedelta
import asyncio
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
import os

# Import from correct path
try:
    from ..data_models.market import Bar, MarketData
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.data_models.market import Bar, MarketData

try:
    from ..adapters.data.enhanced_manager import EnhancedDataManager
except ImportError:
    # Fallback for direct execution
    from src.adapters.data.enhanced_manager import EnhancedDataManager


@dataclass
class CorrelationMetrics:
    """Correlation metrics for a pair of assets"""
    symbol1: str
    symbol2: str
    pearson_correlation: float
    spearman_correlation: float
    rolling_correlation: float
    correlation_strength: str
    correlation_direction: str
    confidence_interval: Tuple[float, float]
    p_value: float
    sample_size: int
    last_updated: datetime


@dataclass
class BitcoinImpactAnalysis:
    """Analysis of Bitcoin's impact on other cryptocurrencies"""
    target_symbol: str
    bitcoin_correlation: float
    bitcoin_beta: float
    bitcoin_alpha: float
    impact_strength: str
    lag_correlation: Dict[int, float]  # Days lag -> correlation
    regime_dependent_correlation: Dict[str, float]  # Market regime -> correlation
    last_updated: datetime


class CryptocurrencyCorrelationAnalyzer:
    """
    Advanced cryptocurrency correlation analysis system
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_manager = None
        self.correlation_cache = {}
        self.correlation_history = {}
        self.bitcoin_impact_cache = {}
        
        # Correlation analysis parameters
        self.correlation_window = config.get('correlation_window', 30)  # days
        self.min_correlation_samples = config.get('min_correlation_samples', 20)
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        self.update_frequency = config.get('update_frequency', 3600)  # seconds
        
        # Bitcoin analysis parameters
        self.bitcoin_symbol = config.get('bitcoin_symbol', 'BTC/USDT')
        self.bitcoin_impact_threshold = config.get('bitcoin_impact_threshold', 0.6)
        self.max_lag_days = config.get('max_lag_days', 7)
        
        # Market regime parameters
        self.regime_thresholds = config.get('regime_thresholds', {
            'bull_market': 0.1,      # 10%+ daily return
            'bear_market': -0.1,     # -10%+ daily return
            'sideways': 0.05         # ±5% daily return
        })
        
        print(f"🔗 Cryptocurrency Correlation Analyzer initialized")
        print(f"   Correlation window: {self.correlation_window} days")
        print(f"   Bitcoin symbol: {self.bitcoin_symbol}")
        print(f"   Update frequency: {self.update_frequency} seconds")
    
    async def initialize(self, data_manager: EnhancedDataManager):
        """Initialize the correlation analyzer with data manager"""
        self.data_manager = data_manager
        print(f"✅ Correlation analyzer initialized with data manager")
    
    async def analyze_all_correlations(self, symbols: List[str] = None) -> Dict[str, CorrelationMetrics]:
        """
        Analyze correlations between all cryptocurrency pairs
        """
        if not self.data_manager:
            raise RuntimeError("Data manager not initialized")
        
        if symbols is None:
            symbols = self.data_manager.symbols[:50]  # Limit to first 50 for performance
        
        print(f"🔗 Analyzing correlations for {len(symbols)} cryptocurrencies...")
        
        correlations = {}
        correlation_matrix = np.zeros((len(symbols), len(symbols)))
        
        # Get historical data for all symbols
        historical_data = {}
        for symbol in symbols:
            try:
                data = await self.data_manager.get_historical_data(
                    symbol, limit=self.correlation_window * 24  # 24 data points per day
                )
                if data and len(data) >= self.min_correlation_samples:
                    historical_data[symbol] = data
                else:
                    print(f"⚠️ Insufficient data for {symbol}: {len(data) if data else 0} bars")
            except Exception as e:
                print(f"❌ Failed to get data for {symbol}: {e}")
                continue
        
        if len(historical_data) < 2:
            print("❌ Insufficient data for correlation analysis")
            return {}
        
        # Calculate correlation matrix
        symbols_with_data = list(historical_data.keys())
        print(f"📊 Calculating correlation matrix for {len(symbols_with_data)} symbols...")
        
        for i, symbol1 in enumerate(symbols_with_data):
            for j, symbol2 in enumerate(symbols_with_data):
                if i <= j:  # Only calculate upper triangle
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                    else:
                        correlation = await self._calculate_correlation(
                            historical_data[symbol1], historical_data[symbol2]
                        )
                        correlation_matrix[i, j] = correlation
                        correlation_matrix[j, i] = correlation
                        
                        # Store correlation metrics
                        if correlation is not None:
                            pair_key = f"{symbol1}_{symbol2}"
                            correlations[pair_key] = self._create_correlation_metrics(
                                symbol1, symbol2, correlation, historical_data[symbol1], historical_data[symbol2]
                            )
        
        # Store correlation matrix
        self.correlation_matrix = pd.DataFrame(
            correlation_matrix, 
            index=symbols_with_data, 
            columns=symbols_with_data
        )
        
        # Cache correlations
        self.correlation_cache = correlations
        self.correlation_history[datetime.now(timezone.utc)] = self.correlation_matrix.copy()
        
        print(f"✅ Correlation analysis completed for {len(correlations)} pairs")
        return correlations
    
    async def analyze_bitcoin_impact(self, target_symbols: List[str] = None) -> Dict[str, BitcoinImpactAnalysis]:
        """
        Analyze Bitcoin's impact on other cryptocurrencies
        """
        if not self.data_manager:
            raise RuntimeError("Data manager not initialized")
        
        if target_symbols is None:
            target_symbols = [s for s in self.data_manager.symbols if s != self.bitcoin_symbol][:30]
        
        print(f"📈 Analyzing Bitcoin impact on {len(target_symbols)} cryptocurrencies...")
        
        bitcoin_impacts = {}
        
        # Get Bitcoin data
        bitcoin_data = await self.data_manager.get_historical_data(
            self.bitcoin_symbol, limit=self.correlation_window * 24
        )
        
        if not bitcoin_data or len(bitcoin_data) < self.min_correlation_samples:
            print("❌ Insufficient Bitcoin data for impact analysis")
            return {}
        
        # Calculate Bitcoin returns
        bitcoin_returns = self._calculate_returns(bitcoin_data)
        
        for symbol in target_symbols:
            try:
                print(f"   Analyzing {symbol} vs Bitcoin...")
                
                # Get target symbol data
                target_data = await self.data_manager.get_historical_data(
                    symbol, limit=self.correlation_window * 24
                )
                
                if not target_data or len(target_data) < self.min_correlation_samples:
                    print(f"   ⚠️ Insufficient data for {symbol}")
                    continue
                
                # Calculate target returns
                target_returns = self._calculate_returns(target_data)
                
                # Align data lengths
                min_length = min(len(bitcoin_returns), len(target_returns))
                if min_length < self.min_correlation_samples:
                    continue
                
                bitcoin_aligned = bitcoin_returns[-min_length:]
                target_aligned = target_returns[-min_length:]
                
                # Calculate correlation and beta
                correlation = np.corrcoef(bitcoin_aligned, target_aligned)[0, 1]
                
                if not np.isnan(correlation):
                    # Calculate beta (market sensitivity)
                    covariance = np.cov(bitcoin_aligned, target_aligned)[0, 1]
                    bitcoin_variance = np.var(bitcoin_aligned)
                    beta = covariance / bitcoin_variance if bitcoin_variance > 0 else 0
                    
                    # Calculate alpha (excess return)
                    target_mean = np.mean(target_aligned)
                    bitcoin_mean = np.mean(bitcoin_aligned)
                    alpha = target_mean - (beta * bitcoin_mean)
                    
                    # Calculate lag correlations
                    lag_correlations = self._calculate_lag_correlations(
                        bitcoin_aligned, target_aligned
                    )
                    
                    # Calculate regime-dependent correlations
                    regime_correlations = self._calculate_regime_correlations(
                        bitcoin_aligned, target_aligned
                    )
                    
                    # Create impact analysis
                    impact_analysis = BitcoinImpactAnalysis(
                        target_symbol=symbol,
                        bitcoin_correlation=correlation,
                        bitcoin_beta=beta,
                        bitcoin_alpha=alpha,
                        impact_strength=self._classify_impact_strength(correlation),
                        lag_correlation=lag_correlations,
                        regime_dependent_correlation=regime_correlations,
                        last_updated=datetime.now(timezone.utc)
                    )
                    
                    bitcoin_impacts[symbol] = impact_analysis
                    
                    print(f"   ✅ {symbol}: Correlation={correlation:.3f}, Beta={beta:.3f}")
                else:
                    print(f"   ⚠️ {symbol}: Invalid correlation")
                    
            except Exception as e:
                print(f"   ❌ Error analyzing {symbol}: {e}")
                continue
        
        # Cache results
        self.bitcoin_impact_cache = bitcoin_impacts
        
        print(f"✅ Bitcoin impact analysis completed for {len(bitcoin_impacts)} symbols")
        return bitcoin_impacts
    
    def _calculate_correlation(self, data1: List[Bar], data2: List[Bar]) -> Optional[float]:
        """Calculate correlation between two time series"""
        try:
            if len(data1) != len(data2) or len(data1) < self.min_correlation_samples:
                return None
            
            # Extract closing prices
            prices1 = [float(bar.close) for bar in data1]
            prices2 = [float(bar.close) for bar in data2]
            
            # Calculate returns
            returns1 = self._calculate_returns_from_prices(prices1)
            returns2 = self._calculate_returns_from_prices(prices2)
            
            # Align lengths
            min_length = min(len(returns1), len(returns2))
            if min_length < self.min_correlation_samples:
                return None
            
            returns1_aligned = returns1[-min_length:]
            returns2_aligned = returns2[-min_length:]
            
            # Calculate Pearson correlation
            correlation = np.corrcoef(returns1_aligned, returns2_aligned)[0, 1]
            
            return correlation if not np.isnan(correlation) else None
            
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            return None
    
    def _calculate_returns(self, data: List[Bar]) -> List[float]:
        """Calculate returns from bar data"""
        if len(data) < 2:
            return []
        
        returns = []
        for i in range(1, len(data)):
            current_price = float(data[i].close)
            previous_price = float(data[i-1].close)
            
            if previous_price > 0:
                return_val = (current_price - previous_price) / previous_price
                returns.append(return_val)
            else:
                returns.append(0.0)
        
        return returns
    
    def _calculate_returns_from_prices(self, prices: List[float]) -> List[float]:
        """Calculate returns from price list"""
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                return_val = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(return_val)
            else:
                returns.append(0.0)
        
        return returns
    
    def _calculate_lag_correlations(self, bitcoin_returns: List[float], 
                                  target_returns: List[float]) -> Dict[int, float]:
        """Calculate correlations with different time lags"""
        lag_correlations = {}
        
        for lag in range(1, min(self.max_lag_days + 1, len(bitcoin_returns) // 2)):
            if len(bitcoin_returns) > lag and len(target_returns) > lag:
                # Bitcoin leads target
                bitcoin_lead = bitcoin_returns[:-lag]
                target_lag = target_returns[lag:]
                
                if len(bitcoin_lead) >= self.min_correlation_samples and len(target_lag) >= self.min_correlation_samples:
                    min_length = min(len(bitcoin_lead), len(target_lag))
                    correlation = np.corrcoef(
                        bitcoin_lead[:min_length], target_lag[:min_length]
                    )[0, 1]
                    
                    if not np.isnan(correlation):
                        lag_correlations[lag] = correlation
        
        return lag_correlations
    
    def _calculate_regime_correlations(self, bitcoin_returns: List[float], 
                                     target_returns: List[float]) -> Dict[str, float]:
        """Calculate correlations in different market regimes"""
        regime_correlations = {}
        
        # Define market regimes based on Bitcoin returns
        bitcoin_returns_array = np.array(bitcoin_returns)
        
        # Bull market (positive returns)
        bull_mask = bitcoin_returns_array > self.regime_thresholds['bull_market']
        if np.sum(bull_mask) >= self.min_correlation_samples:
            bull_correlation = np.corrcoef(
                bitcoin_returns_array[bull_mask], 
                np.array(target_returns)[bull_mask]
            )[0, 1]
            if not np.isnan(bull_correlation):
                regime_correlations['bull_market'] = bull_correlation
        
        # Bear market (negative returns)
        bear_mask = bitcoin_returns_array < self.regime_thresholds['bear_market']
        if np.sum(bear_mask) >= self.min_correlation_samples:
            bear_correlation = np.corrcoef(
                bitcoin_returns_array[bear_mask], 
                np.array(target_returns)[bear_mask]
            )[0, 1]
            if not np.isnan(bear_correlation):
                regime_correlations['bear_market'] = bear_correlation
        
        # Sideways market (low volatility)
        sideways_mask = np.abs(bitcoin_returns_array) <= self.regime_thresholds['sideways']
        if np.sum(sideways_mask) >= self.min_correlation_samples:
            sideways_correlation = np.corrcoef(
                bitcoin_returns_array[sideways_mask], 
                np.array(target_returns)[sideways_mask]
            )[0, 1]
            if not np.isnan(sideways_correlation):
                regime_correlations['sideways'] = sideways_correlation
        
        return regime_correlations
    
    def _create_correlation_metrics(self, symbol1: str, symbol2: str, correlation: float,
                                  data1: List[Bar], data2: List[Bar]) -> CorrelationMetrics:
        """Create comprehensive correlation metrics"""
        
        # Calculate additional statistics
        returns1 = self._calculate_returns(data1)
        returns2 = self._calculate_returns(data2)
        
        # Align lengths
        min_length = min(len(returns1), len(returns2))
        returns1_aligned = returns1[-min_length:]
        returns2_aligned = returns2[-min_length:]
        
        # Spearman correlation (rank correlation)
        spearman_corr = spearmanr(returns1_aligned, returns2_aligned)[0]
        
        # Rolling correlation (recent correlation)
        if min_length >= 10:
            rolling_window = min(10, min_length // 2)
            rolling_corr = np.corrcoef(
                returns1_aligned[-rolling_window:], 
                returns2_aligned[-rolling_window:]
            )[0, 1]
        else:
            rolling_corr = correlation
        
        # Confidence interval (simplified)
        n = min_length
        if n > 3:
            z_score = 1.96  # 95% confidence
            se = np.sqrt((1 - correlation**2) / (n - 2))
            ci_lower = correlation - z_score * se
            ci_upper = correlation + z_score * se
            confidence_interval = (max(-1, ci_lower), min(1, ci_upper))
        else:
            confidence_interval = (-1, 1)
        
        # P-value (simplified)
        if n > 3:
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        else:
            p_value = 1.0
        
        return CorrelationMetrics(
            symbol1=symbol1,
            symbol2=symbol2,
            pearson_correlation=correlation,
            spearman_correlation=spearman_corr,
            rolling_correlation=rolling_corr,
            correlation_strength=self._classify_correlation_strength(correlation),
            correlation_direction="positive" if correlation > 0 else "negative",
            confidence_interval=confidence_interval,
            p_value=p_value,
            sample_size=n,
            last_updated=datetime.now(timezone.utc)
        )
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"
    
    def _classify_impact_strength(self, correlation: float) -> str:
        """Classify Bitcoin impact strength"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "dominant"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "independent"
    
    def get_highly_correlated_pairs(self, threshold: float = None) -> List[Tuple[str, str, float]]:
        """Get pairs with high correlation above threshold"""
        if threshold is None:
            threshold = self.correlation_threshold
        
        if not hasattr(self, 'correlation_matrix'):
            return []
        
        high_correlations = []
        
        for i in range(len(self.correlation_matrix)):
            for j in range(i + 1, len(self.correlation_matrix)):
                correlation = self.correlation_matrix.iloc[i, j]
                if abs(correlation) >= threshold:
                    symbol1 = self.correlation_matrix.index[i]
                    symbol2 = self.correlation_matrix.columns[j]
                    high_correlations.append((symbol1, symbol2, correlation))
        
        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        return high_correlations
    
    def get_bitcoin_dependent_coins(self, threshold: float = None) -> List[Tuple[str, float, float]]:
        """Get coins that are highly dependent on Bitcoin"""
        if threshold is None:
            threshold = self.bitcoin_impact_threshold
        
        bitcoin_dependent = []
        
        for symbol, impact in self.bitcoin_impact_cache.items():
            if abs(impact.bitcoin_correlation) >= threshold:
                bitcoin_dependent.append((
                    symbol, 
                    impact.bitcoin_correlation, 
                    impact.bitcoin_beta
                ))
        
        # Sort by correlation strength
        bitcoin_dependent.sort(key=lambda x: abs(x[1]), reverse=True)
        return bitcoin_dependent
    
    def get_correlation_trading_signals(self) -> Dict[str, Dict]:
        """Generate trading signals based on correlation analysis"""
        signals = {}
        
        if not hasattr(self, 'correlation_matrix') or self.correlation_matrix.empty:
            return signals
        
        # Analyze Bitcoin impact signals
        for symbol, impact in self.bitcoin_impact_cache.items():
            signal = {
                'type': 'correlation_signal',
                'symbol': symbol,
                'bitcoin_correlation': impact.bitcoin_correlation,
                'bitcoin_beta': impact.bitcoin_beta,
                'signal_strength': 'strong' if abs(impact.bitcoin_correlation) > 0.8 else 'moderate',
                'trading_implication': self._get_trading_implication(impact),
                'timestamp': datetime.now(timezone.utc)
            }
            signals[symbol] = signal
        
        # Analyze correlation breakdown signals
        high_corr_pairs = self.get_highly_correlated_pairs(0.8)
        for symbol1, symbol2, correlation in high_corr_pairs[:10]:  # Top 10
            signal = {
                'type': 'correlation_pair_signal',
                'symbol1': symbol1,
                'symbol2': symbol2,
                'correlation': correlation,
                'signal_strength': 'strong',
                'trading_implication': 'Consider pairs trading or correlation-based strategies',
                'timestamp': datetime.now(timezone.utc)
            }
            signals[f"{symbol1}_{symbol2}_pair"] = signal
        
        return signals
    
    def _get_trading_implication(self, impact: BitcoinImpactAnalysis) -> str:
        """Get trading implications based on Bitcoin impact"""
        if abs(impact.bitcoin_correlation) > 0.8:
            if impact.bitcoin_correlation > 0:
                return "High Bitcoin correlation - trade in same direction as Bitcoin"
            else:
                return "High inverse Bitcoin correlation - trade opposite to Bitcoin"
        elif abs(impact.bitcoin_correlation) > 0.6:
            if impact.bitcoin_correlation > 0:
                return "Moderate Bitcoin correlation - Bitcoin movements provide guidance"
            else:
                return "Moderate inverse Bitcoin correlation - consider contrarian positions"
        elif abs(impact.bitcoin_correlation) > 0.4:
            return "Low Bitcoin correlation - trade based on individual coin fundamentals"
        else:
            return "Independent of Bitcoin - focus on coin-specific analysis"
    
    def generate_correlation_report(self) -> str:
        """Generate comprehensive correlation analysis report"""
        if not hasattr(self, 'correlation_matrix') or self.correlation_matrix.empty:
            return "No correlation data available"
        
        report = []
        report.append("🔗 CRYPTOCURRENCY CORRELATION ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append(f"Analysis Window: {self.correlation_window} days")
        report.append(f"Total Symbols: {len(self.correlation_matrix)}")
        report.append("")
        
        # Bitcoin impact summary
        if self.bitcoin_impact_cache:
            report.append("📈 BITCOIN IMPACT ANALYSIS:")
            report.append("-" * 30)
            
            # Top Bitcoin-dependent coins
            bitcoin_dependent = self.get_bitcoin_dependent_coins(0.6)
            report.append(f"High Bitcoin Correlation (≥0.6): {len(bitcoin_dependent)} coins")
            
            for symbol, corr, beta in bitcoin_dependent[:5]:
                report.append(f"  {symbol}: Correlation={corr:.3f}, Beta={beta:.3f}")
            
            # Independent coins
            independent_coins = [s for s, i in self.bitcoin_impact_cache.items() 
                               if abs(i.bitcoin_correlation) < 0.3]
            report.append(f"Independent of Bitcoin (<0.3): {len(independent_coins)} coins")
            report.append("")
        
        # High correlation pairs
        high_corr_pairs = self.get_highly_correlated_pairs(0.8)
        if high_corr_pairs:
            report.append("🔗 HIGHLY CORRELATED PAIRS (≥0.8):")
            report.append("-" * 40)
            
            for symbol1, symbol2, correlation in high_corr_pairs[:10]:
                report.append(f"  {symbol1} ↔ {symbol2}: {correlation:.3f}")
            report.append("")
        
        # Market regime analysis
        if self.bitcoin_impact_cache:
            report.append("🌊 MARKET REGIME CORRELATIONS:")
            report.append("-" * 35)
            
            for symbol, impact in list(self.bitcoin_impact_cache.items())[:5]:
                report.append(f"  {symbol}:")
                for regime, corr in impact.regime_dependent_correlation.items():
                    report.append(f"    {regime}: {corr:.3f}")
                report.append("")
        
        # Trading recommendations
        report.append("💡 TRADING RECOMMENDATIONS:")
        report.append("-" * 30)
        
        signals = self.get_correlation_trading_signals()
        for signal_id, signal in list(signals.items())[:5]:
            if signal['type'] == 'correlation_signal':
                report.append(f"  {signal['symbol']}: {signal['trading_implication']}")
        
        return "\n".join(report)
    
    def save_correlation_data(self, filepath: str):
        """Save correlation data to files"""
        import json
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save correlation matrix
        if hasattr(self, 'correlation_matrix') and not self.correlation_matrix.empty:
            self.correlation_matrix.to_csv(f"{filepath}_correlation_matrix.csv")
            print(f"💾 Saved correlation matrix to {filepath}_correlation_matrix.csv")
        
        # Save Bitcoin impact analysis
        if self.bitcoin_impact_cache:
            bitcoin_impact_data = {}
            for symbol, impact in self.bitcoin_impact_cache.items():
                bitcoin_impact_data[symbol] = {
                    'bitcoin_correlation': impact.bitcoin_correlation,
                    'bitcoin_beta': impact.bitcoin_beta,
                    'bitcoin_alpha': impact.bitcoin_alpha,
                    'impact_strength': impact.impact_strength,
                    'lag_correlation': impact.lag_correlation,
                    'regime_dependent_correlation': impact.regime_dependent_correlation,
                    'last_updated': impact.last_updated.isoformat()
                }
            
            with open(f"{filepath}_bitcoin_impact.json", 'w') as f:
                json.dump(bitcoin_impact_data, f, indent=2)
            print(f"💾 Saved Bitcoin impact analysis to {filepath}_bitcoin_impact.json")
        
        # Save correlation cache
        if self.correlation_cache:
            correlation_data = {}
            for pair_key, metrics in self.correlation_cache.items():
                correlation_data[pair_key] = {
                    'symbol1': metrics.symbol1,
                    'symbol2': metrics.symbol2,
                    'pearson_correlation': metrics.pearson_correlation,
                    'spearman_correlation': metrics.spearman_correlation,
                    'rolling_correlation': metrics.rolling_correlation,
                    'correlation_strength': metrics.correlation_strength,
                    'correlation_direction': metrics.correlation_direction,
                    'confidence_interval': metrics.confidence_interval,
                    'p_value': metrics.p_value,
                    'sample_size': metrics.sample_size,
                    'last_updated': metrics.last_updated.isoformat()
                }
            
            with open(f"{filepath}_correlations.json", 'w') as f:
                json.dump(correlation_data, f, indent=2)
            print(f"💾 Saved correlation data to {filepath}_correlations.json")
    
    async def close(self):
        """Close the correlation analyzer"""
        if self.data_manager:
            await self.data_manager.close()
        print("🔗 Correlation analyzer closed")
