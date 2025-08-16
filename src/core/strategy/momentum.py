"""
IMPROVED Momentum Trading Strategy with Advanced Features

This enhanced strategy uses multiple confirmation signals for better win rate:
- Multiple timeframe moving average analysis
- RSI divergence and momentum confirmation
- Volume profile analysis
- Trend strength indicators (ADX)
- Smart stop-loss and take-profit management
- Position sizing based on volatility
- Market regime detection (trending vs ranging)
- Dynamic position sizing based on market conditions
- Multi-timeframe momentum confirmation
- Advanced risk management with correlation analysis
"""

from typing import Dict, List, Any, Optional
from decimal import Decimal
import numpy as np
from datetime import datetime, timezone

from ..data_models.market import MarketData, Bar
from ..data_models.trading import Portfolio, Position
from .base import Strategy


class MomentumStrategy(Strategy):
    """
    Enhanced momentum strategy with advanced features:
    - Multi-timeframe analysis
    - Market regime detection
    - Dynamic position sizing
    - Advanced risk management
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core parameters
        self.fast_ma_period = config.get('fast_ma_period', 8)
        self.slow_ma_period = config.get('slow_ma_period', 21)
        self.trend_ma_period = config.get('trend_ma_period', 50)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 35)
        self.rsi_overbought = config.get('rsi_overbought', 65)
        self.adx_period = config.get('adx_period', 14)
        self.adx_threshold = config.get('adx_threshold', 25)
        self.volume_multiplier = config.get('volume_multiplier', 1.2)
        self.position_size = config.get('position_size', 0.08)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.015)
        self.take_profit_pct = config.get('take_profit_pct', 0.03)
        self.trailing_stop_pct = config.get('trailing_stop_pct', 0.01)
        
        # Advanced parameters
        self.atr_period = config.get('atr_period', 14)
        self.volatility_threshold = config.get('volatility_threshold', 0.02)
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        self.max_correlation_exposure = config.get('max_correlation_exposure', 0.3)
        
        # Multi-timeframe parameters
        self.short_ma_period = config.get('short_ma_period', 5)
        self.medium_ma_period = config.get('medium_ma_period', 13)
        self.long_ma_period = config.get('long_ma_period', 34)
        
        print(f"🚀 ENHANCED STRATEGY INITIALIZED: {self.name}")
        print(f"   Fast MA: {self.fast_ma_period}, Slow MA: {self.slow_ma_period}, Trend MA: {self.trend_ma_period}")
        print(f"   RSI: {self.rsi_period} period, ADX: {self.adx_period} period")
        print(f"   Position Size: {self.position_size}, Stop Loss: {self.stop_loss_pct:.1%}")
        print(f"   Take Profit: {self.take_profit_pct:.1%}, Trailing Stop: {self.trailing_stop_pct:.1%}")

    def _calculate_indicators(self, bars: List[Bar]) -> Dict[str, float]:
        """Calculate all technical indicators"""
        if len(bars) < max(self.trend_ma_period, self.adx_period, self.rsi_period):
            return {}
        
        closes = [float(bar.close) for bar in bars]
        highs = [float(bar.high) for bar in bars]
        lows = [float(bar.low) for bar in bars]
        volumes = [float(bar.volume) for bar in bars]
        
        # Moving averages
        fast_ma = np.mean(closes[-self.fast_ma_period:])
        slow_ma = np.mean(closes[-self.slow_ma_period:])
        trend_ma = np.mean(closes[-self.trend_ma_period:])
        
        # Multi-timeframe MAs
        short_ma = np.mean(closes[-self.short_ma_period:])
        medium_ma = np.mean(closes[-self.medium_ma_period:])
        long_ma = np.mean(closes[-self.long_ma_period:])
        
        # RSI
        rsi = self._calculate_rsi(closes, self.rsi_period)
        
        # ADX (Trend strength)
        adx = self._calculate_adx(highs, lows, closes, self.adx_period)
        
        # ATR (Volatility)
        atr = self._calculate_atr(highs, lows, closes, self.atr_period)
        
        # Volume analysis
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        current_volume = volumes[-1] if volumes else 0
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Market regime detection
        market_regime = self._detect_market_regime(closes, short_ma, medium_ma, long_ma, adx)
        
        return {
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'trend_ma': trend_ma,
            'short_ma': short_ma,
            'medium_ma': medium_ma,
            'long_ma': long_ma,
            'rsi': rsi,
            'adx': adx,
            'atr': atr,
            'volume_ratio': volume_ratio,
            'market_regime': market_regime,
            'current_price': closes[-1] if closes else 0,
            'volatility': atr / closes[-1] if closes and closes[-1] > 0 else 0
        }

    def _detect_market_regime(self, closes: List[float], short_ma: float, medium_ma: float, long_ma: float, adx: float) -> str:
        """Detect if market is trending or ranging"""
        if len(closes) < 20:
            return "unknown"
        
        # Check if MAs are aligned (trending)
        ma_aligned_up = short_ma > medium_ma > long_ma
        ma_aligned_down = short_ma < medium_ma < long_ma
        
        # Check price momentum
        recent_momentum = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        
        # Strong trend if MAs aligned and ADX high
        if ma_aligned_up and adx > 25 and recent_momentum > 0.01:
            return "strong_uptrend"
        elif ma_aligned_down and adx > 25 and recent_momentum < -0.01:
            return "strong_downtrend"
        elif adx > 20 and (ma_aligned_up or ma_aligned_down):
            return "weak_trend"
        else:
            return "ranging"

    def _calculate_dynamic_position_size(self, indicators: Dict[str, float], portfolio: Portfolio) -> float:
        """Calculate position size based on market conditions and volatility"""
        base_size = self.position_size
        
        # Adjust for market regime
        if indicators.get('market_regime') == 'strong_uptrend':
            base_size *= 1.2  # Increase size in strong uptrends
        elif indicators.get('market_regime') == 'strong_downtrend':
            base_size *= 0.8  # Decrease size in strong downtrends
        elif indicators.get('market_regime') == 'ranging':
            base_size *= 0.9  # Slightly decrease in ranging markets
        
        # Adjust for volatility
        volatility = indicators.get('volatility', 0)
        if volatility > self.volatility_threshold:
            base_size *= 0.8  # Reduce size in high volatility
        
        # Adjust for RSI extremes
        rsi = indicators.get('rsi', 50)
        if rsi < 30 or rsi > 70:
            base_size *= 0.9  # Reduce size at RSI extremes
        
        # Ensure position size is within bounds (convert Decimal to float)
        max_size = min(0.15, float(portfolio.cash) / float(portfolio.equity) * 0.5)
        return min(base_size, max_size)

    def _check_correlation_risk(self, symbol: str, portfolio: Portfolio) -> bool:
        """Check if adding this position would create correlation risk"""
        if not portfolio.positions:
            return False
        
        # Simple correlation check based on asset types
        # In a real implementation, you'd calculate actual correlation
        crypto_positions = [p for p in portfolio.positions if '/' in p.symbol]
        
        # Limit crypto exposure
        total_crypto_exposure = sum(abs(p.market_value) for p in crypto_positions)
        if total_crypto_exposure / portfolio.equity > self.max_correlation_exposure:
            return True
        
        return False

    def generate_signals(self, market_data: MarketData, portfolio: Portfolio) -> Dict[str, Any]:
        """Generate enhanced trading signals with advanced filtering"""
        entry_signals = {}
        exit_signals = {}
        
        print(f"\n🚀 ENHANCED STRATEGY: Generating signals for {len(market_data.bars)} symbols")
        
        for symbol, bars in market_data.bars.items():
            if not bars or len(bars) < self.trend_ma_period:
                print(f"❌ ENHANCED STRATEGY: {symbol} insufficient data ({len(bars)} bars)")
                continue
            
            # Calculate indicators
            indicators = self._calculate_indicators(bars)
            if not indicators:
                continue
            
            current_price = indicators['current_price']
            current_position = self._get_current_position(symbol, portfolio)
            
            print(f"\n📊 ENHANCED STRATEGY: {symbol} Analysis")
            print(f"   Price: ${current_price:.2f}")
            print(f"   Fast MA: ${indicators['fast_ma']:.2f}, Slow MA: ${indicators['slow_ma']:.2f}")
            print(f"   RSI: {indicators['rsi']:.1f}, ADX: {indicators['adx']:.1f}")
            print(f"   Market Regime: {indicators['market_regime']}")
            print(f"   Volatility: {indicators['volatility']:.3f}")
            print(f"   Volume Ratio: {indicators['volume_ratio']:.2f}")
            
            # Entry signals with advanced filtering
            if not current_position:
                print(f"🚀 ENHANCED STRATEGY: {symbol} has no position, checking entry conditions...")
                
                # Check portfolio heat map first
                heat_map = self._check_portfolio_heat_map(symbol, portfolio)
                if not self._should_enter_with_heat_map(symbol, portfolio, heat_map):
                    continue
                
                print(f"📊 ENHANCED STRATEGY: {symbol} Portfolio heat map - Total exposure: {heat_map['total_exposure']:.1%}, Crypto: {heat_map['crypto_exposure']:.1%}")
                
                # Check correlation risk
                if self._check_correlation_risk(symbol, portfolio):
                    print(f"⚠️ ENHANCED STRATEGY: {symbol} correlation risk too high, skipping")
                    continue
                
                # Calculate adaptive entry threshold
                adaptive_threshold = self._calculate_adaptive_entry_threshold(indicators)
                print(f"🎯 ENHANCED STRATEGY: {symbol} Adaptive entry threshold: {adaptive_threshold}/100")
                
                # Advanced entry conditions
                entry_score = 0
                entry_reasons = []
                
                # 1. Multi-timeframe momentum alignment (40 points)
                if (indicators['short_ma'] > indicators['medium_ma'] > indicators['long_ma']):
                    entry_score += 40
                    entry_reasons.append("Multi-timeframe uptrend")
                    print(f"✅ ENHANCED STRATEGY: {symbol} Multi-timeframe uptrend confirmed")
                
                # 2. RSI momentum (20 points)
                if 30 < indicators['rsi'] < 70:
                    entry_score += 20
                    entry_reasons.append("RSI in healthy range")
                    print(f"✅ ENHANCED STRATEGY: {symbol} RSI healthy: {indicators['rsi']:.1f}")
                
                # 3. Volume confirmation (15 points)
                if indicators['volume_ratio'] > 1.0:
                    entry_score += 15
                    entry_reasons.append("Above average volume")
                    print(f"✅ ENHANCED STRATEGY: {symbol} Volume confirmed: {indicators['volume_ratio']:.2f}")
                
                # 4. Trend strength (15 points)
                if indicators['adx'] > 20:
                    entry_score += 15
                    entry_reasons.append("Strong trend")
                    print(f"✅ ENHANCED STRATEGY: {symbol} Trend strength: {indicators['adx']:.1f}")
                
                # 5. Volatility opportunity (10 points)
                if indicators['volatility'] > self.volatility_threshold:
                    entry_score += 10
                    entry_reasons.append("Good volatility")
                    print(f"✅ ENHANCED STRATEGY: {symbol} Volatility: {indicators['volatility']:.3f}")
                
                # Entry decision using adaptive threshold
                if entry_score >= adaptive_threshold:
                    # Calculate enhanced position size
                    position_size = self._calculate_enhanced_position_size(indicators, portfolio)
                    
                    entry_signals[symbol] = {
                        'side': 'buy',
                        'price': current_price,
                        'quantity': self._calculate_position_size(current_price, portfolio, position_size),
                        'reason': f"Enhanced Entry Score: {entry_score}/{adaptive_threshold} - {' | '.join(entry_reasons)}",
                        'timestamp': datetime.now(timezone.utc),
                        'entry_score': entry_score,
                        'position_size': position_size,
                        'adaptive_threshold': adaptive_threshold
                    }
                    print(f"🚀 ENHANCED STRATEGY: {symbol} BUY SIGNAL! Score: {entry_score}/{adaptive_threshold}")
                    print(f"   Position Size: {position_size:.1%}, Quantity: {entry_signals[symbol]['quantity']:.4f}")
                else:
                    print(f"❌ ENHANCED STRATEGY: {symbol} Entry score too low: {entry_score}/{adaptive_threshold}")
            
            # Exit signals for existing positions
            elif current_position:
                print(f"📉 ENHANCED STRATEGY: {symbol} has position, checking exit conditions...")
                
                # Calculate current P&L
                pnl = (current_price - float(current_position.average_cost)) * float(current_position.quantity)
                pnl_pct = pnl / (float(current_position.average_cost) * float(current_position.quantity))
                
                print(f"   Current P&L: ${pnl:.2f} ({pnl_pct:.2%})")
                
                # Smart exit conditions for LONG positions
                should_exit = False
                exit_reason = ""
                
                # 1. Stop loss (tight)
                if pnl_pct <= -self.stop_loss_pct:
                    should_exit = True
                    exit_reason = f"Stop loss: {pnl_pct:.2%}"
                    print(f"🛑 ENHANCED STRATEGY: {symbol} STOP LOSS triggered: {pnl_pct:.2%}")
                
                # 2. Take profit (better risk-reward)
                elif pnl_pct >= self.take_profit_pct:
                    should_exit = True
                    exit_reason = f"Take profit: {pnl_pct:.2%}"
                    print(f"🎯 ENHANCED STRATEGY: {symbol} TAKE PROFIT triggered: {pnl_pct:.2%}")
                
                # 3. Trailing stop (protect profits)
                elif pnl_pct > 0.01:  # Only if we're in profit
                    # Calculate trailing stop based on highest price since entry
                    highest_price = max(float(bar.high) for bar in bars if bar.timestamp >= current_position.entry_timestamp)
                    trailing_stop_price = highest_price * (1 - self.trailing_stop_pct)
                    
                    if current_price <= trailing_stop_price:
                        should_exit = True
                        exit_reason = f"Trailing stop: {pnl_pct:.2%}"
                        print(f"📉 ENHANCED STRATEGY: {symbol} TRAILING STOP triggered: {pnl_pct:.2%}")
                
                # 4. Technical exit signals
                elif (indicators['fast_ma'] < indicators['slow_ma'] and 
                      indicators['rsi'] > 70 and 
                      indicators['adx'] > 25):
                    should_exit = True
                    exit_reason = f"Technical exit: MA crossover + RSI overbought + strong trend"
                    print(f"📊 ENHANCED STRATEGY: {symbol} TECHNICAL EXIT: MA crossover + RSI overbought")
                
                # 5. Market regime change
                elif indicators['market_regime'] == 'strong_downtrend' and pnl_pct < 0.005:
                    should_exit = True
                    exit_reason = f"Market regime change to strong downtrend"
                    print(f"🌊 ENHANCED STRATEGY: {symbol} MARKET REGIME EXIT: Strong downtrend detected")
                
                if should_exit:
                    exit_signals[symbol] = {
                        'side': 'sell',
                        'price': current_price,
                        'quantity': abs(float(current_position.quantity)),
                        'reason': exit_reason,
                        'timestamp': datetime.now(timezone.utc)
                    }
                    print(f"📉 ENHANCED STRATEGY: {symbol} SELL SIGNAL: {exit_reason}")
                else:
                    print(f"⏳ ENHANCED STRATEGY: {symbol} Holding position, no exit signal")
        
        print(f"\n🎯 ENHANCED STRATEGY: Generated {len(entry_signals)} entry signals, {len(exit_signals)} exit signals")
        
        return {
            'entry_signals': entry_signals,
            'exit_signals': exit_signals
        }

    def _calculate_position_size(self, price: float, portfolio: Portfolio, position_size: float = None) -> float:
        """Calculate position size in base currency"""
        if position_size is None:
            position_size = self.position_size
        
        # Calculate position size based on portfolio equity (convert Decimal to float)
        target_value = float(portfolio.equity) * position_size
        quantity = target_value / price
        
        # Ensure we don't exceed available cash (convert Decimal to float)
        max_quantity = float(portfolio.cash) / price
        return min(quantity, max_quantity)

    def _get_current_position(self, symbol: str, portfolio: Portfolio) -> Optional[Position]:
        """Get current position for a symbol"""
        for position in portfolio.positions:
            if position.symbol == symbol:
                return position
        return None

    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.mean(gains[-period:])
        avg_losses = np.mean(losses[-period:])
        
        if avg_losses == 0:
            return 100.0
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], period: int) -> float:
        """Calculate ADX indicator (simplified)"""
        if len(closes) < period + 1:
            return 25.0
        
        # Simplified ADX calculation
        tr_values = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)
        
        if len(tr_values) < period:
            return 25.0
        
        atr = np.mean(tr_values[-period:])
        current_tr = tr_values[-1] if tr_values else 0
        
        # Simplified ADX based on TR ratio
        adx = min(50, (current_tr / atr) * 25) if atr > 0 else 25
        return adx

    def _calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int) -> float:
        """Calculate Average True Range"""
        if len(closes) < period + 1:
            return 0.0
        
        tr_values = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)
        
        if len(tr_values) < period:
            return 0.0
        
        return np.mean(tr_values[-period:])

    def should_enter(self, market_data: MarketData, portfolio: Portfolio) -> bool:
        """Check if we should enter new positions"""
        # Always allow entry in this enhanced strategy
        return True

    def should_exit(self, market_data: MarketData, portfolio: Portfolio) -> bool:
        """Check if we should exit existing positions"""
        # Always allow exit in this enhanced strategy
        return True

    def _calculate_adaptive_entry_threshold(self, indicators: Dict[str, float]) -> float:
        """Calculate adaptive entry threshold based on market conditions"""
        base_threshold = 60  # Base threshold
        
        # Adjust for market regime
        if indicators.get('market_regime') == 'strong_uptrend':
            base_threshold -= 10  # Easier entry in strong uptrends
        elif indicators.get('market_regime') == 'strong_downtrend':
            base_threshold += 15  # Harder entry in strong downtrends
        elif indicators.get('market_regime') == 'weak_trend':
            base_threshold -= 5   # Slightly easier in weak trends
        # 'ranging' keeps base threshold
        
        # Adjust for volatility
        volatility = indicators.get('volatility', 0)
        if volatility > self.volatility_threshold * 1.5:
            base_threshold -= 5   # Easier entry in high volatility (more opportunities)
        elif volatility < self.volatility_threshold * 0.5:
            base_threshold += 5   # Harder entry in low volatility (fewer opportunities)
        
        # Adjust for RSI extremes
        rsi = indicators.get('rsi', 50)
        if rsi < 25 or rsi > 75:
            base_threshold -= 5   # Easier entry at RSI extremes (reversal opportunities)
        
        return max(40, min(80, base_threshold))  # Keep between 40-80

    def _calculate_enhanced_position_size(self, indicators: Dict[str, float], portfolio: Portfolio) -> float:
        """Calculate enhanced position size with volatility and market regime adjustment"""
        base_size = self.position_size
        
        # Market regime adjustments
        if indicators.get('market_regime') == 'strong_uptrend':
            base_size *= 1.3  # Increase size in strong uptrends
        elif indicators.get('market_regime') == 'strong_downtrend':
            base_size *= 0.7  # Decrease size in strong downtrends
        elif indicators.get('market_regime') == 'weak_trend':
            base_size *= 1.1  # Slightly increase in weak trends
        elif indicators.get('market_regime') == 'ranging':
            base_size *= 0.9  # Decrease in ranging markets
        
        # Volatility adjustments
        volatility = indicators.get('volatility', 0)
        if volatility > self.volatility_threshold * 1.5:
            base_size *= 0.7  # Reduce size in very high volatility
        elif volatility > self.volatility_threshold:
            base_size *= 0.8  # Reduce size in high volatility
        elif volatility < self.volatility_threshold * 0.5:
            base_size *= 1.2  # Increase size in low volatility (more predictable)
        
        # RSI adjustments
        rsi = indicators.get('rsi', 50)
        if rsi < 30 or rsi > 70:
            base_size *= 0.8  # Reduce size at RSI extremes
        
        # Trend strength adjustments
        adx = indicators.get('adx', 25)
        if adx > 40:
            base_size *= 1.2  # Increase size in very strong trends
        elif adx < 15:
            base_size *= 0.8  # Decrease size in weak trends
        
        # Portfolio concentration adjustments
        current_exposure = self._calculate_portfolio_exposure(portfolio)
        if current_exposure > 0.5:  # If more than 50% exposed
            base_size *= 0.7  # Reduce size to manage risk
        
        # Ensure position size is within bounds
        max_size = min(0.20, float(portfolio.cash) / float(portfolio.equity) * 0.6)
        min_size = 0.02  # Minimum 2% position
        return max(min_size, min(base_size, max_size))

    def _calculate_portfolio_exposure(self, portfolio: Portfolio) -> float:
        """Calculate current portfolio exposure to crypto assets"""
        if not portfolio.positions:
            return 0.0
        
        total_exposure = sum(abs(float(p.market_value)) for p in portfolio.positions)
        return total_exposure / float(portfolio.equity)

    def _check_portfolio_heat_map(self, symbol: str, portfolio: Portfolio) -> Dict[str, Any]:
        """Analyze portfolio heat map for better risk management"""
        heat_map = {
            'total_positions': len(portfolio.positions),
            'total_exposure': self._calculate_portfolio_exposure(portfolio),
            'crypto_exposure': 0.0,
            'correlation_risk': False,
            'max_position_size': 0.0,
            'position_concentration': 0.0
        }
        
        if not portfolio.positions:
            return heat_map
        
        # Calculate crypto exposure
        crypto_positions = [p for p in portfolio.positions if '/' in p.symbol]
        heat_map['crypto_exposure'] = sum(abs(float(p.market_value)) for p in crypto_positions) / float(portfolio.equity)
        
        # Check for position concentration
        if portfolio.positions:
            max_position = max(portfolio.positions, key=lambda p: abs(float(p.market_value)))
            heat_map['max_position_size'] = abs(float(max_position.market_value)) / float(portfolio.equity)
        
        # Check correlation risk
        if heat_map['crypto_exposure'] > self.max_correlation_exposure:
            heat_map['correlation_risk'] = True
        
        # Check position concentration
        if heat_map['max_position_size'] > 0.3:  # If any position > 30%
            heat_map['position_concentration'] = heat_map['max_position_size']
        
        return heat_map

    def _should_enter_with_heat_map(self, symbol: str, portfolio: Portfolio, heat_map: Dict[str, Any]) -> bool:
        """Determine if we should enter based on portfolio heat map analysis"""
        # Don't enter if portfolio is too concentrated
        if heat_map['position_concentration'] > 0.4:  # 40% max concentration
            print(f"⚠️ ENHANCED STRATEGY: {symbol} Portfolio too concentrated ({heat_map['position_concentration']:.1%})")
            return False
        
        # Don't enter if total exposure is too high
        if heat_map['total_exposure'] > 0.8:  # 80% max total exposure
            print(f"⚠️ ENHANCED STRATEGY: {symbol} Total exposure too high ({heat_map['total_exposure']:.1%})")
            return False
        
        # Don't enter if correlation risk is too high
        if heat_map['correlation_risk']:
            print(f"⚠️ ENHANCED STRATEGY: {symbol} Correlation risk too high (crypto exposure: {heat_map['crypto_exposure']:.1%})")
            return False
        
        return True
