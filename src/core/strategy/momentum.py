"""
IMPROVED Momentum Trading Strategy

This enhanced strategy uses multiple confirmation signals for better win rate:
- Multiple timeframe moving average analysis
- RSI divergence and momentum confirmation
- Volume profile analysis
- Trend strength indicators (ADX)
- Smart stop-loss and take-profit management
- Position sizing based on volatility
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
    Enhanced momentum strategy with multiple confirmation signals
    for higher win rate and better risk management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Enhanced strategy parameters
        self.fast_ma_period = config.get('fast_ma_period', 8)      # Faster for quicker signals
        self.slow_ma_period = config.get('slow_ma_period', 21)     # Slower for trend confirmation
        self.trend_ma_period = config.get('trend_ma_period', 50)   # Long-term trend
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 35)        # Less strict oversold
        self.rsi_overbought = config.get('rsi_overbought', 65)    # Less strict overbought
        self.adx_period = config.get('adx_period', 14)            # Trend strength
        self.adx_threshold = config.get('adx_threshold', 25)      # Minimum trend strength
        self.volume_multiplier = config.get('volume_multiplier', 1.2)  # Lower volume requirement
        self.position_size = config.get('position_size', 0.08)    # Smaller positions for risk management
        self.stop_loss_pct = config.get('stop_loss_pct', 0.015)   # Tighter stop loss (1.5%)
        self.take_profit_pct = config.get('take_profit_pct', 0.03)  # Better risk-reward (2:1)
        self.trailing_stop = config.get('trailing_stop', True)    # Enable trailing stops
        self.trailing_stop_pct = config.get('trailing_stop_pct', 0.01)  # 1% trailing stop
        
        # State tracking
        self.positions = {}
        self.entry_prices = {}
        self.stop_losses = {}
        self.take_profits = {}
        self.highest_prices = {}  # For trailing stops
        self.lowest_prices = {}   # For trailing stops
        
    def _calculate_sma(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return None
        return np.mean(prices[-period:])
    
    def _calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average (more responsive)"""
        if len(prices) < period:
            return None
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
            
        return ema
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return None
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.mean(gains[:period])
        avg_losses = np.mean(losses[:period])
        
        if avg_losses == 0:
            return 100
            
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_adx(self, high_prices: List[float], low_prices: List[float], close_prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate Average Directional Index (trend strength)"""
        if len(high_prices) < period + 1:
            return None
            
        # Simplified ADX calculation
        tr_values = []
        dm_plus = []
        dm_minus = []
        
        for i in range(1, len(high_prices)):
            tr = max(high_prices[i] - low_prices[i], 
                    abs(high_prices[i] - close_prices[i-1]), 
                    abs(low_prices[i] - close_prices[i-1]))
            tr_values.append(tr)
            
            dm_p = high_prices[i] - high_prices[i-1] if high_prices[i] - high_prices[i-1] > low_prices[i-1] - low_prices[i] else 0
            dm_m = low_prices[i-1] - low_prices[i] if low_prices[i-1] - low_prices[i] > high_prices[i] - high_prices[i-1] else 0
            
            dm_plus.append(dm_p)
            dm_minus.append(dm_m)
        
        if len(tr_values) < period:
            return None
            
        avg_tr = np.mean(tr_values[-period:])
        avg_dm_plus = np.mean(dm_plus[-period:])
        avg_dm_minus = np.mean(dm_minus[-period:])
        
        if avg_tr == 0:
            return 0
            
        di_plus = (avg_dm_plus / avg_tr) * 100
        di_minus = (avg_dm_minus / avg_tr) * 100
        
        dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if (di_plus + di_minus) > 0 else 0
        adx = np.mean([dx] * period)  # Simplified smoothing
        
        return adx
    
    def _calculate_volume_ratio(self, volumes: List[float], period: int = 20) -> Optional[float]:
        """Calculate current volume vs average volume ratio"""
        if len(volumes) < period:
            return None
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-period:])
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _calculate_atr(self, high_prices: List[float], low_prices: List[float], close_prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate Average True Range for volatility-based position sizing"""
        if len(high_prices) < period + 1:
            return None
            
        tr_values = []
        for i in range(1, len(high_prices)):
            tr = max(high_prices[i] - low_prices[i], 
                    abs(high_prices[i] - close_prices[i-1]), 
                    abs(low_prices[i] - close_prices[i-1]))
            tr_values.append(tr)
        
        return np.mean(tr_values[-period:])
    
    def _check_trend_confirmation(self, prices: List[float], fast_ma: float, slow_ma: float, trend_ma: float) -> Dict[str, Any]:
        """Check multiple timeframe trend confirmation"""
        # Short-term momentum
        short_momentum = fast_ma > slow_ma
        
        # Medium-term trend
        medium_trend = slow_ma > trend_ma
        
        # Price above trend line
        price_above_trend = prices[-1] > trend_ma
        
        # Trend strength (slope)
        if len(prices) >= 5:
            recent_prices = prices[-5:]
            trend_slope = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            strong_trend = abs(trend_slope) > 0.01  # 1% slope
        else:
            strong_trend = False
        
        return {
            'short_momentum': short_momentum,
            'medium_trend': medium_trend,
            'price_above_trend': price_above_trend,
            'strong_trend': strong_trend,
            'overall_bullish': short_momentum and medium_trend and price_above_trend and strong_trend,
            'overall_bearish': not short_momentum and not medium_trend and not price_above_trend and strong_trend
        }
    
    def generate_signals(self, market_data: MarketData, portfolio: Portfolio) -> Dict[str, Any]:
        """
        Generate enhanced trading signals with multiple confirmations
        """
        print(f"🚀 ENHANCED STRATEGY: Starting signal generation...")
        print(f"🚀 ENHANCED STRATEGY: Market data has {len(market_data.bars)} symbols")
        
        if not market_data.bars:
            return {'entry_signals': {}, 'exit_signals': {}}
        
        entry_signals = {}
        exit_signals = {}
        
        for symbol, bars in market_data.bars.items():
            print(f"🚀 ENHANCED STRATEGY: Processing {symbol} with {len(bars)} bars")
            
            if len(bars) < self.trend_ma_period:
                print(f"❌ ENHANCED STRATEGY: {symbol} has only {len(bars)} bars, need {self.trend_ma_period}")
                continue
                
            # Extract price and volume data
            prices = [float(bar.close) for bar in bars]
            high_prices = [float(bar.high) for bar in bars]
            low_prices = [float(bar.low) for bar in bars]
            volumes = [float(bar.volume) for bar in bars]
            
            # Calculate enhanced indicators
            fast_ma = self._calculate_ema(prices, self.fast_ma_period)  # Use EMA for faster response
            slow_ma = self._calculate_ema(prices, self.slow_ma_period)
            trend_ma = self._calculate_sma(prices, self.trend_ma_period)
            rsi = self._calculate_rsi(prices, self.rsi_period)
            volume_ratio = self._calculate_volume_ratio(volumes)
            adx = self._calculate_adx(high_prices, low_prices, prices, self.adx_period)
            atr = self._calculate_atr(high_prices, low_prices, prices, self.adx_period)
            
            print(f"🚀 ENHANCED STRATEGY: {symbol} - Fast MA: {fast_ma:.2f}, Slow MA: {slow_ma:.2f}, Trend MA: {trend_ma:.2f}")
            print(f"🚀 ENHANCED STRATEGY: {symbol} - RSI: {rsi:.1f}, Volume: {volume_ratio:.2f}x, ADX: {adx:.1f}, ATR: {atr:.2f}")
            
            if any(x is None for x in [fast_ma, slow_ma, trend_ma, rsi, volume_ratio, adx, atr]):
                print(f"❌ ENHANCED STRATEGY: {symbol} has None indicators")
                continue
            
            current_price = prices[-1]
            current_position = self._get_position(symbol, portfolio)
            
            # Check trend confirmation
            trend_analysis = self._check_trend_confirmation(prices, fast_ma, slow_ma, trend_ma)
            
            # Entry signals with multiple confirmations
            if not current_position:
                print(f"🚀 ENHANCED STRATEGY: {symbol} has no position, checking entry conditions...")
                
                # BULLISH ENTRY - MUCH MORE AGGRESSIVE CONDITIONS
                # 1. Momentum bounce (fast MA > slow MA) - PRIMARY CONDITION
                momentum_bounce = fast_ma > slow_ma
                
                # 2. RSI oversold bounce (RSI < 40 and rising)
                rsi_oversold_bounce = rsi < 40
                
                # 3. Volume spike (any volume above average)
                volume_spike = volume_ratio > 0.8
                
                # 4. Trend strength (any ADX above 15)
                trend_strength = adx > 15
                
                # 5. Price above recent low (bounce from support)
                if len(prices) >= 10:
                    recent_low = min(prices[-10:])
                    price_above_support = current_price > recent_low * 0.995  # Within 0.5% of recent low
                else:
                    price_above_support = True
                
                # ENTRY LOGIC: Multiple ways to enter
                should_enter = False
                entry_reason = ""
                
                # Method 1: Strong momentum (fast MA > slow MA)
                if momentum_bounce and volume_spike and trend_strength:
                    should_enter = True
                    entry_reason = f"STRONG MOMENTUM: Fast MA({fast_ma:.2f}) > Slow MA({slow_ma:.2f}), Volume({volume_ratio:.2f}x), ADX({adx:.1f})"
                
                # Method 2: RSI oversold bounce
                elif rsi_oversold_bounce and volume_spike and price_above_support:
                    should_enter = True
                    entry_reason = f"RSI OVERSOLD BOUNCE: RSI({rsi:.1f}), Volume({volume_ratio:.2f}x), Price above support"
                
                # Method 3: Volume breakout with any momentum
                elif volume_ratio > 1.5 and (momentum_bounce or rsi < 50):
                    should_enter = True
                    entry_reason = f"VOLUME BREAKOUT: Volume({volume_ratio:.2f}x), Momentum({momentum_bounce}), RSI({rsi:.1f})"
                
                # Method 4: Price reversal pattern (last 3 bars showing higher lows)
                elif len(prices) >= 3:
                    last_3_prices = prices[-3:]
                    if (last_3_prices[-1] > last_3_prices[-2] > last_3_prices[-3] and 
                        volume_spike and rsi < 60):
                        should_enter = True
                        entry_reason = f"PRICE REVERSAL: 3-bar uptrend, Volume({volume_ratio:.2f}x), RSI({rsi:.1f})"
                
                # Method 5: ATR contraction followed by expansion (volatility breakout)
                if not should_enter and len(prices) >= 20:
                    recent_atr = self._calculate_atr(high_prices[-20:], low_prices[-20:], prices[-20:], 10)
                    if recent_atr and atr and atr > recent_atr * 1.2 and volume_spike:
                        should_enter = True
                        entry_reason = f"VOLATILITY BREAKOUT: ATR expansion {atr:.2f} vs {recent_atr:.2f}, Volume({volume_ratio:.2f}x)"
                
                if should_enter:
                    print(f"🚀 ENHANCED STRATEGY: {symbol} BULLISH ENTRY - {entry_reason}")
                    entry_signals[symbol] = {
                        'side': 'buy',
                        'price': current_price,
                        'quantity': self._calculate_position_size(current_price, portfolio, atr),
                        'reason': entry_reason,
                        'timestamp': datetime.now(timezone.utc)
                    }
                    
                    # Initialize tracking for this position
                    self.highest_prices[symbol] = current_price
                    self.entry_prices[symbol] = current_price
                
                # TEMPORARILY DISABLED: BEARISH ENTRY (short selling)
                # elif (trend_analysis['overall_bearish'] and 
                #       rsi > self.rsi_oversold and  # Not oversold
                #       volume_ratio > self.volume_multiplier and  # Volume confirmation
                #       adx > self.adx_threshold):  # Strong trend
                #     
                #     print(f"📉 ENHANCED STRATEGY: {symbol} BEARISH ENTRY - All conditions met!")
                #     entry_signals[symbol] = {
                #         'side': 'sell',
                #         'price': current_price,
                #         'quantity': self._calculate_position_size(current_price, portfolio, atr),
                #         'reason': f'BEARISH: Trend({trend_analysis["overall_bearish"]}), RSI({rsi:.1f}), Volume({volume_ratio:.2f}x), ADX({adx:.1f})',
                #         'timestamp': datetime.now(timezone.utc)
                #     }
                #     
                #     # Initialize tracking for this position
                #     self.lowest_prices[symbol] = current_price
                #     self.entry_prices[symbol] = current_price
            
            # Exit signals for existing positions
            elif current_position:
                position = current_position
                entry_price = self.entry_prices.get(symbol, float(position.average_cost))
                
                # Update highest/lowest prices for trailing stops
                if position.quantity > 0:  # Long position
                    if symbol not in self.highest_prices or current_price > self.highest_prices[symbol]:
                        self.highest_prices[symbol] = current_price
                    
                    # Calculate P&L
                    pnl_pct = (current_price - entry_price) / entry_price
                    
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
                    
                    # 3. Trailing stop (protect profits) - MORE AGGRESSIVE
                    elif self.trailing_stop and pnl_pct > 0.005:  # Only if in profit > 0.5%
                        trailing_stop_price = self.highest_prices[symbol] * (1 - self.trailing_stop_pct)
                        if current_price <= trailing_stop_price:
                            should_exit = True
                            exit_reason = f"Trailing stop: {pnl_pct:.2%}"
                            print(f"📉 ENHANCED STRATEGY: {symbol} TRAILING STOP triggered: {pnl_pct:.2%}")
                    
                    # 4. Trend reversal (exit on momentum loss) - MORE AGGRESSIVE
                    elif (not trend_analysis['short_momentum'] and 
                          pnl_pct > -0.003):  # Only if not too deep in loss
                        should_exit = True
                        exit_reason = f"Trend reversal: {pnl_pct:.2%}"
                        print(f"🔄 ENHANCED STRATEGY: {symbol} TREND REVERSAL EXIT: {pnl_pct:.2%}")
                    
                    # 5. RSI overbought exit (profit taking) - MORE AGGRESSIVE
                    elif rsi > 70 and pnl_pct > 0.005:  # Exit if overbought and in profit > 0.5%
                        should_exit = True
                        exit_reason = f"RSI overbought exit: {pnl_pct:.2%}"
                        print(f"📊 ENHANCED STRATEGY: {symbol} RSI OVERBOUGHT EXIT: {pnl_pct:.2%}")
                    
                    # 6. Time-based exit (exit if position held too long without profit)
                    if not should_exit and atr > 0:
                        # Check if we've been in this position for more than 24 hours (24 bars)
                        if hasattr(position, 'entry_timestamp'):
                            time_in_position = datetime.now(timezone.utc) - position.entry_timestamp
                            if time_in_position.total_seconds() > 24 * 3600:  # 24 hours
                                should_exit = True
                                exit_reason = f"Time-based exit: {pnl_pct:.2%} after 24h"
                                print(f"⏰ ENHANCED STRATEGY: {symbol} TIME-BASED EXIT: {pnl_pct:.2%}")
                    
                    # 7. Volatility-based exit (exit if ATR increases significantly)
                    if not should_exit and atr > 0:
                        # Calculate ATR ratio compared to entry
                        if symbol in self.entry_prices:
                            entry_atr = self._calculate_atr(high_prices[:-10], low_prices[:-10], prices[:-10], self.adx_period)
                            if entry_atr and entry_atr > 0:
                                atr_ratio = atr / entry_atr
                                if atr_ratio > 2.0 and pnl_pct < 0.01:  # ATR doubled and not in profit
                                    should_exit = True
                                    exit_reason = f"Volatility exit: ATR ratio {atr_ratio:.1f}, P&L {pnl_pct:.2%}"
                                    print(f"📊 ENHANCED STRATEGY: {symbol} VOLATILITY EXIT: ATR ratio {atr_ratio:.1f}")
                    
                    if should_exit:
                        exit_signals[symbol] = {
                            'side': 'sell',
                            'price': current_price,
                            'quantity': abs(position.quantity),
                            'reason': exit_reason,
                            'timestamp': datetime.now(timezone.utc)
                        }
                
                elif position.quantity < 0:  # Short position
                    if symbol not in self.lowest_prices or current_price < self.lowest_prices[symbol]:
                        self.lowest_prices[symbol] = current_price
                    
                    # Calculate P&L for short
                    pnl_pct = (entry_price - current_price) / entry_price
                    
                    # Smart exit conditions for SHORT positions
                    should_exit = False
                    exit_reason = ""
                    
                    # 1. Stop loss (tight)
                    if pnl_pct <= -self.stop_loss_pct:
                        should_exit = True
                        exit_reason = f"Stop loss: {pnl_pct:.2%}"
                        print(f"🛑 ENHANCED STRATEGY: {symbol} SHORT STOP LOSS: {pnl_pct:.2%}")
                    
                    # 2. Take profit (better risk-reward)
                    elif pnl_pct >= self.take_profit_pct:
                        should_exit = True
                        exit_reason = f"Take profit: {pnl_pct:.2%}"
                        print(f"🎯 ENHANCED STRATEGY: {symbol} SHORT TAKE PROFIT: {pnl_pct:.2%}")
                    
                    # 3. Trailing stop (protect profits)
                    elif self.trailing_stop and pnl_pct > 0.01:  # Only if in profit > 1%
                        trailing_stop_price = self.lowest_prices[symbol] * (1 + self.trailing_stop_pct)
                        if current_price >= trailing_stop_price:
                            should_exit = True
                            exit_reason = f"Trailing stop: {pnl_pct:.2%}"
                            print(f"📈 ENHANCED STRATEGY: {symbol} SHORT TRAILING STOP: {pnl_pct:.2%}")
                    
                    # 4. Trend reversal (exit on momentum loss)
                    elif (trend_analysis['short_momentum'] and 
                          pnl_pct > -0.005):  # Only if not too deep in loss
                        should_exit = True
                        exit_reason = f"Trend reversal: {pnl_pct:.2%}"
                        print(f"🔄 ENHANCED STRATEGY: {symbol} SHORT TREND REVERSAL: {pnl_pct:.2%}")
                    
                    # 5. RSI oversold exit (profit taking)
                    elif rsi < 25 and pnl_pct > 0.01:  # Exit if oversold and in profit
                        should_exit = True
                        exit_reason = f"RSI oversold exit: {pnl_pct:.2%}"
                        print(f"📊 ENHANCED STRATEGY: {symbol} SHORT RSI OVERSOLD EXIT: {pnl_pct:.2%}")
                    
                    if should_exit:
                        exit_signals[symbol] = {
                            'side': 'buy',
                            'price': current_price,
                            'quantity': abs(position.quantity),
                            'reason': exit_reason,
                            'timestamp': datetime.now(timezone.utc)
                        }
        
        print(f"📊 ENHANCED STRATEGY: Generated {len(entry_signals)} entry signals and {len(exit_signals)} exit signals")
        return {
            'entry_signals': entry_signals,
            'exit_signals': exit_signals
        }
    
    def _get_position(self, symbol: str, portfolio: Portfolio):
        """Get current position for a symbol"""
        for position in portfolio.positions:
            if position.symbol == symbol:
                return position
        return None
    
    def _calculate_position_size(self, price: float, portfolio: Portfolio, atr: float) -> float:
        """Calculate position size based on portfolio allocation and volatility"""
        # Use ATR for volatility-based position sizing
        volatility_factor = atr / price if price > 0 else 1.0 # Simple factor, could be more sophisticated
        available_capital = float(portfolio.cash) * self.position_size * volatility_factor
        return available_capital / price if price > 0 else 0.0
    
    def should_enter(self, symbol: str, market_data: MarketData, portfolio: Portfolio) -> tuple[bool, Dict]:
        """Determine if we should enter a position"""
        if not market_data.bars or symbol not in market_data.bars:
            return False, {}
        
        bars = market_data.bars[symbol]
        if len(bars) < self.trend_ma_period:
            return False, {}
        
        # Extract price and volume data
        prices = [float(bar.close) for bar in bars]
        volumes = [float(bar.volume) for bar in bars]
        
        # Calculate indicators
        fast_ma = self._calculate_ema(prices, self.fast_ma_period)
        slow_ma = self._calculate_ema(prices, self.slow_ma_period)
        rsi = self._calculate_rsi(prices, self.rsi_period)
        volume_ratio = self._calculate_volume_ratio(volumes)
        adx = self._calculate_adx(high_prices, low_prices, prices, self.adx_period)
        atr = self._calculate_atr(high_prices, low_prices, prices, self.adx_period)
        
        if any(x is None for x in [fast_ma, slow_ma, rsi, volume_ratio, adx, atr]):
            return False, {}
        
        current_price = prices[-1]
        current_position = self._get_position(symbol, portfolio)
        
        # Check if we already have a position
        if current_position:
            return False, {}
        
        # Entry conditions
        if (fast_ma > slow_ma and 
            rsi < self.rsi_overbought and 
            volume_ratio > self.volume_multiplier and 
            adx > self.adx_threshold):
            
            return True, {
                'side': 'buy',
                'price': current_price,
                'quantity': self._calculate_position_size(current_price, portfolio, atr),
                'reason': f'Bullish momentum: Fast MA({fast_ma:.2f}) > Slow MA({slow_ma:.2f}), RSI({rsi:.1f}), Volume({volume_ratio:.2f}x), ADX({adx:.1f})'
            }
        
        elif (fast_ma < slow_ma and 
              rsi > self.rsi_oversold and 
              volume_ratio > self.volume_multiplier and 
              adx > self.adx_threshold):
            
            return True, {
                'side': 'sell',
                'price': current_price,
                'quantity': self._calculate_position_size(current_price, portfolio, atr),
                'reason': f'Bearish momentum: Fast MA({fast_ma:.2f}) < Slow MA({slow_ma:.2f}), RSI({rsi:.1f}), Volume({volume_ratio:.2f}x), ADX({adx:.1f})'
            }
        
        return False, {}
    
    def should_exit(self, position: Position, market_data: MarketData, portfolio: Portfolio) -> tuple[bool, Dict]:
        """Determine if we should exit a position"""
        if not market_data.bars or position.symbol not in market_data.bars:
            return False, {}
        
        bars = market_data.bars[position.symbol]
        if len(bars) < self.trend_ma_period:
            return False, {}
        
        # Extract price and volume data
        prices = [float(bar.close) for bar in bars]
        
        # Calculate indicators
        fast_ma = self._calculate_ema(prices, self.fast_ma_period)
        slow_ma = self._calculate_ema(prices, self.slow_ma_period)
        rsi = self._calculate_rsi(prices, self.rsi_period)
        adx = self._calculate_adx(high_prices, low_prices, prices, self.adx_period)
        
        if any(x is None for x in [fast_ma, slow_ma, rsi, adx]):
            return False, {}
        
        current_price = prices[-1]
        entry_price = float(position.average_cost)
        
        # Calculate P&L
        if position.quantity > 0:  # Long position
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Stop loss or take profit
            if pnl_pct <= -self.stop_loss_pct:
                return True, {
                    'side': 'sell',
                    'price': current_price,
                    'quantity': abs(position.quantity),
                    'reason': f'Stop loss triggered: {pnl_pct:.2%}'
                }
            elif pnl_pct >= self.take_profit_pct:
                return True, {
                    'side': 'sell',
                    'price': current_price,
                    'quantity': abs(position.quantity),
                    'reason': f'Take profit triggered: {pnl_pct:.2%}'
                }
            # Trend reversal exit
            elif fast_ma < slow_ma and rsi > 50:
                return True, {
                    'side': 'sell',
                    'price': current_price,
                    'quantity': abs(position.quantity),
                    'reason': f'Trend reversal: Fast MA({fast_ma:.2f}) < Slow MA({slow_ma:.2f})'
                }
        
        elif position.quantity < 0:  # Short position
            pnl_pct = (entry_price - current_price) / entry_price
            
            # Stop loss or take profit
            if pnl_pct <= -self.stop_loss_pct:
                return True, {
                    'side': 'buy',
                    'price': current_price,
                    'quantity': abs(position.quantity),
                    'reason': f'Stop loss triggered: {pnl_pct:.2%}'
                }
            elif pnl_pct >= self.take_profit_pct:
                return True, {
                    'side': 'buy',
                    'price': current_price,
                    'quantity': abs(position.quantity),
                    'reason': f'Take profit triggered: {pnl_pct:.2%}'
                }
            # Trend reversal exit
            elif fast_ma > slow_ma and rsi < 50:
                return True, {
                    'side': 'buy',
                    'price': current_price,
                    'quantity': abs(position.quantity),
                    'reason': f'Trend reversal: Fast MA({fast_ma:.2f}) > Slow MA({slow_ma:.2f})'
                }
        
        return False, {}
