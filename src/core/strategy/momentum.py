"""
Momentum Trading Strategy

This strategy identifies and follows price trends using:
- Moving average crossovers (fast vs slow)
- Volume confirmation
- RSI momentum confirmation
- Trend strength indicators
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
    Momentum strategy that follows trends using moving average crossovers
    and volume confirmation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Strategy parameters
        self.fast_ma_period = config.get('fast_ma_period', 10)
        self.slow_ma_period = config.get('slow_ma_period', 20)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.volume_multiplier = config.get('volume_multiplier', 1.5)
        self.position_size = config.get('position_size', 0.1)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)
        self.take_profit_pct = config.get('take_profit_pct', 0.04)
        
        # State tracking
        self.positions = {}
        self.entry_prices = {}
        self.stop_losses = {}
        self.take_profits = {}
        
    def _calculate_sma(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return None
        return np.mean(prices[-period:])
    
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
    
    def _calculate_volume_ratio(self, volumes: List[float], period: int = 20) -> Optional[float]:
        """Calculate current volume vs average volume ratio"""
        if len(volumes) < period:
            return None
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-period:])
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def generate_signals(self, market_data: MarketData, portfolio: Portfolio) -> Dict[str, Any]:
        """
        Generate trading signals based on momentum indicators
        """
        print(f"🔍 STRATEGY: Starting signal generation...")
        print(f"🔍 STRATEGY: Market data has {len(market_data.bars)} symbols")
        print(f"🔍 STRATEGY: Bars keys: {list(market_data.bars.keys())}")
        
        if not market_data.bars or len(market_data.bars) < self.slow_ma_period:
            print(f"❌ STRATEGY: Not enough bars or no bars")
            return {'entry_signals': {}, 'exit_signals': {}}
        
        entry_signals = {}
        exit_signals = {}
        
        for symbol, bars in market_data.bars.items():
            print(f"🔍 STRATEGY: Processing {symbol} with {len(bars)} bars")
            
            if len(bars) < self.slow_ma_period:
                print(f"❌ STRATEGY: {symbol} has only {len(bars)} bars, need {self.slow_ma_period}")
                continue
                
            # Extract price and volume data
            prices = [float(bar.close) for bar in bars]
            volumes = [float(bar.volume) for bar in bars]
            
            print(f"🔍 STRATEGY: {symbol} prices: {prices[:5]}... (last: {prices[-1]:.2f})")
            
            # Calculate indicators
            fast_ma = self._calculate_sma(prices, self.fast_ma_period)
            slow_ma = self._calculate_sma(prices, self.slow_ma_period)
            rsi = self._calculate_rsi(prices, self.rsi_period)
            volume_ratio = self._calculate_volume_ratio(volumes)
            
            print(f"🔍 STRATEGY: {symbol} indicators - Fast MA: {fast_ma}, Slow MA: {slow_ma}, RSI: {rsi}, Volume: {volume_ratio}")
            
            if fast_ma is None or slow_ma is None or rsi is None or volume_ratio is None:
                print(f"❌ STRATEGY: {symbol} has None indicators")
                continue
            
            current_price = prices[-1]
            current_position = self._get_position(symbol, portfolio)
            
            print(f"🔍 STRATEGY: {symbol}: Fast MA: {fast_ma:.2f}, Slow MA: {slow_ma:.2f}, RSI: {rsi:.1f}, Volume: {volume_ratio:.2f}x")
            print(f"🔍 STRATEGY: {symbol}: Current position: {current_position}")
            
            # Entry signals - MUCH MORE AGGRESSIVE CONDITIONS
            if not current_position:
                print(f"🔍 STRATEGY: {symbol} has no position, checking entry conditions...")
                
                # Bullish momentum: fast MA above slow MA (relaxed RSI and volume)
                if fast_ma > slow_ma:
                    print(f"🚀 STRATEGY: {symbol} BUY condition met: Fast MA({fast_ma:.2f}) > Slow MA({slow_ma:.2f})")
                    entry_signals[symbol] = {
                        'side': 'buy',
                        'price': current_price,
                        'quantity': self._calculate_position_size(current_price, portfolio),
                        'reason': f'Bullish momentum: Fast MA({fast_ma:.2f}) > Slow MA({slow_ma:.2f}), RSI({rsi:.1f}), Volume({volume_ratio:.2f}x)',
                        'timestamp': datetime.now(timezone.utc)
                    }
                    print(f"🚀 BUY SIGNAL for {symbol}: Fast MA > Slow MA")
                
                # Bearish momentum: fast MA below slow MA (relaxed RSI and volume)
                elif fast_ma < slow_ma:
                    print(f"📉 STRATEGY: {symbol} SELL condition met: Fast MA({fast_ma:.2f}) < Slow MA({slow_ma:.2f})")
                    entry_signals[symbol] = {
                        'side': 'sell',
                        'price': current_price,
                        'quantity': self._calculate_position_size(current_price, portfolio),
                        'reason': f'Bearish momentum: Fast MA({fast_ma:.2f}) < Slow MA({slow_ma:.2f}), RSI({rsi:.1f}), Volume({volume_ratio:.2f}x)',
                        'timestamp': datetime.now(timezone.utc)
                    }
                    print(f"📉 SELL SIGNAL for {symbol}: Fast MA < Slow MA")
            
            # Exit signals for existing positions
            elif current_position:
                position = current_position
                entry_price = float(position.average_cost)
                
                # Calculate P&L
                if position.quantity > 0:  # Long position
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    # Stop loss or take profit
                    if pnl_pct <= -self.stop_loss_pct:
                        exit_signals[symbol] = {
                            'side': 'sell',
                            'price': current_price,
                            'quantity': abs(position.quantity),
                            'reason': f'Stop loss triggered: {pnl_pct:.2%}',
                            'timestamp': datetime.now(timezone.utc)
                        }
                        print(f"🛑 STOP LOSS for {symbol}: {pnl_pct:.2%}")
                    elif pnl_pct >= self.take_profit_pct:
                        exit_signals[symbol] = {
                            'side': 'sell',
                            'price': current_price,
                            'quantity': abs(position.quantity),
                            'reason': f'Take profit triggered: {pnl_pct:.2%}',
                            'timestamp': datetime.now(timezone.utc)
                        }
                        print(f"🎯 TAKE PROFIT for {symbol}: {pnl_pct:.2%}")
                    # Trend reversal exit (more aggressive)
                    elif fast_ma < slow_ma:
                        exit_signals[symbol] = {
                            'side': 'sell',
                            'price': current_price,
                            'quantity': abs(position.quantity),
                            'reason': f'Trend reversal: Fast MA({fast_ma:.2f}) < Slow MA({slow_ma:.2f})',
                            'timestamp': datetime.now(timezone.utc)
                        }
                        print(f"🔄 TREND REVERSAL EXIT for {symbol}")
                
                elif position.quantity < 0:  # Short position
                    pnl_pct = (entry_price - current_price) / entry_price
                    
                    # Stop loss or take profit
                    if pnl_pct <= -self.stop_loss_pct:
                        exit_signals[symbol] = {
                            'side': 'buy',
                            'price': current_price,
                            'quantity': abs(position.quantity),
                            'reason': f'Stop loss triggered: {pnl_pct:.2%}',
                            'timestamp': datetime.now(timezone.utc)
                        }
                        print(f"🛑 STOP LOSS for {symbol}: {pnl_pct:.2%}")
                    elif pnl_pct >= self.take_profit_pct:
                        exit_signals[symbol] = {
                            'side': 'buy',
                            'price': current_price,
                            'quantity': abs(position.quantity),
                            'reason': f'Take profit triggered: {pnl_pct:.2%}',
                            'timestamp': datetime.now(timezone.utc)
                        }
                        print(f"🎯 TAKE PROFIT for {symbol}: {pnl_pct:.2%}")
                    # Trend reversal exit (more aggressive)
                    elif fast_ma > slow_ma:
                        exit_signals[symbol] = {
                            'side': 'buy',
                            'price': current_price,
                            'quantity': abs(position.quantity),
                            'reason': f'Trend reversal: Fast MA({fast_ma:.2f}) > Slow MA({slow_ma:.2f})',
                            'timestamp': datetime.now(timezone.utc)
                        }
                        print(f"🔄 TREND REVERSAL EXIT for {symbol}")
        
        print(f"📊 Generated {len(entry_signals)} entry signals and {len(exit_signals)} exit signals")
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
    
    def _calculate_position_size(self, price: float, portfolio: Portfolio) -> float:
        """Calculate position size based on portfolio allocation"""
        available_capital = float(portfolio.cash) * self.position_size
        return available_capital / price if price > 0 else 0.0
    
    def should_enter(self, symbol: str, market_data: MarketData, portfolio: Portfolio) -> tuple[bool, Dict]:
        """Determine if we should enter a position"""
        if not market_data.bars or symbol not in market_data.bars:
            return False, {}
        
        bars = market_data.bars[symbol]
        if len(bars) < self.slow_ma_period:
            return False, {}
        
        # Extract price and volume data
        prices = [float(bar.close) for bar in bars]
        volumes = [float(bar.volume) for bar in bars]
        
        # Calculate indicators
        fast_ma = self._calculate_sma(prices, self.fast_ma_period)
        slow_ma = self._calculate_sma(prices, self.slow_ma_period)
        rsi = self._calculate_rsi(prices, self.rsi_period)
        volume_ratio = self._calculate_volume_ratio(volumes)
        
        if fast_ma is None or slow_ma is None or rsi is None or volume_ratio is None:
            return False, {}
        
        current_price = prices[-1]
        current_position = self._get_position(symbol, portfolio)
        
        # Check if we already have a position
        if current_position:
            return False, {}
        
        # Entry conditions
        if (fast_ma > slow_ma and 
            rsi < self.rsi_overbought and 
            volume_ratio > self.volume_multiplier):
            
            return True, {
                'side': 'buy',
                'price': current_price,
                'quantity': self._calculate_position_size(current_price, portfolio),
                'reason': f'Bullish momentum: Fast MA({fast_ma:.2f}) > Slow MA({slow_ma:.2f}), RSI({rsi:.1f}), Volume({volume_ratio:.2f}x)'
            }
        
        elif (fast_ma < slow_ma and 
              rsi > self.rsi_oversold and 
              volume_ratio > self.volume_multiplier):
            
            return True, {
                'side': 'sell',
                'price': current_price,
                'quantity': self._calculate_position_size(current_price, portfolio),
                'reason': f'Bearish momentum: Fast MA({fast_ma:.2f}) < Slow MA({slow_ma:.2f}), RSI({rsi:.1f}), Volume({volume_ratio:.2f}x)'
            }
        
        return False, {}
    
    def should_exit(self, position: Position, market_data: MarketData, portfolio: Portfolio) -> tuple[bool, Dict]:
        """Determine if we should exit a position"""
        if not market_data.bars or position.symbol not in market_data.bars:
            return False, {}
        
        bars = market_data.bars[position.symbol]
        if len(bars) < self.slow_ma_period:
            return False, {}
        
        # Extract price and volume data
        prices = [float(bar.close) for bar in bars]
        
        # Calculate indicators
        fast_ma = self._calculate_sma(prices, self.fast_ma_period)
        slow_ma = self._calculate_sma(prices, self.slow_ma_period)
        rsi = self._calculate_rsi(prices, self.rsi_period)
        
        if fast_ma is None or slow_ma is None or rsi is None:
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
