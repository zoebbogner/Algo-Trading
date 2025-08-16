"""Mean reversion trading strategy."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from .base import Strategy
from ..data_models.market import MarketData, Feature
from ..data_models.trading import Position, Portfolio
from ...utils.logging import logger


class MeanReversionStrategy(Strategy):
    """Mean reversion strategy with momentum confirmation."""
    
    def __init__(self, config: Dict):
        """Initialize mean reversion strategy.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        
        # Strategy-specific parameters
        self.momentum_threshold = self.entry_config.get("momentum_threshold", 0.02)
        self.volatility_threshold = self.entry_config.get("volatility_threshold", 0.01)
        self.volume_threshold = self.entry_config.get("volume_threshold", 1000)
        self.trend_confirmation_bars = self.entry_config.get("trend_confirmation_bars", 3)
        
        # Exit parameters
        self.profit_target = self.exit_config.get("profit_target", 0.05)
        self.stop_loss = self.exit_config.get("stop_loss", 0.03)
        self.max_hold_time_hours = self.exit_config.get("max_hold_time_hours", 24)
        self.trailing_stop = self.exit_config.get("trailing_stop", True)
        self.trailing_stop_distance = self.exit_config.get("trailing_stop_distance", 0.02)
        
        logger.logger.info(f"Initialized MeanReversionStrategy: {self.name}")
    
    def generate_signals(self, market_data: MarketData, portfolio: Portfolio) -> Dict:
        """Generate trading signals from market data.
        
        Args:
            market_data: Current market data with features
            portfolio: Current portfolio state
            
        Returns:
            Dictionary containing trading signals
        """
        signals = {
            "entry_signals": {},
            "exit_signals": {},
            "risk_alerts": []
        }
        
        # Check for entry signals
        should_enter, entry_details = self.should_enter(
            market_data.symbol, market_data, portfolio
        )
        
        if should_enter:
            signals["entry_signals"][market_data.symbol] = entry_details
        
        # Check for exit signals on existing positions
        for position in portfolio.positions:
            if position.symbol == market_data.symbol:
                should_exit, exit_details = self.should_exit(position, market_data, portfolio)
                if should_exit:
                    signals["exit_signals"][position.symbol] = exit_details
        
        return signals
    
    def should_enter(self, symbol: str, market_data: MarketData, portfolio: Portfolio) -> Tuple[bool, Dict]:
        """Determine if we should enter a position.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            portfolio: Current portfolio state
            
        Returns:
            Tuple of (should_enter, entry_details)
        """
        # Check if we already have max positions
        if len(portfolio.positions) >= self.max_positions:
            return False, {"reason": "Maximum positions reached"}
        
        # Check if we already have a position in this symbol
        existing_position = next((p for p in portfolio.positions if p.symbol == symbol), None)
        if existing_position:
            return False, {"reason": "Position already exists"}
        
        # Extract features
        features = self._extract_features(market_data)
        if not features:
            return False, {"reason": "Insufficient features"}
        
        # Check entry conditions
        entry_reasons = []
        
        # Momentum check
        if "returns_1m" in features:
            momentum = features["returns_1m"]
            if abs(momentum) > self.momentum_threshold:
                entry_reasons.append(f"Strong momentum: {momentum:.4f}")
        
        # Volatility check
        if "volatility_rolling" in features:
            volatility = features["volatility_rolling"]
            if volatility > self.volatility_threshold:
                entry_reasons.append(f"Sufficient volatility: {volatility:.4f}")
        
        # Volume check
        if "volume_ratio" in features:
            volume_ratio = features["volume_ratio"]
            if volume_ratio > 1.5:  # 50% above average
                entry_reasons.append(f"High volume: {volume_ratio:.2f}x average")
        
        # RSI check (mean reversion signal)
        if "rsi_14" in features:
            rsi = features["rsi_14"]
            if rsi < 30 or rsi > 70:  # Oversold or overbought
                entry_reasons.append(f"RSI extreme: {rsi:.1f}")
        
        # Bollinger Bands check
        if "bb_position" in features:
            bb_position = features["bb_position"]
            if bb_position < 0.1 or bb_position > 0.9:  # Near bands
                entry_reasons.append(f"BB position: {bb_position:.3f}")
        
        # Determine entry direction
        entry_direction = self._determine_entry_direction(features)
        
        # Check if we have enough reasons to enter
        if len(entry_reasons) >= 2 and entry_direction:
            position_size = self.calculate_position_size(symbol, market_data, portfolio)
            
            entry_details = {
                "direction": entry_direction,
                "reasons": entry_reasons,
                "position_size": float(position_size),
                "entry_price": float(market_data.bar.close),
                "timestamp": market_data.timestamp.isoformat()
            }
            
            return True, entry_details
        
        return False, {"reason": "Entry conditions not met", "partial_reasons": entry_reasons}
    
    def should_exit(self, position: Position, market_data: MarketData, portfolio: Portfolio) -> Tuple[bool, Dict]:
        """Determine if we should exit a position.
        
        Args:
            position: Current position
            market_data: Current market data
            portfolio: Current portfolio state
            
        Returns:
            Tuple of (should_exit, exit_details)
        """
        exit_reasons = []
        current_price = float(market_data.bar.close)
        entry_price = float(position.average_cost)
        
        # Calculate current P&L
        if position.is_long:
            pnl_ratio = (current_price - entry_price) / entry_price
        else:
            pnl_ratio = (entry_price - current_price) / entry_price
        
        # Profit target
        if pnl_ratio >= self.profit_target:
            exit_reasons.append(f"Profit target reached: {pnl_ratio:.4f}")
        
        # Stop loss
        if pnl_ratio <= -self.stop_loss:
            exit_reasons.append(f"Stop loss triggered: {pnl_ratio:.4f}")
        
        # Time-based exit
        if position.entry_timestamp:
            hold_time = (market_data.timestamp - position.entry_timestamp).total_seconds() / 3600
            if hold_time >= self.max_hold_time_hours:
                exit_reasons.append(f"Max hold time exceeded: {hold_time:.1f}h")
        
        # Trailing stop (if enabled)
        if self.trailing_stop and position.entry_timestamp:
            # This is a simplified trailing stop - in practice, you'd track the highest/lowest price
            # since entry and adjust the stop accordingly
            if abs(pnl_ratio) > self.trailing_stop_distance:
                # Check if we've moved against the position significantly
                if position.is_long and pnl_ratio < -self.trailing_stop_distance * 0.5:
                    exit_reasons.append(f"Trailing stop: {pnl_ratio:.4f}")
                elif position.is_short and pnl_ratio < -self.trailing_stop_distance * 0.5:
                    exit_reasons.append(f"Trailing stop: {pnl_ratio:.4f}")
        
        # Technical exit signals
        features = self._extract_features(market_data)
        if features:
            # RSI extreme exit
            if "rsi_14" in features:
                rsi = features["rsi_14"]
                if position.is_long and rsi > 80:  # Overbought for long
                    exit_reasons.append(f"RSI overbought exit: {rsi:.1f}")
                elif position.is_short and rsi < 20:  # Oversold for short
                    exit_reasons.append(f"RSI oversold exit: {rsi:.1f}")
            
            # Bollinger Bands exit
            if "bb_position" in features:
                bb_position = features["bb_position"]
                if position.is_long and bb_position > 0.95:  # Near upper band
                    exit_reasons.append(f"BB upper exit: {bb_position:.3f}")
                elif position.is_short and bb_position < 0.05:  # Near lower band
                    exit_reasons.append(f"BB lower exit: {bb_position:.3f}")
        
        if exit_reasons:
            exit_details = {
                "reasons": exit_reasons,
                "current_price": current_price,
                "entry_price": entry_price,
                "pnl_ratio": pnl_ratio,
                "timestamp": market_data.timestamp.isoformat()
            }
            return True, exit_details
        
        return False, {"reason": "Exit conditions not met"}
    
    def _extract_features(self, market_data: MarketData) -> Dict[str, float]:
        """Extract features from market data.
        
        Args:
            market_data: Market data with features
            
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        for feature in market_data.features:
            features[feature.feature_name] = feature.feature_value
        return features
    
    def _determine_entry_direction(self, features: Dict[str, float]) -> Optional[str]:
        """Determine entry direction based on features.
        
        Args:
            features: Extracted features
            
        Returns:
            "long", "short", or None
        """
        # Simple mean reversion logic
        if "rsi_14" in features:
            rsi = features["rsi_14"]
            if rsi < 30:  # Oversold
                return "long"
            elif rsi > 70:  # Overbought
                return "short"
        
        if "bb_position" in features:
            bb_position = features["bb_position"]
            if bb_position < 0.1:  # Near lower band
                return "long"
            elif bb_position > 0.9:  # Near upper band
                return "short"
        
        if "returns_1m" in features:
            returns = features["returns_1m"]
            if returns < -self.momentum_threshold:  # Strong negative momentum
                return "long"  # Mean reversion - expect bounce
            elif returns > self.momentum_threshold:  # Strong positive momentum
                return "short"  # Mean reversion - expect pullback
        
        return None
