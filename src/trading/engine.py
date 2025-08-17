"""Trading engine for executing strategies."""

import logging
from typing import Dict, List, Any, Optional
from .strategies import Strategy
from .portfolio import Portfolio
from .risk import RiskManager

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading engine that executes strategies."""
    
    def __init__(self, portfolio: Portfolio, risk_manager: RiskManager, strategies: Dict[str, Strategy]):
        """Initialize trading engine."""
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.strategies = strategies
        self.signals: List[Dict[str, Any]] = []
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals from all strategies."""
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                strategy_signals = strategy.generate_signals(market_data)
                for signal in strategy_signals:
                    signal["strategy"] = strategy_name
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating signals for {strategy_name}: {e}")
        
        self.signals = signals
        return signals
    
    def execute_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trading signal."""
        try:
            symbol = signal["symbol"]
            action = signal["action"]
            quantity = signal.get("quantity", 0)
            price = signal.get("price", 0)
            
            logger.info(f"Executing signal: {action} {symbol} {quantity} @ {price}")
            
            if action == "BUY":
                # Add long position
                self.portfolio.add_position(symbol, quantity, price, "LONG")
                result = {"status": "SUCCESS", "action": "BUY", "symbol": symbol}
                
            elif action == "SELL":
                # Close long position
                self.portfolio.close_position(symbol, quantity, price)
                result = {"status": "SUCCESS", "action": "SELL", "symbol": symbol}
                
            elif action == "SHORT":
                # Add short position
                self.portfolio.add_position(symbol, quantity, price, "SHORT")
                result = {"status": "SUCCESS", "action": "SHORT", "symbol": symbol}
                
            elif action == "COVER":
                # Close short position
                self.portfolio.close_position(symbol, quantity, price)
                result = {"status": "SUCCESS", "action": "COVER", "symbol": symbol}
                
            else:
                result = {"status": "UNKNOWN_ACTION", "action": action, "symbol": symbol}
                logger.warning(f"Unknown action: {action}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing signal {signal}: {e}")
            return {"status": "ERROR", "error": str(e), "signal": signal}
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """Get trading engine summary."""
        return {
            "total_signals": len(self.signals),
            "strategies_count": len(self.strategies),
            "portfolio_value": self.portfolio.get_total_value({}),
            "positions_count": len(self.portfolio.positions),
            "trades_count": len(self.portfolio.trades)
        }
