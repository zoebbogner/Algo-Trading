"""Risk management for algorithmic trading."""

from dataclasses import dataclass
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio."""
    var_95: float = 0.0  # 95% Value at Risk
    var_99: float = 0.0  # 99% Value at Risk
    max_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0


class RiskManager:
    """Risk management system."""
    
    def __init__(self, max_position_size: float = 0.1, max_portfolio_risk: float = 0.02, stop_loss_pct: float = 0.05):
        """Initialize risk manager."""
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_pct = stop_loss_pct
        self.risk_events: List[Dict[str, Any]] = []
    
    def check_risk_limits(self, portfolio) -> List[Dict[str, Any]]:
        """Check portfolio against risk limits."""
        violations = []
        
        # Check position size limits
        for symbol, position in portfolio.positions.items():
            position_value = portfolio.get_position_value(symbol, 0.0)  # Placeholder price
            portfolio_value = portfolio.get_total_value({})  # Placeholder prices
            
            if portfolio_value > 0:
                position_size_ratio = position_value / portfolio_value
                
                if position_size_ratio > self.max_position_size:
                    violation = {
                        "type": "POSITION_SIZE_LIMIT",
                        "symbol": symbol,
                        "current_ratio": position_size_ratio,
                        "limit": self.max_position_size,
                        "severity": "HIGH"
                    }
                    violations.append(violation)
                    logger.warning(f"Position size limit violated for {symbol}: {position_size_ratio:.2%} > {self.max_position_size:.2%}")
        
        # Check portfolio risk limits
        # This would typically calculate VaR or other risk metrics
        # For now, just check if any positions are at stop loss
        
        # Record risk events
        for violation in violations:
            self.risk_events.append({
                "timestamp": portfolio.timestamp,
                "violation": violation
            })
        
        return violations
    
    def should_stop_loss(self, position, current_price: float) -> bool:
        """Check if position should be stopped out."""
        if position.side == "LONG":
            loss_pct = (float(position.average_cost) - current_price) / float(position.average_cost)
        else:  # SHORT
            loss_pct = (current_price - float(position.average_cost)) / float(position.average_cost)
        
        return loss_pct >= self.stop_loss_pct
    
    def calculate_position_size(self, portfolio, symbol: str, price: float, confidence: float = 0.5) -> float:
        """Calculate optimal position size based on risk parameters."""
        portfolio_value = portfolio.get_total_value({})
        max_position_value = portfolio_value * self.max_position_size
        
        # Adjust for confidence (higher confidence = larger position)
        adjusted_position_value = max_position_value * confidence
        
        # Calculate quantity
        quantity = adjusted_position_value / price
        
        return quantity
    
    def get_risk_metrics(self, portfolio, historical_data: Dict[str, Any]) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        # This would implement actual risk calculations
        # For now, return placeholder metrics
        
        return RiskMetrics(
            var_95=0.02,  # 2% VaR
            var_99=0.03,  # 3% VaR
            max_drawdown=0.05,  # 5% max drawdown
            volatility=0.15,  # 15% volatility
            sharpe_ratio=1.2,  # 1.2 Sharpe ratio
            sortino_ratio=1.5   # 1.5 Sortino ratio
        )
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk management summary."""
        return {
            "max_position_size": self.max_position_size,
            "max_portfolio_risk": self.max_portfolio_risk,
            "stop_loss_pct": self.stop_loss_pct,
            "risk_events_count": len(self.risk_events),
            "last_risk_check": self.risk_events[-1]["timestamp"] if self.risk_events else None
        }
