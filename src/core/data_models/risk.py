"""Risk management data models."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import Field

from .base import BaseEntity, TimeSeriesEntity


class RiskEventType(str, Enum):
    """Risk event type enumeration."""
    POSITION_LIMIT_BREACH = "position_limit_breach"
    EXPOSURE_LIMIT_BREACH = "exposure_limit_breach"
    DRAWDOWN_THRESHOLD = "drawdown_threshold"
    VOLATILITY_SPIKE = "volatility_spike"
    SPREAD_STRESS = "spread_stress"
    CORRELATION_BREACH = "correlation_breach"
    CIRCUIT_BREAKER = "circuit_breaker"
    EMERGENCY_STOP = "emergency_stop"


class RiskEventSeverity(str, Enum):
    """Risk event severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskEvent(BaseEntity):
    """Risk management event."""
    
    event_type: RiskEventType = Field(..., description="Type of risk event")
    severity: RiskEventSeverity = Field(..., description="Event severity")
    timestamp: datetime = Field(..., description="When event occurred")
    symbol: Optional[str] = Field(None, description="Associated symbol")
    details: Dict[str, Any] = Field(default_factory=dict, description="Event details")
    action_taken: Optional[str] = Field(None, description="Action taken in response")
    resolved: bool = Field(default=False, description="Whether event is resolved")
    resolved_timestamp: Optional[datetime] = Field(None, description="When event was resolved")


class RiskMetrics(TimeSeriesEntity):
    """Risk metrics snapshot."""
    
    var_1d: Decimal = Field(..., description="1-day Value at Risk")
    var_1w: Decimal = Field(..., description="1-week Value at Risk")
    var_1m: Decimal = Field(..., description="1-month Value at Risk")
    expected_shortfall: Decimal = Field(..., description="Expected shortfall (CVaR)")
    volatility: Decimal = Field(..., description="Portfolio volatility")
    beta: Decimal = Field(..., description="Portfolio beta")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    sortino_ratio: Optional[float] = Field(None, description="Sortino ratio")
    max_drawdown: Decimal = Field(..., description="Maximum drawdown")
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Position correlation matrix")


class CircuitBreaker(BaseEntity):
    """Circuit breaker configuration and state."""
    
    name: str = Field(..., description="Circuit breaker name")
    enabled: bool = Field(default=True, description="Whether circuit breaker is enabled")
    threshold: Decimal = Field(..., description="Trigger threshold")
    current_value: Decimal = Field(..., description="Current metric value")
    triggered: bool = Field(default=False, description="Whether circuit breaker is triggered")
    triggered_timestamp: Optional[datetime] = Field(None, description="When circuit breaker was triggered")
    actions: list[str] = Field(default_factory=list, description="Actions to take when triggered")
    cooldown_minutes: int = Field(..., description="Cooldown period in minutes")
    cooldown_until: Optional[datetime] = Field(None, description="Cooldown end timestamp")
    
    @property
    def is_active(self) -> bool:
        """Whether circuit breaker is currently active (triggered and in cooldown)."""
        if not self.triggered:
            return False
        if self.cooldown_until is None:
            return False
        return datetime.utcnow() < self.cooldown_until
    
    @property
    def threshold_breached(self) -> bool:
        """Whether the current value breaches the threshold."""
        return self.current_value >= self.threshold
