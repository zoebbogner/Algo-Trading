"""Trading data models."""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import Field

from .base import BaseEntity, TimeSeriesEntity


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Order(TimeSeriesEntity):
    """Trading order."""
    
    run_id: str = Field(..., description="Trading run identifier")
    side: OrderSide = Field(..., description="Order side (buy/sell)")
    order_type: OrderType = Field(..., description="Order type")
    quantity: Decimal = Field(..., description="Order quantity")
    price: Optional[Decimal] = Field(None, description="Limit price (for limit orders)")
    stop_price: Optional[Decimal] = Field(None, description="Stop price (for stop orders)")
    status: OrderStatus = Field(default=OrderStatus.PENDING, description="Order status")
    filled_quantity: Decimal = Field(default=Decimal("0"), description="Filled quantity")
    average_fill_price: Optional[Decimal] = Field(None, description="Average fill price")
    reason_code: Optional[str] = Field(None, description="Reason for order")
    broker_order_id: Optional[str] = Field(None, description="Broker order ID")
    
    @property
    def remaining_quantity(self) -> Decimal:
        """Remaining unfilled quantity."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        """Whether the order is completely filled or final."""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]


class Fill(TimeSeriesEntity):
    """Order fill."""
    
    order_id: str = Field(..., description="Associated order ID")
    fill_price: Decimal = Field(..., description="Fill price")
    fill_quantity: Decimal = Field(..., description="Fill quantity")
    fill_fee: Decimal = Field(default=Decimal("0"), description="Fill fee")
    slippage_bps: Optional[int] = Field(None, description="Slippage in basis points")
    timestamp_filled: datetime = Field(..., description="When fill occurred")
    
    @property
    def fill_value(self) -> Decimal:
        """Total fill value (price * quantity)."""
        return self.fill_price * self.fill_quantity


class Position(TimeSeriesEntity):
    """Trading position."""
    
    quantity: Decimal = Field(..., description="Position quantity (positive for long, negative for short)")
    average_cost: Decimal = Field(..., description="Average cost basis")
    unrealized_pnl: Decimal = Field(default=Decimal("0"), description="Unrealized P&L")
    realized_pnl: Decimal = Field(default=Decimal("0"), description="Realized P&L")
    market_value: Decimal = Field(..., description="Current market value")
    entry_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Position entry timestamp")
    last_update_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last position update timestamp")
    
    @property
    def is_long(self) -> bool:
        """Whether position is long."""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Whether position is short."""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """Whether position is flat (no position)."""
        return self.quantity == 0
    
    @property
    def total_pnl(self) -> Decimal:
        """Total P&L (unrealized + realized)."""
        return self.unrealized_pnl + self.realized_pnl


class Portfolio(BaseEntity):
    """Portfolio snapshot."""
    
    timestamp: datetime = Field(..., description="Portfolio snapshot timestamp")
    cash: Decimal = Field(..., description="Available cash")
    equity: Decimal = Field(..., description="Total portfolio equity")
    exposure_gross: Decimal = Field(..., description="Gross exposure (sum of absolute position values)")
    exposure_net: Decimal = Field(..., description="Net exposure (sum of position values)")
    drawdown: Decimal = Field(default=Decimal("0"), description="Current drawdown from peak")
    peak_equity: Decimal = Field(..., description="Peak equity value")
    total_pnl: Decimal = Field(default=Decimal("0"), description="Total P&L")
    daily_pnl: Decimal = Field(default=Decimal("0"), description="Daily P&L")
    positions: list[Position] = Field(default_factory=list, description="Current positions")
    
    @property
    def leverage(self) -> float:
        """Portfolio leverage (exposure / equity)."""
        if self.equity > 0:
            return float(self.exposure_gross / self.equity)
        return 0.0
    
    @property
    def cash_ratio(self) -> float:
        """Cash ratio (cash / equity)."""
        if self.equity > 0:
            return float(self.cash / self.equity)
        return 0.0
