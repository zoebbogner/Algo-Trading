"""Portfolio management for algorithmic trading."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Trading position."""
    symbol: str
    quantity: Decimal
    average_cost: Decimal
    timestamp: datetime
    side: str = "LONG"  # LONG or SHORT

    def __post_init__(self):
        """Validate position data."""
        if self.quantity <= 0:
            raise ValueError("Position quantity must be positive")
        if self.average_cost <= 0:
            raise ValueError("Average cost must be positive")


@dataclass
class Portfolio:
    """Portfolio management."""
    initial_capital: float
    symbols: list[str]
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Initialize portfolio."""
        self.cash = self.initial_capital
        self.peak_value = self.initial_capital
        self.total_pnl = 0.0

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if portfolio has a position in symbol."""
        return symbol in self.positions

    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Get current value of position."""
        position = self.get_position(symbol)
        if not position:
            return 0.0

        if position.side == "LONG":
            return float(position.quantity) * current_price
        else:  # SHORT
            return float(position.quantity) * (2 * float(position.average_cost) - current_price)

    def get_total_value(self, current_prices: dict[str, float]) -> float:
        """Get total portfolio value."""
        total_value = self.cash

        for symbol in self.symbols:
            if symbol in current_prices:
                total_value += self.get_position_value(symbol, current_prices[symbol])

        return total_value

    def get_unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        """Get unrealized P&L."""
        unrealized_pnl = 0.0

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                if position.side == "LONG":
                    unrealized_pnl += float(position.quantity) * (current_price - float(position.average_cost))
                else:  # SHORT
                    unrealized_pnl += float(position.quantity) * (float(position.average_cost) - current_price)

        return unrealized_pnl

    def get_realized_pnl(self) -> float:
        """Get realized P&L."""
        return self.total_pnl

    def get_total_pnl(self, current_prices: dict[str, float]) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.get_realized_pnl() + self.get_unrealized_pnl(current_prices)

    def get_return_pct(self, current_prices: dict[str, float]) -> float:
        """Get total return percentage."""
        total_value = self.get_total_value(current_prices)
        if self.initial_capital > 0:
            return ((total_value - self.initial_capital) / self.initial_capital) * 100
        return 0.0

    def get_drawdown(self, current_prices: dict[str, float]) -> float:
        """Get current drawdown percentage."""
        total_value = self.get_total_value(current_prices)
        if total_value > self.peak_value:
            self.peak_value = total_value
            return 0.0

        if self.peak_value > 0:
            return ((self.peak_value - total_value) / self.peak_value) * 100
        return 0.0

    def add_position(self, symbol: str, quantity: Decimal, price: Decimal, side: str = "LONG"):
        """Add or update position."""
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not in portfolio symbols")

        if side not in ["LONG", "SHORT"]:
            raise ValueError("Side must be LONG or SHORT")

        # Calculate cost
        cost = float(quantity) * float(price)

        if cost > self.cash:
            raise ValueError(f"Insufficient cash. Need ${cost:.2f}, have ${self.cash:.2f}")

        # Update cash
        self.cash -= cost

        # Update position
        if symbol in self.positions:
            # Average down/up existing position
            existing = self.positions[symbol]
            total_quantity = existing.quantity + quantity
            total_cost = (existing.quantity * existing.average_cost) + (quantity * price)
            new_average_cost = total_cost / total_quantity

            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=total_quantity,
                average_cost=new_average_cost,
                timestamp=datetime.now(),
                side=side
            )
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                average_cost=price,
                timestamp=datetime.now(),
                side=side
            )

        # Record trade
        self.trades.append({
            "timestamp": datetime.now(),
            "symbol": symbol,
            "action": "BUY" if side == "LONG" else "SELL_SHORT",
            "quantity": float(quantity),
            "price": float(price),
            "cost": cost
        })

        logger.info(f"Added {side} position: {symbol} {float(quantity)} @ ${float(price):.2f}")

    def close_position(self, symbol: str, quantity: Decimal, price: Decimal):
        """Close part or all of a position."""
        if symbol not in self.positions:
            raise ValueError(f"No position in {symbol}")

        position = self.positions[symbol]

        if quantity > position.quantity:
            raise ValueError(f"Cannot close {float(quantity)} when position has {float(position.quantity)}")

        # Calculate proceeds
        proceeds = float(quantity) * float(price)

        # Calculate P&L
        if position.side == "LONG":
            pnl = proceeds - (float(quantity) * float(position.average_cost))
        else:  # SHORT
            pnl = (float(quantity) * float(position.average_cost)) - proceeds

        # Update cash
        self.cash += proceeds

        # Update position
        remaining_quantity = position.quantity - quantity

        if remaining_quantity <= 0:
            # Close entire position
            del self.positions[symbol]
        else:
            # Partial close
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=remaining_quantity,
                average_cost=position.average_cost,
                timestamp=datetime.now(),
                side=position.side
            )

        # Update P&L
        self.total_pnl += pnl

        # Record trade
        self.trades.append({
            "timestamp": datetime.now(),
            "symbol": symbol,
            "action": "SELL" if position.side == "LONG" else "BUY_TO_COVER",
            "quantity": float(quantity),
            "price": float(price),
            "proceeds": proceeds,
            "pnl": pnl
        })

        logger.info(f"Closed position: {symbol} {float(quantity)} @ ${float(price):.2f}, P&L: ${pnl:.2f}")

    def update(self, market_data: dict[str, Any]):
        """Update portfolio with current market data."""
        # This would typically update positions with current market prices
        # For now, just update timestamp
        self.timestamp = datetime.now()

    def get_summary(self, current_prices: dict[str, float]) -> dict[str, Any]:
        """Get portfolio summary."""
        total_value = self.get_total_value(current_prices)
        unrealized_pnl = self.get_unrealized_pnl(current_prices)
        total_pnl = self.get_total_pnl(current_prices)
        return_pct = self.get_return_pct(current_prices)
        drawdown = self.get_drawdown(current_prices)

        return {
            "timestamp": self.timestamp.isoformat(),
            "initial_capital": self.initial_capital,
            "current_value": total_value,
            "cash": self.cash,
            "positions_count": len(self.positions),
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": self.total_pnl,
            "total_pnl": total_pnl,
            "return_pct": return_pct,
            "drawdown_pct": drawdown,
            "peak_value": self.peak_value
        }
