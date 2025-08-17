"""Risk management manager."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Optional

from ...utils.logging import logger
from ..data_models.market import MarketData
from ..data_models.risk import (
    CircuitBreaker,
    RiskEvent,
    RiskEventSeverity,
    RiskEventType,
)
from ..data_models.trading import Portfolio, Position


class RiskManager:
    """Comprehensive risk management system."""

    def __init__(self, config: dict):
        """Initialize risk manager.

        Args:
            config: Risk management configuration
        """
        self.config = config
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.risk_events: list[RiskEvent] = []
        self._setup_circuit_breakers()

    def _setup_circuit_breakers(self) -> None:
        """Setup circuit breakers from configuration."""
        circuit_config = self.config.get("circuit_breakers", {})

        # Daily loss circuit breaker
        self.circuit_breakers["daily_loss"] = CircuitBreaker(
            name="DailyLossBreaker",
            threshold=Decimal(str(circuit_config.get("daily_loss_threshold", 0.03))),
            current_value=Decimal("0"),
            actions=["reduce_position_sizes", "pause_new_positions"],
            cooldown_minutes=circuit_config.get("cooldown_minutes", 60),
        )

        # Drawdown circuit breaker
        self.circuit_breakers["drawdown"] = CircuitBreaker(
            name="DrawdownBreaker",
            threshold=Decimal(str(circuit_config.get("drawdown_threshold", 0.10))),
            current_value=Decimal("0"),
            actions=["flatten_positions", "pause_trading"],
            cooldown_minutes=circuit_config.get("cooldown_minutes", 120),
        )

        # Volatility circuit breaker
        self.circuit_breakers["volatility"] = CircuitBreaker(
            name="VolatilityBreaker",
            threshold=Decimal(str(circuit_config.get("volatility_threshold", 0.05))),
            current_value=Decimal("0"),
            actions=["reduce_leverage", "tighten_stops"],
            cooldown_minutes=circuit_config.get("cooldown_minutes", 30),
        )

        # Spread stress circuit breaker
        self.circuit_breakers["spread_stress"] = CircuitBreaker(
            name="SpreadStressBreaker",
            threshold=Decimal(str(circuit_config.get("spread_stress_threshold", 0.02))),
            current_value=Decimal("0"),
            actions=["switch_to_limit_orders", "reduce_trade_frequency"],
            cooldown_minutes=circuit_config.get("cooldown_minutes", 15),
        )

    def check_position_limits(
        self, position: Position, portfolio: Portfolio
    ) -> tuple[bool, Optional[str]]:
        """Check if position violates limits.

        Args:
            position: Position to check
            portfolio: Current portfolio state

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Check per-symbol position size
        max_position_size = self.config.get("position", {}).get(
            "max_size_per_symbol", 0.1
        )
        position_value = abs(float(position.market_value))
        portfolio_value = float(portfolio.equity)

        if portfolio_value > 0:
            position_ratio = position_value / portfolio_value
            if position_ratio > max_position_size:
                return (
                    False,
                    f"Position size {position_ratio:.2%} exceeds limit {max_position_size:.2%}",
                )

        return True, None

    def check_portfolio_limits(
        self, portfolio: Portfolio
    ) -> tuple[bool, Optional[str]]:
        """Check if portfolio violates limits.

        Args:
            portfolio: Portfolio to check

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Check gross exposure
        max_gross_exposure = self.config.get("portfolio", {}).get(
            "max_gross_exposure", 0.5
        )
        if portfolio.equity > 0:
            gross_exposure_ratio = float(portfolio.exposure_gross) / float(
                portfolio.equity
            )
            if gross_exposure_ratio > max_gross_exposure:
                return (
                    False,
                    f"Gross exposure {gross_exposure_ratio:.2%} exceeds limit {max_gross_exposure:.2%}",
                )

        # Check cash reserve
        min_cash_reserve = self.config.get("portfolio", {}).get("min_cash_reserve", 0.1)
        if portfolio.equity > 0:
            cash_ratio = float(portfolio.cash) / float(portfolio.equity)
            if cash_ratio < min_cash_reserve:
                return (
                    False,
                    f"Cash reserve {cash_ratio:.2%} below minimum {min_cash_reserve:.2%}",
                )

        return True, None

    def update_circuit_breakers(
        self, portfolio: Portfolio, market_data: Optional[MarketData] = None
    ) -> list[str]:
        """Update circuit breaker values and check for triggers.

        Args:
            portfolio: Current portfolio state
            market_data: Current market data (optional)

        Returns:
            List of triggered circuit breaker names
        """
        triggered_breakers = []

        # Update daily loss circuit breaker
        if portfolio.equity > 0:
            daily_loss_ratio = abs(float(portfolio.daily_pnl)) / float(portfolio.equity)
            self.circuit_breakers["daily_loss"].current_value = Decimal(
                str(daily_loss_ratio)
            )

            if self.circuit_breakers["daily_loss"].threshold_breached:
                self._trigger_circuit_breaker("daily_loss", daily_loss_ratio)
                triggered_breakers.append("daily_loss")

        # Update drawdown circuit breaker
        if portfolio.equity > 0:
            drawdown_ratio = float(portfolio.drawdown) / float(portfolio.equity)
            self.circuit_breakers["drawdown"].current_value = Decimal(
                str(drawdown_ratio)
            )

            if self.circuit_breakers["drawdown"].threshold_breached:
                self._trigger_circuit_breaker("drawdown", drawdown_ratio)
                triggered_breakers.append("drawdown")

        # Update volatility circuit breaker (if market data available)
        if market_data and hasattr(market_data, "features"):
            volatility = None
            for feature in market_data.features:
                if feature.feature_name == "volatility_rolling":
                    volatility = feature.feature_value
                    break

            if volatility is not None:
                self.circuit_breakers["volatility"].current_value = Decimal(
                    str(volatility)
                )

                if self.circuit_breakers["volatility"].threshold_breached:
                    self._trigger_circuit_breaker("volatility", volatility)
                    triggered_breakers.append("volatility")

        # Update spread stress circuit breaker
        if market_data and market_data.spread_relative is not None:
            spread_ratio = market_data.spread_relative
            self.circuit_breakers["spread_stress"].current_value = Decimal(
                str(spread_ratio)
            )

            if self.circuit_breakers["spread_stress"].threshold_breached:
                self._trigger_circuit_breaker("spread_stress", spread_ratio)
                triggered_breakers.append("spread_stress")

        return triggered_breakers

    def _trigger_circuit_breaker(self, breaker_name: str, value: float) -> None:
        """Trigger a circuit breaker.

        Args:
            breaker_name: Name of the circuit breaker
            value: Current value that triggered the breaker
        """
        breaker = self.circuit_breakers[breaker_name]

        if not breaker.triggered:
            breaker.triggered = True
            breaker.triggered_timestamp = datetime.now(UTC)

            # Log risk event
            event = RiskEvent(
                event_type=RiskEventType.CIRCUIT_BREAKER,
                severity=RiskEventSeverity.HIGH,
                timestamp=datetime.now(UTC),
                details={
                    "breaker_name": breaker_name,
                    "threshold": float(breaker.threshold),
                    "current_value": value,
                    "actions": breaker.actions,
                },
            )

            self.risk_events.append(event)
            logger.log_risk_event(event.dict())

            logger.logger.warning(
                f"Circuit breaker {breaker_name} triggered: {value:.4f} >= {float(breaker.threshold):.4f}"
            )

    def get_risk_metrics(self, portfolio: Portfolio) -> dict:
        """Calculate current risk metrics.

        Args:
            portfolio: Current portfolio state

        Returns:
            Dictionary of risk metrics
        """
        metrics = {}

        if portfolio.equity > 0:
            # Position concentration
            total_position_value = sum(
                abs(float(pos.market_value)) for pos in portfolio.positions
            )
            metrics["position_concentration"] = total_position_value / float(
                portfolio.equity
            )

            # Leverage
            metrics["leverage"] = float(portfolio.leverage)

            # Cash ratio
            metrics["cash_ratio"] = float(portfolio.cash_ratio)

            # Exposure ratios
            metrics["gross_exposure_ratio"] = float(portfolio.exposure_gross) / float(
                portfolio.equity
            )
            metrics["net_exposure_ratio"] = float(portfolio.exposure_net) / float(
                portfolio.equity
            )

        # Circuit breaker status
        metrics["circuit_breakers"] = {
            name: {
                "triggered": breaker.triggered,
                "active": breaker.is_active,
                "current_value": float(breaker.current_value),
                "threshold": float(breaker.threshold),
            }
            for name, breaker in self.circuit_breakers.items()
        }

        return metrics

    def can_trade(self, portfolio: Portfolio) -> tuple[bool, Optional[str]]:
        """Check if trading is allowed.

        Args:
            portfolio: Current portfolio state

        Returns:
            Tuple of (can_trade, reason_if_not)
        """
        # Check if any circuit breakers are active
        active_breakers = [
            name for name, breaker in self.circuit_breakers.items() if breaker.is_active
        ]

        if active_breakers:
            return (
                False,
                f"Trading blocked by active circuit breakers: {', '.join(active_breakers)}",
            )

        # Check portfolio limits
        portfolio_valid, portfolio_reason = self.check_portfolio_limits(portfolio)
        if not portfolio_valid:
            return False, f"Portfolio limits violated: {portfolio_reason}"

        return True, None

    def get_position_size_limit(self, symbol: str, portfolio: Portfolio) -> Decimal:
        """Get maximum position size for a symbol.

        Args:
            symbol: Trading symbol
            portfolio: Current portfolio state

        Returns:
            Maximum position size in USD
        """
        max_position_ratio = self.config.get("position", {}).get(
            "max_size_per_symbol", 0.1
        )
        max_position_value = portfolio.equity * Decimal(str(max_position_ratio))

        # Reduce position size if circuit breakers are active
        active_breakers = [
            name for name, breaker in self.circuit_breakers.items() if breaker.is_active
        ]
        if active_breakers:
            # Reduce by 50% if any circuit breaker is active
            max_position_value *= Decimal("0.5")

        return max_position_value
