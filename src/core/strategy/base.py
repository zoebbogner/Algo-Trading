"""Base strategy class for trading strategies."""

import logging
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Optional

from ..data_models.market import MarketData
from ..data_models.trading import Portfolio, Position

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, config: dict):
        """Initialize strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.name = config.get("name", "UnknownStrategy")
        self.version = config.get("version", "1.0.0")
        self.enabled = config.get("enabled", True)

        # Strategy parameters
        self.lookback_period = config.get("lookback_period", 100)
        self.max_positions = config.get("max_positions", 5)

        # Entry/exit conditions
        self.entry_config = config.get("entry", {})
        self.exit_config = config.get("exit", {})
        self.sizing_config = config.get("sizing", {})

        # Performance tracking
        self.trades: list[dict] = []
        self.performance_metrics: dict = {}

        logger.info(f"Initialized strategy: {self.name} v{self.version}")

    @abstractmethod
    def generate_signals(self, market_data: MarketData, portfolio: Portfolio) -> dict:
        """Generate trading signals from market data.

        Args:
            market_data: Current market data with features
            portfolio: Current portfolio state

        Returns:
            Dictionary containing trading signals
        """

    @abstractmethod
    def should_enter(
        self, symbol: str, market_data: MarketData, portfolio: Portfolio
    ) -> tuple[bool, dict]:
        """Determine if we should enter a position.

        Args:
            symbol: Trading symbol
            market_data: Current market data
            portfolio: Current portfolio state

        Returns:
            Tuple of (should_enter, entry_details)
        """

    @abstractmethod
    def should_exit(
        self, position: Position, market_data: MarketData, portfolio: Portfolio
    ) -> tuple[bool, dict]:
        """Determine if we should exit a position.

        Args:
            position: Current position
            market_data: Current market data
            portfolio: Current portfolio state

        Returns:
            Tuple of (should_exit, exit_details)
        """

    def calculate_position_size(
        self, symbol: str, market_data: MarketData, portfolio: Portfolio
    ) -> Decimal:
        """Calculate position size for entry.

        Args:
            symbol: Trading symbol
            market_data: Current market data
            portfolio: Current portfolio state

        Returns:
            Position size in base currency
        """
        sizing_method = self.sizing_config.get("method", "fixed")
        base_size = self.sizing_config.get("base_size", 0.02)
        max_size = self.sizing_config.get("max_size", 0.1)

        if sizing_method == "fixed":
            # Fixed percentage of portfolio
            position_value = portfolio.equity * Decimal(str(base_size))
            return position_value

        elif sizing_method == "volatility":
            # Adjust size based on volatility
            volatility = self._get_volatility(market_data)
            if volatility and volatility > 0:
                # Reduce size for high volatility
                adjusted_size = base_size / volatility
                adjusted_size = min(adjusted_size, max_size)
                position_value = portfolio.equity * Decimal(str(adjusted_size))
                return position_value

        # Default to fixed sizing
        position_value = portfolio.equity * Decimal(str(base_size))
        return position_value

    def _get_volatility(self, market_data: MarketData) -> Optional[float]:
        """Get volatility from market data features.

        Args:
            market_data: Market data with features

        Returns:
            Volatility value or None if not available
        """
        for feature in market_data.features:
            if feature.feature_name == "volatility_rolling":
                return feature.feature_value
        return None

    def update_performance(self, trade: dict) -> None:
        """Update performance metrics with new trade.

        Args:
            trade: Trade information dictionary
        """
        self.trades.append(trade)
        self._calculate_performance_metrics()

    def _calculate_performance_metrics(self) -> None:
        """Calculate performance metrics from trades."""
        if not self.trades:
            return

        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.get("pnl", 0) > 0])
        losing_trades = len([t for t in self.trades if t.get("pnl", 0) < 0])

        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # P&L metrics
        total_pnl = sum(t.get("pnl", 0) for t in self.trades)
        avg_win = (
            sum(t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) > 0)
            / winning_trades
            if winning_trades > 0
            else 0
        )
        avg_loss = (
            sum(t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) < 0)
            / losing_trades
            if losing_trades > 0
            else 0
        )

        # Risk metrics
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        self.performance_metrics = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
        }

    def get_performance_summary(self) -> dict:
        """Get performance summary.

        Returns:
            Performance metrics dictionary
        """
        return self.performance_metrics.copy()

    def reset_performance(self) -> None:
        """Reset performance tracking."""
        self.trades.clear()
        self.performance_metrics.clear()
        logger.info(f"Reset performance tracking for strategy: {self.name}")

    def is_enabled(self) -> bool:
        """Check if strategy is enabled.

        Returns:
            True if strategy is enabled
        """
        return self.enabled

    def enable(self) -> None:
        """Enable the strategy."""
        self.enabled = True
        logger.info(f"Enabled strategy: {self.name}")

    def disable(self) -> None:
        """Disable the strategy."""
        self.enabled = False
        logger.info(f"Disabled strategy: {self.name}")

    def get_config(self) -> dict:
        """Get strategy configuration.

        Returns:
            Strategy configuration dictionary
        """
        return self.config.copy()

    def update_config(self, new_config: dict) -> None:
        """Update strategy configuration.

        Args:
            new_config: New configuration values
        """
        self.config.update(new_config)
        logger.info(f"Updated configuration for strategy: {self.name}")
