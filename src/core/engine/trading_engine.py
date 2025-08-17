"""Main trading engine that coordinates all components."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Optional

from ...utils.config import config_manager
from ...utils.logging import logger
from ..data_models.market import Bar, Feature, MarketData
from ..data_models.trading import Order, OrderSide, OrderType, Portfolio, Position
from ..features.technical import FeatureEngine
from ..risk.manager import RiskManager
from ..strategy.base import Strategy


class TradingEngine:
    """Main trading engine that coordinates all components."""

    def __init__(self, config: dict):
        """Initialize trading engine.

        Args:
            config: Trading engine configuration
        """
        self.config = config
        self.enabled = True
        self.running = False

        # Core components
        self.feature_engine = FeatureEngine()
        self.risk_manager = RiskManager(config.get("risk", {}))

        # Strategies
        self.strategies: dict[str, Strategy] = {}
        self._setup_strategies()

        # Portfolio and state
        self.portfolio: Optional[Portfolio] = None
        self.market_data_cache: dict[str, list[Bar]] = {}
        self.active_orders: dict[str, Order] = {}
        self.positions: dict[str, Position] = {}

        # Performance tracking
        self.trades: list[dict] = []
        self.performance_metrics: dict = {}

        logger.logger.info("Trading engine initialized")

    def _setup_strategies(self) -> None:
        """Setup trading strategies from configuration."""
        strategy_configs = config_manager.get("strategy", {})

        # For now, we'll create a default mean reversion strategy
        # In a full implementation, you'd load strategies dynamically
        from ..strategy.mean_reversion import MeanReversionStrategy

        default_strategy = MeanReversionStrategy(strategy_configs)
        self.strategies[default_strategy.name] = default_strategy

        logger.logger.info(f"Loaded strategy: {default_strategy.name}")

    def initialize_portfolio(self, initial_capital: Decimal) -> None:
        """Initialize portfolio with starting capital.

        Args:
            initial_capital: Starting capital in USD
        """
        self.portfolio = Portfolio(
            timestamp=datetime.now(UTC),
            cash=initial_capital,
            equity=initial_capital,
            exposure_gross=Decimal("0"),
            exposure_net=Decimal("0"),
            peak_equity=initial_capital,
        )

        logger.logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")

    def add_market_data(self, symbol: str, bar: Bar) -> None:
        """Add new market data bar.

        Args:
            symbol: Trading symbol
            bar: OHLCV bar data
        """
        if symbol not in self.market_data_cache:
            self.market_data_cache[symbol] = []

        self.market_data_cache[symbol].append(bar)

        # Keep only recent bars (configurable lookback)
        max_bars = self.config.get("data", {}).get("feature_lookback_bars", 100)
        if len(self.market_data_cache[symbol]) > max_bars:
            self.market_data_cache[symbol] = self.market_data_cache[symbol][-max_bars:]

    def process_market_data(self, symbol: str) -> Optional[MarketData]:
        """Process market data and generate features.

        Args:
            symbol: Trading symbol

        Returns:
            MarketData with features, or None if insufficient data
        """
        if symbol not in self.market_data_cache:
            return None

        bars = self.market_data_cache[symbol]
        if len(bars) < 20:  # Need minimum bars for features
            return None

        # Calculate features
        features_dict = self.feature_engine.calculate_features(bars)

        # Convert to Feature objects
        features = []
        for feature_name, feature_value in features_dict.items():
            if not isinstance(feature_value, (int, float)):
                continue

            # Check for NaN values
            try:
                if feature_value != feature_value:  # NaN check
                    continue
            except:
                continue

            feature = Feature(
                timestamp=bars[-1].timestamp,
                symbol=symbol,
                feature_name=feature_name,
                feature_value=float(feature_value),
                source="technical",
            )
            features.append(feature)

        # Create MarketData object
        latest_bar = bars[-1]
        market_data = MarketData(
            timestamp=latest_bar.timestamp,
            symbol=symbol,
            bar=latest_bar,
            features=features,
            timestamp_received=datetime.now(UTC),
        )

        return market_data

    def generate_signals(self, symbol: str) -> dict:
        """Generate trading signals for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary of trading signals
        """
        if not self.portfolio:
            return {}

        # Process market data
        market_data = self.process_market_data(symbol)
        if not market_data:
            return {}

        # Generate signals from all strategies
        all_signals = {}
        for strategy_name, strategy in self.strategies.items():
            if not strategy.is_enabled():
                continue

            try:
                signals = strategy.generate_signals(market_data, self.portfolio)
                if signals:
                    all_signals[strategy_name] = signals
            except Exception as e:
                logger.logger.error(
                    f"Error generating signals for {strategy_name}: {e}"
                )

        return all_signals

    def execute_signals(self, signals: dict) -> list[Order]:
        """Execute trading signals.

        Args:
            signals: Trading signals from strategies

        Returns:
            List of created orders
        """
        if not self.portfolio or not self.enabled:
            return []

        orders = []

        # Check if trading is allowed
        can_trade, reason = self.risk_manager.can_trade(self.portfolio)
        if not can_trade:
            logger.logger.warning(f"Trading blocked: {reason}")
            return []

        # Process entry signals
        for strategy_name, strategy_signals in signals.items():
            entry_signals = strategy_signals.get("entry_signals", {})

            for symbol, entry_details in entry_signals.items():
                try:
                    order = self._create_entry_order(
                        symbol, entry_details, strategy_name
                    )
                    if order:
                        orders.append(order)
                        self.active_orders[order.id] = order
                except Exception as e:
                    logger.logger.error(f"Error creating entry order: {e}")

        # Process exit signals
        for strategy_name, strategy_signals in signals.items():
            exit_signals = strategy_signals.get("exit_signals", {})

            for symbol, exit_details in exit_signals.items():
                try:
                    order = self._create_exit_order(symbol, exit_details, strategy_name)
                    if order:
                        orders.append(order)
                        self.active_orders[order.id] = order
                except Exception as e:
                    logger.logger.error(f"Error creating exit order: {e}")

        return orders

    def _create_entry_order(
        self, symbol: str, entry_details: dict, strategy_name: str
    ) -> Optional[Order]:
        """Create entry order based on signal.

        Args:
            symbol: Trading symbol
            entry_details: Entry signal details
            strategy_name: Name of strategy that generated signal

        Returns:
            Order object or None if creation failed
        """
        direction = entry_details.get("direction")
        position_size = entry_details.get("position_size", 0)

        if not direction or position_size <= 0:
            return None

        # Check position size limits
        max_size = self.risk_manager.get_position_size_limit(symbol, self.portfolio)
        if position_size > float(max_size):
            position_size = float(max_size)
            logger.logger.warning(f"Position size reduced to limit: {position_size}")

        # Create order
        order = Order(
            timestamp=datetime.now(UTC),
            symbol=symbol,
            run_id=f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            side=OrderSide.BUY if direction == "long" else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal(str(position_size)),
            reason_code=f"{strategy_name}_entry",
        )

        logger.logger.info(
            f"Created {direction} entry order for {symbol}: {position_size}"
        )
        return order

    def _create_exit_order(
        self, symbol: str, exit_details: dict, strategy_name: str
    ) -> Optional[Order]:
        """Create exit order based on signal.

        Args:
            symbol: Trading symbol
            exit_details: Exit signal details
            strategy_name: Name of strategy that generated signal

        Returns:
            Order object or None if creation failed
        """
        # Find existing position
        position = self.positions.get(symbol)
        if not position or position.is_flat:
            return None

        # Create exit order
        order = Order(
            timestamp=datetime.now(UTC),
            symbol=symbol,
            run_id=f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            side=OrderSide.SELL if position.is_long else OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=abs(position.quantity),
            reason_code=f"{strategy_name}_exit",
        )

        logger.logger.info(f"Created exit order for {symbol}: {abs(position.quantity)}")
        return order

    def update_portfolio(self, market_data: MarketData) -> None:
        """Update portfolio with current market data.

        Args:
            market_data: Current market data
        """
        if not self.portfolio:
            return

        # Update position values
        for position in self.portfolio.positions:
            if position.symbol == market_data.symbol:
                current_price = market_data.bar.close
                position.market_value = current_price * abs(position.quantity)

                # Calculate unrealized P&L
                if position.is_long:
                    position.unrealized_pnl = (
                        current_price - position.average_cost
                    ) * position.quantity
                else:
                    position.unrealized_pnl = (
                        position.average_cost - current_price
                    ) * abs(position.quantity)

        # Recalculate portfolio metrics
        self._recalculate_portfolio_metrics()

        # Update risk metrics
        self.risk_manager.update_circuit_breakers(self.portfolio, market_data)

    def _recalculate_portfolio_metrics(self) -> None:
        """Recalculate portfolio metrics."""
        if not self.portfolio:
            return

        # Calculate exposure
        total_position_value = sum(
            abs(float(pos.market_value)) for pos in self.portfolio.positions
        )
        self.portfolio.exposure_gross = Decimal(str(total_position_value))

        # Calculate net exposure
        net_exposure = sum(float(pos.market_value) for pos in self.portfolio.positions)
        self.portfolio.exposure_net = Decimal(str(net_exposure))

        # Calculate equity
        total_pnl = sum(float(pos.unrealized_pnl) for pos in self.portfolio.positions)
        self.portfolio.equity = self.portfolio.cash + Decimal(str(total_pnl))

        # Update peak equity
        if self.portfolio.equity > self.portfolio.peak_equity:
            self.portfolio.peak_equity = self.portfolio.equity

        # Calculate drawdown
        if self.portfolio.peak_equity > 0:
            drawdown = (
                self.portfolio.peak_equity - self.portfolio.equity
            ) / self.portfolio.peak_equity
            self.portfolio.drawdown = drawdown

    def get_performance_summary(self) -> dict:
        """Get performance summary.

        Returns:
            Performance metrics dictionary
        """
        summary = {
            "portfolio": {
                "equity": float(self.portfolio.equity) if self.portfolio else 0,
                "cash": float(self.portfolio.cash) if self.portfolio else 0,
                "exposure": float(self.portfolio.exposure_gross)
                if self.portfolio
                else 0,
                "drawdown": float(self.portfolio.drawdown) if self.portfolio else 0,
            },
            "risk_metrics": self.risk_manager.get_risk_metrics(self.portfolio)
            if self.portfolio
            else {},
            "strategies": {},
        }

        # Add strategy performance
        for strategy_name, strategy in self.strategies.items():
            summary["strategies"][strategy_name] = strategy.get_performance_summary()

        return summary

    def start(self) -> None:
        """Start the trading engine."""
        if self.running:
            logger.logger.warning("Trading engine already running")
            return

        self.running = True
        logger.logger.info("Trading engine started")

    def stop(self) -> None:
        """Stop the trading engine."""
        if not self.running:
            logger.logger.warning("Trading engine not running")
            return

        self.running = False
        logger.logger.info("Trading engine stopped")

    def is_running(self) -> bool:
        """Check if trading engine is running.

        Returns:
            True if running
        """
        return self.running
