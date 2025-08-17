"""Main trading bot that orchestrates the entire trading process."""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.config import TradingConfig
from src.data import DataCollector, DataStorage
from src.ml.models import MLModelManager
from src.reporting import ReportGenerator
from src.trading.engine import TradingEngine
from src.trading.portfolio import Portfolio
from src.trading.risk import RiskManager
from src.trading.strategies import Strategy

logger = logging.getLogger(__name__)


class TradingBot:
    """Main algorithmic trading bot that runs continuously."""

    def __init__(self, config: TradingConfig):
        """Initialize the trading bot."""
        self.config = config
        self.running = False
        self.data_collector: Optional[DataCollector] = None
        self.data_storage: Optional[DataStorage] = None
        self.trading_engine: Optional[TradingEngine] = None
        self.portfolio: Optional[Portfolio] = None
        self.risk_manager: Optional[RiskManager] = None
        self.ml_models: Optional[MLModelManager] = None
        self.report_generator: Optional[ReportGenerator] = None

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._setup_logging()
        self._initialize_components()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Clear previous logs
        for log_file in log_dir.glob("*.log"):
            log_file.unlink()

        # Setup new log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"trading_bot_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        logger.info(f"Trading bot logging initialized: {log_file}")

    def _initialize_components(self):
        """Initialize all trading components."""
        logger.info("Initializing trading components...")

        try:
            # Initialize data components
            self.data_storage = DataStorage(self.config.data.base_path)
            self.data_collector = DataCollector(self.config.data.collectors)

            # Initialize trading components
            self.portfolio = Portfolio(
                initial_capital=self.config.trading.initial_capital,
                symbols=self.config.trading.symbols
            )

            self.risk_manager = RiskManager(
                max_position_size=self.config.risk.max_position_size,
                max_portfolio_risk=self.config.risk.max_portfolio_risk,
                stop_loss_pct=self.config.risk.stop_loss_pct
            )

            self.trading_engine = TradingEngine(
                portfolio=self.portfolio,
                risk_manager=self.risk_manager,
                strategies=self._load_strategies()
            )

            # Initialize ML models
            self.ml_models = MLModelManager(self.config.ml)

            # Initialize reporting
            self.report_generator = ReportGenerator(
                output_dir=self.config.reporting.output_dir
            )

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def _load_strategies(self) -> dict[str, Strategy]:
        """Load and initialize trading strategies."""
        strategies = {}

        for strategy_config in self.config.trading.strategies:
            strategy_name = strategy_config.name
            strategy_class = self._get_strategy_class(strategy_config.type)

            if strategy_class:
                strategy = strategy_class(strategy_config)
                strategies[strategy_name] = strategy
                logger.info(f"Loaded strategy: {strategy_name}")
            else:
                logger.warning(f"Unknown strategy type: {strategy_config.type}")

        return strategies

    def _get_strategy_class(self, strategy_type: str):
        """Get strategy class by type."""
        strategy_map = {
            "momentum": "MomentumStrategy",
            "mean_reversion": "MeanReversionStrategy"
        }

        class_name = strategy_map.get(strategy_type)
        if class_name:
            # Import strategy class dynamically
            module = __import__("src.trading.strategies", fromlist=[class_name])
            return getattr(module, class_name)

        return None

    async def run(self):
        """Main trading bot loop."""
        logger.info("Starting trading bot...")
        self.running = True

        try:
            # Clear previous run data
            self.data_storage.clear_previous_run()

            # Collect historical data
            await self._collect_historical_data()

            # Train ML models
            await self._train_ml_models()

            # Main trading loop
            await self._trading_loop()

        except Exception as e:
            logger.error(f"Trading bot error: {e}")
            raise
        finally:
            await self._cleanup()

    async def _collect_historical_data(self):
        """Collect historical data for training and testing."""
        logger.info("Collecting historical data...")

        # Calculate date ranges: 5 years total, 4 for training, 1 for testing
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5 * 365)  # 5 years
        train_end_date = start_date + timedelta(days=4 * 365)  # 4 years

        logger.info(f"Data collection period: {start_date.date()} to {end_date.date()}")
        logger.info(f"Training period: {start_date.date()} to {train_end_date.date()}")
        logger.info(f"Testing period: {train_end_date.date()} to {end_date.date()}")

        # Collect data for all symbols
        async with self.data_collector as collector:
            results = await collector.collect_multiple_symbols(
                symbols=self.config.trading.symbols,
                start_date=start_date,
                end_date=end_date,
                interval=self.config.data.interval
            )

        # Save collected data
        for symbol, response in results.items():
            if response.success and response.data.bars:
                self.data_storage.save_market_data(symbol, response.data)
                logger.info(f"Saved {len(response.data.bars)} bars for {symbol}")
            else:
                logger.warning(f"Failed to collect data for {symbol}: {response.error_message}")

        # Generate data summary
        summary = self.data_storage.export_data_summary()
        logger.info(f"Data collection summary: {summary}")

    async def _train_ml_models(self):
        """Train ML models on historical data."""
        logger.info("Training ML models...")

        try:
            await self.ml_models.train_all_models()
            logger.info("ML models trained successfully")
        except Exception as e:
            logger.error(f"ML model training failed: {e}")

    async def _trading_loop(self):
        """Main trading loop."""
        logger.info("Starting main trading loop...")

        while self.running:
            try:
                # Get current market data
                current_data = await self._get_current_market_data()

                if current_data:
                    # Generate trading signals
                    signals = self.trading_engine.generate_signals(current_data)

                    # Execute trades
                    if signals:
                        await self._execute_trades(signals)

                    # Update portfolio
                    self.portfolio.update(current_data)

                    # Risk management checks
                    self.risk_manager.check_risk_limits(self.portfolio)

                    # Generate reports
                    await self._generate_reports()

                # Wait for next iteration
                await asyncio.sleep(self.config.trading.interval_seconds)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _get_current_market_data(self):
        """Get current market data for all symbols."""
        # This would typically fetch real-time data
        # For now, return the latest collected data
        current_data = {}

        for symbol in self.config.trading.symbols:
            data = self.data_storage.load_market_data(symbol)
            if data:
                current_data[symbol] = data

        return current_data

    async def _execute_trades(self, signals):
        """Execute trading signals."""
        for trade_signal in signals:
            try:
                logger.info(f"Executing signal: {trade_signal}")
                # Execute trade through trading engine
                result = self.trading_engine.execute_signal(trade_signal)
                logger.info(f"Trade executed: {result}")
            except Exception as e:
                logger.error(f"Failed to execute signal {trade_signal}: {e}")

    async def _generate_reports(self):
        """Generate trading reports."""
        try:
            await self.report_generator.generate_portfolio_report(self.portfolio)
            await self.report_generator.generate_performance_report(self.portfolio)
            await self.report_generator.generate_risk_report(self.portfolio)
        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")

    async def _cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up trading bot...")

        if self.data_collector:
            await self.data_collector.__aexit__(None, None, None)

        # Generate final reports
        if self.report_generator:
            await self.report_generator.generate_final_report(self.portfolio)

        logger.info("Trading bot cleanup completed")

    def _signal_handler(self, signum, _frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def stop(self):
        """Stop the trading bot."""
        logger.info("Stopping trading bot...")
        self.running = False
