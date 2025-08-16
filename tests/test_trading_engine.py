"""Tests for the trading engine."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone

from src.core.engine.trading_engine import TradingEngine
from src.core.data_models.market import Bar
from src.core.data_models.trading import Portfolio


class TestTradingEngine:
    """Test trading engine functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.config = {
            "risk": {
                "circuit_breakers": {
                    "daily_loss_threshold": 0.03,
                    "drawdown_threshold": 0.10,
                    "volatility_threshold": 0.05,
                    "spread_stress_threshold": 0.02,
                    "cooldown_minutes": 60
                }
            },
            "data": {
                "feature_lookback_bars": 50
            }
        }
        
        self.engine = TradingEngine(self.config)
        self.engine.initialize_portfolio(Decimal("10000"))
    
    def test_engine_initialization(self):
        """Test trading engine initialization."""
        assert self.engine is not None
        assert self.engine.portfolio is not None
        assert self.engine.feature_engine is not None
        assert self.engine.risk_manager is not None
        assert len(self.engine.strategies) > 0
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        portfolio = self.engine.portfolio
        assert portfolio.cash == Decimal("10000")
        assert portfolio.equity == Decimal("10000")
        assert portfolio.peak_equity == Decimal("10000")
        assert portfolio.exposure_gross == Decimal("0")
        assert portfolio.exposure_net == Decimal("0")
    
    def test_add_market_data(self):
        """Test adding market data."""
        # Create sample bar
        bar = Bar(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USD",
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("1000"),
            interval="1m"
        )
        
        self.engine.add_market_data("BTC/USD", bar)
        assert "BTC/USD" in self.engine.market_data_cache
        assert len(self.engine.market_data_cache["BTC/USD"]) == 1
    
    def test_market_data_processing_insufficient_data(self):
        """Test market data processing with insufficient data."""
        # Add only 10 bars (less than required 20)
        for i in range(10):
            bar = Bar(
                timestamp=datetime.now(timezone.utc),
                symbol="BTC/USD",
                open=Decimal("50000"),
                high=Decimal("51000"),
                low=Decimal("49000"),
                close=Decimal("50500"),
                volume=Decimal("1000"),
                interval="1m"
            )
            self.engine.add_market_data("BTC/USD", bar)
        
        # Should return None due to insufficient data
        market_data = self.engine.process_market_data("BTC/USD")
        assert market_data is None
    
    def test_market_data_processing_sufficient_data(self):
        """Test market data processing with sufficient data."""
        # Add 25 bars (more than required 20)
        for i in range(25):
            bar = Bar(
                timestamp=datetime.now(timezone.utc),
                symbol="BTC/USD",
                open=Decimal("50000"),
                high=Decimal("51000"),
                low=Decimal("49000"),
                close=Decimal("50500"),
                volume=Decimal("1000"),
                interval="1m"
            )
            self.engine.add_market_data("BTC/USD", bar)
        
        # Should process successfully
        market_data = self.engine.process_market_data("BTC/USD")
        assert market_data is not None
        assert market_data.symbol == "BTC/USD"
        assert len(market_data.features) > 0
    
    def test_signal_generation_no_data(self):
        """Test signal generation with no market data."""
        signals = self.engine.generate_signals("BTC/USD")
        assert signals == {}
    
    def test_engine_start_stop(self):
        """Test engine start/stop functionality."""
        assert not self.engine.is_running()
        
        self.engine.start()
        assert self.engine.is_running()
        
        self.engine.stop()
        assert not self.engine.is_running()
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        summary = self.engine.get_performance_summary()
        
        assert "portfolio" in summary
        assert "risk_metrics" in summary
        assert "strategies" in summary
        
        portfolio_info = summary["portfolio"]
        assert portfolio_info["equity"] == 10000.0
        assert portfolio_info["cash"] == 10000.0
        assert portfolio_info["exposure"] == 0.0
        assert portfolio_info["drawdown"] == 0.0
    
    def test_market_data_cache_limit(self):
        """Test market data cache size limit."""
        # Add more bars than the limit
        max_bars = self.config["data"]["feature_lookback_bars"]
        
        for i in range(max_bars + 10):
            bar = Bar(
                timestamp=datetime.now(timezone.utc),
                symbol="BTC/USD",
                open=Decimal("50000"),
                high=Decimal("51000"),
                low=Decimal("49000"),
                close=Decimal("50500"),
                volume=Decimal("1000"),
                interval="1m"
            )
            self.engine.add_market_data("BTC/USD", bar)
        
        # Should only keep the most recent bars
        assert len(self.engine.market_data_cache["BTC/USD"]) == max_bars
