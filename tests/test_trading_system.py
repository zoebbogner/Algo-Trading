"""Tests for the trading system."""

import pytest
from decimal import Decimal
from datetime import datetime
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.trading.portfolio import Portfolio, Position
from src.trading.risk import RiskManager
from src.trading.strategies import MomentumStrategy, MeanReversionStrategy
from src.trading.engine import TradingEngine


class TestPortfolio:
    """Test portfolio management."""
    
    def test_portfolio_creation(self):
        """Test creating a portfolio."""
        portfolio = Portfolio(
            initial_capital=100000.0,
            symbols=["BTC", "ETH", "ADA"]
        )
        
        assert portfolio.initial_capital == 100000.0
        assert portfolio.cash == 100000.0
        assert len(portfolio.symbols) == 3
        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 0
    
    def test_add_position(self):
        """Test adding a position."""
        portfolio = Portfolio(
            initial_capital=100000.0,
            symbols=["BTC", "ETH"]
        )
        
        # Add BTC position
        portfolio.add_position("BTC", Decimal("1.0"), Decimal("50000.0"), "LONG")
        
        assert len(portfolio.positions) == 1
        assert "BTC" in portfolio.positions
        assert portfolio.cash == 50000.0  # 100000 - 50000
        
        position = portfolio.positions["BTC"]
        assert position.symbol == "BTC"
        assert position.quantity == Decimal("1.0")
        assert position.average_cost == Decimal("50000.0")
        assert position.side == "LONG"
    
    def test_close_position(self):
        """Test closing a position."""
        portfolio = Portfolio(
            initial_capital=100000.0,
            symbols=["BTC"]
        )
        
        # Add position
        portfolio.add_position("BTC", Decimal("1.0"), Decimal("50000.0"), "LONG")
        
        # Close position at higher price
        portfolio.close_position("BTC", Decimal("1.0"), Decimal("55000.0"))
        
        assert len(portfolio.positions) == 0  # Position closed
        assert portfolio.cash == 105000.0  # 50000 + 55000
        assert portfolio.total_pnl == 5000.0  # 55000 - 50000
    
    def test_portfolio_summary(self):
        """Test portfolio summary generation."""
        portfolio = Portfolio(
            initial_capital=100000.0,
            symbols=["BTC"]
        )
        
        # Add position
        portfolio.add_position("BTC", Decimal("1.0"), Decimal("50000.0"), "LONG")
        
        # Get summary with current prices
        current_prices = {"BTC": 55000.0}
        summary = portfolio.get_summary(current_prices)
        
        assert summary["initial_capital"] == 100000.0
        assert summary["current_value"] == 105000.0  # 50000 cash + 55000 BTC
        assert summary["cash"] == 50000.0
        assert summary["positions_count"] == 1
        assert summary["unrealized_pnl"] == 5000.0  # 55000 - 50000
        assert summary["return_pct"] == 5.0  # (105000 - 100000) / 100000 * 100


class TestRiskManager:
    """Test risk management."""
    
    def test_risk_manager_creation(self):
        """Test creating a risk manager."""
        risk_manager = RiskManager(
            max_position_size=0.1,
            max_portfolio_risk=0.02,
            stop_loss_pct=0.05
        )
        
        assert risk_manager.max_position_size == 0.1
        assert risk_manager.max_portfolio_risk == 0.02
        assert risk_manager.stop_loss_pct == 0.05
        assert len(risk_manager.risk_events) == 0
    
    def test_position_size_calculation(self):
        """Test position size calculation."""
        portfolio = Portfolio(
            initial_capital=100000.0,
            symbols=["BTC"]
        )
        
        risk_manager = RiskManager(max_position_size=0.1)
        
        # Calculate position size for BTC at $50000
        quantity = risk_manager.calculate_position_size(portfolio, "BTC", 50000.0, 0.8)
        
        # Max position value = 100000 * 0.1 * 0.8 = 8000
        # Quantity = 8000 / 50000 = 0.16
        expected_quantity = (100000.0 * 0.1 * 0.8) / 50000.0
        assert abs(quantity - expected_quantity) < 0.001


class TestStrategies:
    """Test trading strategies."""
    
    def test_momentum_strategy_creation(self):
        """Test creating a momentum strategy."""
        config = {
            "name": "test_momentum",
            "parameters": {
                "fast_period": 10,
                "slow_period": 30,
                "rsi_period": 14
            }
        }
        
        strategy = MomentumStrategy(config)
        
        assert strategy.name == "test_momentum"
        assert strategy.fast_period == 10
        assert strategy.slow_period == 30
        assert strategy.rsi_period == 14
    
    def test_mean_reversion_strategy_creation(self):
        """Test creating a mean reversion strategy."""
        config = {
            "name": "test_mean_reversion",
            "parameters": {
                "lookback_period": 20,
                "std_dev_threshold": 2.0,
                "rsi_period": 14
            }
        }
        
        strategy = MeanReversionStrategy(config)
        
        assert strategy.name == "test_mean_reversion"
        assert strategy.lookback_period == 20
        assert strategy.std_dev_threshold == 2.0
        assert strategy.rsi_period == 14
    
    def test_strategy_info(self):
        """Test strategy information retrieval."""
        config = {
            "name": "test_strategy",
            "parameters": {"param1": "value1"}
        }
        
        strategy = MomentumStrategy(config)
        info = strategy.get_strategy_info()
        
        assert info["name"] == "test_strategy"
        assert info["type"] == "MomentumStrategy"
        assert info["parameters"] == {"param1": "value1"}


class TestTradingEngine:
    """Test trading engine."""
    
    def test_trading_engine_creation(self):
        """Test creating a trading engine."""
        portfolio = Portfolio(
            initial_capital=100000.0,
            symbols=["BTC"]
        )
        
        risk_manager = RiskManager()
        
        strategies = {
            "momentum": MomentumStrategy({"name": "test", "parameters": {}})
        }
        
        engine = TradingEngine(portfolio, risk_manager, strategies)
        
        assert engine.portfolio == portfolio
        assert engine.risk_manager == risk_manager
        assert len(engine.strategies) == 1
        assert len(engine.signals) == 0
    
    def test_trading_summary(self):
        """Test trading engine summary."""
        portfolio = Portfolio(
            initial_capital=100000.0,
            symbols=["BTC"]
        )
        
        risk_manager = RiskManager()
        strategies = {}
        
        engine = TradingEngine(portfolio, risk_manager, strategies)
        
        summary = engine.get_trading_summary()
        
        assert summary["total_signals"] == 0
        assert summary["strategies_count"] == 0
        assert summary["positions_count"] == 0
        assert summary["trades_count"] == 0
