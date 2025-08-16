"""Tests for data models."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone

from src.core.data_models.base import BaseEntity, TimeSeriesEntity, ConfigurableEntity
from src.core.data_models.market import Bar, Feature, MarketData
from src.core.data_models.trading import Order, Fill, Position, Portfolio, OrderSide, OrderType, OrderStatus
from src.core.data_models.risk import RiskEvent, RiskEventType, RiskEventSeverity, RiskMetrics, CircuitBreaker


class TestBaseEntity:
    """Test base entity functionality."""
    
    def test_base_entity_creation(self):
        """Test base entity creation with default values."""
        entity = BaseEntity()
        assert entity.id is not None
        assert entity.created_at is not None
        assert entity.updated_at is not None
        assert entity.metadata == {}
    
    def test_update_timestamp(self):
        """Test timestamp update functionality."""
        entity = BaseEntity()
        old_timestamp = entity.updated_at
        entity.update_timestamp()
        assert entity.updated_at > old_timestamp


class TestTimeSeriesEntity:
    """Test time series entity functionality."""
    
    def test_time_series_entity_creation(self):
        """Test time series entity creation."""
        timestamp = datetime.now(timezone.utc)
        entity = TimeSeriesEntity(timestamp=timestamp, symbol="BTC/USD")
        assert entity.timestamp == timestamp
        assert entity.symbol == "BTC/USD"


class TestConfigurableEntity:
    """Test configurable entity functionality."""
    
    def test_configurable_entity_creation(self):
        """Test configurable entity creation."""
        entity = ConfigurableEntity(
            name="TestStrategy",
            version="1.0.0",
            description="Test strategy",
            config={"param1": "value1"}
        )
        assert entity.name == "TestStrategy"
        assert entity.version == "1.0.0"
        assert entity.description == "Test strategy"
        assert entity.enabled is True
        assert entity.config == {"param1": "value1"}


class TestBar:
    """Test bar data model."""
    
    def test_bar_creation(self):
        """Test bar creation."""
        timestamp = datetime.now(timezone.utc)
        bar = Bar(
            timestamp=timestamp,
            symbol="BTC/USD",
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("100"),
            interval="1m"
        )
        assert bar.open == Decimal("50000")
        assert bar.high == Decimal("51000")
        assert bar.low == Decimal("49000")
        assert bar.close == Decimal("50500")
        assert bar.volume == Decimal("100")
        assert bar.interval == "1m"
    
    def test_bar_properties(self):
        """Test bar computed properties."""
        timestamp = datetime.now(timezone.utc)
        bar = Bar(
            timestamp=timestamp,
            symbol="BTC/USD",
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("100"),
            interval="1m"
        )
        assert bar.range == Decimal("2000")  # 51000 - 49000
        assert bar.body == Decimal("500")    # 50500 - 50000
        assert bar.is_bullish is True        # 50500 > 50000
        assert bar.is_bearish is False


class TestOrder:
    """Test order data model."""
    
    def test_order_creation(self):
        """Test order creation."""
        timestamp = datetime.now(timezone.utc)
        order = Order(
            timestamp=timestamp,
            symbol="BTC/USD",
            run_id="test_run_001",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == Decimal("1.0")
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == Decimal("0")
    
    def test_order_properties(self):
        """Test order computed properties."""
        timestamp = datetime.now(timezone.utc)
        order = Order(
            timestamp=timestamp,
            symbol="BTC/USD",
            run_id="test_run_001",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )
        order.filled_quantity = Decimal("0.5")
        assert order.remaining_quantity == Decimal("0.5")
        assert order.is_complete is False
        
        order.status = OrderStatus.FILLED
        assert order.is_complete is True


class TestPosition:
    """Test position data model."""
    
    def test_position_creation(self):
        """Test position creation."""
        timestamp = datetime.now(timezone.utc)
        position = Position(
            timestamp=timestamp,
            symbol="BTC/USD",
            quantity=Decimal("1.0"),
            average_cost=Decimal("50000"),
            market_value=Decimal("50500")
        )
        assert position.quantity == Decimal("1.0")
        assert position.average_cost == Decimal("50000")
        assert position.market_value == Decimal("50500")
        assert position.is_long is True
        assert position.is_short is False
        assert position.is_flat is False
    
    def test_position_properties(self):
        """Test position computed properties."""
        timestamp = datetime.now(timezone.utc)
        position = Position(
            timestamp=timestamp,
            symbol="BTC/USD",
            quantity=Decimal("1.0"),
            average_cost=Decimal("50000"),
            market_value=Decimal("50500"),
            unrealized_pnl=Decimal("500"),
            realized_pnl=Decimal("100")
        )
        assert position.total_pnl == Decimal("600")


class TestPortfolio:
    """Test portfolio data model."""
    
    def test_portfolio_creation(self):
        """Test portfolio creation."""
        timestamp = datetime.now(timezone.utc)
        portfolio = Portfolio(
            timestamp=timestamp,
            cash=Decimal("50000"),
            equity=Decimal("100000"),
            exposure_gross=Decimal("50000"),
            exposure_net=Decimal("50000"),
            peak_equity=Decimal("100000")
        )
        assert portfolio.cash == Decimal("50000")
        assert portfolio.equity == Decimal("100000")
        assert portfolio.exposure_gross == Decimal("50000")
        assert portfolio.exposure_net == Decimal("50000")
    
    def test_portfolio_properties(self):
        """Test portfolio computed properties."""
        timestamp = datetime.now(timezone.utc)
        portfolio = Portfolio(
            timestamp=timestamp,
            cash=Decimal("50000"),
            equity=Decimal("100000"),
            exposure_gross=Decimal("50000"),
            exposure_net=Decimal("50000"),
            peak_equity=Decimal("100000")
        )
        assert portfolio.leverage == 0.5  # 50000 / 100000
        assert portfolio.cash_ratio == 0.5  # 50000 / 100000


class TestRiskEvent:
    """Test risk event data model."""
    
    def test_risk_event_creation(self):
        """Test risk event creation."""
        timestamp = datetime.now(timezone.utc)
        event = RiskEvent(
            event_type=RiskEventType.POSITION_LIMIT_BREACH,
            severity=RiskEventSeverity.HIGH,
            timestamp=timestamp,
            symbol="BTC/USD",
            details={"position_size": "0.15", "limit": "0.10"}
        )
        assert event.event_type == RiskEventType.POSITION_LIMIT_BREACH
        assert event.severity == RiskEventSeverity.HIGH
        assert event.symbol == "BTC/USD"
        assert event.details == {"position_size": "0.15", "limit": "0.10"}
        assert event.resolved is False


class TestCircuitBreaker:
    """Test circuit breaker data model."""
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation."""
        breaker = CircuitBreaker(
            name="DailyLossBreaker",
            threshold=Decimal("0.05"),
            current_value=Decimal("0.03"),
            actions=["reduce_position_sizes", "pause_new_positions"],
            cooldown_minutes=60
        )
        assert breaker.name == "DailyLossBreaker"
        assert breaker.threshold == Decimal("0.05")
        assert breaker.current_value == Decimal("0.03")
        assert breaker.enabled is True
        assert breaker.triggered is False
    
    def test_circuit_breaker_properties(self):
        """Test circuit breaker computed properties."""
        breaker = CircuitBreaker(
            name="DailyLossBreaker",
            threshold=Decimal("0.05"),
            current_value=Decimal("0.06"),
            actions=["reduce_position_sizes"],
            cooldown_minutes=60
        )
        assert breaker.threshold_breached is True  # 0.06 >= 0.05
