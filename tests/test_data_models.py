"""Tests for data models."""

import pytest
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.data.models import Bar, Feature, MarketData, DataRequest, DataResponse


class TestBar:
    """Test Bar data model."""
    
    def test_bar_creation(self):
        """Test creating a valid bar."""
        timestamp = datetime.now()
        bar = Bar(
            timestamp=timestamp,
            open=Decimal("50000.00"),
            high=Decimal("51000.00"),
            low=Decimal("49000.00"),
            close=Decimal("50500.00"),
            volume=Decimal("100.5"),
            symbol="BTC"
        )
        
        assert bar.timestamp == timestamp
        assert bar.open == Decimal("50000.00")
        assert bar.high == Decimal("51000.00")
        assert bar.low == Decimal("49000.00")
        assert bar.close == Decimal("50500.00")
        assert bar.volume == Decimal("100.5")
        assert bar.symbol == "BTC"
        assert bar.id is not None
    
    def test_bar_validation_high_low(self):
        """Test bar validation for high/low values."""
        timestamp = datetime.now()
        
        # High should be >= max(open, close)
        with pytest.raises(ValueError, match="High must be >= max\\(open, close\\)"):
            Bar(
                timestamp=timestamp,
                open=Decimal("50000.00"),
                high=Decimal("49000.00"),  # Too low
                low=Decimal("48000.00"),
                close=Decimal("51000.00"),
                volume=Decimal("100.0"),
                symbol="BTC"
            )
        
        # Low should be <= min(open, close)
        with pytest.raises(ValueError, match="Low must be <= min\\(open, close\\)"):
            Bar(
                timestamp=timestamp,
                open=Decimal("50000.00"),
                high=Decimal("51000.00"),
                low=Decimal("52000.00"),  # Too high
                close=Decimal("49000.00"),
                volume=Decimal("100.0"),
                symbol="BTC"
            )
    
    def test_bar_validation_volume(self):
        """Test bar validation for volume."""
        timestamp = datetime.now()
        
        with pytest.raises(ValueError, match="Volume must be >= 0"):
            Bar(
                timestamp=timestamp,
                open=Decimal("50000.00"),
                high=Decimal("51000.00"),
                low=Decimal("49000.00"),
                close=Decimal("50500.00"),
                volume=Decimal("-10.0"),  # Negative volume
                symbol="BTC"
            )


class TestFeature:
    """Test Feature data model."""
    
    def test_feature_creation(self):
        """Test creating a valid feature."""
        timestamp = datetime.now()
        feature = Feature(
            name="RSI",
            value=65.5,
            timestamp=timestamp,
            symbol="BTC",
            parameters={"period": 14}
        )
        
        assert feature.name == "RSI"
        assert feature.value == 65.5
        assert feature.timestamp == timestamp
        assert feature.symbol == "BTC"
        assert feature.parameters == {"period": 14}
        assert feature.id is not None


class TestMarketData:
    """Test MarketData model."""
    
    def test_market_data_creation(self):
        """Test creating market data."""
        timestamp = datetime.now()
        bars = {
            timestamp: Bar(
                timestamp=timestamp,
                open=Decimal("50000.00"),
                high=Decimal("51000.00"),
                low=Decimal("49000.00"),
                close=Decimal("50500.00"),
                volume=Decimal("100.0"),
                symbol="BTC"
            )
        }
        
        market_data = MarketData(
            timestamp=timestamp,
            bars=bars
        )
        
        assert market_data.timestamp == timestamp
        assert len(market_data.bars) == 1
        assert market_data.features == {}
        assert market_data.metadata == {}
        assert market_data.id is not None


class TestDataRequest:
    """Test DataRequest model."""
    
    def test_data_request_creation(self):
        """Test creating a valid data request."""
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 1, 1)
        
        request = DataRequest(
            symbol="BTC",
            start_date=start_date,
            end_date=end_date,
            interval="1d",
            features=["RSI", "MACD"]
        )
        
        assert request.symbol == "BTC"
        assert request.start_date == start_date
        assert request.end_date == end_date
        assert request.interval == "1d"
        assert request.features == ["RSI", "MACD"]
    
    def test_data_request_validation_dates(self):
        """Test date validation in data request."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2020, 1, 1)  # End before start
        
        with pytest.raises(ValueError, match="Start date must be before end date"):
            DataRequest(
                symbol="BTC",
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
    
    def test_data_request_validation_interval(self):
        """Test interval validation in data request."""
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 1, 1)
        
        with pytest.raises(ValueError, match="Invalid interval"):
            DataRequest(
                symbol="BTC",
                start_date=start_date,
                end_date=end_date,
                interval="invalid"
            )


class TestDataResponse:
    """Test DataResponse model."""
    
    def test_data_response_creation(self):
        """Test creating a data response."""
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 1, 1)
        
        request = DataRequest(
            symbol="BTC",
            start_date=start_date,
            end_date=end_date,
            interval="1d"
        )
        
        market_data = MarketData(
            timestamp=datetime.now(),
            bars={}
        )
        
        response = DataResponse(
            request=request,
            data=market_data,
            success=True
        )
        
        assert response.request == request
        assert response.data == market_data
        assert response.success is True
        assert response.error_message is None
        assert response.collection_timestamp is not None
