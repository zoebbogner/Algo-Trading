"""Tests for feature engineering."""

import pytest
import numpy as np
from decimal import Decimal
from datetime import datetime, timezone

from src.core.features.technical import TechnicalIndicators, FeatureEngine
from src.core.data_models.market import Bar


class TestTechnicalIndicators:
    """Test technical indicators calculation."""
    
    def test_sma(self):
        """Test Simple Moving Average calculation."""
        prices = [10, 20, 30, 40, 50]
        sma = TechnicalIndicators.sma(prices, 3)
        assert sma == 40.0  # (30 + 40 + 50) / 3
    
    def test_sma_insufficient_data(self):
        """Test SMA with insufficient data."""
        prices = [10, 20]
        sma = TechnicalIndicators.sma(prices, 3)
        assert np.isnan(sma)
    
    def test_ema(self):
        """Test Exponential Moving Average calculation."""
        prices = [10, 20, 30, 40, 50]
        ema = TechnicalIndicators.ema(prices, 3)
        # EMA calculation depends on alpha and previous values
        assert not np.isnan(ema)
        assert ema > 0
    
    def test_rsi(self):
        """Test RSI calculation."""
        prices = [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.34, 44.09, 44.15, 43.61]
        rsi = TechnicalIndicators.rsi(prices, 9)
        # RSI should be between 0 and 100
        assert 0 <= rsi <= 100
        assert not np.isnan(rsi)
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        prices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        bb = TechnicalIndicators.bollinger_bands(prices, 5)
        
        assert "upper" in bb
        assert "middle" in bb
        assert "lower" in bb
        assert bb["upper"] > bb["middle"] > bb["lower"]
    
    def test_atr(self):
        """Test Average True Range calculation."""
        highs = [10, 12, 15, 14, 16]
        lows = [8, 10, 12, 11, 13]
        closes = [9, 11, 13, 12, 14]
        
        atr = TechnicalIndicators.atr(highs, lows, closes, 3)
        assert not np.isnan(atr)
        assert atr > 0


class TestFeatureEngine:
    """Test feature engineering engine."""
    
    def setup_method(self):
        """Setup test data."""
        self.feature_engine = FeatureEngine()
        
        # Create sample bars
        self.bars = []
        base_price = 100.0
        for i in range(50):
            bar = Bar(
                timestamp=datetime.now(timezone.utc),
                symbol="BTC/USD",
                open=Decimal(str(base_price + i)),
                high=Decimal(str(base_price + i + 1)),
                low=Decimal(str(base_price + i - 1)),
                close=Decimal(str(base_price + i + 0.5)),
                volume=Decimal("1000"),
                interval="1m"
            )
            self.bars.append(bar)
    
    def test_calculate_features(self):
        """Test feature calculation."""
        features = self.feature_engine.calculate_features(self.bars)
        
        # Check that features were calculated
        assert len(features) > 0
        
        # Check specific features
        if "sma_20" in features:
            assert not np.isnan(features["sma_20"])
        
        if "rsi_14" in features:
            assert 0 <= features["rsi_14"] <= 100
        
        if "returns_1m" in features:
            assert isinstance(features["returns_1m"], float)
    
    def test_calculate_features_empty_data(self):
        """Test feature calculation with empty data."""
        features = self.feature_engine.calculate_features([])
        assert features == {}
    
    def test_calculate_features_insufficient_data(self):
        """Test feature calculation with insufficient data."""
        # Only 5 bars (not enough for most indicators)
        few_bars = self.bars[:5]
        features = self.feature_engine.calculate_features(few_bars)
        
        # Should still calculate some basic features
        assert "returns_1m" in features
        # But not complex indicators
        assert "sma_20" not in features
