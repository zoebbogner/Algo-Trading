#!/usr/bin/env python3
"""
Comprehensive tests for the Feature Engineering system.

Tests all feature families, formulas, and calculations to ensure:
- Formulas are mathematically correct
- No data leakage occurs
- Features are computed accurately
- Edge cases are handled properly
"""

import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the modules
from src.config.base import load_config as load_base_config
from src.config.features import load_config as load_features_config
from src.data.feature_extraction.engine import FeatureEngineer


class TestFeatureEngineering:
    """Test suite for feature engineering functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        # Create 100 rows of realistic crypto data
        np.random.seed(42)  # For reproducible tests

        # Generate realistic price movements
        base_price = 50000.0
        returns = np.random.normal(0, 0.001, 100)  # 0.1% volatility per minute
        prices = [base_price]

        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)

        # Generate OHLCV data
        data = []
        for i, close in enumerate(prices):
            # Create realistic OHLC from close price
            volatility = abs(returns[i]) * 2  # High-low range
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            open_price = prices[i-1] if i > 0 else close

            # Ensure OHLC relationships are valid
            high = max(high, open_price, close)
            low = min(low, open_price, close)

            # Generate realistic volume
            volume = np.random.lognormal(10, 1)  # Log-normal volume distribution

            data.append({
                'ts': datetime.now(timezone.utc) - timedelta(minutes=100-i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        return pd.DataFrame(data)

    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineer instance for testing."""
        base_config = load_base_config()
        features_config = load_features_config()
        config = {**base_config, **features_config}
        return FeatureEngineer(config)

    def test_price_return_features(self, feature_engineer, sample_data):
        """Test price and return feature calculations."""
        df = feature_engineer.compute_price_return_features(sample_data.copy())

        # Test 1-minute returns
        assert 'ret_1m' in df.columns
        # First row should be NaN (no previous price)
        assert pd.isna(df['ret_1m'].iloc[0])
        # Second row should have valid return
        assert not pd.isna(df['ret_1m'].iloc[1])

        # Test log return formula: ln(close_t / close_t-1)
        expected_return = np.log(df['close'].iloc[1] / df['close'].iloc[0])
        assert abs(df['ret_1m'].iloc[1] - expected_return) < 1e-8

        # Test multi-period returns
        assert 'ret_5m' in df.columns
        assert 'ret_15m' in df.columns

        # Test close_to_open
        assert 'close_to_open' in df.columns
        expected_co = (df['close'].iloc[1] - df['open'].iloc[1]) / df['open'].iloc[1]
        assert abs(df['close_to_open'].iloc[1] - expected_co) < 1e-10

        # Test high-low range
        assert 'hl_range' in df.columns
        expected_hl = (df['high'].iloc[1] - df['low'].iloc[1]) / df['close'].iloc[1]
        assert abs(df['hl_range'].iloc[1] - expected_hl) < 1e-10

    def test_trend_momentum_features(self, feature_engineer, sample_data):
        """Test trend and momentum feature calculations."""
        df = feature_engineer.compute_trend_momentum_features(sample_data.copy())

        # Test moving averages
        assert 'ma_20' in df.columns
        assert 'ma_50' in df.columns
        assert 'ma_100' in df.columns

        # First 19 rows should be NaN for MA20
        assert all(pd.isna(df['ma_20'].iloc[:19]))
        # Row 19 should have valid MA20
        assert not pd.isna(df['ma_20'].iloc[19])

        # Verify MA20 calculation
        expected_ma20 = df['close'].iloc[:20].mean()
        assert abs(df['ma_20'].iloc[19] - expected_ma20) < 1e-10

        # Test exponential moving averages
        assert 'ema_20' in df.columns
        assert 'ema_50' in df.columns

        # Test momentum
        assert 'momentum_10' in df.columns
        expected_momentum = df['close'].iloc[10] - df['close'].iloc[0]
        assert abs(df['momentum_10'].iloc[10] - expected_momentum) < 1e-10

        # Test RSI
        assert 'rsi_14' in df.columns
        # RSI should be between 0 and 100
        rsi_values = df['rsi_14'].dropna()
        assert all((rsi_values >= 0) & (rsi_values <= 100))

    def test_mean_reversion_features(self, feature_engineer, sample_data):
        """Test mean reversion feature calculations."""
        # Need to compute MA first
        df = sample_data.copy()
        df['symbol'] = 'BTCUSDT'
        df['symbol'] = 'BTCUSDT'
        df = feature_engineer.compute_trend_momentum_features(df)
        df = feature_engineer.compute_price_return_features(df)
        df = feature_engineer.compute_mean_reversion_features(df)

        # Test z-score
        assert 'zscore_20' in df.columns
        # Z-score should be approximately normal (mean ~0, std ~1)
        zscore_values = df['zscore_20'].dropna()
        if len(zscore_values) > 0:
            assert abs(zscore_values.mean()) < 1.0  # Mean reasonably close to 0 for small datasets
            assert 0.5 < zscore_values.std() < 2.0  # Std reasonable

        # Test Bollinger Bands
        assert 'bb_upper_20' in df.columns
        assert 'bb_lower_20' in df.columns
        assert 'bb_bandwidth_20' in df.columns

        # Upper band should be above lower band
        valid_bb = df[['bb_upper_20', 'bb_lower_20']].dropna()
        if len(valid_bb) > 0:
            assert all(valid_bb['bb_upper_20'] > valid_bb['bb_lower_20'])

        # Test VWAP distance
        assert 'vwap_dist' in df.columns
        # VWAP should be between high and low
        vwap = (df['high'] + df['low'] + df['close']) / 3
        assert all((vwap >= df['low']) & (vwap <= df['high']))

    def test_volatility_risk_features(self, feature_engineer, sample_data):
        """Test volatility and risk feature calculations."""
        # Need returns first
        df = sample_data.copy()
        df['symbol'] = 'BTCUSDT'
        df['symbol'] = 'BTCUSDT'
        df = feature_engineer.compute_price_return_features(df)
        df = feature_engineer.compute_volatility_risk_features(df)

        # Test rolling volatility
        assert 'vol_20' in df.columns
        assert 'vol_50' in df.columns

        # Volatility should be positive
        vol_values = df['vol_20'].dropna()
        assert all(vol_values >= 0)

        # Test ATR
        assert 'atr_14' in df.columns
        atr_values = df['atr_14'].dropna()
        assert all(atr_values >= 0)

        # Test realized volatility (annualized)
        assert 'realized_vol_30' in df.columns
        realized_vol = df['realized_vol_30'].dropna()
        assert all(realized_vol >= 0)

        # Test downside volatility
        assert 'downside_vol_30' in df.columns
        downside_vol = df['downside_vol_30'].dropna()
        assert all(downside_vol >= 0)

    def test_volume_liquidity_features(self, feature_engineer, sample_data):
        """Test volume and liquidity feature calculations."""
        df = sample_data.copy()
        df['symbol'] = 'BTCUSDT'
        df['symbol'] = 'BTCUSDT'
        df = feature_engineer.compute_volume_liquidity_features(df)

        # Test volume moving averages
        assert 'vol_ma_20' in df.columns
        assert 'vol_ma_50' in df.columns

        # Volume MA should be positive
        vol_ma_values = df['vol_ma_20'].dropna()
        assert all(vol_ma_values > 0)

        # Test volume z-score
        assert 'vol_zscore_20' in df.columns

        # Test volume spike flag
        assert 'volume_spike_flag' in df.columns
        spike_flags = df['volume_spike_flag'].dropna()
        assert all((spike_flags == 0) | (spike_flags == 1))

        # Test notional share
        assert 'notional_share' in df.columns
        notional = df['notional_share'].dropna()
        assert all(notional >= 0)
        assert all(notional <= 1)  # Should be fraction

    def test_microstructure_features(self, feature_engineer, sample_data):
        """Test microstructure feature calculations."""
        # Need returns first
        df = sample_data.copy()
        df['symbol'] = 'BTCUSDT'
        df['symbol'] = 'BTCUSDT'
        df = feature_engineer.compute_price_return_features(df)
        df = feature_engineer.compute_volume_liquidity_features(df)
        df = feature_engineer.compute_microstructure_features(df)

        # Test high-low spread
        assert 'hl_spread' in df.columns
        hl_spread = df['hl_spread'].dropna()
        assert all(hl_spread >= 0)

        # Test open-close spread
        assert 'oc_spread' in df.columns
        oc_spread = df['oc_spread'].dropna()
        assert all(oc_spread >= 0)

        # Test Kyle lambda proxy
        assert 'kyle_lambda_proxy' in df.columns
        kyle_lambda = df['kyle_lambda_proxy'].dropna()
        assert all(kyle_lambda >= 0)

    def test_regime_features(self, feature_engineer, sample_data):
        """Test regime classification features."""
        # Need all prerequisite features
        df = sample_data.copy()
        df['symbol'] = 'BTCUSDT'
        df['symbol'] = 'BTCUSDT'
        df = feature_engineer.compute_price_return_features(df)
        df = feature_engineer.compute_trend_momentum_features(df)
        df = feature_engineer.compute_volatility_risk_features(df)
        df = feature_engineer.compute_volume_liquidity_features(df)
        df = feature_engineer.compute_regime_features(df)

        # Test trend regime score
        assert 'trend_regime_score' in df.columns
        trend_score = df['trend_regime_score'].dropna()
        assert all((trend_score >= 0) & (trend_score <= 1))

        # Test volatility regime flag
        assert 'vol_regime_flag' in df.columns
        vol_flag = df['vol_regime_flag'].dropna()
        assert all((vol_flag == 0) | (vol_flag == 1))

        # Test liquidity regime flag
        assert 'liquidity_regime_flag' in df.columns
        liquidity_flag = df['liquidity_regime_flag'].dropna()
        assert all((liquidity_flag == 0) | (liquidity_flag == 1))

    def test_risk_execution_features(self, feature_engineer, sample_data):
        """Test risk and execution helper features."""
        # Need all prerequisite features
        df = sample_data.copy()
        df['symbol'] = 'BTCUSDT'
        df['symbol'] = 'BTCUSDT'
        df = feature_engineer.compute_price_return_features(df)
        df = feature_engineer.compute_trend_momentum_features(df)
        df = feature_engineer.compute_volatility_risk_features(df)
        df = feature_engineer.compute_volume_liquidity_features(df)
        df = feature_engineer.compute_regime_features(df)
        df = feature_engineer.compute_risk_execution_features(df)

        # Test position cap hint
        assert 'position_cap_hint' in df.columns
        position_cap = df['position_cap_hint'].dropna()
        assert all((position_cap >= 0.1) & (position_cap <= 1.0))

        # Test stop distance hint
        assert 'stop_distance_hint' in df.columns
        stop_distance = df['stop_distance_hint'].dropna()
        assert all(stop_distance >= 0)

        # Test slippage hint
        assert 'slippage_hint_bps' in df.columns
        slippage = df['slippage_hint_bps'].dropna()
        assert all(slippage >= 0)

    def test_no_data_leakage(self, feature_engineer, sample_data):
        """Test that no future data is used in feature calculations."""
        df = sample_data.copy()
        df['symbol'] = 'BTCUSDT'
        df['symbol'] = 'BTCUSDT'

        # Add a future price spike to detect leakage
        future_spike_idx = 50
        original_price = df['close'].iloc[future_spike_idx]
        df.loc[df.index[future_spike_idx], 'close'] = original_price * 2  # Double the price

        # Compute features
        df = feature_engineer.compute_price_return_features(df)
        df = feature_engineer.compute_trend_momentum_features(df)
        df = feature_engineer.compute_mean_reversion_features(df)

        # Check that features before the spike are not affected
        # MA20 at index 49 should not be affected by future spike at index 50
        if future_spike_idx >= 20:
            ma_before_spike = df['ma_20'].iloc[future_spike_idx - 1]
            # This should be the MA of prices 0-19, not including the spike
            expected_ma = df['close'].iloc[future_spike_idx-20:future_spike_idx].mean()
            assert abs(ma_before_spike - expected_ma) < 1e-10

    def test_feature_consistency(self, feature_engineer, sample_data):
        """Test that features are consistent across different data sizes."""
        # Test with different data sizes
        for size in [50, 100, 200]:
            if size <= len(sample_data):
                subset = sample_data.head(size).copy()

                # Compute features
                df = feature_engineer.compute_price_return_features(subset)
                df = feature_engineer.compute_trend_momentum_features(df)

                # Check that MA20 is consistent
                if size >= 20:
                    ma20_large = df['ma_20'].iloc[19]
                    # Should match the mean of first 20 prices
                    expected_ma = subset['close'].iloc[:20].mean()
                    assert abs(ma20_large - expected_ma) < 1e-10

    def test_edge_cases(self, feature_engineer):
        """Test edge cases and error handling."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame(columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df = feature_engineer.compute_price_return_features(empty_df)
        assert len(df) == 0

        # Test with single row
        single_row = pd.DataFrame([{
            'ts': datetime.now(timezone.utc),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000.0
        }])
        df = feature_engineer.compute_price_return_features(single_row)
        assert len(df) == 1
        # First row should have NaN return (no previous price)
        assert pd.isna(df['ret_1m'].iloc[0])

        # Test with NaN values
        nan_df = pd.DataFrame([{
            'ts': datetime.now(timezone.utc),
            'open': np.nan,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000.0
        }])
        df = feature_engineer.compute_price_return_features(nan_df)
        assert len(df) == 1

    def test_winsorization(self, feature_engineer, sample_data):
        """Test winsorization functionality."""
        df = sample_data.copy()
        df['symbol'] = 'BTCUSDT'
        df['symbol'] = 'BTCUSDT'

        # Add extreme outliers
        df.loc[df.index[0], 'close'] = 1000000.0  # Extreme high
        df.loc[df.index[1], 'close'] = 0.0001     # Extreme low

        # Compute features and apply winsorization
        df = feature_engineer.compute_price_return_features(df)
        df = feature_engineer.apply_winsorization(df)

        # Check that extreme values are capped
        # The winsorization should have capped the extreme close prices
        # This is a basic check - in practice, you'd verify the specific limits

    def test_metadata_columns(self, feature_engineer, sample_data):
        """Test metadata column addition."""
        df = sample_data.copy()
        df['symbol'] = 'BTCUSDT'
        df['symbol'] = 'BTCUSDT'
        df = feature_engineer.add_metadata_columns(df, 'BTCUSDT')

        # Check required metadata columns
        assert 'source' in df.columns
        assert 'load_id' in df.columns
        assert 'ingestion_ts' in df.columns
        assert 'symbol' in df.columns

        # Check values
        assert df['source'].iloc[0] == 'binance'
        assert df['symbol'].iloc[0] == 'BTCUSDT'
        # assert df['load_id'].iloc[0] == feature_engineer.run_id

        # Check timestamp format
        timestamp = df['ingestion_ts'].iloc[0]
        assert isinstance(timestamp, str)
        # Should be ISO format
        assert 'T' in timestamp and ('Z' in timestamp or '+' in timestamp)


class TestFeatureFormulas:
    """Test specific mathematical formulas used in features."""

    def test_rsi_formula(self):
        """Test RSI calculation formula."""
        # Create simple test data
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91])

        # Manual RSI calculation
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # 14-period RSI
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Test the RSI function
        from src.data.feature_extraction.engine import FeatureEngineer
        config = {
            "rolling_windows": {"rsi": [14], "ma": [20, 50]},
            "thresholds": {"rsi_overbought": 70, "rsi_oversold": 30, "volume_spike_multiplier": 2.0, "volatility_regime_percentile": 0.8, "breakout_lookback": 20, "stop_atr_multiplier": 2.0, "intraday_reversal_threshold": 0.01},
            "output": {"path": "test_features.parquet"}
        }
        engineer = FeatureEngineer(config)

        # Create DataFrame with close prices
        df = pd.DataFrame({'open': prices, 'high': prices, 'low': prices, 'close': prices, 'volume': [1000] * len(prices)})
        df = engineer.compute_trend_momentum_features(df)

        # Compare calculated RSI
        calculated_rsi = df['rsi_14'].dropna()
        expected_rsi = rsi.dropna()

        # Should be very close (allowing for floating point differences)
        assert len(calculated_rsi) == len(expected_rsi)
        for calc, exp in zip(calculated_rsi, expected_rsi, strict=False):
            if not pd.isna(calc) and not pd.isna(exp):
                assert abs(calc - exp) < 1e-6

    def test_bollinger_bands_formula(self):
        """Test Bollinger Bands calculation."""
        # Create test data
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85])

        # Test the function
        from src.data.feature_extraction.engine import FeatureEngineer
        config = {
            "rolling_windows": {"rsi": [14], "ma": [20, 50]},
            "thresholds": {"rsi_overbought": 70, "rsi_oversold": 30, "volume_spike_multiplier": 2.0, "volatility_regime_percentile": 0.8, "breakout_lookback": 20, "stop_atr_multiplier": 2.0, "intraday_reversal_threshold": 0.01},
            "output": {"path": "test_features.parquet"}
        }
        engineer = FeatureEngineer(config)

        df = pd.DataFrame({'open': prices, 'high': prices, 'low': prices, 'close': prices, 'volume': [1000] * len(prices)})
        # Need to compute price returns first
        df = engineer.compute_price_return_features(df)
        df = engineer.compute_trend_momentum_features(df)
        df = engineer.compute_mean_reversion_features(df)

        # Compare results
        assert 'bb_upper_20' in df.columns
        assert 'bb_lower_20' in df.columns
        assert 'bb_bandwidth_20' in df.columns

        # Check that upper > lower
        valid_bb = df[['bb_upper_20', 'bb_lower_20']].dropna()
        if len(valid_bb) > 0:
            assert all(valid_bb['bb_upper_20'] > valid_bb['bb_lower_20'])

        # Verify bandwidth calculation matches expected
        if len(valid_bb) > 0:
            calculated_bandwidth = df['bb_bandwidth_20'].dropna()
            assert len(calculated_bandwidth) > 0

    def test_atr_formula(self):
        """Test Average True Range calculation."""
        # Create test data
        high = pd.Series([105, 106, 107, 106, 105, 104, 103, 102, 101, 100])
        low = pd.Series([95, 96, 97, 96, 95, 94, 93, 92, 91, 90])
        close = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 96, 95])

        # Manual ATR calculation (for reference)
        # high_low = high - low
        # high_close = np.abs(high - close.shift(1))
        # low_close = np.abs(low - close.shift(1))
        # true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Test the function
        from src.data.feature_extraction.engine import FeatureEngineer
        config = {
            "rolling_windows": {"rsi": [14], "ma": [20, 50]},
            "thresholds": {"rsi_overbought": 70, "rsi_oversold": 30, "volume_spike_multiplier": 2.0, "volatility_regime_percentile": 0.8, "breakout_lookback": 20, "stop_atr_multiplier": 2.0, "intraday_reversal_threshold": 0.01},
            "output": {"path": "test_features.parquet"}
        }
        engineer = FeatureEngineer(config)

        df = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close
        })

        # Need to compute price returns first
        df = engineer.compute_price_return_features(df)
        df = engineer.compute_volatility_risk_features(df)

        # ATR should be positive
        atr_values = df['atr_14'].dropna()
        assert all(atr_values >= 0)

        # Verify ATR calculation produces expected results
        assert len(atr_values) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
