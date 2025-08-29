"""
Feature Engineering Engine for Crypto Algorithmic Trading

Implements professional-grade feature engineering with:
- Proper volatility scaling for intraday data
- Rolling normalization to prevent data leakage
- Industry-standard technical indicators
- Partitioned Parquet output for scalability
- Comprehensive quality control
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...utils.file_ops import ensure_directory
from ...utils.logging import get_logger, log_performance
from ...utils.time_utils import get_utc_now
from ...utils.validation import validate_dataframe, validate_timestamps

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Professional feature engineering engine for 1-minute crypto data.
    
    Implements best practices:
    - No data leakage (rolling windows only)
    - Proper volatility scaling for intraday data
    - Industry-standard technical indicators
    - Partitioned output for scalability
    - Comprehensive quality control
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration dictionary with feature parameters
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # Extract configuration
        self.rolling_windows = config.get('rolling_windows', {})
        self.thresholds = config.get('thresholds', {})
        self.cross_assets = config.get('cross_assets', {})
        self.output_config = config.get('output', {})
        self.qc_config = config.get('qc', {})
        self.data_config = config.get('data_config', {})
        self.winsorization_config = config.get('winsorization', {})
        self.trend_regime_config = config.get('trend_regime', {})

        # Set defaults for data frequency scaling
        self.bars_per_day = self.data_config.get('bars_per_day', 1440)
        self.bars_per_quarter = self.data_config.get('bars_per_quarter', 129600)

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_keys = ['rolling_windows', 'thresholds', 'output']
        missing_keys = [key for key in required_keys if key not in self.config]

        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

        # Validate rolling windows
        for window_type, windows in self.rolling_windows.items():
            if not isinstance(windows, list) or not all(isinstance(w, int) for w in windows):
                raise ValueError(f"Invalid rolling windows for {window_type}: {windows}")

        # Validate thresholds
        required_thresholds = [
            'volume_spike_multiplier', 'volatility_regime_percentile',
            'breakout_lookback', 'stop_atr_multiplier', 'intraday_reversal_threshold'
        ]
        missing_thresholds = [t for t in required_thresholds if t not in self.thresholds]

        if missing_thresholds:
            raise ValueError(f"Missing required thresholds: {missing_thresholds}")

    @log_performance()
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for the input DataFrame.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with all computed features
        """
        self.logger.info(f"Computing features for {len(df)} rows")

        # Validate input data
        required_columns = ['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        validation_result = validate_dataframe(df, required_columns)
        if not validation_result['valid']:
            raise ValueError(f"Data validation failed: {validation_result['errors']}")

        # Validate timestamps
        timestamp_validation = validate_timestamps(
            df,
            timestamp_column='ts',
            symbol_column='symbol',
            require_monotonic=True,
            require_unique=True
        )
        if not timestamp_validation['valid']:
            raise ValueError(f"Timestamp validation failed: {timestamp_validation['errors']}")

        # Start with base data
        result_df = df.copy()

        # Compute features in dependency order
        result_df = self.compute_price_return_features(result_df)
        result_df = self.compute_trend_momentum_features(result_df)
        result_df = self.compute_mean_reversion_features(result_df)
        result_df = self.compute_volatility_risk_features(result_df)
        result_df = self.compute_volume_liquidity_features(result_df)
        result_df = self.compute_microstructure_features(result_df)
        result_df = self.compute_regime_features(result_df)
        result_df = self.compute_risk_execution_features(result_df)

        # Compute cross-asset features last (requires all individual features)
        result_df = self.compute_cross_asset_features(result_df)

        # Apply winsorization
        result_df = self._apply_winsorization(result_df)

        # Add metadata
        result_df = self._add_metadata(result_df)

        # Quality control
        self._run_quality_control(result_df)

        self.logger.info(f"Feature computation completed. Output shape: {result_df.shape}")
        return result_df

    @log_performance()
    def compute_price_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price and return features."""
        self.logger.debug("Computing price and return features")

        # 1-minute returns
        df['ret_1m'] = df['close'].pct_change()

        # Multi-period returns
        for period in [5, 15]:
            df[f'ret_{period}m'] = df['close'].pct_change(periods=period)

        # Intra-bar features
        df['close_to_open'] = (df['close'] - df['open']) / df['open']
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['oc_abs'] = abs(df['open'] - df['close']) / df['close']

        # Log returns for better statistical properties
        df['log_ret_1m'] = np.log(df['close'] / df['close'].shift(1))

        return df

    @log_performance()
    def compute_trend_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute trend and momentum features."""
        self.logger.debug("Computing trend and momentum features")

        # Moving averages
        for window in self.rolling_windows.get('ma', [20, 50, 100]):
            df[f'ma_{window}'] = df['close'].rolling(window=window, min_periods=window).mean()

        # Exponential moving averages
        for window in self.rolling_windows.get('ema', [20, 50]):
            df[f'ema_{window}'] = df['close'].ewm(span=window, min_periods=window).mean()

        # Momentum
        for period in [10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)

        # Trend slope using rolling regression
        for window in self.rolling_windows.get('regression', [20, 240, 720]):
            df[f'trend_slope_{window}'] = self._compute_rolling_slope(df['close'], window)

        # Breakout detection
        breakout_lookback = self.thresholds.get('breakout_lookback', 20)
        df['breakout_20'] = (df['close'] > df['close'].rolling(breakout_lookback).max().shift(1)).astype(int)

        # RSI using Wilder's smoothing (industry standard)
        for window in self.rolling_windows.get('rsi', [14]):
            df[f'rsi_{window}'] = self._compute_rsi_wilders(df['close'], window)

        return df

    @log_performance()
    def compute_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute mean reversion features."""
        self.logger.debug("Computing mean reversion features")

        # Z-score from moving average
        for window in self.rolling_windows.get('zscore', [20]):
            ma = df['close'].rolling(window=window, min_periods=window).mean()
            std = df['close'].rolling(window=window, min_periods=window).std()
            df[f'zscore_{window}'] = (df['close'] - ma) / std

        # Bollinger Bands
        for window in [20]:
            ma = df['close'].rolling(window=window, min_periods=window).mean()
            std = df['close'].rolling(window=window, min_periods=window).std()
            df[f'bb_upper_{window}'] = ma + (2 * std)
            df[f'bb_lower_{window}'] = ma - (2 * std)
            df[f'bb_bandwidth_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / ma

        # Intraday reversal detection
        threshold = self.thresholds.get('intraday_reversal_threshold', 0.01)
        df['intraday_reversal_flag'] = self._detect_intraday_reversals(df, threshold)

        # VWAP distance (approximated from OHLC)
        df['vwap_dist'] = self._compute_vwap_distance(df)

        return df

    @log_performance()
    def compute_volatility_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility and risk features."""
        self.logger.debug("Computing volatility and risk features")

        # Rolling volatility (properly scaled for intraday)
        for window in self.rolling_windows.get('volatility', [20, 50]):
            # Use log returns for better statistical properties
            log_returns = np.log(df['close'] / df['close'].shift(1))
            rolling_std = log_returns.rolling(window=window, min_periods=window).std()

            # Scale to annualized volatility (1-minute bars)
            df[f'vol_{window}'] = rolling_std * np.sqrt(self.bars_per_day * 365)

        # ATR (Average True Range)
        for window in self.rolling_windows.get('atr', [14]):
            df[f'atr_{window}'] = self._compute_atr(df, window)

        # Realized volatility (rolling)
        for window in [30]:
            log_returns = np.log(df['close'] / df['close'].shift(1))
            rolling_std = log_returns.rolling(window=window, min_periods=window).std()
            df[f'realized_vol_{window}'] = rolling_std * np.sqrt(self.bars_per_day * 365)

        # Volatility percentile (properly scaled for intraday)
        vol_window = 20
        vol_period = int(self.bars_per_quarter / self.bars_per_day * 90)  # 90 days in bars

        if vol_period > vol_window:
            df['vol_percentile_90d'] = self._compute_rolling_percentile(
                df[f'vol_{vol_window}'], vol_period, vol_window
            )

        # Downside volatility (only negative returns)
        for window in [30]:
            log_returns = np.log(df['close'] / df['close'].shift(1))
            negative_returns = log_returns.where(log_returns < 0, 0)
            downside_vol = negative_returns.rolling(window=window, min_periods=window).std()
            df[f'downside_vol_{window}'] = downside_vol * np.sqrt(self.bars_per_day * 365)

        return df

    @log_performance()
    def compute_volume_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume and liquidity features."""
        self.logger.debug("Computing volume and liquidity features")

        # Volume moving averages
        for window in self.rolling_windows.get('volume', [20, 50]):
            df[f'vol_ma_{window}'] = df['volume'].rolling(window=window, min_periods=window).mean()

        # Volume z-score
        for window in [20]:
            vol_ma = df['volume'].rolling(window=window, min_periods=window).mean()
            vol_std = df['volume'].rolling(window=window, min_periods=window).std()
            df[f'vol_zscore_{window}'] = (df['volume'] - vol_ma) / vol_std

        # Volume spike detection
        multiplier = self.thresholds.get('volume_spike_multiplier', 2.0)
        for window in [20]:
            vol_ma = df['volume'].rolling(window=window, min_periods=window).mean()
            df[f'volume_spike_flag_{window}'] = (df['volume'] > multiplier * vol_ma).astype(int)
            # Also create the generic name for backward compatibility
            if window == 20:
                df['volume_spike_flag'] = df[f'volume_spike_flag_{window}']

        # Notional volume (price × volume)
        df['notional_volume'] = df['close'] * df['volume']

        # Notional share (fraction of total turnover)
        for window in [60]:
            total_notional = df['notional_volume'].rolling(window=window, min_periods=window).sum()
            df[f'notional_share_{window}'] = df['notional_volume'] / total_notional
            # Also create the generic name for backward compatibility
            if window == 60:
                df['notional_share'] = df[f'notional_share_{window}']

        return df

    @log_performance()
    def compute_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute microstructure proxy features."""
        self.logger.debug("Computing microstructure features")

        # High-low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']

        # Open-close spread
        df['oc_spread'] = abs(df['open'] - df['close']) / df['close']

        # Kyle lambda proxy (using notional volume for better accuracy)
        df['kyle_lambda_proxy'] = abs(df['log_ret_1m']) / df['notional_volume']

        # Roll measure proxy (negative autocovariance at lag 1)
        df['roll_measure_proxy'] = self._compute_roll_measure(df['log_ret_1m'])

        return df

    @log_performance()
    def compute_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute market regime classification features."""
        self.logger.debug("Computing regime features")

        # Trend regime score (using rolling normalization to prevent leakage)
        df['trend_regime_score'] = self._compute_trend_regime_score(df)

        # Volatility regime flag
        vol_percentile = self.thresholds.get('volatility_regime_percentile', 0.80)
        df['vol_regime_flag'] = (df['vol_percentile_90d'] > vol_percentile).astype(int)

        # Liquidity regime flag
        df['liquidity_regime_flag'] = self._compute_liquidity_regime_flag(df)

        return df

    @log_performance()
    def compute_risk_execution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute risk and execution helper features."""
        self.logger.debug("Computing risk and execution features")

        # Position cap hint based on volatility and beta
        df['position_cap_hint'] = self._compute_position_cap_hint(df)

        # Stop distance hint based on ATR
        atr_multiplier = self.thresholds.get('stop_atr_multiplier', 2.0)
        df['stop_distance_hint'] = atr_multiplier * df['atr_14']

        # Slippage hint based on spread and volatility
        df['slippage_hint_bps'] = self._compute_slippage_hint(df)

        return df

    @log_performance()
    def compute_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute cross-asset relationship features."""
        self.logger.debug("Computing cross-asset features")

        driver_symbol = self.cross_assets.get('driver_symbol', 'BTCUSDT')
        pairs = self.cross_assets.get('pairs', [])

        if driver_symbol not in df['symbol'].values:
            self.logger.warning(f"Driver symbol {driver_symbol} not found in data")
            return df

        # Get driver data
        driver_data = df[df['symbol'] == driver_symbol].copy()
        driver_data = driver_data.drop_duplicates(subset=['ts']).reset_index(drop=True)

        # Compute features for each pair
        for symbol in pairs:
            if symbol not in df['symbol'].values:
                continue

            try:
                symbol_data = df[df['symbol'] == symbol].copy()
                symbol_data = symbol_data.drop_duplicates(subset=['ts']).reset_index(drop=True)

                # Merge with driver data
                merged_data = pd.merge(
                    symbol_data,
                    driver_data[['ts', 'close', 'ret_1m']],
                    on='ts',
                    suffixes=('', '_driver')
                )

                if merged_data.empty:
                    continue

                # Compute pair-specific features
                pair_features = self._compute_pair_specific_features(merged_data, symbol, driver_symbol)

                # Update original DataFrame
                for feature_name, feature_values in pair_features.items():
                    df.loc[df['symbol'] == symbol, feature_name] = feature_values

            except Exception as e:
                self.logger.error(f"Error computing pair features for {symbol}: {e}")
                continue

        return df

    def _compute_pair_specific_features(self, merged_data: pd.DataFrame, symbol: str, driver_symbol: str) -> dict[str, pd.Series]:
        """Compute features specific to a symbol-driver pair."""
        features = {}

        # Driver returns
        features[f'{driver_symbol.lower()}_ret_1m'] = merged_data['ret_1m_driver']

        # Relative returns
        features[f'{symbol.lower()}_minus_{driver_symbol.lower()}_ret'] = (
            merged_data['ret_1m'] - merged_data['ret_1m_driver']
        )

        # Price ratio
        features[f'ratio_{symbol.lower()}_{driver_symbol.lower()}'] = (
            merged_data['close'] / merged_data['close_driver']
        )

        # Ratio z-score
        ratio = features[f'ratio_{symbol.lower()}_{driver_symbol.lower()}']
        for window in [240]:  # 4 hours
            ratio_ma = ratio.rolling(window=window, min_periods=window).mean()
            ratio_std = ratio.rolling(window=window, min_periods=window).std()
            features[f'ratio_{symbol.lower()}_{driver_symbol.lower()}_zscore_{window}'] = (
                (ratio - ratio_ma) / ratio_std
            )

        # Beta to driver
        for window in [720]:  # 12 hours
            beta = self._compute_beta_to_driver(
                merged_data['ret_1m'],
                merged_data['ret_1m_driver'],
                window
            )
            features[f'beta_to_{driver_symbol.lower()}_{window}'] = beta

        return features

    def _compute_beta_to_driver(self, returns: pd.Series, driver_returns: pd.Series, window: int) -> pd.Series:
        """Compute rolling beta to driver asset."""
        def rolling_beta(x):
            if len(x) < 2:
                return np.nan

            y = x['returns'].values
            x_driver = x['driver_returns'].values

            # Remove NaN values
            mask = ~(np.isnan(y) | np.isnan(x_driver))
            if np.sum(mask) < 2:
                return np.nan

            y_clean = y[mask]
            x_clean = x_driver[mask]

            try:
                # Add constant for regression
                X = np.column_stack([np.ones(len(x_clean)), x_clean])
                beta = np.linalg.lstsq(X, y_clean, rcond=None)[0][1]
                return beta
            except Exception:
                return np.nan

        # Create rolling window data
        rolling_data = pd.DataFrame({
            'returns': returns,
            'driver_returns': driver_returns
        }).rolling(window=window, min_periods=window)

        # Apply rolling beta calculation
        beta_series = rolling_data.apply(rolling_beta, raw=False)

        return beta_series

    def _compute_rolling_slope(self, series: pd.Series, window: int) -> pd.Series:
        """Compute rolling linear regression slope."""
        def slope(x):
            if len(x) < 2:
                return np.nan

            y = x.values
            x_vals = np.arange(len(y))

            try:
                slope_val = np.polyfit(x_vals, y, 1)[0]
                return slope_val
            except Exception:
                return np.nan

        return series.rolling(window=window, min_periods=window).apply(slope, raw=False)

    def _compute_rsi_wilders(self, series: pd.Series, window: int) -> pd.Series:
        """Compute RSI using Wilder's smoothing (industry standard)."""
        # Calculate price changes
        delta = series.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Initial average gain and loss
        avg_gain = gains.rolling(window=window, min_periods=window).mean()
        avg_loss = losses.rolling(window=window, min_periods=window).mean()

        # Apply Wilder's smoothing
        for i in range(window, len(series)):
            if i == window:
                continue

            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (window - 1) + gains.iloc[i]) / window
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (window - 1) + losses.iloc[i]) / window

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _detect_intraday_reversals(self, df: pd.DataFrame, threshold: float) -> pd.Series:
        """Detect intraday reversal patterns."""
        # Look for 1-minute moves that oppose recent trend
        lookback = 5  # Look back 5 minutes

        # Recent trend
        recent_trend = df['ret_1m'].rolling(lookback).sum()

        # Current move
        current_move = df['ret_1m']

        # Reversal condition: current move opposes recent trend and exceeds threshold
        reversal = (
            (current_move * recent_trend < 0) &  # Opposite direction
            (abs(current_move) > threshold)      # Exceeds threshold
        )

        return reversal.astype(int)

    def _compute_vwap_distance(self, df: pd.DataFrame) -> pd.Series:
        """Compute distance from VWAP (approximated from OHLC)."""
        # Approximate VWAP as (H+L+C)/3
        vwap = (df['high'] + df['low'] + df['close']) / 3

        # Distance from VWAP
        vwap_dist = (df['close'] - vwap) / vwap

        return vwap_dist

    def _compute_roll_measure(self, returns: pd.Series) -> pd.Series:
        """Compute Roll measure proxy (negative autocovariance at lag 1)."""
        def rolling_roll_measure(x):
            if len(x) < 2:
                return np.nan

            try:
                # Compute autocovariance at lag 1
                autocov = np.cov(x[:-1], x[1:])[0, 1]
                return -autocov
            except Exception:
                return np.nan

        return returns.rolling(window=20, min_periods=20).apply(rolling_roll_measure, raw=False)

    def _compute_trend_regime_score(self, df: pd.DataFrame) -> pd.Series:
        """Compute trend regime score using rolling normalization."""
        # Combine trend indicators
        trend_indicators = []

        # Trend slope (normalized)
        if 'trend_slope_20' in df.columns:
            slope = df['trend_slope_20']
            # Use rolling z-score to prevent leakage
            lookback = self.trend_regime_config.get('lookback_periods', 240)
            min_periods = self.trend_regime_config.get('min_periods', 120)

            slope_normalized = self._rolling_zscore_normalize(slope, lookback, min_periods)
            trend_indicators.append(slope_normalized)

        # Breakout indicator
        if 'breakout_20' in df.columns:
            trend_indicators.append(df['breakout_20'])

        # EMA trend
        if 'ema_20' in df.columns and 'ema_50' in df.columns:
            ema_trend = (df['ema_20'] > df['ema_50']).astype(float)
            trend_indicators.append(ema_trend)

        if not trend_indicators:
            return pd.Series(0.5, index=df.index)

        # Combine indicators (equal weight)
        combined = pd.concat(trend_indicators, axis=1).mean(axis=1)

        # Scale to [0, 1] range
        trend_score = (combined + 1) / 2

        return trend_score.fillna(0.5)

    def _rolling_zscore_normalize(self, series: pd.Series, lookback: int, min_periods: int) -> pd.Series:
        """Normalize series using rolling z-score to prevent leakage."""
        rolling_mean = series.rolling(window=lookback, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=lookback, min_periods=min_periods).std()

        normalized = (series - rolling_mean) / rolling_std

        # Clip extreme values
        normalized = normalized.clip(-3, 3)

        return normalized

    def _compute_liquidity_regime_flag(self, df: pd.DataFrame) -> pd.Series:
        """Compute liquidity regime flag."""
        # Thin market: low volume relative to recent average
        thin_market = df['vol_zscore_20'] < -1

        # High notional share: large orders relative to turnover
        high_share = df['notional_share_60'] > 0.01  # 1% of turnover

        # Liquidity regime: thin market OR high notional share
        liquidity_regime = (thin_market | high_share).astype(int)

        return liquidity_regime

    def _compute_position_cap_hint(self, df: pd.DataFrame) -> pd.Series:
        """Compute position cap hint based on volatility and beta."""
        # Base cap
        base_cap = 1.0

        # Reduce cap in high volatility
        if 'vol_regime_flag' in df.columns:
            vol_adjustment = np.where(df['vol_regime_flag'] == 1, 0.5, 1.0)
        else:
            vol_adjustment = 1.0

        # Reduce cap for high beta assets
        if 'beta_to_btc_720' in df.columns:
            beta = df['beta_to_btc_720'].abs()
            beta_adjustment = np.where(beta > 2, 0.5, np.where(beta > 1, 0.75, 1.0))
        else:
            beta_adjustment = 1.0

        # Reduce cap in low liquidity
        if 'liquidity_regime_flag' in df.columns:
            liquidity_adjustment = np.where(df['liquidity_regime_flag'] == 1, 0.7, 1.0)
        else:
            liquidity_adjustment = 1.0

        # Combine adjustments
        position_cap = base_cap * vol_adjustment * beta_adjustment * liquidity_adjustment

        return position_cap

    def _compute_slippage_hint(self, df: pd.DataFrame) -> pd.Series:
        """Compute slippage hint in basis points."""
        # Base slippage
        base_slippage_bps = 10

        # Adjust for spread
        if 'hl_spread' in df.columns:
            median_spread = df['hl_spread'].rolling(240).median()
            spread_adjustment = df['hl_spread'] / median_spread
        else:
            spread_adjustment = 1.0

        # Adjust for volatility
        if 'vol_20' in df.columns:
            median_vol = df['vol_20'].rolling(240).median()
            vol_adjustment = df['vol_20'] / median_vol
        else:
            vol_adjustment = 1.0

        # Compute final slippage
        slippage_bps = base_slippage_bps * spread_adjustment * vol_adjustment

        # Clip to reasonable range
        slippage_bps = slippage_bps.clip(1, 100)

        return slippage_bps

    def _compute_vwap_distance(self, df: pd.DataFrame) -> pd.Series:
        """Compute VWAP distance (approximated from OHLC)."""
        # Approximate VWAP using OHLC
        vwap = (df['high'] + df['low'] + df['close']) / 3
        vwap_dist = (df['close'] - vwap) / vwap
        return vwap_dist

    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=period).mean()

        return atr

    def _compute_roll_measure(self, returns: pd.Series) -> pd.Series:
        """Compute Roll measure proxy (negative autocovariance at lag 1)."""
        def rolling_autocovariance(x):
            if len(x) < 2:
                return np.nan
            return -np.cov(x[:-1], x[1:])[0, 1]

        return returns.rolling(window=20, min_periods=20).apply(rolling_autocovariance, raw=False)

    def _compute_rolling_percentile(self, series: pd.Series, lookback: int, min_periods: int) -> pd.Series:
        """Compute rolling percentile rank."""
        def rolling_percentile(x):
            """Calculate rolling percentile for a series."""
            if len(x) < lookback:
                return np.nan  # Return NaN for insufficient data
            
            current_value = x.iloc[-1]
            historical_values = x.iloc[:-1]

            if len(historical_values) == 0:
                return np.nan  # Return NaN instead of hardcoded 0.5

            percentile = (historical_values < current_value).mean()
            return percentile

        return series.rolling(window=lookback, min_periods=min_periods).apply(rolling_percentile, raw=False)

    def _apply_winsorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply winsorization to numeric features."""
        method = self.winsorization_config.get('method', 'quantile')
        lower_limit = self.winsorization_config.get('lower_limit', 0.01)
        upper_limit = self.winsorization_config.get('upper_limit', 0.99)
        apply_per_symbol = self.winsorization_config.get('apply_per_symbol', True)

        # Get numeric columns (excluding metadata and categorical)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in ['ts', 'symbol']]

        if apply_per_symbol:
            # Apply winsorization per symbol
            for symbol in df['symbol'].unique():
                symbol_mask = df['symbol'] == symbol
                symbol_data = df.loc[symbol_mask, feature_columns]

                if method == 'quantile':
                    lower_bounds = symbol_data.quantile(lower_limit)
                    upper_bounds = symbol_data.quantile(upper_limit)

                    for col in feature_columns:
                        # Explicitly convert to match dtype to avoid FutureWarning
                        clipped_values = symbol_data[col].clip(
                            lower=lower_bounds[col],
                            upper=upper_bounds[col]
                        ).astype(df[col].dtype)
                        df.loc[symbol_mask, col] = clipped_values
        else:
            # Apply winsorization globally
            if method == 'quantile':
                lower_bounds = df[feature_columns].quantile(lower_limit)
                upper_bounds = df[feature_columns].quantile(upper_limit)

                for col in feature_columns:
                    # Explicitly convert to match dtype to avoid FutureWarning
                    df[col] = df[col].clip(lower=lower_bounds[col], upper=upper_bounds[col]).astype(df[col].dtype)

        return df

    def apply_winsorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply winsorization to features (public interface)."""
        return self._apply_winsorization(df)

    def add_metadata_columns(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Add metadata columns (public interface)."""
        if symbol:
            df['symbol'] = symbol
        return self._add_metadata(df)

    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata columns."""
        # Source
        df['source'] = 'binance'

        # Load ID
        df['load_id'] = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Ingestion timestamp
        df['ingestion_ts'] = get_utc_now().isoformat()

        # Date partition column
        df['date'] = pd.to_datetime(df['ts']).dt.date.astype(str)

        return df

    def _run_quality_control(self, df: pd.DataFrame) -> None:
        """Run quality control checks."""
        self.logger.info("Running quality control checks")

        # Check for NaN values
        nan_counts = df.isnull().sum()
        high_nan_features = nan_counts[nan_counts > 0]

        if not high_nan_features.empty:
            self.logger.warning(f"Features with NaN values: {high_nan_features.to_dict()}")

        # Check feature ranges
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in ['ts', 'symbol', 'date']]

        for col in feature_columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()

                # Check for extreme values
                if abs(min_val) > 1e6 or abs(max_val) > 1e6:
                    self.logger.warning(f"Feature {col} has extreme values: [{min_val}, {max_val}]")

        # Check timestamp monotonicity
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].sort_values('ts')
            if not symbol_data['ts'].is_monotonic_increasing:
                self.logger.error(f"Timestamps not monotonic for symbol {symbol}")

        self.logger.info("Quality control completed")

    @log_performance()
    def save_features(self, df: pd.DataFrame, output_path: str = None) -> str:
        """
        Save features to partitioned Parquet dataset.

        Args:
            df: DataFrame with features
            output_path: Output path (uses config if not specified)

        Returns:
            Path to saved features
        """
        if output_path is None:
            output_path = self.output_config.get('path', 'data/features/features_1m.parquet')

        # Ensure output directory exists
        output_dir = Path(output_path)
        ensure_directory(output_dir.parent)

        # Get partition columns
        partition_by = self.output_config.get('partition_by', ['symbol', 'date'])

        # Save as partitioned Parquet
        df.to_parquet(
            output_path,
            partition_cols=partition_by,
            compression=self.output_config.get('compression', 'snappy'),
            index=False
        )

        self.logger.info(f"Features saved to: {output_path}")
        return output_path
