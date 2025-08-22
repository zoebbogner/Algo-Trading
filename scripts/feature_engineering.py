#!/usr/bin/env python3
"""
Professional Feature Engineering Engine

Implements a comprehensive feature layer for 1-minute crypto data following
professional quant desk standards. Features include price returns, trend momentum,
mean reversion, volatility, volume, cross-asset relationships, and regime classification.

Usage:
    python3 scripts/feature_engineering.py --symbols BTCUSDT ETHUSDT --config configs/features.yaml
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import yaml
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.base import load_config as load_base_config
from configs.features import load_config as load_features_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_engineering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Professional-grade feature engineering engine for crypto data."""
    
    def __init__(self, config: dict):
        """Initialize the feature engineer with configuration."""
        self.config = config
        self.feature_config = load_features_config()
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ensure logs directory exists
        Path('logs').mkdir(exist_ok=True)
        
        # Ensure reports directory exists
        Path('reports/runs').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Feature Engineer with run ID: {self.run_id}")
        logger.info(f"Feature families: {', '.join(self.feature_config['feature_families'])}")
    
    def load_processed_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Load processed data for feature engineering."""
        logger.info(f"Loading processed data for {len(symbols)} symbols")
        
        data_dict = {}
        for symbol in symbols:
            try:
                file_path = f"data/processed/binance/{symbol.lower()}_bars_1m.parquet"
                if Path(file_path).exists():
                    df = pd.read_parquet(file_path)
                    df['symbol'] = symbol
                    data_dict[symbol] = df
                    logger.info(f"Loaded {len(df)} rows for {symbol}")
                else:
                    logger.warning(f"Processed data file not found for {symbol}: {file_path}")
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
        
        return data_dict
    
    def compute_price_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute core price and return features."""
        logger.info("Computing price and return features")
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns: {required_cols}")
            return df
        
        # 1-minute returns (log returns)
        df['ret_1m'] = np.log(df['close'] / df['close'].shift(1))
        
        # Multi-period cumulative returns
        df['ret_5m'] = np.log(df['close'] / df['close'].shift(5))
        df['ret_15m'] = np.log(df['close'] / df['close'].shift(15))
        
        # Intra-bar directionality
        df['close_to_open'] = (df['close'] - df['open']) / df['open']
        
        # High-low range (realized volatility proxy)
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        
        # Open-close absolute difference
        df['oc_abs'] = np.abs(df['open'] - df['close']) / df['close']
        
        return df
    
    def compute_trend_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute trend and momentum features."""
        logger.info("Computing trend and momentum features")
        
        windows = self.feature_config['rolling_windows']
        
        # Moving averages
        for period in windows['ma']:
            df[f'ma_{period}'] = df['close'].rolling(window=period, min_periods=period).mean()
        
        # Exponential moving averages
        for period in windows['ema']:
            df[f'ema_{period}'] = df['close'].ewm(span=period, min_periods=period).mean()
        
        # Price momentum
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        # Trend slope (OLS regression over rolling window)
        df['trend_slope_20'] = self._compute_trend_slope(df['close'], 20)
        
        # Breakout indicator
        df['breakout_20'] = self._compute_breakout(df['close'], 20)
        
        # RSI
        df['rsi_14'] = self._compute_rsi(df['close'], 14)
        
        return df
    
    def compute_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute mean reversion features."""
        logger.info("Computing mean reversion features")
        
        # Z-score from moving average
        df['zscore_20'] = (df['close'] - df['ma_20']) / df['close'].rolling(window=20, min_periods=20).std()
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_upper_20'] = df['ma_20'] + (bb_std * df['close'].rolling(window=bb_period, min_periods=bb_period).std())
        df['bb_lower_20'] = df['ma_20'] - (bb_std * df['close'].rolling(window=bb_period, min_periods=bb_period).std())
        df['bb_bandwidth_20'] = (df['bb_upper_20'] - df['bb_lower_20']) / df['ma_20']
        
        # Intraday reversal flag
        threshold = self.feature_config['thresholds']['intraday_reversal_threshold']
        df['intraday_reversal_flag'] = self._compute_intraday_reversal_flag(df, threshold)
        
        # VWAP distance (approximated from OHLC)
        df['vwap_dist'] = self._compute_vwap_distance(df)
        
        return df
    
    def compute_volatility_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility and risk features."""
        logger.info("Computing volatility and risk features")
        
        windows = self.feature_config['rolling_windows']
        
        # Rolling volatility (standard deviation of returns)
        for period in windows['volatility']:
            df[f'vol_{period}'] = df['ret_1m'].rolling(window=period, min_periods=period).std()
        
        # ATR (Average True Range)
        df['atr_14'] = self._compute_atr(df, 14)
        
        # Realized volatility (annualized)
        df['realized_vol_30'] = df['ret_1m'].rolling(window=30, min_periods=30).std() * np.sqrt(525600)
        
        # Volatility percentile (90-day lookback)
        df['vol_percentile_90d'] = self._compute_volatility_percentile(df, 90)
        
        # Downside volatility (only negative returns)
        df['downside_vol_30'] = self._compute_downside_volatility(df, 30)
        
        return df
    
    def compute_volume_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume and liquidity features."""
        logger.info("Computing volume and liquidity features")
        
        windows = self.feature_config['rolling_windows']
        
        # Volume moving averages
        for period in windows['volume']:
            df[f'vol_ma_{period}'] = df['volume'].rolling(window=period, min_periods=period).mean()
        
        # Volume z-score
        df['vol_zscore_20'] = (df['volume'] - df['vol_ma_20']) / df['volume'].rolling(window=20, min_periods=20).std()
        
        # Volume spike flag
        multiplier = self.feature_config['thresholds']['volume_spike_multiplier']
        df['volume_spike_flag'] = (df['volume'] > (multiplier * df['vol_ma_20'])).astype(int)
        
        # Notional share (position sizing hint)
        df['notional_share'] = self._compute_notional_share(df, 60)
        
        return df
    
    def compute_cross_asset_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Compute cross-asset relationship features."""
        logger.info("Computing cross-asset features")
        
        driver_symbol = self.feature_config['cross_assets']['driver_symbol']
        pairs = self.feature_config['cross_assets']['pairs']
        
        if driver_symbol not in data_dict:
            logger.warning(f"Driver symbol {driver_symbol} not found in data")
            return data_dict
        
        driver_df = data_dict[driver_symbol].copy()
        
        # Ensure driver_df has unique timestamps
        driver_df = driver_df.drop_duplicates(subset=['ts']).reset_index(drop=True)
        
        # Add driver symbol returns to all other symbols
        for symbol, df in data_dict.items():
            if symbol != driver_symbol:
                # Ensure this symbol also has unique timestamps
                df = df.drop_duplicates(subset=['ts']).reset_index(drop=True)
                
                # Align timestamps and merge driver data
                merged = pd.merge(
                    df, 
                    driver_df[['ts', 'ret_1m', 'close']], 
                    on='ts', 
                    suffixes=('', f'_{driver_symbol.lower()}')
                )
                
                # Rename columns for clarity
                merged[f'{driver_symbol.lower()}_ret_1m'] = merged[f'ret_1m_{driver_symbol.lower()}']
                merged[f'{driver_symbol.lower()}_close'] = merged[f'close_{driver_symbol.lower()}']
                
                # Cross-asset specific features for major pairs
                if symbol in pairs:
                    merged = self._compute_pair_specific_features(merged, driver_df, symbol, driver_symbol)
                
                data_dict[symbol] = merged
        
        return data_dict
    
    def compute_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute microstructure proxy features."""
        logger.info("Computing microstructure features")
        
        # High-low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Open-close spread
        df['oc_spread'] = np.abs(df['open'] - df['close']) / df['close']
        
        # Kyle lambda proxy (price impact)
        df['kyle_lambda_proxy'] = np.abs(df['ret_1m']) / (df['volume'] + 1e-8)  # Avoid division by zero
        
        # Roll measure proxy (microstructure noise)
        df['roll_measure_proxy'] = self._compute_roll_measure(df)
        
        return df
    
    def compute_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute market regime classification features."""
        logger.info("Computing regime features")
        
        # Trend regime score (0-1)
        df['trend_regime_score'] = self._compute_trend_regime_score(df)
        
        # Volatility regime flag
        vol_threshold = self.feature_config['thresholds']['volatility_regime_percentile']
        df['vol_regime_flag'] = (df['vol_percentile_90d'] > vol_threshold).astype(int)
        
        # Liquidity regime flag
        df['liquidity_regime_flag'] = self._compute_liquidity_regime_flag(df)
        
        return df
    
    def compute_risk_execution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute risk and execution helper features."""
        logger.info("Computing risk and execution features")
        
        # Position cap hint
        df['position_cap_hint'] = self._compute_position_cap_hint(df)
        
        # Stop distance hint
        df['stop_distance_hint'] = self._compute_stop_distance_hint(df)
        
        # Slippage hint
        df['slippage_hint_bps'] = self._compute_slippage_hint(df)
        
        return df
    
    def _compute_trend_slope(self, series: pd.Series, window: int) -> pd.Series:
        """Compute trend slope using OLS regression."""
        def slope(x):
            if len(x) < window:
                return np.nan
            y = x.values
            x_vals = np.arange(len(y))
            try:
                slope_val = np.polyfit(x_vals, y, 1)[0]
                return slope_val
            except:
                return np.nan
        
        return series.rolling(window=window, min_periods=window).apply(slope)
    
    def _compute_breakout(self, series: pd.Series, lookback: int) -> pd.Series:
        """Compute breakout indicator."""
        return (series > series.rolling(window=lookback, min_periods=lookback).max().shift(1)).astype(int)
    
    def _compute_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Compute RSI (Relative Strength Index)."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _compute_intraday_reversal_flag(self, df: pd.DataFrame, threshold: float) -> pd.Series:
        """Compute intraday reversal flag."""
        # Look at last 5 bars for trend
        trend = df['ret_1m'].rolling(window=5, min_periods=5).sum()
        current_ret = df['ret_1m']
        
        # Reversal: current return opposes trend and magnitude > threshold
        reversal = ((current_ret * trend < 0) & (np.abs(current_ret) > threshold)).astype(int)
        return reversal
    
    def _compute_vwap_distance(self, df: pd.DataFrame) -> pd.Series:
        """Compute VWAP distance (approximated from OHLC)."""
        # Approximate VWAP as (H+L+C)/3
        vwap = (df['high'] + df['low'] + df['close']) / 3
        return (df['close'] - vwap) / vwap
    
    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=period).mean()
        return atr
    
    def _compute_volatility_percentile(self, df: pd.DataFrame, days: int) -> pd.Series:
        """Compute volatility percentile over lookback period."""
        def percentile_rank(x):
            if len(x) < days:
                return np.nan
            current_vol = x.iloc[-1]
            return (x < current_vol).mean()
        
        return df['vol_20'].rolling(window=days, min_periods=days).apply(percentile_rank)
    
    def _compute_downside_volatility(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute downside volatility (only negative returns)."""
        negative_returns = df['ret_1m'].where(df['ret_1m'] < 0)
        return negative_returns.rolling(window=period, min_periods=period).std()
    
    def _compute_notional_share(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Compute notional share for position sizing."""
        notional = df['close'] * df['volume']
        rolling_sum = notional.rolling(window=window, min_periods=window).sum()
        return notional / (rolling_sum + 1e-8)
    
    def _compute_pair_specific_features(self, df: pd.DataFrame, driver_df: pd.DataFrame, symbol: str, driver: str) -> pd.DataFrame:
        """Compute pair-specific cross-asset features."""
        try:
            # Ensure we have the required columns
            required_cols = ['ret_1m', f'ret_1m_{driver.lower()}', f'close_{driver.lower()}']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns for {symbol} pair features: {required_cols}")
                return df
            
            # ETH minus BTC return
            if symbol == 'ETHUSDT':
                df['eth_minus_btc_ret'] = df['ret_1m'] - df[f'ret_1m_{driver.lower()}']
                
                # ETH/BTC ratio (need close prices from both)
                if 'close' in df.columns and f'close_{driver.lower()}' in df.columns:
                    df['ratio_eth_btc'] = df['close'] / df[f'close_{driver.lower()}']
                    
                    # Ratio z-score over 240 bars
                    df['ratio_eth_btc_zscore_240'] = self._compute_ratio_zscore(df['ratio_eth_btc'], 240)
                    
                    # Beta to BTC over 720 bars
                    df['beta_to_btc_720'] = self._compute_beta_to_driver(df, 720)
                else:
                    logger.warning(f"Missing close price columns for {symbol} ratio features")
            
            return df
            
        except Exception as e:
            logger.error(f"Error computing pair features for {symbol}: {e}")
            return df
    
    def _compute_ratio_zscore(self, ratio: pd.Series, window: int) -> pd.Series:
        """Compute z-score of ratio over rolling window."""
        return (ratio - ratio.rolling(window=window, min_periods=window).mean()) / ratio.rolling(window=window, min_periods=window).std()
    
    def _compute_beta_to_driver(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Compute beta to driver symbol over rolling window."""
        try:
            def beta(x):
                if len(x) < window:
                    return np.nan
                try:
                    # Get returns for this window
                    rets = x['ret_1m'].iloc[-window:]
                    driver_rets = x['ret_1m_btcusdt'].iloc[-window:]
                    
                    # Compute beta
                    covariance = np.cov(rets, driver_rets)[0, 1]
                    driver_variance = np.var(driver_rets)
                    
                    if driver_variance == 0:
                        return np.nan
                    
                    return covariance / driver_variance
                except:
                    return np.nan
            
            # Create a DataFrame with both return series for rolling apply
            temp_df = pd.DataFrame({
                'ret_1m': df['ret_1m'],
                'ret_1m_btcusdt': df[f'ret_1m_btcusdt']
            })
            
            # Use rolling apply with raw=False to avoid reindexing issues
            return temp_df.rolling(window=window, min_periods=window).apply(beta, raw=False)
            
        except Exception as e:
            logger.error(f"Error computing beta: {e}")
            return pd.Series(np.nan, index=df.index)
    
    def _compute_roll_measure(self, df: pd.DataFrame) -> pd.Series:
        """Compute Roll measure proxy (microstructure noise)."""
        def roll_measure(x):
            if len(x) < 2:
                return np.nan
            returns = x.values
            # Negative autocovariance at lag 1
            if len(returns) >= 2:
                return -np.cov(returns[:-1], returns[1:])[0, 1]
            return np.nan
        
        return df['ret_1m'].rolling(window=20, min_periods=20).apply(roll_measure)
    
    def _compute_trend_regime_score(self, df: pd.DataFrame) -> pd.Series:
        """Compute trend regime score (0-1)."""
        # Combine multiple trend indicators
        trend_indicators = []
        
        # Trend slope (normalized)
        if 'trend_slope_20' in df.columns:
            slope_norm = (df['trend_slope_20'] - df['trend_slope_20'].rolling(window=100, min_periods=100).mean()) / (df['trend_slope_20'].rolling(window=100, min_periods=100).std() + 1e-8)
            trend_indicators.append(slope_norm)
        
        # Breakout indicator
        if 'breakout_20' in df.columns:
            trend_indicators.append(df['breakout_20'])
        
        # EMA trend
        if 'ema_20' in df.columns and 'ema_50' in df.columns:
            ema_trend = (df['ema_20'] > df['ema_50']).astype(float)
            trend_indicators.append(ema_trend)
        
        if trend_indicators:
            # Combine and normalize to 0-1
            combined = pd.concat(trend_indicators, axis=1).mean(axis=1)
            # Normalize to 0-1 range
            combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
            return combined
        
        return pd.Series(0.5, index=df.index)
    
    def _compute_liquidity_regime_flag(self, df: pd.DataFrame) -> pd.Series:
        """Compute liquidity regime flag."""
        flags = []
        
        # Thin market flag
        if 'vol_zscore_20' in df.columns:
            thin_market = (df['vol_zscore_20'] < -1).astype(int)
            flags.append(thin_market)
        
        # High notional share flag
        if 'notional_share' in df.columns:
            high_share = (df['notional_share'] > df['notional_share'].rolling(window=100, min_periods=100).quantile(0.9)).astype(int)
            flags.append(high_share)
        
        if flags:
            return pd.concat(flags, axis=1).max(axis=1)
        
        return pd.Series(0, index=df.index)
    
    def _compute_position_cap_hint(self, df: pd.DataFrame) -> pd.Series:
        """Compute position cap hint for risk management."""
        cap = 1.0  # Base cap
        
        # Reduce cap in high volatility
        if 'vol_regime_flag' in df.columns:
            cap = cap * (1 - 0.3 * df['vol_regime_flag'])
        
        # Reduce cap in low liquidity
        if 'liquidity_regime_flag' in df.columns:
            cap = cap * (1 - 0.2 * df['liquidity_regime_flag'])
        
        # Reduce cap for high beta assets
        if 'beta_to_btc_720' in df.columns:
            beta_factor = np.minimum(df['beta_to_btc_720'].abs(), 2.0) / 2.0
            cap = cap * (1 - 0.2 * beta_factor)
        
        return np.maximum(cap, 0.1)  # Minimum 10% cap
    
    def _compute_stop_distance_hint(self, df: pd.DataFrame) -> pd.Series:
        """Compute stop distance hint based on ATR."""
        multiplier = self.feature_config['thresholds']['stop_atr_multiplier']
        
        if 'atr_14' in df.columns:
            return multiplier * df['atr_14']
        
        return pd.Series(0.02, index=df.index)  # Default 2%
    
    def _compute_slippage_hint(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute slippage hint based on market conditions."""
        base_slippage = 10  # Base 10 bps
        
        if 'hl_spread' in df.columns:
            # Adjust for current volatility vs historical
            median_spread = df['hl_spread'].rolling(window=100, min_periods=100).median()
            volatility_adjustment = df['hl_spread'] / (median_spread + 1e-8)
            
            # Cap adjustment factor
            volatility_adjustment = np.minimum(volatility_adjustment, 3.0)
            
            return base_slippage * (1 + volatility_adjustment)
        
        return pd.Series(base_slippage, index=df.index)
    
    def add_metadata_columns(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add required metadata columns."""
        df['source'] = 'binance'
        df['load_id'] = self.run_id
        df['ingestion_ts'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        return df
    
    def apply_winsorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply winsorization to numeric features."""
        logger.info("Applying winsorization to features")
        
        limits = self.feature_config['thresholds']['winsorization_limits']
        
        # Get numeric columns (exclude metadata and timestamp columns)
        exclude_cols = ['ts', 'symbol', 'source', 'load_id', 'ingestion_ts']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for col in numeric_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                lower_limit = df[col].quantile(limits[0])
                upper_limit = df[col].quantile(limits[1])
                df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
        
        return df
    
    def run_feature_engineering(self, symbols: List[str]) -> Dict[str, Any]:
        """Run complete feature engineering pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING PROFESSIONAL FEATURE ENGINEERING")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Load processed data
            data_dict = self.load_processed_data(symbols)
            
            if not data_dict:
                raise ValueError("No data loaded for any symbols")
            
            # Process each symbol
            processed_symbols = []
            total_features = 0
            
            for symbol, df in data_dict.items():
                logger.info(f"Processing features for {symbol}")
                
                # Sort by timestamp
                df = df.sort_values('ts').reset_index(drop=True)
                
                # Compute all feature families
                df = self.compute_price_return_features(df)
                df = self.compute_trend_momentum_features(df)
                df = self.compute_mean_reversion_features(df)
                df = self.compute_volatility_risk_features(df)
                df = self.compute_volume_liquidity_features(df)
                df = self.compute_microstructure_features(df)
                df = self.compute_regime_features(df)
                df = self.compute_risk_execution_features(df)
                
                # Add metadata columns
                df = self.add_metadata_columns(df, symbol)
                
                # Apply winsorization
                df = self.apply_winsorization(df)
                
                # Store processed data
                data_dict[symbol] = df
                processed_symbols.append(symbol)
                total_features += len(df)
                
                logger.info(f"Completed {symbol}: {len(df)} rows, {len(df.columns)} columns")
            
            # Compute cross-asset features
            data_dict = self.compute_cross_asset_features(data_dict)
            
            # Save features
            self._save_features(data_dict)
            
            # Generate QC report
            qc_report = self._generate_qc_report(data_dict)
            
            # Calculate duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Final summary
            logger.info("=" * 80)
            logger.info("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
            logger.info(f"Processed Symbols: {len(processed_symbols)}")
            logger.info(f"Total Feature Rows: {total_features}")
            logger.info(f"Duration: {duration:.1f} seconds")
            logger.info("=" * 80)
            
            return {
                'status': 'success',
                'run_id': self.run_id,
                'processed_symbols': processed_symbols,
                'total_features': total_features,
                'duration_seconds': duration,
                'qc_report': qc_report
            }
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'run_id': self.run_id
            }
    
    def _save_features(self, data_dict: Dict[str, pd.DataFrame]):
        """Save features to parquet files."""
        logger.info("Saving features to parquet files")
        
        # Ensure features directory exists
        features_dir = Path(self.feature_config['output']['path']).parent
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each symbol separately for now (can be optimized later)
        for symbol, df in data_dict.items():
            # Add date partition column
            df['date'] = pd.to_datetime(df['ts']).dt.date.astype(str)
            
            # Save to parquet
            output_path = features_dir / f"{symbol.lower()}_features_1m.parquet"
            df.to_parquet(output_path, compression='snappy', index=False)
            logger.info(f"Saved features for {symbol} to {output_path}")
    
    def _generate_qc_report(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate quality control report."""
        logger.info("Generating QC report")
        
        qc_report = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'symbols_processed': list(data_dict.keys()),
            'feature_summary': {},
            'quality_metrics': {}
        }
        
        # Analyze each symbol
        for symbol, df in data_dict.items():
            # Feature summary
            feature_summary = {}
            for col in df.columns:
                if col not in ['ts', 'symbol', 'source', 'load_id', 'ingestion_ts', 'date']:
                    if df[col].dtype in ['float64', 'int64']:
                        feature_summary[col] = {
                            'non_na_pct': (df[col].notna().sum() / len(df)) * 100,
                            'mean': df[col].mean(),
                            'std': df[col].std(),
                            'p1': df[col].quantile(0.01),
                            'p50': df[col].quantile(0.50),
                            'p99': df[col].quantile(0.99)
                        }
            
            qc_report['feature_summary'][symbol] = feature_summary
            
            # Quality metrics
            qc_report['quality_metrics'][symbol] = {
                'total_rows': len(df),
                'total_features': len(feature_summary),
                'timestamp_coverage': df['ts'].nunique(),
                'monotonic_timestamps': df['ts'].is_monotonic_increasing,
                'unique_symbol_timestamp': df.groupby(['ts', 'symbol']).size().max() == 1
            }
        
        # Save QC report
        qc_file = Path('reports/runs') / f'feature_engineering_qc_{self.run_id}.yaml'
        with open(qc_file, 'w') as f:
            yaml.dump(qc_report, f, default_flow_style=False, indent=2)
        
        logger.info(f"QC report saved to: {qc_file}")
        return qc_report

def main():
    """Main function to run feature engineering."""
    parser = argparse.ArgumentParser(description='Run professional feature engineering for crypto data')
    parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to process')
    parser.add_argument('--config', default='configs/features.yaml', help='Feature configuration file')
    
    args = parser.parse_args()
    
    # Load configurations
    try:
        base_config = load_base_config()
        features_config = load_features_config()
        config = {**base_config, **features_config}
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Create feature engineer and run
    engineer = FeatureEngineer(config)
    results = engineer.run_feature_engineering(args.symbols)
    
    # Return exit code based on success
    if results['status'] == 'success':
        logger.info("Feature engineering completed successfully!")
        return 0
    else:
        logger.error("Feature engineering failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
