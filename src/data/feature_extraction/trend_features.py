"""Trend and Momentum Feature Computation Module."""

import numpy as np
import pandas as pd
from typing import List

from ...utils.logging import get_logger

logger = get_logger(__name__)


class TrendFeatureComputer:
    """Computes trend and momentum-related features."""
    
    def __init__(self, ma_windows: List[int] = None, ema_windows: List[int] = None, 
                 rsi_windows: List[int] = None, regression_windows: List[int] = None):
        """Initialize trend feature computer."""
        self.ma_windows = ma_windows or [20, 50, 100]
        self.ema_windows = ema_windows or [20, 50]
        self.rsi_windows = rsi_windows or [14]
        self.regression_windows = regression_windows or [20, 240, 720]
        
        logger.debug(f"TrendFeatureComputer initialized with windows: MA={self.ma_windows}, EMA={self.ema_windows}")
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all trend and momentum features."""
        logger.debug("Computing trend and momentum features")
        
        df = df.copy()
        df = self._add_moving_averages(df)
        df = self._add_exponential_moving_averages(df)
        df = self._add_momentum_features(df)
        df = self._add_trend_slopes(df)
        df = self._add_breakout_detection(df)
        df = self._add_rsi_features(df)
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple moving averages."""
        for window in self.ma_windows:
            df[f'ma_{window}'] = df['close'].rolling(
                window=window, min_periods=window
            ).mean()
        return df
    
    def _add_exponential_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add exponential moving averages."""
        for window in self.ema_windows:
            df[f'ema_{window}'] = df['close'].ewm(
                span=window, min_periods=window
            ).mean()
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features."""
        for period in [10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        return df
    
    def _add_trend_slopes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend slope features using rolling regression."""
        for window in self.regression_windows:
            df[f'trend_slope_{window}'] = self._compute_rolling_slope(df['close'], window)
        return df
    
    def _add_breakout_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add breakout detection features."""
        breakout_lookback = 20  # Could be configurable
        
        df['breakout_20'] = (
            df['close'] > df['close'].rolling(breakout_lookback).max().shift(1)
        ).astype(int)
        return df
    
    def _add_rsi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI features using Wilder's smoothing."""
        for window in self.rsi_windows:
            df[f'rsi_{window}'] = self._compute_rsi_wilders(df['close'], window)
        return df
    
    def _compute_rolling_slope(self, series: pd.Series, window: int) -> pd.Series:
        """Compute rolling slope using linear regression."""
        def slope(x):
            if len(x) < 2:
                return np.nan
            return np.polyfit(range(len(x)), x, 1)[0]
        
        return series.rolling(window=window).apply(slope)
    
    def _compute_rsi_wilders(self, series: pd.Series, window: int) -> pd.Series:
        """Compute RSI using Wilder's smoothing method."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
