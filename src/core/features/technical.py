"""Technical indicators for feature engineering."""

import numpy as np
import pandas as pd
from typing import Optional, Union, List

from ..data_models.market import Bar


class TechnicalIndicators:
    """Technical indicators calculator."""
    
    @staticmethod
    def sma(prices: Union[List[float], np.ndarray, pd.Series], period: int) -> float:
        """Calculate Simple Moving Average.
        
        Args:
            prices: Price data
            period: Lookback period
            
        Returns:
            SMA value
        """
        if len(prices) < period:
            return np.nan
        
        return np.mean(prices[-period:])
    
    @staticmethod
    def ema(prices: Union[List[float], np.ndarray, pd.Series], period: int) -> float:
        """Calculate Exponential Moving Average.
        
        Args:
            prices: Price data
            period: Lookback period
            
        Returns:
            EMA value
        """
        if len(prices) < period:
            return np.nan
        
        alpha = 2.0 / (period + 1)
        ema_value = prices[0]
        
        for price in prices[1:]:
            ema_value = alpha * price + (1 - alpha) * ema_value
        
        return ema_value
    
    @staticmethod
    def rsi(prices: Union[List[float], np.ndarray, pd.Series], period: int = 14) -> float:
        """Calculate Relative Strength Index.
        
        Args:
            prices: Price data
            period: Lookback period (default: 14)
            
        Returns:
            RSI value (0-100)
        """
        if len(prices) < period + 1:
            return np.nan
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(prices: Union[List[float], np.ndarray, pd.Series], 
             fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> dict:
        """Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price data
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
            
        Returns:
            Dictionary with MACD, signal, and histogram values
        """
        if len(prices) < slow_period:
            return {"macd": np.nan, "signal": np.nan, "histogram": np.nan}
        
        fast_ema = TechnicalIndicators.ema(prices, fast_period)
        slow_ema = TechnicalIndicators.ema(prices, slow_period)
        
        if np.isnan(fast_ema) or np.isnan(slow_ema):
            return {"macd": np.nan, "signal": np.nan, "histogram": np.nan}
        
        macd_line = fast_ema - slow_ema
        
        # For signal line, we'd need to track MACD values over time
        # For now, return just the MACD line
        return {
            "macd": macd_line,
            "signal": np.nan,  # Would need historical MACD values
            "histogram": np.nan
        }
    
    @staticmethod
    def bollinger_bands(prices: Union[List[float], np.ndarray, pd.Series], 
                        period: int = 20, std_dev: float = 2.0) -> dict:
        """Calculate Bollinger Bands.
        
        Args:
            prices: Price data
            period: Lookback period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        if len(prices) < period:
            return {"upper": np.nan, "middle": np.nan, "lower": np.nan}
        
        middle = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return {
            "upper": upper,
            "middle": middle,
            "lower": lower
        }
    
    @staticmethod
    def atr(highs: Union[List[float], np.ndarray, pd.Series],
             lows: Union[List[float], np.ndarray, pd.Series],
             closes: Union[List[float], np.ndarray, pd.Series],
             period: int = 14) -> float:
        """Calculate Average True Range.
        
        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            period: Lookback period (default: 14)
            
        Returns:
            ATR value
        """
        if len(highs) < period + 1:
            return np.nan
        
        true_ranges = []
        for i in range(1, len(highs)):
            high_low = highs[i] - lows[i]
            high_close_prev = abs(highs[i] - closes[i-1])
            low_close_prev = abs(lows[i] - closes[i-1])
            
            true_range = max(high_low, high_close_prev, low_close_prev)
            true_ranges.append(true_range)
        
        if len(true_ranges) < period:
            return np.nan
        
        return np.mean(true_ranges[-period:])
    
    @staticmethod
    def stochastic(highs: Union[List[float], np.ndarray, pd.Series],
                   lows: Union[List[float], np.ndarray, pd.Series],
                   closes: Union[List[float], np.ndarray, pd.Series],
                   k_period: int = 14, d_period: int = 3) -> dict:
        """Calculate Stochastic Oscillator.
        
        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)
            
        Returns:
            Dictionary with %K and %D values
        """
        if len(closes) < k_period:
            return {"k": np.nan, "d": np.nan}
        
        k_values = []
        for i in range(k_period):
            if i >= len(closes):
                break
            
            highest_high = max(highs[-k_period+i:])
            lowest_low = min(lows[-k_period+i:])
            
            if highest_high == lowest_low:
                k = 50.0
            else:
                k = ((closes[-(i+1)] - lowest_low) / (highest_high - lowest_low)) * 100
            
            k_values.append(k)
        
        if len(k_values) < d_period:
            return {"k": np.nan, "d": np.nan}
        
        k = k_values[0]  # Current %K
        d = np.mean(k_values[:d_period])  # Current %D
        
        return {"k": k, "d": d}


class FeatureEngine:
    """Feature engineering engine for market data."""
    
    def __init__(self):
        """Initialize feature engine."""
        self.indicators = TechnicalIndicators()
    
    def calculate_features(self, bars: List[Bar]) -> dict:
        """Calculate features from a list of bars.
        
        Args:
            bars: List of OHLCV bars
            
        Returns:
            Dictionary of calculated features
        """
        if not bars:
            return {}
        
        # Extract price data
        closes = [float(bar.close) for bar in bars]
        highs = [float(bar.high) for bar in bars]
        lows = [float(bar.low) for bar in bars]
        volumes = [float(bar.volume) for bar in bars]
        
        features = {}
        
        # Technical indicators
        if len(closes) >= 20:
            features["sma_20"] = self.indicators.sma(closes, 20)
            features["sma_50"] = self.indicators.sma(closes, 50) if len(closes) >= 50 else np.nan
            features["rsi_14"] = self.indicators.rsi(closes, 14)
            
            bb = self.indicators.bollinger_bands(closes, 20)
            features["bb_upper"] = bb["upper"]
            features["bb_middle"] = bb["middle"]
            features["bb_lower"] = bb["lower"]
            
            features["bb_position"] = (closes[-1] - bb["lower"]) / (bb["upper"] - bb["lower"])
        
        if len(closes) >= 26:
            macd = self.indicators.macd(closes)
            features["macd"] = macd["macd"]
        
        if len(closes) >= 14:
            features["atr_14"] = self.indicators.atr(highs, lows, closes, 14)
            stoch = self.indicators.stochastic(highs, lows, closes)
            features["stoch_k"] = stoch["k"]
            features["stoch_d"] = stoch["d"]
        
        # Price-based features
        if len(closes) >= 2:
            features["returns_1m"] = (closes[-1] - closes[-2]) / closes[-2]
            
            if len(closes) >= 6:
                features["returns_5m"] = (closes[-1] - closes[-6]) / closes[-6]
            
            if len(closes) >= 16:
                features["returns_15m"] = (closes[-1] - closes[-16]) / closes[-16]
        
        # Volatility features
        if len(closes) >= 20:
            returns = np.diff(closes) / closes[:-1]
            features["volatility_rolling"] = np.std(returns[-20:]) * np.sqrt(20)
        
        # Volume features
        if len(volumes) >= 20:
            features["volume_sma_20"] = self.indicators.sma(volumes, 20)
            features["volume_ratio"] = volumes[-1] / features["volume_sma_20"] if features["volume_sma_20"] > 0 else 1.0
        
        # Momentum features
        if len(closes) >= 10:
            features["price_momentum"] = (closes[-1] - closes[-10]) / closes[-10]
        
        return features
