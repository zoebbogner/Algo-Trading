"""Machine Learning Feature Engineering for Trading."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class MLFeatureEngineer:
    """Advanced feature engineering for ML-based trading strategies."""
    
    def __init__(self, config: Dict):
        """Initialize ML feature engineer."""
        self.config = config
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        self.feature_columns = []
        self.is_fitted = False
        
    def create_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive ML features from raw data."""
        logger.info("Creating ML features...")
        
        # Ensure we have required columns
        required_cols = ['close', 'volume', 'ts']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Price-based features
        df = self._add_price_features(df)
        
        # Volume-based features
        df = self._add_volume_features(df)
        
        # Technical indicator features
        df = self._add_technical_features(df)
        
        # Statistical features
        df = self._add_statistical_features(df)
        
        # Market microstructure features
        df = self._add_microstructure_features(df)
        
        # Time-based features
        df = self._add_time_features(df)
        
        # Cross-asset features (if multiple symbols)
        df = self._add_cross_asset_features(df)
        
        # Feature interactions
        df = self._add_feature_interactions(df)
        
        # Remove any infinite or NaN values
        df = self._clean_features(df)
        
        logger.info(f"Created {len(self.feature_columns)} ML features")
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced price-based features."""
        # Returns at different timeframes
        for period in [1, 2, 3, 5, 10, 20]:
            df[f'return_{period}m'] = df['close'].pct_change(period)
            df[f'log_return_{period}m'] = np.log(df['close'] / df['close'].shift(period))
        
        # Price momentum
        for period in [5, 10, 20, 50]:
            df[f'momentum_{period}m'] = df['close'] / df['close'].shift(period) - 1
            df[f'price_acceleration_{period}m'] = df[f'momentum_{period}m'].diff()
        
        # Price volatility
        for period in [10, 20, 50]:
            df[f'volatility_{period}m'] = df['close'].rolling(period).std()
            df[f'volatility_ratio_{period}m'] = df[f'volatility_{period}m'] / df[f'volatility_{period}m'].rolling(period).mean()
        
        # Price levels
        df['price_position_20'] = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min())
        df['price_position_50'] = (df['close'] - df['close'].rolling(50).min()) / (df['close'].rolling(50).max() - df['close'].rolling(50).min())
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume indicators
        for period in [5, 10, 20, 50]:
            df[f'volume_ma_{period}m'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}m'] = df['volume'] / df[f'volume_ma_{period}m']
            df[f'volume_momentum_{period}m'] = df['volume'].pct_change(period)
        
        # Volume-price relationship
        df['volume_price_trend'] = (df['close'] * df['volume']).rolling(20).sum()
        df['volume_weighted_avg_price'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Volume volatility
        df['volume_volatility_20'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()
        
        return df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features."""
        # Moving averages
        for period in [5, 10, 20, 50, 100]:
            df[f'ma_{period}'] = df['close'].rolling(period).mean()
            df[f'ma_ratio_{period}'] = df['close'] / df[f'ma_{period}']
        
        # RSI variations
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
            df[f'rsi_momentum_{period}'] = df[f'rsi_{period}'].diff()
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        for period in [20, 50]:
            df[f'bb_upper_{period}'], df[f'bb_lower_{period}'], df[f'bb_width_{period}'] = self._calculate_bollinger_bands(df['close'], period)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df, 14)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        # Statistical moments
        for period in [20, 50]:
            df[f'skewness_{period}'] = df['close'].rolling(period).skew()
            df[f'kurtosis_{period}'] = df['close'].rolling(period).kurt()
        
        # Percentiles
        for period in [20, 50]:
            for p in [10, 25, 75, 90]:
                df[f'percentile_{p}_{period}'] = df['close'].rolling(period).quantile(p/100)
        
        # Z-scores
        for period in [20, 50]:
            df[f'zscore_{period}'] = (df['close'] - df['close'].rolling(period).mean()) / df['close'].rolling(period).std()
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        # Bid-ask spread proxy (using high-low)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Price efficiency
        df['price_efficiency_20'] = abs(df['close'] - df['close'].shift(20)) / df['close'].rolling(20).std()
        
        # Market impact
        df['market_impact'] = df['volume'] * df['close'].pct_change().abs()
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        # Convert timestamp to datetime if needed
        if 'ts' in df.columns and df['ts'].dtype == 'object':
            df['ts'] = pd.to_datetime(df['ts'])
        
        # Time features
        df['hour'] = df['ts'].dt.hour
        df['day_of_week'] = df['ts'].dt.dayofweek
        df['day_of_month'] = df['ts'].dt.day
        df['month'] = df['ts'].dt.month
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-asset correlation features."""
        if 'symbol' in df.columns and df['symbol'].nunique() > 1:
            # Calculate correlation with other assets
            symbols = df['symbol'].unique()
            for symbol in symbols:
                if symbol != df['symbol'].iloc[0]:  # Skip first symbol
                    symbol_data = df[df['symbol'] == symbol]['close']
                    if len(symbol_data) > 0:
                        # Align data
                        aligned_data = pd.concat([df['close'], symbol_data], axis=1).dropna()
                        if len(aligned_data) > 20:
                            correlation = aligned_data.iloc[:, 0].rolling(20).corr(aligned_data.iloc[:, 1])
                            df[f'corr_{symbol}'] = correlation
        
        return df
    
    def _add_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature interaction terms."""
        # Price-volume interactions
        df['price_volume_interaction'] = df['close'].pct_change() * df['volume'].pct_change()
        
        # RSI-momentum interactions
        if 'rsi_14' in df.columns and 'momentum_20m' in df.columns:
            df['rsi_momentum_interaction'] = df['rsi_14'] * df['momentum_20m']
        
        # Volatility-price interactions
        if 'volatility_20m' in df.columns:
            df['volatility_price_interaction'] = df['volatility_20m'] * df['close'].pct_change()
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean features by removing infinite and NaN values."""
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill NaN values
        df = df.fillna(method='ffill')
        
        # Drop any remaining NaN rows
        df = df.dropna()
        
        # Store feature columns (excluding metadata)
        metadata_cols = ['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        self.feature_columns = [col for col in df.columns if col not in metadata_cols]
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        width = (upper - lower) / ma
        return upper, lower, width
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator."""
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d = k.rolling(3).mean()
        return k, d
    
    def fit_scaler(self, df: pd.DataFrame) -> None:
        """Fit the scaler on training data."""
        if not self.feature_columns:
            raise ValueError("No features available. Run create_ml_features first.")
        
        feature_data = df[self.feature_columns]
        self.scaler.fit(feature_data)
        self.is_fitted = True
        logger.info("Scaler fitted successfully")
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Run fit_scaler first.")
        
        df_scaled = df.copy()
        df_scaled[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        return df_scaled
    
    def apply_pca(self, df: pd.DataFrame, n_components: Optional[int] = None) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction."""
        if not self.feature_columns:
            raise ValueError("No features available. Run create_ml_features first.")
        
        feature_data = df[self.feature_columns]
        
        if n_components:
            self.pca = PCA(n_components=n_components)
        
        # Fit PCA if not already fitted
        if not hasattr(self.pca, 'components_'):
            self.pca.fit(feature_data)
        
        # Transform features
        pca_features = self.pca.transform(feature_data)
        
        # Create new dataframe with PCA features
        pca_columns = [f'pca_{i}' for i in range(pca_features.shape[1])]
        df_pca = df.copy()
        
        # Replace original features with PCA features
        df_pca = df_pca.drop(columns=self.feature_columns)
        for i, col in enumerate(pca_columns):
            df_pca[col] = pca_features[:, i]
        
        logger.info(f"PCA applied: {len(self.feature_columns)} features reduced to {len(pca_columns)}")
        return df_pca
