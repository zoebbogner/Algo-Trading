"""Refactored Feature Engineering Engine for Crypto Algorithmic Trading."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import numpy as np

from ...utils.file_ops import ensure_directory
from ...utils.logging import get_logger, log_performance
from ...utils.time_utils import get_utc_now
from ...utils.validation import validate_dataframe, validate_timestamps

from .price_features import PriceFeatureComputer
from .trend_features import TrendFeatureComputer

logger = get_logger(__name__)


class FeatureEngineerRefactored:
    """
    Refactored feature engineering engine with modular design.
    
    Implements best practices:
    - No data leakage (rolling windows only)
    - Proper volatility scaling for intraday data
    - Industry-standard technical indicators
    - Partitioned output for scalability
    - Comprehensive quality control
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the refactored feature engineer."""
        self._setup_configuration(config)
        self._setup_feature_computers()
        self._validate_configuration()
        
        logger.info("FeatureEngineerRefactored initialized successfully")

    def _setup_configuration(self, config: dict[str, Any]) -> None:
        """Setup configuration parameters."""
        self.config = config
        self.rolling_windows = config.get('rolling_windows', {})
        self.thresholds = config.get('thresholds', {})
        self.output_config = config.get('output', {})
        self.qc_config = config.get('qc', {})
        self.data_config = config.get('data_config', {})

        # Set defaults for data frequency scaling
        self.bars_per_day = self.data_config.get('bars_per_day', 1440)
        self.bars_per_quarter = self.data_config.get('bars_per_quarter', 129600)

    def _setup_feature_computers(self) -> None:
        """Setup modular feature computers."""
        self.price_computer = PriceFeatureComputer(
            return_periods=[1, 5, 15]
        )
        
        self.trend_computer = TrendFeatureComputer(
            ma_windows=self.rolling_windows.get('ma', [20, 50, 100]),
            ema_windows=self.rolling_windows.get('ema', [20, 50]),
            rsi_windows=self.rolling_windows.get('rsi', [14]),
            regression_windows=self.rolling_windows.get('regression', [20, 240, 720])
        )

    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        self._validate_required_keys()
        self._validate_rolling_windows()

    def _validate_required_keys(self) -> None:
        """Validate that required configuration keys are present."""
        required_keys = ['rolling_windows', 'thresholds', 'output']
        missing_keys = [key for key in required_keys if key not in self.config]

        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

    def _validate_rolling_windows(self) -> None:
        """Validate rolling window configurations."""
        for window_type, windows in self.rolling_windows.items():
            if not isinstance(windows, list) or not all(isinstance(w, int) for w in windows):
                raise ValueError(f"Invalid rolling windows for {window_type}: {windows}")

    @log_performance()
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for the input DataFrame."""
        logger.info(f"Computing features for {len(df)} rows")

        df = self._validate_input_data(df)
        df = self._compute_feature_groups(df)
        df = self._add_metadata(df)
        df = self._run_quality_control(df)

        logger.info(f"Feature computation completed. Output shape: {df.shape}")
        return df

    def _validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate input data requirements."""
        required_columns = ['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        validation_result = validate_dataframe(df, required_columns)
        
        if not validation_result['valid']:
            raise ValueError(f"Data validation failed: {validation_result['errors']}")

        timestamp_validation = validate_timestamps(
            df,
            timestamp_column='ts',
            symbol_column='symbol',
            require_monotonic=True,
            require_unique=True
        )
        
        if not timestamp_validation['valid']:
            raise ValueError(f"Timestamp validation failed: {timestamp_validation['errors']}")

        return df.copy()

    def _compute_feature_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features using modular components."""
        # Compute features in dependency order
        df = self.price_computer.compute_features(df)
        df = self.trend_computer.compute_features(df)
        
        # Add basic volatility features
        df = self._add_basic_volatility_features(df)
        
        return df

    def _add_basic_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic volatility features."""
        logger.debug("Adding basic volatility features")
        
        # Simple volatility (could be expanded with dedicated volatility computer)
        for window in [20, 60]:
            returns = df['close'].pct_change()
            df[f'volatility_{window}'] = returns.rolling(window=window).std()
        
        return df

    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata columns."""
        df['feature_computed_at'] = get_utc_now()
        df['feature_version'] = '2.0.0'  # Refactored version
        df['config_hash'] = str(hash(str(self.config)))
        
        return df

    def _run_quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run quality control checks."""
        if not self.qc_config.get('enabled', False):
            return df

        self._check_data_quality(df)
        return df

    def _check_data_quality(self, df: pd.DataFrame) -> None:
        """Check data quality metrics."""
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Missing values detected: {missing_counts[missing_counts > 0]}")

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=['number']).columns
        inf_counts = df[numeric_cols].apply(lambda x: np.isinf(x).sum())
        if inf_counts.sum() > 0:
            logger.warning(f"Infinite values detected: {inf_counts[inf_counts > 0]}")

    def save_features(self, df: pd.DataFrame, output_path: str) -> None:
        """Save computed features to partitioned Parquet files."""
        output_dir = Path(output_path)
        ensure_directory(output_dir)

        # Partition by symbol and date
        df['date'] = pd.to_datetime(df['ts']).dt.date
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            
            for date in symbol_data['date'].unique():
                date_data = symbol_data[symbol_data['date'] == date]
                
                # Create partition directory
                partition_dir = output_dir / f"symbol={symbol}" / f"date={date}"
                partition_dir.mkdir(parents=True, exist_ok=True)
                
                # Save partition
                partition_file = partition_dir / f"{symbol}_{date}.parquet"
                date_data.to_parquet(partition_file, index=False)
                
                logger.debug(f"Saved partition: {partition_file}")

        logger.info(f"Features saved to {output_dir}")

    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of computed features."""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        summary = {
            'total_rows': len(df),
            'total_symbols': df['symbol'].nunique(),
            'feature_columns': len(numeric_cols),
            'missing_values': df.isnull().sum().to_dict(),
            'feature_ranges': {}
        }
        
        for col in numeric_cols:
            if col in ['ts', 'symbol']:
                continue
                
            summary['feature_ranges'][col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            }
        
        return summary
