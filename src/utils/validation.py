"""
Data validation utilities for the crypto algorithmic trading system.

Provides validation functions for:
- DataFrame structure and content
- Timestamp validation and alignment
- Data quality checks
- Schema validation
"""

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class DataValidator:
    """Comprehensive data validator with detailed reporting."""

    def __init__(self, strict_mode: bool = True):
        """
        Initialize the data validator.

        Args:
            strict_mode: If True, raise exceptions on validation failures
        """
        self.strict_mode = strict_mode
        self.validation_results = []

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        required_columns: list[str],
        expected_dtypes: dict[str, str | np.dtype] = None,
        min_rows: int = 1,
        max_rows: int = None,
        allow_duplicates: bool = False
    ) -> dict[str, Any]:
        """
        Validate DataFrame structure and content.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            expected_dtypes: Dictionary of column names to expected dtypes
            min_rows: Minimum number of rows required
            max_rows: Maximum number of rows allowed
            allow_duplicates: Whether to allow duplicate rows

        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }

        # Check if DataFrame is empty
        if df is None or df.empty:
            error_msg = "DataFrame is None or empty"
            results['errors'].append(error_msg)
            results['valid'] = False
            if self.strict_mode:
                raise ValidationError(error_msg)
            return results

        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            results['errors'].append(error_msg)
            results['valid'] = False

        # Check data types
        if expected_dtypes:
            for col, expected_dtype in expected_dtypes.items():
                if col in df.columns:
                    actual_dtype = df[col].dtype
                    if not pd.api.types.is_dtype_equal(actual_dtype, expected_dtype):
                        warning_msg = f"Column {col}: expected {expected_dtype}, got {actual_dtype}"
                        results['warnings'].append(warning_msg)

        # Check row count
        row_count = len(df)
        if row_count < min_rows:
            error_msg = f"Row count {row_count} is below minimum {min_rows}"
            results['errors'].append(error_msg)
            results['valid'] = False

        if max_rows and row_count > max_rows:
            warning_msg = f"Row count {row_count} exceeds maximum {max_rows}"
            results['warnings'].append(warning_msg)

        # Check for duplicates
        if not allow_duplicates:
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                warning_msg = f"Found {duplicate_count} duplicate rows"
                results['warnings'].append(warning_msg)

        # Add info
        results['info'] = {
            'row_count': row_count,
            'column_count': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'null_counts': df.isnull().sum().to_dict()
        }

        # Store results
        self.validation_results.append(results)

        if self.strict_mode and not results['valid']:
            raise ValidationError(f"Validation failed: {results['errors']}")

        return results

    def validate_timestamps(
        self,
        df: pd.DataFrame,
        timestamp_column: str = 'ts',
        symbol_column: str = 'symbol',
        require_monotonic: bool = True,
        require_unique: bool = True,
        timezone_aware: bool = True
    ) -> dict[str, Any]:
        """
        Validate timestamp data in DataFrame.

        Args:
            df: DataFrame to validate
            timestamp_column: Name of timestamp column
            symbol_column: Name of symbol column
            require_monotonic: Whether timestamps should be monotonically increasing
            require_unique: Whether timestamps should be unique per symbol
            timezone_aware: Whether timestamps should be timezone-aware

        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }

        if timestamp_column not in df.columns:
            error_msg = f"Timestamp column '{timestamp_column}' not found"
            results['errors'].append(error_msg)
            results['valid'] = False
            if self.strict_mode:
                raise ValidationError(error_msg)
            return results

        # Check timestamp data type
        ts_series = df[timestamp_column]
        if not pd.api.types.is_datetime64_any_dtype(ts_series):
            error_msg = f"Timestamp column '{timestamp_column}' is not datetime type"
            results['errors'].append(error_msg)
            results['valid'] = False

        # Check timezone awareness
        if timezone_aware:
            if not ts_series.dt.tz:
                warning_msg = f"Timestamp column '{timestamp_column}' is not timezone-aware"
                results['warnings'].append(warning_msg)

        # Check for future timestamps
        now = datetime.now(UTC)
        future_timestamps = ts_series[ts_series > now]
        if len(future_timestamps) > 0:
            warning_msg = f"Found {len(future_timestamps)} future timestamps"
            results['warnings'].append(warning_msg)

        # Check monotonicity per symbol
        if require_monotonic and symbol_column in df.columns:
            for symbol in df[symbol_column].unique():
                symbol_data = df[df[symbol_column] == symbol].sort_values(timestamp_column)
                if not symbol_data[timestamp_column].is_monotonic_increasing:
                    error_msg = f"Timestamps not monotonic for symbol {symbol}"
                    results['errors'].append(error_msg)
                    results['valid'] = False

        # Check uniqueness per symbol
        if require_unique and symbol_column in df.columns:
            duplicate_timestamps = df.groupby([symbol_column, timestamp_column]).size()
            duplicate_count = (duplicate_timestamps > 1).sum()
            if duplicate_count > 0:
                warning_msg = f"Found {duplicate_count} duplicate timestamp-symbol combinations"
                results['warnings'].append(warning_msg)

        # Add info
        results['info'] = {
            'min_timestamp': ts_series.min().isoformat() if not ts_series.empty else None,
            'max_timestamp': ts_series.max().isoformat() if not ts_series.empty else None,
            'timezone_info': str(ts_series.dt.tz) if ts_series.dt.tz else 'None',
            'future_timestamp_count': len(future_timestamps)
        }

        # Store results
        self.validation_results.append(results)

        if self.strict_mode and not results['valid']:
            raise ValidationError(f"Timestamp validation failed: {results['errors']}")

        return results

    def validate_ohlcv_data(
        self,
        df: pd.DataFrame,
        price_columns: list[str] = None,
        volume_column: str = 'volume'
    ) -> dict[str, Any]:
        """
        Validate OHLCV data integrity.

        Args:
            df: DataFrame to validate
            price_columns: List of price column names
            volume_column: Name of volume column

        Returns:
            Validation results dictionary
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }

        # Check price columns exist
        missing_price_cols = set(price_columns) - set(df.columns)
        if missing_price_cols:
            error_msg = f"Missing price columns: {missing_price_cols}"
            results['errors'].append(error_msg)
            results['valid'] = False

        # Check volume column exists
        if volume_column not in df.columns:
            error_msg = f"Volume column '{volume_column}' not found"
            results['errors'].append(error_msg)
            results['valid'] = False

        if not results['valid']:
            if self.strict_mode:
                raise ValidationError(f"OHLCV validation failed: {results['errors']}")
            return results

        # Initialize default price columns if not provided
        if price_columns is None:
            price_columns = ['open', 'high', 'low', 'close']

        # Validate OHLC relationships
        if all(col in df.columns for col in price_columns):
            # High should be >= Low
            invalid_high_low = df[df['high'] < df['low']]
            if len(invalid_high_low) > 0:
                error_msg = f"Found {len(invalid_high_low)} rows where high < low"
                results['errors'].append(error_msg)
                results['valid'] = False

            # High should be >= Open and Close
            invalid_high = df[
                (df['high'] < df['open']) | (df['high'] < df['close'])
            ]
            if len(invalid_high) > 0:
                error_msg = f"Found {len(invalid_high)} rows where high < open or close"
                results['errors'].append(error_msg)
                results['valid'] = False

            # Low should be <= Open and Close
            invalid_low = df[
                (df['low'] > df['open']) | (df['low'] > df['close'])
            ]
            if len(invalid_low) > 0:
                error_msg = f"Found {len(invalid_low)} rows where low > open or close"
                results['errors'].append(error_msg)
                results['valid'] = False

        # Validate volume
        if volume_column in df.columns:
            negative_volume = df[df[volume_column] < 0]
            if len(negative_volume) > 0:
                error_msg = f"Found {len(negative_volume)} rows with negative volume"
                results['errors'].append(error_msg)
                results['valid'] = False

        # Add info
        results['info'] = {
            'price_range': {
                'min': df[price_columns].min().to_dict() if price_columns else {},
                'max': df[price_columns].max().to_dict() if price_columns else {}
            },
            'volume_stats': {
                'min': df[volume_column].min() if volume_column in df.columns else None,
                'max': df[volume_column].max() if volume_column in df.columns else None,
                'mean': df[volume_column].mean() if volume_column in df.columns else None
            } if volume_column in df.columns else {}
        }

        # Store results
        self.validation_results.append(results)

        if self.strict_mode and not results['valid']:
            raise ValidationError(f"OHLCV validation failed: {results['errors']}")

        return results

    def get_validation_summary(self) -> dict[str, Any]:
        """Get summary of all validation results."""
        if not self.validation_results:
            return {'message': 'No validations performed'}

        total_validations = len(self.validation_results)
        successful_validations = sum(1 for r in self.validation_results if r['valid'])
        failed_validations = total_validations - successful_validations

        all_errors = []
        all_warnings = []

        for result in self.validation_results:
            all_errors.extend(result.get('errors', []))
            all_warnings.extend(result.get('warnings', []))

        return {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'failed_validations': failed_validations,
            'total_errors': len(all_errors),
            'total_warnings': len(all_warnings),
            'success_rate': successful_validations / total_validations if total_validations > 0 else 0,
            'errors': all_errors,
            'warnings': all_warnings
        }


# Convenience functions for common validations
def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    **kwargs
) -> dict[str, Any]:
    """Validate DataFrame structure and content."""
    validator = DataValidator()
    return validator.validate_dataframe(df, required_columns, **kwargs)


def validate_timestamps(
    df: pd.DataFrame,
    timestamp_column: str = 'ts',
    **kwargs
) -> dict[str, Any]:
    """Validate timestamp data in DataFrame."""
    validator = DataValidator()
    return validator.validate_timestamps(df, timestamp_column, **kwargs)


def validate_ohlcv_data(
    df: pd.DataFrame,
    **kwargs
) -> dict[str, Any]:
    """Validate OHLCV data integrity."""
    validator = DataValidator()
    return validator.validate_ohlcv_data(df, **kwargs)
