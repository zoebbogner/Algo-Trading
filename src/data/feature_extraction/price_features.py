"""Price and Return Feature Computation Module."""

import numpy as np
import pandas as pd
from typing import List

from ...utils.logging import get_logger

logger = get_logger(__name__)


class PriceFeatureComputer:
    """Computes price and return-related features."""
    
    def __init__(self, return_periods: List[int] = None):
        """Initialize price feature computer."""
        self.return_periods = return_periods or [1, 5, 15]
        logger.debug(f"PriceFeatureComputer initialized with periods: {self.return_periods}")
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all price and return features."""
        logger.debug("Computing price and return features")
        
        df = df.copy()
        df = self._add_basic_returns(df)
        df = self._add_intra_bar_features(df)
        df = self._add_log_returns(df)
        
        return df
    
    def _add_basic_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic return features."""
        # 1-minute returns
        df['ret_1m'] = df['close'].pct_change()
        
        # Multi-period returns
        for period in self.return_periods[1:]:  # Skip 1m as it's already added
            df[f'ret_{period}m'] = df['close'].pct_change(periods=period)
        
        return df
    
    def _add_intra_bar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add intra-bar price features."""
        df['close_to_open'] = (df['close'] - df['open']) / df['open']
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['oc_abs'] = abs(df['open'] - df['close']) / df['close']
        
        return df
    
    def _add_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add log returns for better statistical properties."""
        df['log_ret_1m'] = np.log(df['close'] / df['close'].shift(1))
        return df
