"""
Feature extraction module.

Handles all feature engineering operations including:
- Price and return features
- Trend and momentum indicators
- Mean reversion signals
- Volatility and risk metrics
- Volume and liquidity features
- Cross-asset relationships
- Microstructure proxies
- Regime classification
- Risk and execution helpers
"""

from .engine import FeatureEngineer

__all__ = [
    "FeatureEngineer"
]
