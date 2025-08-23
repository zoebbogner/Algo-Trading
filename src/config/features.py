"""Feature configuration module for the crypto feature engineering system."""

from pathlib import Path
from typing import Any

import yaml


def load_config() -> dict[str, Any]:
    """Load feature configuration from YAML file."""
    # Read from root configs directory
    config_file = Path(__file__).parent.parent.parent / "configs" / "features.yaml"

    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f)

    # Return default configuration if file doesn't exist
    return {
        "rolling_windows": {
            "ma": [20, 50, 100],
            "ema": [20, 50],
            "zscore": [20],
            "volatility": [20, 50],
            "regression": [20, 240, 720],
            "volume": [20, 50],
            "rsi": [14],
            "atr": [14]
        },
        "thresholds": {
            "volume_spike_multiplier": 2.0,
            "volatility_regime_percentile": 0.80,
            "breakout_lookback": 20,
            "stop_atr_multiplier": 2.0,
            "intraday_reversal_threshold": 0.01,
            "winsorization_limits": [0.01, 0.99]
        },
        "cross_assets": {
            "driver_symbol": "BTCUSDT",
            "pairs": ["ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "TRXUSDT", "DOTUSDT"]
        },
        "output": {
            "path": "data/features/features_1m.parquet",
            "partition_by": ["symbol", "date"],
            "compression": "snappy"
        },
        "qc": {
            "min_non_na_percentage": 0.95,
            "min_periods_required": True,
            "check_monotonic_timestamps": True,
            "check_unique_symbol_timestamp": True,
            "validate_feature_ranges": True
        },
        "feature_families": [
            "price_returns",
            "trend_momentum",
            "mean_reversion",
            "volatility_risk",
            "volume_liquidity",
            "cross_asset",
            "microstructure",
            "regime",
            "risk_execution"
        ]
    }
