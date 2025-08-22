"""Data history configuration for the crypto historical data collection system."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """Load data history configuration from YAML file."""
    config_file = Path(__file__).parent / "data.history.yaml"
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    # Return default configuration if file doesn't exist
    return {
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "timeframe": "1m",
        "data_format": "csv",
        "raw_dir": "data/raw/binance",
        "processed_dir": "data/processed/binance",
        "features_dir": "data/features",
        "history": {
            "start_ts_utc": "2024-07-01T00:00:00Z",
            "end_ts_utc": "2025-08-20T23:59:00Z"
        },
        "holdout": {
            "start_ts_utc": "2025-07-01T00:00:00Z",
            "end_ts_utc": "2025-08-20T23:59:00Z"
        },
        "sources": {
            "primary": "binance:dump",
            "secondary": "binance:rest"
        },
        "collection": {
            "csv_download_parallel": 3,
            "rest_batch_size": 1000,
            "rest_rate_limit_ms": 100,
            "gap_fill_threshold_hours": 24
        },
        "quality": {
            "min_coverage_pct": 99.9,
            "max_duplicates": 0,
            "require_monotonic_ts": True,
            "validate_price_ranges": True,
            "validate_volume_ranges": True
        }
    }
