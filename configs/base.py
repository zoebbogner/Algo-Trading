"""Base configuration for the crypto historical data collection system."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """Load base configuration from YAML file."""
    config_file = Path(__file__).parent / "base.yaml"
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    # Return default configuration if file doesn't exist
    return {
        "data_root": "data",
        "reports_root": "reports", 
        "logs_root": "logs",
        "tz": "UTC",
        "processing": {
            "batch_size": 10000,
            "max_workers": 4,
            "cache_ttl_hours": 24
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_rotation": "1 day",
            "max_log_files": 30
        },
        "qc": {
            "min_coverage_pct": 99.9,
            "max_price_deviation": 0.5,
            "min_volume_threshold": 0.0,
            "max_gap_minutes": 60
        }
    }
