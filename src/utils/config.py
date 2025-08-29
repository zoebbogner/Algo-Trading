"""Consolidated configuration loader for the crypto historical data collection system."""

import hashlib
import json
from pathlib import Path
from typing import Any

from src.config import base, data_history, features


def load_merged_config() -> dict[str, Any]:
    """Load and merge all configurations from the root configs directory."""
    # Load all configs
    base_config = base.load_config()
    data_config = data_history.load_config()
    features_config = features.load_config()

    # Merge configs (base config takes precedence)
    merged_config = {
        **base_config,
        "data": data_config,
        "features": features_config
    }

    return merged_config


def get_config_hash(config: dict[str, Any]) -> str:
    """Generate a hash of the configuration for tracking changes."""
    # Sort keys to ensure consistent hashing
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def load_and_log_config() -> dict[str, Any]:
    """Load configuration and log the config hash for tracking."""
    from src.utils.logging import get_logger

    logger = get_logger(__name__)

    # Load merged config
    config = load_merged_config()

    # Generate and log config hash
    config_hash = get_config_hash(config)
    logger.info(f"Configuration loaded with hash: {config_hash}")

    # Log key configuration details
    logger.info(f"Data root: {config.get('data_root', 'N/A')}")
    logger.info(f"Symbols: {config.get('data', {}).get('symbols', [])}")
    logger.info(f"Feature families: {config.get('features', {}).get('feature_families', [])}")

    return config


def get_config_path() -> Path:
    """Get the path to the root configs directory."""
    return Path(__file__).parent.parent.parent / "configs"


def validate_config_files() -> bool:
    """Validate that all required config files exist."""
    config_path = get_config_path()
    required_files = [
        "base.yaml",
        "data.history.yaml",
        "features.yaml",
        "costs.sim.yaml"
    ]

    missing_files = []
    for file_name in required_files:
        if not (config_path / file_name).exists():
            missing_files.append(file_name)

    if missing_files:
        print(f"❌ Missing required config files: {missing_files}")
        return False

    print("✅ All required config files found")
    return True
