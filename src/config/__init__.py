"""Configuration module for the crypto historical data collection system."""

from .base import load_config as load_base_config
from .data_history import load_config as load_data_config

__all__ = ["load_base_config", "load_data_config"]
