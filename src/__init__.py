"""
Crypto Algorithmic Trading System

A professional-grade system for historical data collection, feature engineering,
and algorithmic trading strategy development.
"""

__version__ = "1.0.0"
__author__ = "Zoe Bogner"
__description__ = "Professional crypto algorithmic trading system with data-first approach"

from . import config, data, utils

__all__ = [
    "data",
    "config",
    "utils"
]
