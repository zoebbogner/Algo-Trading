"""
API module for the Algo-Trading system.

Provides REST API endpoints for:
- Data collection and management
- Feature engineering operations
- Backtesting execution and analysis
- System monitoring and management
"""

from .backtest_api import backtest_router
from .data_api import data_router
from .features_api import features_router
from .server import app, create_app
from .system_api import system_router

__all__ = [
    'create_app',
    'app',
    'data_router',
    'features_router',
    'backtest_router',
    'system_router'
]
