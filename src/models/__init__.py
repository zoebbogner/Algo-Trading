"""
Data models for the Algo-Trading system.

Provides structured data models for:
- Market data (OHLCV, ticks, order book)
- Features and indicators
- Trading signals and decisions
- Performance metrics and reports
"""

from .features import (
    FeatureMetadata,
    FeatureQualityMetrics,
    FeatureSet,
    create_feature_metadata,
    validate_feature_dataframe,
)
from .market_data import OHLCV, OrderBook, Tick
from .performance import (
    BacktestResult,
    PerformanceMetrics,
    PortfolioMetrics,
    RiskMetrics,
    TradeRecord,
    calculate_performance_metrics,
    create_backtest_report,
)
from .signals import (
    SignalMetadata,
    SignalSet,
    SignalSource,
    SignalStrength,
    SignalType,
    TradingSignal,
    calculate_signal_confidence,
    create_signal_id,
)

__all__ = [
    # Market data models
    'OHLCV', 'Tick', 'OrderBook',

    # Feature models
    'FeatureMetadata', 'FeatureQualityMetrics', 'FeatureSet',
    'validate_feature_dataframe', 'create_feature_metadata',

    # Signal models
    'SignalType', 'SignalStrength', 'SignalSource',
    'SignalMetadata', 'TradingSignal', 'SignalSet',
    'create_signal_id', 'calculate_signal_confidence',

    # Performance models
    'PerformanceMetrics', 'RiskMetrics', 'PortfolioMetrics',
    'TradeRecord', 'BacktestResult',
    'calculate_performance_metrics', 'create_backtest_report'
]
