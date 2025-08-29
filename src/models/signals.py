"""
Trading signal models for the Algo-Trading system.

Provides structured data models for:
- Trading signals and decisions
- Signal metadata and confidence
- Signal validation and quality metrics
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class SignalType(Enum):
    """Types of trading signals."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    SCALP = "SCALP"
    SWING = "SWING"
    POSITION_SIZE = "POSITION_SIZE"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    EXTREME = "EXTREME"


class SignalSource(Enum):
    """Sources of trading signals."""
    TECHNICAL_INDICATOR = "TECHNICAL_INDICATOR"
    PATTERN_RECOGNITION = "PATTERN_RECOGNITION"
    MACHINE_LEARNING = "MACHINE_LEARNING"
    FUNDAMENTAL = "FUNDAMENTAL"
    SENTIMENT = "SENTIMENT"
    REGIME_DETECTION = "REGIME_DETECTION"
    RISK_MANAGEMENT = "RISK_MANAGEMENT"
    PORTFOLIO_OPTIMIZATION = "PORTFOLIO_OPTIMIZATION"


@dataclass
class SignalMetadata:
    """Metadata for trading signals."""

    signal_id: str
    signal_type: SignalType
    signal_strength: SignalStrength
    signal_source: SignalSource
    symbol: str
    timestamp: datetime
    confidence_score: float  # 0.0 to 1.0
    description: str = ""
    reasoning: str = ""
    feature_values: dict[str, float] = None
    model_version: str = ""
    strategy_name: str = ""
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.feature_values is None:
            self.feature_values = {}

        # Validate confidence score
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, got {self.confidence_score}")


@dataclass
class TradingSignal:
    """Complete trading signal with all components."""

    metadata: SignalMetadata
    entry_price: float | None = None
    exit_price: float | None = None
    position_size: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_reward_ratio: float | None = None
    expected_return: float | None = None
    max_drawdown: float | None = None
    holding_period: str | None = None

    def calculate_risk_metrics(self) -> dict[str, float]:
        """Calculate risk metrics for the signal."""
        metrics = {}

        if self.entry_price and self.stop_loss:
            metrics["risk_per_share"] = abs(self.entry_price - self.stop_loss)
            metrics["risk_pct"] = abs(self.entry_price - self.stop_loss) / self.entry_price

        if self.entry_price and self.take_profit:
            metrics["reward_per_share"] = abs(self.take_profit - self.entry_price)
            metrics["reward_pct"] = abs(self.take_profit - self.entry_price) / self.entry_price

        if "risk_per_share" in metrics and "reward_per_share" in metrics:
            if metrics["risk_per_share"] > 0:
                metrics["risk_reward_ratio"] = metrics["reward_per_share"] / metrics["risk_per_share"]

        return metrics

    def validate_signal(self) -> dict[str, Any]:
        """Validate the trading signal for consistency."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # Check required fields based on signal type
        if self.metadata.signal_type in [SignalType.BUY, SignalType.SELL]:
            if self.entry_price is None:
                validation_result["errors"].append("Entry price required for BUY/SELL signals")
                validation_result["valid"] = False

            if self.position_size is None:
                validation_result["warnings"].append("Position size recommended for BUY/SELL signals")

        # Check stop loss and take profit
        if self.stop_loss and self.take_profit:
            if self.metadata.signal_type == SignalType.BUY:
                if self.stop_loss >= self.entry_price:
                    validation_result["errors"].append("Stop loss must be below entry price for BUY signals")
                    validation_result["valid"] = False
                if self.take_profit <= self.entry_price:
                    validation_result["errors"].append("Take profit must be above entry price for BUY signals")
                    validation_result["valid"] = False

            elif self.metadata.signal_type == SignalType.SELL:
                if self.stop_loss <= self.entry_price:
                    validation_result["errors"].append("Stop loss must be above entry price for SELL signals")
                    validation_result["valid"] = False
                if self.take_profit >= self.entry_price:
                    validation_result["errors"].append("Take profit must be below entry price for SELL signals")
                    validation_result["valid"] = False

        # Check confidence score
        if self.metadata.confidence_score < 0.5:
            validation_result["warnings"].append("Low confidence signal (confidence < 0.5)")

        return validation_result


@dataclass
class SignalSet:
    """Collection of trading signals with analysis capabilities."""

    signals: list[TradingSignal]
    start_date: datetime
    end_date: datetime
    symbols: list[str]

    def get_signals_by_type(self, signal_type: SignalType) -> list[TradingSignal]:
        """Filter signals by type."""
        return [s for s in self.signals if s.metadata.signal_type == signal_type]

    def get_signals_by_symbol(self, symbol: str) -> list[TradingSignal]:
        """Filter signals by symbol."""
        return [s for s in self.signals if s.metadata.symbol == symbol]

    def get_signals_by_strength(self, strength: SignalStrength) -> list[TradingSignal]:
        """Filter signals by strength."""
        return [s for s in self.signals if s.metadata.signal_strength == strength]

    def get_signals_by_source(self, source: SignalSource) -> list[TradingSignal]:
        """Filter signals by source."""
        return [s for s in self.signals if s.metadata.signal_source == source]

    def get_high_confidence_signals(self, threshold: float = 0.7) -> list[TradingSignal]:
        """Get signals above confidence threshold."""
        return [s for s in self.signals if s.metadata.confidence_score >= threshold]

    def get_signal_summary(self) -> dict[str, Any]:
        """Get summary statistics for all signals."""
        if not self.signals:
            return {"total_signals": 0}

        summary = {
            "total_signals": len(self.signals),
            "signals_by_type": {},
            "signals_by_strength": {},
            "signals_by_source": {},
            "confidence_stats": {
                "mean": np.mean([s.metadata.confidence_score for s in self.signals]),
                "std": np.std([s.metadata.confidence_score for s in self.signals]),
                "min": np.min([s.metadata.confidence_score for s in self.signals]),
                "max": np.max([s.metadata.confidence_score for s in self.signals])
            },
            "symbols": list({s.metadata.symbol for s in self.signals}),
            "date_range": {
                "start": min(s.metadata.timestamp for s in self.signals),
                "end": max(s.metadata.timestamp for s in self.signals)
            }
        }

        # Count by type
        for signal_type in SignalType:
            summary["signals_by_type"][signal_type.value] = len(
                self.get_signals_by_type(signal_type)
            )

        # Count by strength
        for strength in SignalStrength:
            summary["signals_by_strength"][strength.value] = len(
                self.get_signals_by_strength(strength)
            )

        # Count by source
        for source in SignalSource:
            summary["signals_by_source"][source.value] = len(
                self.get_signals_by_source(source)
            )

        return summary

    def to_dataframe(self) -> pd.DataFrame:
        """Convert signals to DataFrame for analysis."""
        if not self.signals:
            return pd.DataFrame()

        data = []
        for signal in self.signals:
            row = {
                "signal_id": signal.metadata.signal_id,
                "signal_type": signal.metadata.signal_type.value,
                "signal_strength": signal.metadata.signal_strength.value,
                "signal_source": signal.metadata.signal_source.value,
                "symbol": signal.metadata.symbol,
                "timestamp": signal.metadata.timestamp,
                "confidence_score": signal.metadata.confidence_score,
                "description": signal.metadata.description,
                "strategy_name": signal.metadata.strategy_name,
                "entry_price": signal.entry_price,
                "exit_price": signal.exit_price,
                "position_size": signal.position_size,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "risk_reward_ratio": signal.risk_reward_ratio,
                "expected_return": signal.expected_return,
                "max_drawdown": signal.max_drawdown,
                "holding_period": signal.holding_period,
                "created_at": signal.metadata.created_at
            }
            data.append(row)

        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

        return df


def create_signal_id(symbol: str, timestamp: datetime, signal_type: SignalType) -> str:
    """Create a unique signal ID."""
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{symbol}_{signal_type.value}_{timestamp_str}"


def calculate_signal_confidence(
    feature_values: dict[str, float],
    weights: dict[str, float],
    base_confidence: float = 0.5
) -> float:
    """Calculate signal confidence based on feature values and weights."""
    if not feature_values or not weights:
        return base_confidence

    weighted_sum = 0.0
    total_weight = 0.0

    for feature, weight in weights.items():
        if feature in feature_values:
            weighted_sum += feature_values[feature] * weight
            total_weight += weight

    if total_weight == 0:
        return base_confidence

    # Normalize to 0-1 range and combine with base confidence
    feature_confidence = weighted_sum / total_weight
    combined_confidence = (base_confidence + feature_confidence) / 2

    # Ensure bounds
    return max(0.0, min(1.0, combined_confidence))
