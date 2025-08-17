"""
Centralized Strategy Configuration

This module contains all configuration parameters, constants, and settings
for the algorithmic trading system, organized by feature and strategy type.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass
class MLConfig:
    """Machine Learning configuration parameters"""

    # Model parameters
    confidence_threshold: float = 0.7
    lookback_periods: list[int] = field(default_factory=lambda: [5, 10, 20])

    # Random Forest parameters
    random_forest: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
        }
    )

    # Gradient Boosting parameters
    gradient_boosting: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
        }
    )

    # Training parameters
    min_training_samples: int = 100
    validation_split: float = 0.2
    model_save_path: str = "models/signal_confirmation"


@dataclass
class AdvancedRiskConfig:
    """Advanced Risk Modeling configuration parameters"""

    # VaR parameters
    var_confidence_level: float = 0.95
    var_time_horizon: int = 1  # days
    var_methods: list[str] = field(
        default_factory=lambda: ["historical", "parametric", "monte_carlo"]
    )

    # Kelly Criterion parameters
    kelly_max_fraction: float = 0.25  # Max 25% of portfolio
    kelly_risk_free_rate: float = 0.02  # 2% annual

    # Stress testing parameters
    stress_scenarios: dict[str, float] = field(
        default_factory=lambda: {
            "market_crash": -0.20,  # 20% market decline
            "volatility_spike": 0.50,  # 50% volatility increase
            "correlation_breakdown": 0.8,  # 80% correlation increase
            "liquidity_crisis": -0.15,  # 15% liquidity reduction
        }
    )

    # Risk thresholds
    max_portfolio_var: float = 0.05  # 5% max portfolio VaR
    max_correlation_exposure: float = 0.3  # 30% max crypto exposure


@dataclass
class PortfolioOptimizationConfig:
    """Portfolio Optimization configuration parameters"""

    # Optimization parameters
    optimization_method: str = "efficient_frontier"
    risk_free_rate: float = 0.02
    max_iterations: int = 1000
    constraint_tolerance: float = 1e-6

    # Portfolio constraints
    min_weight: float = 0.01  # 1% minimum position
    max_weight: float = 0.40  # 40% maximum position
    target_volatility: float = None
    target_return: float = None

    # Rebalancing parameters
    rebalancing_threshold: float = 0.05  # 5% threshold
    rebalancing_frequency: str = "monthly"

    # Available methods
    available_methods: list[str] = field(
        default_factory=lambda: [
            "efficient_frontier",
            "risk_parity",
            "maximum_sharpe",
            "minimum_variance",
            "black_litterman",
        ]
    )


@dataclass
class TechnicalIndicatorsConfig:
    """Technical Indicators configuration parameters"""

    # Moving Average periods
    fast_ma_period: int = 8
    slow_ma_period: int = 21
    trend_ma_period: int = 50

    # Multi-timeframe periods
    short_ma_period: int = 5
    medium_ma_period: int = 13
    long_ma_period: int = 34

    # RSI parameters
    rsi_period: int = 14
    rsi_oversold: float = 35
    rsi_overbought: float = 65

    # ADX parameters
    adx_period: int = 14
    adx_threshold: int = 25

    # ATR parameters
    atr_period: int = 14

    # Volume parameters
    volume_multiplier: float = 1.2
    volume_lookback: int = 20


@dataclass
class RiskManagementConfig:
    """Risk Management configuration parameters"""

    # Position sizing
    position_size: float = 0.08
    min_position_size: float = 0.02  # 2% minimum
    max_position_size: float = 0.20  # 20% maximum

    # Stop loss and take profit
    stop_loss_pct: float = 0.015  # 1.5%
    take_profit_pct: float = 0.03  # 3%
    trailing_stop_pct: float = 0.01  # 1%

    # Portfolio limits
    max_portfolio_exposure: float = 0.80  # 80% max total exposure
    max_position_concentration: float = 0.40  # 40% max single position

    # Volatility thresholds
    volatility_threshold: float = 0.02
    max_volatility_multiplier: float = 1.5

    # Correlation limits
    correlation_threshold: float = 0.7
    max_correlation_exposure: float = 0.3


@dataclass
class EntryExitConfig:
    """Entry and Exit configuration parameters"""

    # Entry scoring
    base_entry_threshold: int = 60
    min_entry_threshold: int = 40
    max_entry_threshold: int = 80

    # Entry condition weights
    multi_timeframe_weight: int = 40
    rsi_weight: int = 20
    volume_weight: int = 15
    trend_strength_weight: int = 15
    volatility_weight: int = 10
    ml_confirmation_bonus: int = 10
    ml_rejection_penalty: int = 20

    # Market regime adjustments
    strong_uptrend_threshold_adjustment: int = -10
    strong_downtrend_threshold_adjustment: int = 15
    weak_trend_threshold_adjustment: int = -5

    # Volatility adjustments
    high_volatility_threshold_adjustment: int = -5
    low_volatility_threshold_adjustment: int = 5

    # RSI extreme adjustments
    rsi_extreme_threshold_adjustment: int = -5


@dataclass
class DataConfig:
    """Data and Market configuration parameters"""

    # Data requirements
    min_bars_required: int = 50
    min_training_samples: int = 100
    max_historical_data: int = 1000
    max_returns_history: int = 500

    # Timeframes
    default_timeframe: str = "1h"
    available_timeframes: list[str] = field(
        default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"]
    )

    # Symbols
    default_symbols: list[str] = field(
        default_factory=lambda: ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
    )

    # Data quality
    min_data_quality_score: float = 0.8
    max_missing_data_pct: float = 0.1


@dataclass
class BacktestingConfig:
    """Backtesting configuration parameters"""

    # Capital and portfolio
    initial_capital: Decimal = Decimal("10000")
    min_capital: Decimal = Decimal("1000")

    # Performance thresholds
    min_acceptable_return: float = 0.02  # 2%
    max_acceptable_drawdown: float = 0.15  # 15%
    min_acceptable_sharpe: float = 0.5

    # Reporting
    generate_detailed_reports: bool = True
    save_trade_history: bool = True
    performance_analysis: bool = True

    # Validation
    out_of_sample_testing: bool = True
    walk_forward_analysis: bool = True
    monte_carlo_simulation: bool = True


@dataclass
class UltraEnhancedStrategyConfig:
    """Complete configuration for Ultra-Enhanced Momentum Strategy"""

    # Strategy identity
    name: str = "Ultra-Enhanced Momentum Strategy"
    version: str = "3.0"
    description: str = (
        "Advanced momentum strategy with ML confirmation and risk management"
    )

    # Feature flags
    use_ml_confirmation: bool = True
    use_advanced_risk: bool = True
    use_portfolio_optimization: bool = True
    use_adaptive_thresholds: bool = True
    use_portfolio_heat_mapping: bool = True

    # Sub-configurations
    ml: MLConfig = field(default_factory=MLConfig)
    advanced_risk: AdvancedRiskConfig = field(default_factory=AdvancedRiskConfig)
    portfolio_optimization: PortfolioOptimizationConfig = field(
        default_factory=PortfolioOptimizationConfig
    )
    technical_indicators: TechnicalIndicatorsConfig = field(
        default_factory=TechnicalIndicatorsConfig
    )
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    entry_exit: EntryExitConfig = field(default_factory=EntryExitConfig)
    data: DataConfig = field(default_factory=DataConfig)
    backtesting: BacktestingConfig = field(default_factory=BacktestingConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary format"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "use_ml_confirmation": self.use_ml_confirmation,
            "use_advanced_risk": self.use_advanced_risk,
            "use_portfolio_optimization": self.use_portfolio_optimization,
            "use_adaptive_thresholds": self.use_adaptive_thresholds,
            "use_portfolio_heat_mapping": self.use_portfolio_heat_mapping,
            "ml_config": self.ml.__dict__,
            "risk_config": self.advanced_risk.__dict__,
            "optimization_config": self.portfolio_optimization.__dict__,
            "fast_ma_period": self.technical_indicators.fast_ma_period,
            "slow_ma_period": self.technical_indicators.slow_ma_period,
            "trend_ma_period": self.technical_indicators.trend_ma_period,
            "rsi_period": self.technical_indicators.rsi_period,
            "rsi_oversold": self.technical_indicators.rsi_oversold,
            "rsi_overbought": self.technical_indicators.rsi_overbought,
            "adx_period": self.technical_indicators.adx_period,
            "adx_threshold": self.technical_indicators.adx_threshold,
            "volume_multiplier": self.technical_indicators.volume_multiplier,
            "position_size": self.risk_management.position_size,
            "stop_loss_pct": self.risk_management.stop_loss_pct,
            "take_profit_pct": self.risk_management.take_profit_pct,
            "trailing_stop_pct": self.risk_management.trailing_stop_pct,
            "short_ma_period": self.technical_indicators.short_ma_period,
            "medium_ma_period": self.technical_indicators.medium_ma_period,
            "long_ma_period": self.technical_indicators.long_ma_period,
            "atr_period": self.technical_indicators.atr_period,
            "volatility_threshold": self.risk_management.volatility_threshold,
            "correlation_threshold": self.advanced_risk.correlation_threshold,
            "max_correlation_exposure": self.advanced_risk.max_correlation_exposure,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "UltraEnhancedStrategyConfig":
        """Create configuration from dictionary"""
        # Extract sub-configurations
        ml_config = MLConfig(**config_dict.get("ml_config", {}))
        risk_config = AdvancedRiskConfig(**config_dict.get("risk_config", {}))
        opt_config = PortfolioOptimizationConfig(
            **config_dict.get("optimization_config", {})
        )

        # Create technical indicators config
        tech_config = TechnicalIndicatorsConfig(
            fast_ma_period=config_dict.get("fast_ma_period", 8),
            slow_ma_period=config_dict.get("slow_ma_period", 21),
            trend_ma_period=config_dict.get("trend_ma_period", 50),
            rsi_period=config_dict.get("rsi_period", 14),
            rsi_oversold=config_dict.get("rsi_oversold", 35),
            rsi_overbought=config_dict.get("rsi_overbought", 65),
            adx_period=config_dict.get("adx_period", 14),
            adx_threshold=config_dict.get("adx_threshold", 25),
            volume_multiplier=config_dict.get("volume_multiplier", 1.2),
            short_ma_period=config_dict.get("short_ma_period", 5),
            medium_ma_period=config_dict.get("medium_ma_period", 13),
            long_ma_period=config_dict.get("long_ma_period", 34),
            atr_period=config_dict.get("atr_period", 14),
        )

        # Create risk management config
        risk_mgmt_config = RiskManagementConfig(
            position_size=config_dict.get("position_size", 0.08),
            stop_loss_pct=config_dict.get("stop_loss_pct", 0.015),
            take_profit_pct=config_dict.get("take_profit_pct", 0.03),
            trailing_stop_pct=config_dict.get("trailing_stop_pct", 0.01),
            volatility_threshold=config_dict.get("volatility_threshold", 0.02),
            correlation_threshold=config_dict.get("correlation_threshold", 0.7),
            max_correlation_exposure=config_dict.get("max_correlation_exposure", 0.3),
        )

        return cls(
            name=config_dict.get("name", "Ultra-Enhanced Momentum Strategy"),
            version=config_dict.get("version", "3.0"),
            use_ml_confirmation=config_dict.get("use_ml_confirmation", True),
            use_advanced_risk=config_dict.get("use_advanced_risk", True),
            use_portfolio_optimization=config_dict.get(
                "use_portfolio_optimization", True
            ),
            ml=ml_config,
            advanced_risk=risk_config,
            portfolio_optimization=opt_config,
            technical_indicators=tech_config,
            risk_management=risk_mgmt_config,
        )


# Default configuration instance
DEFAULT_ULTRA_ENHANCED_CONFIG = UltraEnhancedStrategyConfig()

# Configuration presets for different market conditions
CONFIG_PRESETS = {
    "conservative": UltraEnhancedStrategyConfig(
        name="Conservative Ultra-Enhanced Strategy",
        risk_management=RiskManagementConfig(
            position_size=0.05,
            stop_loss_pct=0.01,
            take_profit_pct=0.02,
            max_portfolio_exposure=0.60,
        ),
        advanced_risk=AdvancedRiskConfig(
            var_confidence_level=0.99, kelly_max_fraction=0.15
        ),
    ),
    "aggressive": UltraEnhancedStrategyConfig(
        name="Aggressive Ultra-Enhanced Strategy",
        risk_management=RiskManagementConfig(
            position_size=0.12,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            max_portfolio_exposure=0.90,
        ),
        advanced_risk=AdvancedRiskConfig(
            var_confidence_level=0.90, kelly_max_fraction=0.35
        ),
    ),
    "balanced": UltraEnhancedStrategyConfig(
        name="Balanced Ultra-Enhanced Strategy",
        risk_management=RiskManagementConfig(
            position_size=0.08,
            stop_loss_pct=0.015,
            take_profit_pct=0.03,
            max_portfolio_exposure=0.80,
        ),
        advanced_risk=AdvancedRiskConfig(
            var_confidence_level=0.95, kelly_max_fraction=0.25
        ),
    ),
}


def get_config_preset(preset_name: str) -> UltraEnhancedStrategyConfig:
    """Get a configuration preset by name"""
    if preset_name not in CONFIG_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(CONFIG_PRESETS.keys())}"
        )

    return CONFIG_PRESETS[preset_name]


def validate_config(config: UltraEnhancedStrategyConfig) -> list[str]:
    """Validate configuration and return list of issues"""
    issues = []

    # Validate position size
    if config.risk_management.position_size < config.risk_management.min_position_size:
        issues.append(
            f"Position size {config.risk_management.position_size} below minimum {config.risk_management.min_position_size}"
        )

    if config.risk_management.position_size > config.risk_management.max_position_size:
        issues.append(
            f"Position size {config.risk_management.position_size} above maximum {config.risk_management.max_position_size}"
        )

    # Validate entry thresholds
    if (
        config.entry_exit.min_entry_threshold < 0
        or config.entry_exit.max_entry_threshold > 100
    ):
        issues.append("Entry thresholds must be between 0 and 100")

    # Validate ML confidence threshold
    if config.ml.confidence_threshold < 0 or config.ml.confidence_threshold > 1:
        issues.append("ML confidence threshold must be between 0 and 1")

    # Validate VaR confidence level
    if (
        config.advanced_risk.var_confidence_level < 0.5
        or config.advanced_risk.var_confidence_level > 0.999
    ):
        issues.append("VaR confidence level must be between 0.5 and 0.999")

    return issues
