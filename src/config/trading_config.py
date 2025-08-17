"""Trading configuration using dataclasses."""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class DataConfig:
    """Data collection configuration."""
    base_path: str = "data"
    interval: str = "1d"
    collectors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    initial_capital: float = 100000.0
    symbols: List[str] = field(default_factory=lambda: ["BTC", "ETH", "ADA"])
    interval_seconds: int = 3600
    strategies: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size: float = 0.1
    max_portfolio_risk: float = 0.02
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15


@dataclass
class MLConfig:
    """Machine learning configuration."""
    enabled: bool = True
    models: List[str] = field(default_factory=lambda: ["random_forest", "gradient_boosting"])
    training: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportingConfig:
    """Reporting configuration."""
    output_dir: str = "reports"
    generate_reports: bool = True
    report_interval: int = 3600
    include_charts: bool = True
    include_metrics: bool = True


@dataclass
class TradingBotConfig:
    """Main trading bot configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> "TradingBotConfig":
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            # Return default configuration
            return cls()
        
        try:
            with config_file.open('r') as f:
                config_data = yaml.safe_load(f)
            
            # Parse configuration sections
            data_config = DataConfig(**config_data.get("data", {}))
            trading_config = TradingConfig(**config_data.get("trading", {}))
            risk_config = RiskConfig(**config_data.get("risk", {}))
            ml_config = MLConfig(**config_data.get("ml", {}))
            reporting_config = ReportingConfig(**config_data.get("reporting", {}))
            
            return cls(
                data=data_config,
                trading=trading_config,
                risk=risk_config,
                ml=ml_config,
                reporting=reporting_config
            )
            
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            print("Using default configuration")
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "data": {
                "base_path": self.data.base_path,
                "interval": self.data.interval,
                "collectors": self.data.collectors
            },
            "trading": {
                "initial_capital": self.trading.initial_capital,
                "symbols": self.trading.symbols,
                "interval_seconds": self.trading.interval_seconds,
                "strategies": self.trading.strategies
            },
            "risk": {
                "max_position_size": self.risk.max_position_size,
                "max_portfolio_risk": self.risk.max_portfolio_risk,
                "stop_loss_pct": self.risk.stop_loss_pct,
                "take_profit_pct": self.risk.take_profit_pct
            },
            "ml": {
                "enabled": self.ml.enabled,
                "models": self.ml.models,
                "training": self.ml.training
            },
            "reporting": {
                "output_dir": self.reporting.output_dir,
                "generate_reports": self.reporting.generate_reports,
                "report_interval": self.reporting.report_interval,
                "include_charts": self.reporting.include_charts,
                "include_metrics": self.reporting.include_metrics
            }
        }
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with config_file.open('w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)


# Alias for backward compatibility
TradingConfig = TradingBotConfig
