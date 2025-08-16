"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Trading mode
    mode: str = Field(default="SIM", description="Trading mode: SIM, PAPER, or LIVE")
    
    # Broker configuration
    broker_key_id: Optional[str] = Field(default=None, description="Broker API key ID")
    broker_secret: Optional[str] = Field(default=None, description="Broker API secret")
    broker_base_url: Optional[str] = Field(default=None, description="Broker base URL")
    broker_sandbox_url: Optional[str] = Field(default=None, description="Broker sandbox URL")
    
    # Data configuration
    data_cache_dir: str = Field(default="Algo-Trading/data/cache", description="Data cache directory")
    data_raw_dir: str = Field(default="Algo-Trading/data/raw", description="Raw data directory")
    data_processed_dir: str = Field(default="Algo-Trading/data/processed", description="Processed data directory")
    data_features_dir: str = Field(default="Algo-Trading/data/features", description="Features directory")
    
    # LLM configuration
    llm_instruct_model_path: str = Field(
        default="/Users/zoe/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf",
        description="LLM instruction model path"
    )
    embeddings_model_path: str = Field(
        default="/Applications/gpt4all/bin/gpt4all.app/Contents/Resources/nomic-embed-text-v1.5.f16.gguf",
        description="Embeddings model path"
    )
    llm_max_tokens: int = Field(default=2048, description="LLM max tokens")
    llm_temperature: float = Field(default=0.1, description="LLM temperature")
    
    # Timezone
    tz: str = Field(default="Asia/Jerusalem", description="Timezone")
    
    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format")
    log_retention_days: int = Field(default=14, description="Log retention days")
    
    # Risk management
    max_position_size: float = Field(default=0.1, description="Max position size")
    max_gross_exposure: float = Field(default=0.5, description="Max gross exposure")
    max_daily_drawdown: float = Field(default=0.05, description="Max daily drawdown")
    drawdown_cooldown_minutes: int = Field(default=60, description="Drawdown cooldown minutes")
    per_trade_risk: float = Field(default=0.005, description="Per trade risk")
    
    # Trading configuration
    default_symbols: str = Field(default="BTC/USD,ETH/USD", description="Default trading symbols")
    bar_interval: str = Field(default="1m", description="Bar interval")
    market_order_timeout_seconds: int = Field(default=30, description="Market order timeout")
    limit_order_timeout_seconds: int = Field(default=300, description="Limit order timeout")
    
    # State management
    state_dir: str = Field(default="Algo-Trading/state", description="State directory")
    portfolio_dir: str = Field(default="Algo-Trading/state/portfolios", description="Portfolio directory")
    checkpoint_dir: str = Field(default="Algo-Trading/state/checkpoints", description="Checkpoint directory")
    
    # Reports
    reports_dir: str = Field(default="Algo-Trading/reports", description="Reports directory")
    daily_reports_dir: str = Field(default="Algo-Trading/reports/daily", description="Daily reports directory")
    run_reports_dir: str = Field(default="Algo-Trading/reports/runs", description="Run reports directory")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class ConfigManager:
    """Configuration manager for loading and merging YAML configs."""
    
    def __init__(self, config_dir: str = "configs"):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Any] = {}
        self._loaded = False
    
    def load_configs(self) -> Dict[str, Any]:
        """Load all configuration files.
        
        Returns:
            Merged configuration dictionary
        """
        if self._loaded:
            return self._configs
        
        # Load base config first
        base_config = self._load_yaml("base.yaml")
        if base_config:
            self._configs.update(base_config)
        
        # Load mode-specific configs
        mode = os.getenv("MODE", "SIM").lower()
        mode_configs = [
            f"broker.{mode}.yaml",
            "strategy.default.yaml",
            "risk.default.yaml",
            "llm.policy.yaml"
        ]
        
        for config_file in mode_configs:
            config = self._load_yaml(config_file)
            if config:
                self._merge_config(self._configs, config)
        
        self._loaded = True
        return self._configs
    
    def _load_yaml(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load a YAML configuration file.
        
        Args:
            filename: Configuration file name
            
        Returns:
            Configuration dictionary or None if file not found
        """
        file_path = self.config_dir / filename
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config {filename}: {e}")
            return None
    
    def _merge_config(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Recursively merge configuration dictionaries.
        
        Args:
            base: Base configuration dictionary
            update: Configuration to merge
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if not self._loaded:
            self.load_configs()
        
        keys = key.split('.')
        value = self._configs
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def reload(self) -> None:
        """Reload configuration files."""
        self._loaded = False
        self._configs.clear()
        self.load_configs()


# Global configuration instances
settings = AppSettings()
config_manager = ConfigManager()
