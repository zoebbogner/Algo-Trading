"""Structured logging configuration."""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler

from .config import settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class StructuredLogger:
    """Structured logger with file rotation and multiple outputs."""
    
    def __init__(self, name: str = "algo_trading"):
        """Initialize structured logger.
        
        Args:
            name: Logger name
        """
        self.name = name
        self.logger = None
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Create logger
        self.logger = structlog.get_logger(self.name)
        
        # Setup handlers
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup logging handlers."""
        # Console handler with rich formatting
        console_handler = RichHandler(
            console=Console(),
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True
        )
        console_handler.setLevel(getattr(logging, settings.log_level.upper()))
        
        # File handler with JSON formatting
        app_log_path = Path("logs/app.log")
        file_handler = logging.handlers.RotatingFileHandler(
            app_log_path,
            maxBytes=settings.log_retention_days * 1024 * 1024,  # MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(JSONFormatter())
        file_handler.setLevel(logging.INFO)
        
        # JSONL handlers for specific log types
        trades_handler = self._create_jsonl_handler("logs/trades.log.jsonl")
        risk_handler = self._create_jsonl_handler("logs/risk.log.jsonl")
        llm_handler = self._create_jsonl_handler("logs/llm_decisions/decisions.jsonl")
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        # Configure specific loggers
        self._configure_logger("trades", trades_handler)
        self._configure_logger("risk", risk_handler)
        self._configure_logger("llm", llm_handler)
    
    def _create_jsonl_handler(self, log_path: str) -> logging.handlers.RotatingFileHandler:
        """Create a rotating JSONL file handler.
        
        Args:
            log_path: Path to log file
            
        Returns:
            Rotating file handler
        """
        # Create directory if needed
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=5,
            encoding="utf-8"
        )
        handler.setFormatter(JSONFormatter())
        handler.setLevel(logging.INFO)
        return handler
    
    def _configure_logger(self, name: str, handler: logging.Handler) -> None:
        """Configure a specific logger.
        
        Args:
            name: Logger name
            handler: Logging handler
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False  # Prevent duplicate logs
    
    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """Log trade information.
        
        Args:
            trade_data: Trade data dictionary
        """
        logger = logging.getLogger("trades")
        logger.info("Trade executed", extra={"extra_fields": trade_data})
    
    def log_risk_event(self, risk_data: Dict[str, Any]) -> None:
        """Log risk management event.
        
        Args:
            risk_data: Risk event data dictionary
        """
        logger = logging.getLogger("risk")
        logger.warning("Risk event", extra={"extra_fields": risk_data})
    
    def log_llm_decision(self, decision_data: Dict[str, Any]) -> None:
        """Log LLM decision.
        
        Args:
            decision_data: LLM decision data dictionary
        """
        logger = logging.getLogger("llm")
        logger.info("LLM decision", extra={"extra_fields": decision_data})
    
    def get_logger(self, name: Optional[str] = None) -> Any:
        """Get a logger instance.
        
        Args:
            name: Logger name (optional)
            
        Returns:
            Logger instance
        """
        if name:
            return structlog.get_logger(name)
        return self.logger


# Global logger instance
logger = StructuredLogger()
