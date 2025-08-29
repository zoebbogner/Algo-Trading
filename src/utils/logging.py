"""
Global logging system for the crypto algorithmic trading system.

Provides centralized logging with:
- Structured logging with context
- Multiple output handlers (file, console, remote)
- Log rotation and retention
- Performance monitoring
- Error tracking
- Run ID tracking for provenance
- Audit logging for LLM-assisted decisions
"""

import json
import logging
import logging.handlers
import sys
import time
import traceback
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

# Global run ID for tracking operations
_RUN_ID = None


def get_run_id() -> str:
    """Get the current run ID for tracking operations."""
    global _RUN_ID
    if _RUN_ID is None:
        _RUN_ID = str(uuid.uuid4())[:8]
    return _RUN_ID


def set_run_id(run_id: str = None) -> str:
    """Set a custom run ID or generate a new one."""
    global _RUN_ID
    if run_id is None:
        _RUN_ID = str(uuid.uuid4())[:8]
    else:
        _RUN_ID = run_id
    return _RUN_ID


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""

    def format(self, record):
        """Format log record with structured data."""
        # Base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'run_id': get_run_id()  # Always include run_id
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        # Add performance metrics if present
        if hasattr(record, 'duration_ms'):
            log_entry['duration_ms'] = record.duration_ms

        return json.dumps(log_entry, default=str)


class AuditLogger:
    """Audit logger for LLM-assisted decisions and critical operations."""

    def __init__(self, audit_file: str = "logs/llm_audit.jsonl"):
        self.audit_file = Path(audit_file)
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)

    def log_llm_decision(self,
                         decision_type: str,
                         input_data: dict,
                         output_data: dict,
                         confidence: float = None,
                         reasoning: str = None,
                         metadata: dict = None) -> None:
        """Log an LLM-assisted decision for audit purposes."""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'run_id': get_run_id(),
            'decision_type': decision_type,
            'input_data': input_data,
            'output_data': output_data,
            'confidence': confidence,
            'reasoning': reasoning,
            'metadata': metadata or {}
        }

        # Write to audit file
        with open(self.audit_file, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')

    def log_trading_decision(self,
                            symbol: str,
                            action: str,
                            quantity: float,
                            price: float,
                            timestamp: datetime,
                            features: dict,
                            model_output: dict,
                            metadata: dict = None) -> None:
        """Log a trading decision for audit purposes."""
        audit_entry = {
            'timestamp': timestamp.isoformat(),
            'run_id': get_run_id(),
            'decision_type': 'trading_decision',
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'features': features,
            'model_output': model_output,
            'metadata': metadata or {}
        }

        # Write to audit file
        with open(self.audit_file, 'a') as f:
            f.write(json.dumps(audit_entry) + '\n')


class PerformanceLogger:
    """Decorator for logging function performance."""

    def __init__(self, logger_name: str = None):
        self.logger_name = logger_name

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(self.logger_name or func.__module__)
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                logger.info(
                    f"Function {func.__name__} completed successfully",
                    extra={
                        'function_name': func.__name__,
                        'duration_ms': round(duration_ms, 2),
                        'status': 'success'
                    }
                )

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"Function {func.__name__} failed: {str(e)}",
                    extra={
                        'function_name': func.__name__,
                        'duration_ms': round(duration_ms, 2),
                        'status': 'error',
                        'error': str(e)
                    },
                    exc_info=True
                )
                raise

        return wrapper


class ContextLogger:
    """Logger that maintains context across operations."""

    def __init__(self, logger_name: str, context: dict[str, Any] = None):
        self.logger = get_logger(logger_name)
        self.context = context or {}

    def add_context(self, **kwargs):
        """Add context to the logger."""
        self.context.update(kwargs)

    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log message with context."""
        extra_fields = {**self.context, **kwargs}

        if level.upper() == 'DEBUG':
            self.logger.debug(message, extra={'extra_fields': extra_fields})
        elif level.upper() == 'INFO':
            self.logger.info(message, extra={'extra_fields': extra_fields})
        elif level.upper() == 'WARNING':
            self.logger.warning(message, extra={'extra_fields': extra_fields})
        elif level.upper() == 'ERROR':
            self.logger.error(message, extra={'extra_fields': extra_fields})
        elif level.upper() == 'CRITICAL':
            self.logger.critical(message, extra={'extra_fields': extra_fields})

    def debug(self, message: str, **kwargs):
        self._log_with_context('DEBUG', message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log_with_context('INFO', message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log_with_context('WARNING', message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log_with_context('ERROR', message, **kwargs)

    def critical(self, message: str, **kwargs):
        self._log_with_context('CRITICAL', message, **kwargs)

    def log_operation_start(self, operation: str, **kwargs):
        """Log the start of an operation."""
        self.info(f"Starting operation: {operation}", operation=operation, **kwargs)

    def log_operation_complete(self, operation: str, duration_ms: float, **kwargs):
        """Log the completion of an operation."""
        self.info(
            f"Operation completed: {operation}",
            operation=operation,
            duration_ms=duration_ms,
            **kwargs
        )

    def log_operation_failed(self, operation: str, error: str, duration_ms: float, **kwargs):
        """Log the failure of an operation."""
        self.error(
            f"Operation failed: {operation} - {error}",
            operation=operation,
            error=error,
            duration_ms=duration_ms,
            **kwargs
        )


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    max_log_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
    file_output: bool = True
) -> None:
    """
    Set up the global logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        max_log_size: Maximum size of log files before rotation
        backup_count: Number of backup log files to keep
        console_output: Whether to output to console
        file_output: Whether to output to files
    """
    # Create log directory FIRST before any logging setup
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatters
    detailed_formatter = StructuredFormatter()
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)

    # File handlers
    if file_output:
        # Main application log
        app_log_file = log_path / "app.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=max_log_size,
            backupCount=backup_count
        )
        app_handler.setLevel(logging.DEBUG)
        app_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(app_handler)

        # Error log (only errors and critical)
        error_log_file = log_path / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_log_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)

        # Performance log
        perf_log_file = log_path / "performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=max_log_size,
            backupCount=backup_count
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(perf_handler)

    # Set specific logger levels
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def get_audit_logger() -> AuditLogger:
    """Get the audit logger instance."""
    return AuditLogger()


def get_context_logger(name: str, context: dict[str, Any] = None) -> ContextLogger:
    """
    Get a context logger instance.

    Args:
        name: Logger name
        context: Initial context dictionary

    Returns:
        Context logger instance
    """
    return ContextLogger(name, context)


# Performance monitoring decorator
def log_performance(logger_name: str = None):
    """
    Decorator to log function performance.

    Args:
        logger_name: Optional logger name

    Returns:
        Decorated function
    """
    return PerformanceLogger(logger_name)


# Initialize logging when module is imported
# Ensure logs directory exists before setting up logging
if not logging.getLogger().handlers:
    setup_logging()
