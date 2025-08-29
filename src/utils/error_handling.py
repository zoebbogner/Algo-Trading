#!/usr/bin/env python3
"""
Error handling utilities for the crypto historical data collection system.

This module provides:
- Custom exception hierarchy
- Retry mechanisms with exponential backoff
- Error context enrichment
- Error tracking and monitoring
"""

import sys
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timedelta
from functools import wraps
from typing import Any

from src.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


class SystemError(Exception):
    """Base exception for system errors."""

    def __init__(self, message: str, context: dict[str, Any] = None, original_error: Exception = None):
        super().__init__(message)
        self.context = context or {}
        self.original_error = original_error
        self.timestamp = datetime.now()

    def __str__(self):
        base_msg = super().__str__()
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{base_msg} (Context: {context_str})"
        return base_msg


class DataError(SystemError):
    """Exception for data-related errors."""
    pass


class NetworkError(SystemError):
    """Exception for network-related errors."""
    pass


class ValidationError(SystemError):
    """Exception for validation errors."""
    pass


class ConfigurationError(SystemError):
    """Exception for configuration errors."""
    pass


def handle_exception(
    error_types: type[Exception] | list[type[Exception]] = Exception,
    default_return: Any = None,
    log_error: bool = True,
    reraise: bool = False,
    context: dict[str, Any] = None
):
    """
    Decorator for handling exceptions gracefully.

    Args:
        error_types: Exception types to catch
        default_return: Value to return on error
        log_error: Whether to log the error
        reraise: Whether to re-raise the error
        context: Additional context for error handling

    Returns:
        Decorated function
    """
    if not isinstance(error_types, list):
        error_types = [error_types]

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except tuple(error_types) as e:
                # Add context
                error_context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args': str(args),
                    'kwargs': str(kwargs),
                    'timestamp': datetime.now().isoformat()
                }

                if context:
                    error_context.update(context)

                # Log error if requested
                if log_error:
                    logger.error(
                        f"Exception in {func.__name__}: {str(e)}",
                        extra={'error_context': error_context},
                        exc_info=True
                    )

                # Re-raise if requested
                if reraise:
                    raise

                return default_return

        return wrapper

    return decorator


def retry_on_failure(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_backoff: bool = True,
    error_types: type[Exception] | list[type[Exception]] = Exception,
    on_retry: Callable = None,
    on_final_failure: Callable = None
):
    """
    Decorator for retrying functions on failure.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_backoff: Whether to use exponential backoff
        error_types: Exception types to retry on
        on_retry: Callback function called before each retry
        on_final_failure: Callback function called after all retries fail

    Returns:
        Decorated function
    """
    if not isinstance(error_types, list):
        error_types = [error_types]

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except tuple(error_types) as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        # Calculate delay
                        if exponential_backoff:
                            delay = min(base_delay * (2 ** attempt), max_delay)
                        else:
                            delay = base_delay

                        # Log retry attempt
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )

                        # Call retry callback if provided
                        if on_retry:
                            try:
                                on_retry(attempt + 1, max_attempts, e, delay)
                            except Exception as callback_error:
                                logger.error(f"Error in retry callback: {callback_error}")

                        # Wait before retry
                        time.sleep(delay)

                    else:
                        # Final attempt failed
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}. "
                            f"Final error: {str(e)}"
                        )

                        # Call final failure callback if provided
                        if on_final_failure:
                            try:
                                on_final_failure(e, max_attempts)
                            except Exception as callback_error:
                                logger.error(f"Error in final failure callback: {callback_error}")

                        # Re-raise the last exception
                        raise last_exception from last_exception

            # This should never be reached
            return None

        return wrapper

    return decorator


class ErrorTracker:
    """Tracks and analyzes errors for system health monitoring."""

    def __init__(self, max_errors: int = 1000):
        """
        Initialize error tracker.

        Args:
            max_errors: Maximum number of errors to track
        """
        self.max_errors = max_errors
        self.errors = []
        self.error_counts = {}
        self.start_time = datetime.now()

    def track_error(
        self,
        error: Exception,
        context: dict[str, Any] = None,
        severity: str = 'error'
    ) -> None:
        """
        Track an error occurrence.

        Args:
            error: The exception that occurred
            context: Additional context about the error
            severity: Error severity level
        """
        error_info = {
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'severity': severity,
            'context': context or {},
            'traceback': traceback.format_exc()
        }

        # Add to errors list
        self.errors.append(error_info)

        # Limit list size
        if len(self.errors) > self.max_errors:
            self.errors.pop(0)

        # Update error counts
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Log error
        logger.error(
            f"Error tracked: {error_type} - {str(error)}",
            extra={'error_info': error_info}
        )

    def get_error_summary(self, hours: int = 24) -> dict[str, Any]:
        """
        Get error summary for the specified time period.

        Args:
            hours: Number of hours to look back

        Returns:
            Error summary dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filter errors by time
        recent_errors = [
            error for error in self.errors
            if error['timestamp'] >= cutoff_time
        ]

        # Count by severity
        severity_counts = {}
        for error in recent_errors:
            severity = error['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Count by error type
        type_counts = {}
        for error in recent_errors:
            error_type = error['error_type']
            type_counts[error_type] = type_counts.get(error_type, 0) + 1

        return {
            'period_hours': hours,
            'total_errors': len(recent_errors),
            'severity_counts': severity_counts,
            'type_counts': type_counts,
            'error_rate_per_hour': len(recent_errors) / hours if hours > 0 else 0,
            'most_common_error': max(type_counts.items(), key=lambda x: x[1]) if type_counts else None
        }

    def get_health_score(self) -> float:
        """
        Calculate system health score based on error patterns.

        Returns:
            Health score from 0.0 (unhealthy) to 1.0 (healthy)
        """
        if not self.errors:
            return 1.0

        # Get recent error summary
        summary = self.get_error_summary(hours=1)

        # Calculate health score based on error rate and severity
        error_rate = summary['error_rate_per_hour']
        critical_errors = summary['severity_counts'].get('critical', 0)

        # Base score starts at 1.0
        score = 1.0

        # Reduce score based on error rate
        if error_rate > 10:
            score -= 0.5
        elif error_rate > 5:
            score -= 0.3
        elif error_rate > 1:
            score -= 0.1

        # Reduce score based on critical errors
        if critical_errors > 0:
            score -= 0.3

        # Ensure score is between 0.0 and 1.0
        return max(0.0, min(1.0, score))

    def clear_errors(self) -> None:
        """Clear all tracked errors."""
        self.errors.clear()
        self.error_counts.clear()
        logger.info("All tracked errors cleared")


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    error_handler: Callable = None,
    **kwargs
) -> Any:
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Value to return on error
        error_handler: Custom error handler function
        **kwargs: Function keyword arguments

    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)

    except Exception as e:
        if error_handler:
            try:
                return error_handler(e, func, args, kwargs)
            except Exception as handler_error:
                logger.error(f"Error in custom error handler: {handler_error}")

        logger.error(f"Error executing {func.__name__}: {e}")
        return default_return


def create_error_context(
    function_name: str,
    module_name: str,
    **kwargs
) -> dict[str, Any]:
    """
    Create standardized error context.

    Args:
        function_name: Name of the function where error occurred
        module_name: Name of the module
        **kwargs: Additional context variables

    Returns:
        Error context dictionary
    """
    context = {
        'function': function_name,
        'module': module_name,
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': sys.platform
    }
    context.update(kwargs)
    return context


# Global error tracker instance
error_tracker = ErrorTracker()


def track_global_error(
    error: Exception,
    context: dict[str, Any] = None,
    severity: str = 'error'
) -> None:
    """
    Track an error using the global error tracker.

    Args:
        error: The exception that occurred
        context: Additional context about the error
        severity: Error severity level
    """
    error_tracker.track_error(error, context, severity)
