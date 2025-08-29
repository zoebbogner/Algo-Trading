#!/usr/bin/env python3
"""
Base data fetcher class for the crypto historical data collection system.

This module provides the abstract base class for all data fetchers.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


class BaseDataFetcher(ABC):
    """Base class for all data fetchers."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the base fetcher."""
        self.config = config
        self.session = None
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Ensure output directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure required directories exist."""
        Path('logs').mkdir(exist_ok=True)
        Path('reports/runs').mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def fetch_data(self, symbols: list[str], **kwargs) -> dict[str, Any]:
        """Fetch data for the given symbols."""
        pass

    @abstractmethod
    async def __aenter__(self):
        """Async context manager entry."""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass

    def save_report(self, report: dict[str, Any], filename: str):
        """Save a report to the reports directory."""
        report_path = Path('reports/runs') / filename
        with open(report_path, 'w') as f:
            import yaml
            yaml.dump(report, f, default_flow_style=False, indent=2)
        logger.info(f"Report saved to: {report_path}")

    def log_summary(self, operation: str, results: dict[str, Any]):
        """Log a summary of the operation."""
        logger.info("=" * 80)
        logger.info(f"{operation.upper()} COMPLETED")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Results: {results}")
        logger.info("=" * 80)
