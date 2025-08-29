"""
Data collection and processing module.

This module handles all data operations including:
- Fetching data from external sources
- Data processing and normalization
- Feature engineering
- Data quality control
"""

from . import feature_extraction, fetch, processing

__all__ = [
    "fetch",
    "processing",
    "feature_extraction"
]
