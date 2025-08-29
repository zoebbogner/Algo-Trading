"""
Data processing module.

Handles data normalization, cleaning, and quality control:
- Raw data normalization
- Data quality checks
- Schema validation
- Data cleaning and deduplication
"""

from .normalizer import DataNormalizer

__all__ = [
    "DataNormalizer"
]
