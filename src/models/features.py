"""
Feature data models for the Algo-Trading system.

Provides structured data models for:
- Feature data structures and validation
- Feature metadata and lineage
- Feature quality metrics
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FeatureMetadata:
    """Metadata for feature computation and lineage tracking."""

    feature_name: str
    feature_family: str
    rolling_window: int | None = None
    computation_method: str = "rolling"
    dependencies: list[str] = None
    description: str = ""
    units: str = ""
    expected_range: tuple = (None, None)
    created_at: str = ""
    config_hash: str = ""

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class FeatureQualityMetrics:
    """Quality metrics for feature validation."""

    non_na_percentage: float
    mean_value: float
    std_value: float
    min_value: float
    max_value: float
    p1_percentile: float
    p50_percentile: float
    p99_percentile: float
    coverage_status: str = "PASS"
    quality_score: float = 1.0

    def __post_init__(self):
        if self.non_na_percentage < 0.95:
            self.coverage_status = "FAIL"
            self.quality_score = max(0.0, self.non_na_percentage / 0.95)


@dataclass
class FeatureSet:
    """Complete set of features with metadata and quality metrics."""

    features_df: pd.DataFrame
    metadata: dict[str, FeatureMetadata]
    quality_metrics: dict[str, FeatureQualityMetrics]
    feature_families: list[str]
    symbols: list[str]
    date_range: tuple
    config_hash: str

    def validate_features(self) -> dict[str, Any]:
        """Validate the feature set against quality gates."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "coverage_summary": {},
            "quality_summary": {}
        }

        # Check coverage requirements
        for feature_name, metrics in self.quality_metrics.items():
            if metrics.coverage_status == "FAIL":
                validation_results["valid"] = False
                validation_results["errors"].append(
                    f"Feature {feature_name} has insufficient coverage: {metrics.non_na_percentage:.1%}"
                )

            validation_results["coverage_summary"][feature_name] = {
                "coverage": metrics.non_na_percentage,
                "status": metrics.coverage_status
            }

        # Check feature completeness
        expected_features = set(self.metadata.keys())
        actual_features = set(self.features_df.columns)
        missing_features = expected_features - actual_features

        if missing_features:
            validation_results["valid"] = False
            validation_results["errors"].append(
                f"Missing expected features: {missing_features}"
            )

        # Check data integrity
        if not self.features_df.index.is_monotonic_increasing:
            validation_results["valid"] = False
            validation_results["errors"].append("Feature timestamps are not monotonic")

        # Check for duplicates
        if self.features_df.index.duplicated().any():
            validation_results["valid"] = False
            validation_results["errors"].append("Duplicate timestamps found in features")

        return validation_results

    def get_feature_summary(self) -> pd.DataFrame:
        """Get a summary of all features with their metadata."""
        summary_data = []

        for feature_name, metadata in self.metadata.items():
            quality = self.quality_metrics.get(feature_name)

            summary_data.append({
                "feature_name": feature_name,
                "feature_family": metadata.feature_family,
                "rolling_window": metadata.rolling_window,
                "computation_method": metadata.computation_method,
                "dependencies": ", ".join(metadata.dependencies),
                "description": metadata.description,
                "units": metadata.units,
                "coverage_pct": quality.non_na_percentage if quality else None,
                "quality_status": quality.coverage_status if quality else "UNKNOWN",
                "mean_value": quality.mean_value if quality else None,
                "std_value": quality.std_value if quality else None
            })

        return pd.DataFrame(summary_data)

    def filter_by_family(self, family: str) -> pd.DataFrame:
        """Filter features by family."""
        family_features = [
            name for name, meta in self.metadata.items()
            if meta.feature_family == family
        ]
        return self.features_df[family_features]

    def get_feature_dependencies(self, feature_name: str) -> list[str]:
        """Get the dependency chain for a specific feature."""
        if feature_name not in self.metadata:
            return []

        dependencies = set()
        to_process = [feature_name]

        while to_process:
            current = to_process.pop(0)
            if current in self.metadata:
                deps = self.metadata[current].dependencies
                for dep in deps:
                    if dep not in dependencies:
                        dependencies.add(dep)
                        to_process.append(dep)

        return list(dependencies)


def validate_feature_dataframe(df: pd.DataFrame, required_columns: list[str]) -> dict[str, Any]:
    """Validate a feature DataFrame against requirements."""
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }

    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        validation_result["valid"] = False
        validation_result["errors"].append(
            f"Missing required columns: {missing_columns}"
        )

    # Check for infinite values
    if df.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).any().any():
        validation_result["warnings"].append("Infinite values detected in numeric features")

    # Check for extreme outliers (beyond 6 standard deviations)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                if std_val > 0:
                    z_scores = np.abs((values - mean_val) / std_val)
                    extreme_outliers = (z_scores > 6).sum()
                    if extreme_outliers > 0:
                        validation_result["warnings"].append(
                            f"Feature {col} has {extreme_outliers} extreme outliers (|z| > 6)"
                        )

    return validation_result


def create_feature_metadata(
    feature_name: str,
    feature_family: str,
    rolling_window: int | None = None,
    dependencies: list[str] = None,
    description: str = "",
    units: str = "",
    expected_range: tuple = (None, None)
) -> FeatureMetadata:
    """Create feature metadata with standard defaults."""
    if dependencies is None:
        dependencies = []

    return FeatureMetadata(
        feature_name=feature_name,
        feature_family=feature_family,
        rolling_window=rolling_window,
        dependencies=dependencies,
        description=description,
        units=units,
        expected_range=expected_range
    )
