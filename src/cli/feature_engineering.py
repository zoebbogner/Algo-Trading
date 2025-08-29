#!/usr/bin/env python3
"""
Feature Engineering CLI for the Algo-Trading system.

Provides command-line interface for:
- Feature computation and engineering
- Feature validation and quality control
- Feature dataset management
- Performance monitoring
"""

import sys
from pathlib import Path

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.base import load_config as load_base_config
from src.config.features import load_config as load_features_config
from src.data.feature_extraction.engine import FeatureEngineer
from src.utils.logging import get_logger, setup_logging


@click.group()
@click.option('--config', '-c', default='configs/features.yaml',
              help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def feature_engineering(ctx, config: str, verbose: bool):
    """Feature Engineering CLI for Algo-Trading system."""
    # Setup logging
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(log_level=log_level)

    # Load configuration
    base_config = load_base_config()
    features_config = load_features_config()
    ctx.obj = {**base_config, **features_config}

    # Ensure output directory exists
    output_path = Path(ctx.obj['output']['path']).parent
    output_path.mkdir(parents=True, exist_ok=True)


@feature_engineering.command()
@click.option('--input-dir', '-i', default='data/processed', help='Input directory with processed data')
@click.option('--output-file', '-o', help='Output file path (overrides config)')
@click.option('--symbols', '-s', multiple=True, help='Specific symbols to process (all if not specified)')
@click.option('--start-date', help='Start date for feature computation (YYYY-MM-DD)')
@click.option('--end-date', help='End date for feature computation (YYYY-MM-DD)')
@click.pass_context
def compute_features(ctx, input_dir: str, output_file: str, symbols: list[str] | None,
                    start_date: str, end_date: str):
    """Compute features for the specified data."""
    logger = get_logger(__name__)
    config = ctx.obj

    # Override output path if specified
    if output_file:
        config['output']['path'] = output_file

    logger.info("Starting feature computation")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output file: {config['output']['path']}")

    try:
        # Initialize feature engineer
        engineer = FeatureEngineer(config)

        # Load data
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            sys.exit(1)

        # Find data files
        if symbols:
            data_files = [input_path / f"{symbol.lower()}_bars_1m.parquet" for symbol in symbols]
        else:
            data_files = list(input_path.glob("*_bars_1m.parquet"))

        logger.info(f"Found {len(data_files)} data files to process")

        # Process each file
        all_features = []
        for file_path in data_files:
            if file_path.exists():
                logger.info(f"Processing {file_path.name}")

                try:
                    import pandas as pd
                    df = pd.read_parquet(file_path)

                    # Filter by date range if specified
                    if start_date:
                        df = df[df['ts'] >= start_date]
                    if end_date:
                        df = df[df['ts'] <= end_date]

                    if len(df) == 0:
                        logger.warning(f"No data found in {file_path.name} for specified date range")
                        continue

                    # Compute features
                    features_df = engineer.compute_all_features(df)

                    # Add to collection
                    all_features.append(features_df)

                    logger.info(f"✅ Computed features for {file_path.name}: {len(features_df)} rows")

                except Exception as e:
                    logger.error(f"❌ Error processing {file_path.name}: {e}")
                    continue
            else:
                logger.warning(f"File not found: {file_path}")

        if not all_features:
            logger.error("No features computed from any files")
            sys.exit(1)

        # Combine all features
        logger.info("Combining features from all files")
        combined_features = pd.concat(all_features, ignore_index=True)

        # Sort by timestamp and symbol
        combined_features = combined_features.sort_values(['symbol', 'ts'])

        # Save features
        output_path = Path(config['output']['path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        combined_features.to_parquet(output_path, index=False)

        logger.info(f"✅ Features saved to {output_path}")
        logger.info(f"📊 Total features computed: {len(combined_features)} rows")
        logger.info(f"🔧 Feature columns: {list(combined_features.columns)}")

    except Exception as e:
        logger.error(f"Error during feature computation: {e}")
        sys.exit(1)


@feature_engineering.command()
@click.option('--features-file', '-f', required=True, help='Features file to validate')
@click.pass_context
def validate_features(ctx, features_file: str):
    """Validate computed features for quality and integrity."""
    logger = get_logger(__name__)

    logger.info("Starting feature validation")
    logger.info(f"Features file: {features_file}")

    try:
        import pandas as pd

        # Load features
        features_path = Path(features_file)
        if not features_path.exists():
            logger.error(f"Features file does not exist: {features_file}")
            sys.exit(1)

        df = pd.read_parquet(features_file)
        logger.info(f"Loaded {len(df)} feature rows")

        # Basic structure validation
        required_columns = ['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            sys.exit(1)

        # Feature quality checks
        feature_columns = [col for col in df.columns if col not in required_columns + ['source', 'load_id', 'ingestion_ts', 'date']]
        logger.info(f"Found {len(feature_columns)} feature columns")

        # Check for NaN values in features
        nan_counts = {}
        for col in feature_columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                nan_counts[col] = nan_count

        if nan_counts:
            logger.warning("Found NaN values in features:")
            for col, count in nan_counts.items():
                percentage = (count / len(df)) * 100
                logger.warning(f"  {col}: {count} NaN values ({percentage:.2f}%)")
        else:
            logger.info("✅ No NaN values found in features")

        # Check feature ranges
        logger.info("Checking feature value ranges...")
        for col in feature_columns:
            if df[col].dtype in ['float64', 'int64']:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                std_val = df[col].std()

                logger.info(f"  {col}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}")

        # Check for infinite values
        inf_counts = {}
        for col in feature_columns:
            if df[col].dtype in ['float64']:
                inf_count = pd.isinf(df[col]).sum()
                if inf_count > 0:
                    inf_counts[col] = inf_count

        if inf_counts:
            logger.warning("Found infinite values in features:")
            for col, count in inf_counts.items():
                logger.warning(f"  {col}: {count} infinite values")
        else:
            logger.info("✅ No infinite values found in features")

        # Summary
        logger.info("Feature validation completed")
        logger.info(f"📊 Total rows: {len(df)}")
        logger.info(f"🔧 Feature columns: {len(feature_columns)}")
        logger.info(f"📅 Date range: {df['ts'].min()} to {df['ts'].max()}")
        logger.info(f"🏷️ Symbols: {df['symbol'].nunique()}")

    except Exception as e:
        logger.error(f"Error during feature validation: {e}")
        sys.exit(1)


@feature_engineering.command()
@click.option('--features-file', '-f', required=True, help='Features file to analyze')
@click.option('--output-dir', '-o', default='reports/features', help='Output directory for analysis')
@click.pass_context
def analyze_features(ctx, features_file: str, output_dir: str):
    """Analyze feature distributions and correlations."""
    logger = get_logger(__name__)

    logger.info("Starting feature analysis")
    logger.info(f"Features file: {features_file}")
    logger.info(f"Output directory: {output_dir}")

    try:
        import pandas as pd

        # Load features
        df = pd.read_parquet(features_file)
        logger.info(f"Loaded {len(df)} feature rows")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Feature columns (exclude metadata)
        feature_columns = [col for col in df.columns if col not in ['ts', 'symbol', 'source', 'load_id', 'ingestion_ts', 'date']]

        # Basic statistics
        logger.info("Computing basic statistics...")
        stats_df = df[feature_columns].describe()
        stats_df.to_csv(output_path / 'feature_statistics.csv')
        logger.info(f"✅ Statistics saved to {output_path / 'feature_statistics.csv'}")

        # Correlation analysis
        logger.info("Computing feature correlations...")
        correlation_matrix = df[feature_columns].corr()
        correlation_matrix.to_csv(output_path / 'feature_correlations.csv')
        logger.info(f"✅ Correlations saved to {output_path / 'feature_correlations.csv'}")

        # High correlation pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_val
                    })

        if high_corr_pairs:
            logger.warning(f"Found {len(high_corr_pairs)} high correlation pairs (>0.8):")
            for pair in high_corr_pairs:
                logger.warning(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")

        # Feature importance (simple variance-based)
        logger.info("Computing feature importance...")
        feature_variance = df[feature_columns].var().sort_values(ascending=False)
        feature_variance.to_csv(output_path / 'feature_variance.csv')
        logger.info(f"✅ Feature variance saved to {output_path / 'feature_variance.csv'}")

        # Top features by variance
        top_features = feature_variance.head(10)
        logger.info("Top 10 features by variance:")
        for feature, variance in top_features.items():
            logger.info(f"  {feature}: {variance:.6f}")

        # Summary report
        summary = {
            'total_rows': len(df),
            'total_features': len(feature_columns),
            'date_range': f"{df['ts'].min()} to {df['ts'].max()}",
            'symbols': df['symbol'].nunique(),
            'high_correlation_pairs': len(high_corr_pairs),
            'nan_percentage': (df[feature_columns].isna().sum().sum() / (len(df) * len(feature_columns))) * 100
        }

        import json
        with open(output_path / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"✅ Analysis summary saved to {output_path / 'analysis_summary.json'}")
        logger.info("Feature analysis completed successfully")

    except Exception as e:
        logger.error(f"Error during feature analysis: {e}")
        sys.exit(1)


if __name__ == '__main__':
    feature_engineering()

