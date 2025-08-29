#!/usr/bin/env python3
"""
Features API endpoints for the Algo-Trading system.

Provides REST API for:
- Feature computation and engineering
- Feature validation and quality control
- Feature analysis and statistics
"""

import sys
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.base import load_config as load_base_config
from src.config.features import load_config as load_features_config
from src.data.feature_extraction.engine import FeatureEngineer
from src.utils.logging import get_logger

# Create blueprint
features_router = Blueprint('features', __name__, url_prefix='/api/features')
logger = get_logger(__name__)

# Load configuration
base_config = load_base_config()
features_config = load_features_config()
config = {**base_config, **features_config}


@features_router.route('/health', methods=['GET'])
def health_check():
    """Check features API health."""
    return jsonify({
        'status': 'healthy',
        'service': 'features-api',
        'config_loaded': bool(config)
    })


@features_router.route('/compute', methods=['POST'])
def compute_features():
    """Compute features for the specified data."""
    try:
        data = request.get_json()
        input_dir = data.get('input_dir', 'data/processed')
        output_file = data.get('output_file')
        symbols = data.get('symbols', [])
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        logger.info("Starting feature computation")

        # Override output path if specified
        if output_file:
            config['output']['path'] = output_file

        # Initialize feature engineer
        engineer = FeatureEngineer(config)

        # Load data
        input_path = Path(input_dir)
        if not input_path.exists():
            return jsonify({
                'success': False,
                'error': f'Input directory does not exist: {input_dir}'
            }), 400

        # Find data files
        if symbols:
            data_files = [input_path / f"{symbol.lower()}_bars_1m.parquet" for symbol in symbols]
        else:
            data_files = list(input_path.glob("*_bars_1m.parquet"))

        logger.info(f"Found {len(data_files)} data files to process")

        # Process each file
        all_features = []
        processed_files = []
        failed_files = []

        for file_path in data_files:
            if file_path.exists():
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
                    processed_files.append({
                        'file': file_path.name,
                        'rows': len(features_df),
                        'features': len([col for col in features_df.columns if col not in ['ts', 'symbol', 'source', 'load_id', 'ingestion_ts', 'date']])
                    })

                    logger.info(f"✅ Computed features for {file_path.name}: {len(features_df)} rows")

                except Exception as e:
                    logger.error(f"❌ Error processing {file_path.name}: {e}")
                    failed_files.append({
                        'file': file_path.name,
                        'error': str(e)
                    })
                    continue
            else:
                logger.warning(f"File not found: {file_path}")
                failed_files.append({
                    'file': str(file_path),
                    'error': 'File not found'
                })

        if not all_features:
            return jsonify({
                'success': False,
                'error': 'No features computed from any files'
            }), 500

        # Combine all features
        logger.info("Combining features from all files")
        import pandas as pd
        combined_features = pd.concat(all_features, ignore_index=True)

        # Sort by timestamp and symbol
        combined_features = combined_features.sort_values(['symbol', 'ts'])

        # Save features
        output_path = Path(config['output']['path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        combined_features.to_parquet(output_path, index=False)

        # Summary
        feature_columns = [col for col in combined_features.columns if col not in ['ts', 'symbol', 'source', 'load_id', 'ingestion_ts', 'date']]

        return jsonify({
            'success': True,
            'message': 'Feature computation completed successfully',
            'output_file': str(output_path),
            'summary': {
                'total_rows': len(combined_features),
                'total_features': len(feature_columns),
                'symbols': combined_features['symbol'].nunique(),
                'date_range': {
                    'start': str(combined_features['ts'].min()),
                    'end': str(combined_features['ts'].max())
                }
            },
            'processed_files': processed_files,
            'failed_files': failed_files
        })

    except Exception as e:
        logger.error(f"Error during feature computation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@features_router.route('/validate', methods=['POST'])
def validate_features():
    """Validate computed features for quality and integrity."""
    try:
        data = request.get_json()
        features_file = data.get('features_file')

        if not features_file:
            return jsonify({
                'success': False,
                'error': 'features_file is required'
            }), 400

        logger.info("Starting feature validation")

        import pandas as pd

        # Load features
        features_path = Path(features_file)
        if not features_path.exists():
            return jsonify({
                'success': False,
                'error': f'Features file does not exist: {features_file}'
            }), 400

        df = pd.read_parquet(features_file)
        logger.info(f"Loaded {len(df)} feature rows")

        # Basic structure validation
        required_columns = ['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return jsonify({
                'success': False,
                'error': f'Missing required columns: {missing_columns}'
            }), 400

        # Feature quality checks
        feature_columns = [col for col in df.columns if col not in required_columns + ['source', 'load_id', 'ingestion_ts', 'date']]
        logger.info(f"Found {len(feature_columns)} feature columns")

        # Check for NaN values in features
        nan_counts = {}
        for col in feature_columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                nan_counts[col] = nan_count

        # Check feature ranges
        feature_stats = {}
        for col in feature_columns:
            if df[col].dtype in ['float64', 'int64']:
                feature_stats[col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'nan_count': int(df[col].isna().sum())
                }

        # Check for infinite values
        inf_counts = {}
        for col in feature_columns:
            if df[col].dtype in ['float64']:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    inf_counts[col] = inf_count

        # Summary
        validation_summary = {
            'total_rows': len(df),
            'total_features': len(feature_columns),
            'date_range': {
                'start': str(df['ts'].min()),
                'end': str(df['ts'].max())
            },
            'symbols': df['symbol'].nunique(),
            'nan_features': len(nan_counts),
            'inf_features': len(inf_counts),
            'validation_status': 'valid' if not nan_counts and not inf_counts else 'warnings'
        }

        return jsonify({
            'success': True,
            'message': 'Feature validation completed',
            'validation_summary': validation_summary,
            'nan_counts': nan_counts,
            'inf_counts': inf_counts,
            'feature_stats': feature_stats
        })

    except Exception as e:
        logger.error(f"Error during feature validation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@features_router.route('/analyze', methods=['POST'])
def analyze_features():
    """Analyze feature distributions and correlations."""
    try:
        data = request.get_json()
        features_file = data.get('features_file')
        output_dir = data.get('output_dir', 'reports/features')

        if not features_file:
            return jsonify({
                'success': False,
                'error': 'features_file is required'
            }), 400

        logger.info("Starting feature analysis")

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
        stats_file = output_path / 'feature_statistics.csv'
        stats_df.to_csv(stats_file)

        # Correlation analysis
        logger.info("Computing feature correlations...")
        correlation_matrix = df[feature_columns].corr()
        corr_file = output_path / 'feature_correlations.csv'
        correlation_matrix.to_csv(corr_file)

        # High correlation pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': float(corr_val)
                    })

        # Feature importance (simple variance-based)
        logger.info("Computing feature importance...")
        feature_variance = df[feature_columns].var().sort_values(ascending=False)
        variance_file = output_path / 'feature_variance.csv'
        feature_variance.to_csv(variance_file)

        # Top features by variance
        top_features = feature_variance.head(10).to_dict()

        # Summary report
        summary = {
            'total_rows': len(df),
            'total_features': len(feature_columns),
            'date_range': f"{df['ts'].min()} to {df['ts'].max()}",
            'symbols': df['symbol'].nunique(),
            'high_correlation_pairs': len(high_corr_pairs),
            'nan_percentage': float((df[feature_columns].isna().sum().sum() / (len(df) * len(feature_columns))) * 100),
            'output_files': [
                str(stats_file),
                str(corr_file),
                str(variance_file)
            ]
        }

        summary_file = output_path / 'analysis_summary.json'
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        return jsonify({
            'success': True,
            'message': 'Feature analysis completed successfully',
            'output_directory': str(output_path),
            'summary': summary,
            'high_correlation_pairs': high_corr_pairs,
            'top_features': top_features
        })

    except Exception as e:
        logger.error(f"Error during feature analysis: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@features_router.route('/status', methods=['GET'])
def get_features_status():
    """Get current features status and statistics."""
    try:
        features_dir = Path(config['output']['path']).parent

        status = {
            'features_directory': str(features_dir),
            'features_file': config['output']['path'],
            'exists': False,
            'file_info': {}
        }

        if features_dir.exists():
            # Check for features files
            features_files = list(features_dir.glob("*.parquet"))

            if features_files:
                # Get info about the main features file
                main_file = Path(config['output']['path'])
                if main_file.exists():
                    import pandas as pd
                    df = pd.read_parquet(main_file)

                    feature_columns = [col for col in df.columns if col not in ['ts', 'symbol', 'source', 'load_id', 'ingestion_ts', 'date']]

                    status.update({
                        'exists': True,
                        'file_info': {
                            'rows': len(df),
                            'symbols': df['symbol'].nunique(),
                            'features': len(feature_columns),
                            'date_range': {
                                'start': str(df['ts'].min()),
                                'end': str(df['ts'].max())
                            },
                            'size_mb': round(main_file.stat().st_size / (1024 * 1024), 2)
                        }
                    })

            status['total_files'] = len(features_files)
            status['files'] = [f.name for f in features_files]

        return jsonify({
            'success': True,
            'data': status
        })

    except Exception as e:
        logger.error(f"Error getting features status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500





