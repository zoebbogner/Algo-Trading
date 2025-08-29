"""CLI for Machine Learning Model Training and Validation."""

import click
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
import json
from datetime import datetime

from src.config.base import load_config as load_base_config
from src.core.ml import MLFeatureEngineer, MLSignalGenerator, ModelTrainer, ModelValidator
from src.utils.logging import get_logger

logger = get_logger(__name__)


@click.group()
@click.pass_context
def ml(ctx):
    """Machine Learning commands for trading system."""
    ctx.ensure_object(dict)
    base_config = load_base_config()
    ctx.obj['base_config'] = base_config


@ml.command()
@click.option('--data-file', required=True, help='Path to data file (Parquet)')
@click.option('--config-file', default='configs/ml.yaml', help='ML configuration file')
@click.option('--output-dir', default='models/ml', help='Output directory for trained models')
@click.option('--target-method', default='future_return', 
              type=click.Choice(['future_return', 'trend_following', 'mean_reversion']),
              help='Method for creating target variable')
@click.option('--test-size', default=0.2, help='Test set size (0.0-1.0)')
@click.option('--cv-folds', default=5, help='Cross-validation folds')
@click.pass_context
def train(ctx, data_file, config_file, output_dir, target_method, test_size, cv_folds):
    """Train ML models for trading signal generation."""
    try:
        # Load ML configuration
        with open(config_file, 'r') as f:
            ml_config = yaml.safe_load(f)
        
        # Update config with CLI options
        ml_config['ml_training']['test_size'] = test_size
        ml_config['ml_training']['cv_folds'] = cv_folds
        ml_config['ml_training']['target_method'] = target_method
        
        logger.info(f"Starting ML model training with config: {config_file}")
        logger.info(f"Data file: {data_file}")
        logger.info(f"Output directory: {output_dir}")
        
        # Load data
        logger.info("Loading data...")
        data = pd.read_parquet(data_file)
        logger.info(f"Loaded {len(data)} rows, {len(data.columns)} columns")
        
        # Initialize ML components
        feature_engineer = MLFeatureEngineer(ml_config)
        model_trainer = ModelTrainer(ml_config)
        
        # Create ML features
        logger.info("Creating ML features...")
        data_with_features = feature_engineer.create_ml_features(data)
        logger.info(f"Created {len(feature_engineer.feature_columns)} features")
        
        # Prepare target variable
        logger.info(f"Preparing target variable using method: {target_method}")
        data_with_target = model_trainer.prepare_target_variable(data_with_features, target_method)
        logger.info(f"Target distribution: {data_with_target['target'].value_counts().to_dict()}")
        
        # Split data into train/test
        train_size = int(len(data_with_target) * (1 - test_size))
        train_data = data_with_target.iloc[:train_size]
        test_data = data_with_target.iloc[train_size:]
        
        logger.info(f"Train set: {len(train_data)} samples, Test set: {len(test_data)} samples")
        
        # Train models
        logger.info("Training models with hyperparameter tuning...")
        trained_models = model_trainer.train_all_models(train_data, 'target')
        
        if not trained_models:
            logger.error("No models were trained successfully")
            return
        
        # Save training results
        logger.info("Saving training results...")
        model_trainer.save_training_results(output_dir)
        
        # Validate models on test set
        logger.info("Validating models on test set...")
        validator = ModelValidator(ml_config)
        
        # Prepare test features
        metadata_cols = ['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'target']
        feature_columns = [col for col in test_data.columns if col not in metadata_cols]
        
        X_test = test_data[feature_columns].values
        y_test = test_data['target'].values
        
        # Remove NaN values
        mask = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
        X_test = X_test[mask]
        y_test = y_test[mask]
        
        logger.info(f"Test data shape: {X_test.shape}")
        
        # Validate models
        validation_results = validator.validate_models(trained_models, X_test, y_test)
        
        # Save validation results
        logger.info("Saving validation results...")
        validator.save_validation_results(output_dir)
        
        # Create performance plots
        logger.info("Creating performance plots...")
        validator.create_performance_plots(output_dir)
        
        # Print summary
        logger.info("=== TRAINING SUMMARY ===")
        logger.info(f"Models trained: {len(trained_models)}")
        logger.info(f"Features created: {len(feature_engineer.feature_columns)}")
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Test samples: {len(test_data)}")
        logger.info(f"Output directory: {output_dir}")
        
        # Print best model performance
        if validation_results:
            logger.info("\n=== MODEL PERFORMANCE ===")
            for model_name, results in validation_results.items():
                metrics = results['metrics']
                logger.info(f"{model_name}:")
                logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  Recall: {metrics['recall']:.4f}")
                logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
                if metrics.get('roc_auc'):
                    logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        
        logger.info("ML training completed successfully!")
        
    except Exception as e:
        logger.error(f"ML training failed: {e}")
        raise


@ml.command()
@click.option('--model-dir', required=True, help='Directory containing trained models')
@click.option('--data-file', required=True, help='Path to data file for prediction')
@click.option('--config-file', default='configs/ml.yaml', help='ML configuration file')
@click.option('--output-file', help='Output file for predictions')
@click.pass_context
def predict(ctx, model_dir, data_file, config_file, output_file):
    """Generate predictions using trained ML models."""
    try:
        # Load ML configuration
        with open(config_file, 'r') as f:
            ml_config = yaml.safe_load(f)
        
        logger.info(f"Generating predictions using models from: {model_dir}")
        logger.info(f"Data file: {data_file}")
        
        # Load data
        logger.info("Loading data...")
        data = pd.read_parquet(data_file)
        logger.info(f"Loaded {len(data)} rows, {len(data.columns)} columns")
        
        # Initialize ML components
        feature_engineer = MLFeatureEngineer(ml_config)
        ml_signal_generator = MLSignalGenerator(ml_config)
        
        # Create ML features
        logger.info("Creating ML features...")
        data_with_features = feature_engineer.create_ml_features(data)
        logger.info(f"Created {len(feature_engineer.feature_columns)} features")
        
        # Load trained models
        logger.info("Loading trained models...")
        ml_signal_generator.load_models(model_dir)
        
        # Generate predictions
        logger.info("Generating ML predictions...")
        data_with_predictions = ml_signal_generator.generate_signals(data_with_features)
        
        # Save results
        if output_file:
            logger.info(f"Saving predictions to: {output_file}")
            data_with_predictions.to_parquet(output_file, index=False)
        else:
            # Save to default location
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            default_output = f"predictions_ml_{timestamp}.parquet"
            logger.info(f"Saving predictions to: {default_output}")
            data_with_predictions.to_parquet(default_output, index=False)
        
        # Print summary
        logger.info("=== PREDICTION SUMMARY ===")
        logger.info(f"Input samples: {len(data)}")
        logger.info(f"Features created: {len(feature_engineer.feature_columns)}")
        logger.info(f"Predictions generated: {len(data_with_predictions)}")
        
        # Show prediction distribution
        if 'ml_signal' in data_with_predictions.columns:
            signal_counts = data_with_predictions['ml_signal'].value_counts()
            logger.info(f"Signal distribution: {signal_counts.to_dict()}")
        
        if 'ml_confidence' in data_with_predictions.columns:
            confidence_stats = data_with_predictions['ml_confidence'].describe()
            logger.info(f"Confidence statistics:")
            logger.info(f"  Mean: {confidence_stats['mean']:.4f}")
            logger.info(f"  Std: {confidence_stats['std']:.4f}")
            logger.info(f"  Min: {confidence_stats['min']:.4f}")
            logger.info(f"  Max: {confidence_stats['max']:.4f}")
        
        logger.info("ML predictions completed successfully!")
        
    except Exception as e:
        logger.error(f"ML prediction failed: {e}")
        raise


@ml.command()
@click.option('--model-dir', required=True, help='Directory containing trained models')
@click.option('--config-file', default='configs/ml.yaml', help='ML configuration file')
@click.pass_context
def evaluate(ctx, model_dir, config_file):
    """Evaluate trained ML models."""
    try:
        # Load ML configuration
        with open(config_file, 'r') as f:
            ml_config = yaml.safe_load(f)
        
        logger.info(f"Evaluating models from: {model_dir}")
        
        # Check if models exist
        model_path = Path(model_dir)
        if not model_path.exists():
            logger.error(f"Model directory does not exist: {model_dir}")
            return
        
        # List available models
        model_files = list(model_path.glob("*.joblib"))
        logger.info(f"Found {len(model_files)} model files:")
        for model_file in model_files:
            logger.info(f"  - {model_file.name}")
        
        # Load and evaluate models
        ml_signal_generator = MLSignalGenerator(ml_config)
        ml_signal_generator.load_models(model_dir)
        
        logger.info("Models loaded successfully!")
        logger.info(f"Feature columns: {len(ml_signal_generator.feature_columns)}")
        
        # Check model performance files
        performance_files = list(model_path.glob("*.json"))
        if performance_files:
            logger.info("\n=== PERFORMANCE FILES ===")
            for perf_file in performance_files:
                logger.info(f"  - {perf_file.name}")
                
                # Load and display performance
                try:
                    with open(perf_file, 'r') as f:
                        perf_data = json.load(f)
                    
                    if 'training_history' in perf_file.name:
                        logger.info(f"Training History:")
                        for model_name, history in perf_data.items():
                            logger.info(f"  {model_name}: {history}")
                    
                    elif 'cv_results' in perf_file.name:
                        logger.info(f"Cross-Validation Results:")
                        for model_name, results in perf_data.items():
                            logger.info(f"  {model_name}: Best Score = {results['best_score']:.4f}")
                    
                    elif 'validation_results' in perf_file.name:
                        logger.info(f"Validation Results:")
                        for model_name, results in perf_data.items():
                            metrics = results['metrics']
                            logger.info(f"  {model_name}:")
                            logger.info(f"    Accuracy: {metrics['accuracy']:.4f}")
                            logger.info(f"    Precision: {metrics['precision']:.4f}")
                            logger.info(f"    Recall: {metrics['recall']:.4f}")
                            logger.info(f"    F1-Score: {metrics['f1_score']:.4f}")
                
                except Exception as e:
                    logger.warning(f"Could not read {perf_file}: {e}")
        
        logger.info("Model evaluation completed!")
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise


if __name__ == '__main__':
    ml()
