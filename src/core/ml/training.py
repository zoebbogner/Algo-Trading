"""Machine Learning Training and Validation Module."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import json
from datetime import datetime
import joblib

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Machine Learning model trainer with hyperparameter optimization."""
    
    def __init__(self, config: Dict):
        """Initialize model trainer."""
        self.config = config
        self.best_models = {}
        self.training_history = {}
        self.cv_results = {}
        
        # Training configuration
        self.train_config = config.get('ml_training', {})
        self.cv_folds = self.train_config.get('cv_folds', 5)
        self.test_size = self.train_config.get('test_size', 0.2)
        self.random_state = self.train_config.get('random_state', 42)
        
        # Hyperparameter grids
        self.param_grids = self._get_parameter_grids()
    
    def _get_parameter_grids(self) -> Dict:
        """Get hyperparameter grids for different models."""
        return {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 0.9, 1.0]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
    
    def prepare_target_variable(self, df: pd.DataFrame, method: str = 'future_return') -> pd.DataFrame:
        """Prepare target variable for ML training."""
        df = df.copy()
        
        if method == 'future_return':
            # Target: 1 if future return > threshold, 0 otherwise
            future_returns = df['close'].pct_change(5).shift(-5)  # 5-period ahead return
            threshold = future_returns.std() * 0.5  # Dynamic threshold
            
            df['target'] = np.where(future_returns > threshold, 1, 0)
            df['target'] = np.where(future_returns < -threshold, 2, df['target'])  # 2 for sell
            
            # Remove rows where we can't calculate future returns
            df = df.dropna(subset=['target'])
            
        elif method == 'trend_following':
            # Target: 1 if price above MA, 0 if below
            ma_period = 20
            df['ma'] = df['close'].rolling(ma_period).mean()
            df['target'] = np.where(df['close'] > df['ma'], 1, 0)
            
        elif method == 'mean_reversion':
            # Target: 1 if oversold, 0 if overbought, 2 if neutral
            rsi = self._calculate_rsi(df['close'], 14)
            df['target'] = np.where(rsi < 30, 1, np.where(rsi > 70, 0, 2))
        
        logger.info(f"Created target variable using method: {method}")
        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def train_with_hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, model_type: str) -> Any:
        """Train model with hyperparameter tuning using time series cross-validation."""
        logger.info(f"Training {model_type} with hyperparameter tuning...")
        
        # Get model class and parameter grid
        model_class = self._get_model_class(model_type)
        param_grid = self.param_grids.get(model_type, {})
        
        if not param_grid:
            logger.warning(f"No parameter grid found for {model_type}, using default parameters")
            model = model_class()
            model.fit(X, y)
            return model
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # Grid search with time series CV
        grid_search = GridSearchCV(
            estimator=model_class(),
            param_grid=param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X, y)
        
        # Store results
        self.cv_results[model_type] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
        logger.info(f"Best CV score for {model_type}: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _get_model_class(self, model_type: str):
        """Get model class by type."""
        model_classes = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'logistic_regression': LogisticRegression,
            'svm': SVC,
            'neural_network': MLPClassifier
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model_classes[model_type]
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train_all_models(self, df: pd.DataFrame, target_column: str = 'target') -> Dict:
        """Train all models with hyperparameter tuning."""
        logger.info("Training all models with hyperparameter tuning...")
        
        # Prepare features and target
        metadata_cols = ['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume', target_column]
        feature_columns = [col for col in df.columns if col not in metadata_cols]
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Remove NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Training data shape: {X.shape}")
        
        # Train each model
        for model_type in self.param_grids.keys():
            try:
                best_model = self.train_with_hyperparameter_tuning(X, y, model_type)
                self.best_models[model_type] = best_model
                
                # Calculate training accuracy
                train_accuracy = best_model.score(X, y)
                self.training_history[model_type] = {
                    'train_accuracy': train_accuracy,
                    'training_date': datetime.now().isoformat()
                }
                
                logger.info(f"{model_type} training completed. Accuracy: {train_accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                continue
        
        logger.info(f"Training completed for {len(self.best_models)} models")
        return self.best_models
    
    def save_training_results(self, output_dir: str) -> None:
        """Save training results and models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save best models
        for model_name, model in self.best_models.items():
            model_path = output_path / f"{model_name}_best.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save training history
        history_path = output_path / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
        
        # Save CV results
        cv_results_path = output_path / "cv_results.json"
        # Convert numpy arrays to lists for JSON serialization
        cv_results_serializable = {}
        for model_type, results in self.cv_results.items():
            cv_results_serializable[model_type] = {
                'best_params': results['best_params'],
                'best_score': float(results['best_score']),
                'cv_results': {
                    k: v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in results['cv_results'].items()
                }
            }
        
        with open(cv_results_path, 'w') as f:
            json.dump(cv_results_serializable, f, indent=2)
        logger.info(f"Saved CV results to {cv_results_path}")


class ModelValidator:
    """Model validation and performance evaluation."""
    
    def __init__(self, config: Dict):
        """Initialize model validator."""
        self.config = config
        self.validation_results = {}
        
    def validate_models(self, models: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Validate trained models on test data."""
        logger.info("Validating models on test data...")
        
        for model_name, model in models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                
                # Store results
                self.validation_results[model_name] = {
                    'metrics': metrics,
                    'predictions': y_pred.tolist(),
                    'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
                }
                
                logger.info(f"{model_name} validation completed:")
                logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  Recall: {metrics['recall']:.4f}")
                logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
                
            except Exception as e:
                logger.error(f"Error validating {model_name}: {e}")
                continue
        
        return self.validation_results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """Calculate comprehensive performance metrics."""
        # Basic metrics
        accuracy = np.mean(y_true == y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Precision, recall, F1-score
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # ROC AUC (if probabilities available)
        roc_auc = None
        if y_pred_proba is not None and y_pred_proba.shape[1] > 1:
            try:
                # Convert to binary classification for ROC AUC
                y_true_binary = (y_true == 1).astype(int)
                y_pred_proba_binary = y_pred_proba[:, 1]
                roc_auc = roc_auc_score(y_true_binary, y_pred_proba_binary)
            except:
                roc_auc = None
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'roc_auc': roc_auc,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
    
    def create_performance_plots(self, output_dir: str) -> None:
        """Create performance visualization plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Model comparison plot
        self._plot_model_comparison(output_path)
        
        # Confusion matrix plots
        self._plot_confusion_matrices(output_path)
        
        # Training history plots
        self._plot_training_history(output_path)
        
        logger.info(f"Performance plots saved to {output_path}")
    
    def _plot_model_comparison(self, output_path: Path) -> None:
        """Plot model performance comparison."""
        if not self.validation_results:
            logger.warning("No validation results available for plotting")
            return
        
        # Extract metrics for comparison
        models = list(self.validation_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Create comparison dataframe
        comparison_data = []
        for model in models:
            row = {'model': model}
            for metric in metrics:
                row[metric] = self.validation_results[model]['metrics'].get(metric, 0)
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        df_comparison.set_index('model')[metrics].plot(kind='bar')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / "model_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison plot saved to {plot_path}")
    
    def _plot_confusion_matrices(self, output_path: Path) -> None:
        """Plot confusion matrices for all models."""
        if not self.validation_results:
            return
        
        n_models = len(self.validation_results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (model_name, results) in enumerate(self.validation_results.items()):
            if i >= len(axes):
                break
                
            cm = np.array(results['metrics']['confusion_matrix'])
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / "confusion_matrices.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrices plot saved to {plot_path}")
    
    def _plot_training_history(self, output_path: Path) -> None:
        """Plot training history if available."""
        # This would plot training curves, loss functions, etc.
        # Implementation depends on available training history data
        pass
    
    def save_validation_results(self, output_dir: str) -> None:
        """Save validation results to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save validation results
        results_path = output_path / "validation_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for model_name, results in self.validation_results.items():
            results_serializable[model_name] = {
                'metrics': results['metrics'],
                'predictions': results['predictions'],
                'probabilities': results['probabilities']
            }
        
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Validation results saved to {results_path}")
