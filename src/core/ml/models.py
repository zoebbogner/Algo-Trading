"""Machine Learning Models for Trading Signal Generation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MLSignalGenerator:
    """Machine Learning-based trading signal generator."""
    
    def __init__(self, config: Dict):
        """Initialize ML signal generator."""
        self.config = config
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.feature_columns = []
        
        # Model configuration
        self.model_config = config.get('ml_models', {})
        self.ensemble_method = self.model_config.get('ensemble_method', 'voting')
        self.confidence_threshold = self.model_config.get('confidence_threshold', 0.6)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize individual ML models."""
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Logistic Regression
        self.models['logistic_regression'] = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )
        
        # Support Vector Machine
        self.models['svm'] = SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=42
        )
        
        # Neural Network
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        
        # Create ensemble
        if self.ensemble_method == 'voting':
            self.models['ensemble'] = VotingClassifier(
                estimators=[
                    ('rf', self.models['random_forest']),
                    ('gb', self.models['gradient_boosting']),
                    ('lr', self.models['logistic_regression']),
                    ('svm', self.models['svm']),
                    ('nn', self.models['neural_network'])
                ],
                voting='soft'
            )
        
        # Initialize scalers for each model
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def prepare_training_data(self, df: pd.DataFrame, target_column: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models."""
        # Ensure target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        # Get feature columns (exclude metadata and target)
        metadata_cols = ['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume', target_column]
        self.feature_columns = [col for col in df.columns if col not in metadata_cols]
        
        # Prepare features and target
        X = df[self.feature_columns].values
        y = df[target_column].values
        
        # Remove any rows with NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train_models(self, df: pd.DataFrame, target_column: str = 'target') -> None:
        """Train all ML models."""
        logger.info("Training ML models...")
        
        # Prepare training data
        X, y = self.prepare_training_data(df, target_column)
        
        # Train each model
        for model_name, model in self.models.items():
            if model_name == 'ensemble':
                continue  # Ensemble will be trained after individual models
                
            logger.info(f"Training {model_name}...")
            
            # Scale features
            X_scaled = self.scalers[model_name].fit_transform(X)
            
            # Train model
            model.fit(X_scaled, y)
            
            # Calculate training accuracy
            train_accuracy = model.score(X_scaled, y)
            logger.info(f"{model_name} training accuracy: {train_accuracy:.4f}")
        
        # Train ensemble if using voting
        if self.ensemble_method == 'voting':
            logger.info("Training ensemble model...")
            # Use scaled features from one of the scalers (they should be similar)
            X_scaled = self.scalers['random_forest'].transform(X)
            self.models['ensemble'].fit(X_scaled, y)
            
            # Calculate ensemble accuracy
            ensemble_accuracy = self.models['ensemble'].score(X_scaled, y)
            logger.info(f"Ensemble training accuracy: {ensemble_accuracy:.4f}")
        
        self.is_trained = True
        logger.info("All models trained successfully")
    
    def generate_signals(self, df: pd.DataFrame, use_ensemble: bool = True) -> pd.DataFrame:
        """Generate trading signals using trained ML models."""
        if not self.is_trained:
            raise ValueError("Models not trained. Run train_models first.")
        
        logger.info("Generating ML trading signals...")
        
        # Prepare features
        X = df[self.feature_columns].values
        
        # Remove any rows with NaN values
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]
        df_clean = df[mask].copy()
        
        # Generate predictions from each model
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models.items():
            if model_name == 'ensemble' and not use_ensemble:
                continue
                
            # Scale features
            X_scaled = self.scalers[model_name].transform(X_clean)
            
            # Get predictions and probabilities
            pred = model.predict(X_scaled)
            prob = model.predict_proba(X_scaled)
            
            predictions[f'{model_name}_pred'] = pred
            probabilities[f'{model_name}_prob'] = prob[:, 1]  # Probability of positive class
        
        # Add predictions to dataframe
        for col, pred in predictions.items():
            df_clean[col] = pred
        
        # Add probabilities to dataframe
        for col, prob in probabilities.items():
            df_clean[col] = prob
        
        # Generate ensemble signal
        if use_ensemble and 'ensemble' in self.models:
            df_clean['ml_signal'] = self._generate_ensemble_signal(df_clean)
        else:
            df_clean['ml_signal'] = self._generate_individual_signal(df_clean)
        
        # Add confidence score
        df_clean['ml_confidence'] = self._calculate_confidence(df_clean)
        
        logger.info(f"Generated ML signals for {len(df_clean)} samples")
        return df_clean
    
    def _generate_ensemble_signal(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signal using ensemble model."""
        # Get ensemble prediction
        signal = df['ensemble_pred'].map({0: 'sell', 1: 'buy', 2: 'hold'})
        
        # Apply confidence threshold
        confidence = df['ensemble_prob']
        signal[confidence < self.confidence_threshold] = 'hold'
        
        return signal
    
    def _generate_individual_signal(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signal using individual model consensus."""
        # Get predictions from all models (excluding ensemble)
        pred_columns = [col for col in df.columns if col.endswith('_pred') and 'ensemble' not in col]
        
        # Count buy/sell votes
        buy_votes = (df[pred_columns] == 1).sum(axis=1)
        sell_votes = (df[pred_columns] == 0).sum(axis=1)
        
        # Generate signal based on majority vote
        signal = pd.Series('hold', index=df.index)
        signal[buy_votes > sell_votes] = 'buy'
        signal[sell_votes > buy_votes] = 'sell'
        
        # Apply confidence threshold
        avg_confidence = df[[col for col in df.columns if col.endswith('_prob') and 'ensemble' not in col]].mean(axis=1)
        signal[avg_confidence < self.confidence_threshold] = 'hold'
        
        return signal
    
    def _calculate_confidence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate confidence score for predictions."""
        prob_columns = [col for col in df.columns if col.endswith('_prob')]
        
        if prob_columns:
            # Use average probability as confidence
            confidence = df[prob_columns].mean(axis=1)
        else:
            # Use prediction consistency as confidence
            pred_columns = [col for col in df.columns if col.endswith('_pred') and 'ensemble' not in col]
            if pred_columns:
                # Calculate how many models agree
                mode_pred = df[pred_columns].mode(axis=1)[0]
                agreement = (df[pred_columns] == mode_pred).sum(axis=1)
                confidence = agreement / len(pred_columns)
            else:
                confidence = pd.Series(0.5, index=df.index)
        
        return confidence
    
    def save_models(self, output_dir: str) -> None:
        """Save trained models and scalers."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = output_path / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = output_path / f"{scaler_name}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved {scaler_name} scaler to {scaler_path}")
        
        # Save feature columns
        features_path = output_path / "feature_columns.txt"
        with open(features_path, 'w') as f:
            for col in self.feature_columns:
                f.write(f"{col}\n")
        logger.info(f"Saved feature columns to {features_path}")
    
    def load_models(self, model_dir: str) -> None:
        """Load trained models and scalers."""
        model_path = Path(model_dir)
        
        # Load models
        for model_name in self.models.keys():
            model_file = model_path / f"{model_name}.joblib"
            if model_file.exists():
                self.models[model_name] = joblib.load(model_file)
                logger.info(f"Loaded {model_name} from {model_file}")
        
        # Load scalers
        for scaler_name in self.scalers.keys():
            scaler_file = model_path / f"{scaler_name}_scaler.joblib"
            if scaler_file.exists():
                self.scalers[scaler_name] = joblib.load(scaler_file)
                logger.info(f"Loaded {scaler_name} scaler from {scaler_file}")
        
        # Load feature columns
        features_file = model_path / "feature_columns.txt"
        if features_file.exists():
            with open(features_file, 'r') as f:
                self.feature_columns = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(self.feature_columns)} feature columns")
        
        self.is_trained = True


class EnsembleMLStrategy:
    """Ensemble ML strategy combining multiple approaches."""
    
    def __init__(self, config: Dict):
        """Initialize ensemble ML strategy."""
        self.config = config
        self.ml_signal_generator = MLSignalGenerator(config)
        self.strategy_weights = config.get('strategy_weights', {
            'ml': 0.6,
            'technical': 0.3,
            'sentiment': 0.1
        })
    
    def generate_ensemble_signals(self, df: pd.DataFrame, technical_signals: pd.Series) -> pd.DataFrame:
        """Generate ensemble signals combining ML and technical analysis."""
        # Generate ML signals
        df_with_ml = self.ml_signal_generator.generate_signals(df)
        
        # Combine signals
        df_with_ml['ensemble_signal'] = self._combine_signals(
            df_with_ml['ml_signal'],
            technical_signals,
            df_with_ml['ml_confidence']
        )
        
        return df_with_ml
    
    def _combine_signals(self, ml_signals: pd.Series, technical_signals: pd.Series, ml_confidence: pd.Series) -> pd.Series:
        """Combine ML and technical signals with weighted approach."""
        # Convert signals to numeric
        signal_map = {'sell': -1, 'hold': 0, 'buy': 1}
        
        ml_numeric = ml_signals.map(signal_map)
        tech_numeric = technical_signals.map(signal_map)
        
        # Weighted combination
        ml_weight = self.strategy_weights['ml']
        tech_weight = self.strategy_weights['technical']
        
        # Adjust ML weight by confidence
        adjusted_ml_weight = ml_weight * ml_confidence
        
        # Combine signals
        combined_score = (adjusted_ml_weight * ml_numeric + tech_weight * tech_numeric) / (adjusted_ml_weight + tech_weight)
        
        # Convert back to signals
        ensemble_signals = pd.Series('hold', index=combined_score.index)
        ensemble_signals[combined_score > 0.2] = 'buy'
        ensemble_signals[combined_score < -0.2] = 'sell'
        
        return ensemble_signals
