"""
Machine Learning Signal Confirmation Module

This module provides ML-based confirmation for trading signals using:
- Feature engineering from technical indicators
- Ensemble methods (Random Forest, Gradient Boosting)
- Signal strength scoring and confidence levels
- Out-of-sample validation for unseen data
"""

import os

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from ..data_models.market import Bar


class MLSignalConfirmation:
    """
    Machine Learning-based signal confirmation system
    """

    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}

        # ML model parameters
        self.rf_params = config.get(
            "random_forest",
            {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
        )

        self.gb_params = config.get(
            "gradient_boosting",
            {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
            },
        )

        # Feature engineering parameters
        self.lookback_periods = config.get("lookback_periods", [5, 10, 20])
        self.confidence_threshold = config.get("confidence_threshold", 0.7)

        print(
            f"🤖 ML Signal Confirmation initialized with confidence threshold: {self.confidence_threshold}"
        )

    def engineer_features(self, bars: list[Bar], indicators: dict) -> np.ndarray:
        """
        Engineer comprehensive features for ML model training
        """
        if len(bars) < max(self.lookback_periods):
            return np.array([])

        features = []

        # Extract price and volume data
        closes = [float(bar.close) for bar in bars]
        highs = [float(bar.high) for bar in bars]
        lows = [float(bar.low) for bar in bars]
        volumes = [float(bar.volume) for bar in bars]

        # Current indicators
        features.extend(
            [
                indicators.get("rsi", 50),
                indicators.get("adx", 25),
                indicators.get("volatility", 0.02),
                indicators.get("volume_ratio", 1.0),
            ]
        )

        # Price momentum features
        for period in self.lookback_periods:
            if len(closes) >= period:
                # Price momentum
                momentum = (closes[-1] - closes[-period]) / closes[-period]
                features.append(momentum)

                # Volatility
                returns = np.diff(closes[-period:]) / closes[-period:-1]
                volatility = np.std(returns) if len(returns) > 0 else 0
                features.append(volatility)

                # Volume momentum
                volume_momentum = (volumes[-1] - np.mean(volumes[-period:])) / np.mean(
                    volumes[-period:]
                )
                features.append(volume_momentum)

                # High-Low spread
                hl_spread = (max(highs[-period:]) - min(lows[-period:])) / np.mean(
                    closes[-period:]
                )
                features.append(hl_spread)
            else:
                features.extend([0, 0, 0, 0])

        # Technical indicator features
        features.extend(
            [
                indicators.get("fast_ma", 0),
                indicators.get("slow_ma", 0),
                indicators.get("trend_ma", 0),
                indicators.get("short_ma", 0),
                indicators.get("medium_ma", 0),
                indicators.get("long_ma", 0),
            ]
        )

        # Market regime encoding
        market_regime = indicators.get("market_regime", "ranging")
        regime_encoding = {
            "strong_uptrend": 1.0,
            "weak_trend": 0.5,
            "ranging": 0.0,
            "weak_downtrend": -0.5,
            "strong_downtrend": -1.0,
        }
        features.append(regime_encoding.get(market_regime, 0.0))

        # Trend strength features
        if len(closes) >= 20:
            # Linear trend strength
            x = np.arange(len(closes[-20:]))
            y = np.array(closes[-20:])
            if len(y) > 1:
                slope = np.polyfit(x, y, 1)[0]
                trend_strength = slope / np.mean(y) if np.mean(y) > 0 else 0
            else:
                trend_strength = 0
            features.append(trend_strength)

            # Price acceleration
            if len(closes) >= 40:
                recent_momentum = (closes[-1] - closes[-20]) / closes[-20]
                past_momentum = (closes[-20] - closes[-40]) / closes[-40]
                acceleration = recent_momentum - past_momentum
                features.append(acceleration)
            else:
                features.append(0)
        else:
            features.extend([0, 0])

        return np.array(features).reshape(1, -1)

    def prepare_training_data(
        self, historical_data: list[tuple], lookback_days: int = 30
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from historical market data
        """
        X, y = [], []

        for i in range(lookback_days, len(historical_data)):
            # Get historical bars and indicators
            bars = historical_data[i - lookback_days : i]
            current_indicators = bars[-1][1]  # Latest indicators

            # Engineer features
            features = self.engineer_features(
                [bar[0] for bar in bars], current_indicators
            )
            if features.size == 0:
                continue

            # Create labels (1 for profitable trade, 0 for loss)
            # This is a simplified approach - in practice, you'd use actual trade outcomes
            if i + 5 < len(historical_data):
                future_price = float(historical_data[i + 5][0].close)
                current_price = float(bars[-1][0].close)
                price_change = (future_price - current_price) / current_price

                # Label: 1 if price increases by 1%, 0 otherwise
                label = 1 if price_change > 0.01 else 0
            else:
                label = 0

            X.append(features.flatten())
            y.append(label)

        return np.array(X), np.array(y)

    def train_models(self, historical_data: list[tuple], validation_split: float = 0.2):
        """
        Train ML models on historical data
        """
        print("🤖 Training ML models for signal confirmation...")

        # Prepare training data
        X, y = self.prepare_training_data(historical_data)

        if len(X) < 100:
            print("⚠️ Insufficient training data. Need at least 100 samples.")
            return False

        # Split data (time series split)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        self.scalers["main"] = scaler

        # Train Random Forest
        print("🌲 Training Random Forest...")
        rf_model = RandomForestClassifier(**self.rf_params)
        rf_model.fit(X_train_scaled, y_train)

        # Train Gradient Boosting
        print("📈 Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(**self.gb_params)
        gb_model.fit(X_train_scaled, y_train)

        # Store models
        self.models["random_forest"] = rf_model
        self.models["gradient_boosting"] = gb_model

        # Evaluate models
        self._evaluate_models(X_val_scaled, y_val)

        # Store feature importance
        self.feature_importance["random_forest"] = rf_model.feature_importances_
        self.feature_importance["gradient_boosting"] = gb_model.feature_importances_

        print("✅ ML models trained successfully!")
        return True

    def _evaluate_models(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Evaluate model performance on validation set
        """
        for name, model in self.models.items():
            y_pred = model.predict(X_val)

            metrics = {
                "accuracy": accuracy_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred, zero_division=0),
                "f1": f1_score(y_val, y_pred, zero_division=0),
            }

            self.performance_metrics[name] = metrics

            print(f"📊 {name} Performance:")
            print(f"   Accuracy: {metrics['accuracy']:.3f}")
            print(f"   Precision: {metrics['precision']:.3f}")
            print(f"   Recall: {metrics['recall']:.3f}")
            print(f"   F1-Score: {metrics['f1']:.3f}")

    def confirm_signal(
        self, bars: list[Bar], indicators: dict
    ) -> tuple[bool, float, dict]:
        """
        Confirm trading signal using ML models
        """
        if not self.models:
            return True, 1.0, {"ml_available": False}

        # Engineer features
        features = self.engineer_features(bars, indicators)
        if features.size == 0:
            return True, 1.0, {"ml_available": False}

        # Scale features
        features_scaled = self.scalers["main"].transform(features)

        # Get predictions from all models
        predictions = {}
        confidences = {}

        for name, model in self.models.items():
            # Get prediction probability
            proba = model.predict_proba(features_scaled)[0]
            prediction = model.predict(features_scaled)[0]

            predictions[name] = prediction
            confidences[name] = (
                proba[1] if len(proba) > 1 else proba[0]
            )  # Probability of positive class

        # Ensemble decision
        avg_confidence = np.mean(list(confidences.values()))
        ensemble_prediction = np.mean(list(predictions.values()))

        # Signal confirmation logic
        signal_confirmed = (
            ensemble_prediction > 0.5 and avg_confidence > self.confidence_threshold
        )

        # Additional confidence boost for strong signals
        if avg_confidence > 0.8:
            signal_confirmed = True

        result = {
            "ml_available": True,
            "ensemble_prediction": ensemble_prediction,
            "avg_confidence": avg_confidence,
            "individual_predictions": predictions,
            "individual_confidences": confidences,
            "signal_confirmed": signal_confirmed,
            "confidence_threshold": self.confidence_threshold,
        }

        return signal_confirmed, avg_confidence, result

    def save_models(self, filepath: str):
        """
        Save trained models to disk
        """
        if not self.models:
            print("⚠️ No models to save")
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save models
        for name, model in self.models.items():
            model_path = f"{filepath}_{name}.joblib"
            joblib.dump(model, model_path)
            print(f"💾 Saved {name} model to {model_path}")

        # Save scalers
        for name, scaler in self.scalers.items():
            scaler_path = f"{filepath}_{name}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            print(f"💾 Saved {name} scaler to {scaler_path}")

    def load_models(self, filepath: str):
        """
        Load trained models from disk
        """
        try:
            # Load models
            for name in ["random_forest", "gradient_boosting"]:
                model_path = f"{filepath}_{name}.joblib"
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                    print(f"📂 Loaded {name} model from {model_path}")

            # Load scalers
            scaler_path = f"{filepath}_main_scaler.joblib"
            if os.path.exists(scaler_path):
                self.scalers["main"] = joblib.load(scaler_path)
                print(f"📂 Loaded main scaler from {scaler_path}")

            return len(self.models) > 0
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
