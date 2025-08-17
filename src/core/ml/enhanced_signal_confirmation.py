"""
Enhanced Machine Learning Signal Confirmation System

Advanced ML-based confirmation with:
- Ensemble learning with multiple algorithms
- Feature selection and engineering
- Real-time model adaptation
- Out-of-sample validation
- Confidence scoring and uncertainty quantification
"""

import os
from datetime import UTC, datetime

import joblib
import numpy as np
from scipy import stats
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from ..data_models.market import Bar


class EnhancedMLSignalConfirmation:
    """
    Enhanced ML-based signal confirmation system with advanced features
    """

    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        self.model_weights = {}
        self.adaptation_history = []

        # Enhanced ML model parameters
        self.ensemble_config = config.get(
            "ensemble",
            {"voting_method": "soft", "use_weights": True, "cross_validation_folds": 5},
        )

        # Advanced model configurations
        self.model_configs = {
            "random_forest": config.get(
                "random_forest",
                {
                    "n_estimators": 200,
                    "max_depth": 15,
                    "min_samples_split": 3,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "random_state": 42,
                    "class_weight": "balanced",
                },
            ),
            "gradient_boosting": config.get(
                "gradient_boosting",
                {
                    "n_estimators": 200,
                    "max_depth": 8,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "random_state": 42,
                },
            ),
            "adaboost": config.get(
                "adaboost",
                {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
            ),
            "extra_trees": config.get(
                "extra_trees",
                {
                    "n_estimators": 200,
                    "max_depth": 15,
                    "min_samples_split": 3,
                    "random_state": 42,
                },
            ),
            "neural_network": config.get(
                "neural_network",
                {
                    "hidden_layer_sizes": (100, 50, 25),
                    "activation": "relu",
                    "solver": "adam",
                    "alpha": 0.001,
                    "max_iter": 1000,
                    "random_state": 42,
                },
            ),
            "svm": config.get(
                "svm",
                {
                    "C": 1.0,
                    "kernel": "rbf",
                    "gamma": "scale",
                    "probability": True,
                    "random_state": 42,
                },
            ),
        }

        # Feature engineering parameters
        self.feature_config = config.get(
            "feature_engineering",
            {
                "lookback_periods": [3, 5, 8, 13, 21, 34, 55],
                "use_technical_indicators": True,
                "use_market_microstructure": True,
                "use_sentiment_features": False,
                "feature_selection_method": "mutual_info",
                "max_features": 50,
            },
        )

        # Confidence and uncertainty parameters
        self.confidence_config = config.get(
            "confidence",
            {
                "confidence_threshold": 0.7,
                "uncertainty_threshold": 0.3,
                "ensemble_variance_threshold": 0.2,
                "use_bayesian_uncertainty": True,
            },
        )

        # Real-time adaptation parameters
        self.adaptation_config = config.get(
            "adaptation",
            {
                "online_learning": True,
                "adaptation_frequency": 100,  # Adapt every 100 predictions
                "forgetting_factor": 0.95,  # Exponential forgetting
                "performance_threshold": 0.6,  # Minimum performance to adapt
            },
        )

        print("🤖 Enhanced ML Signal Confirmation initialized")
        print(f"   Ensemble Method: {self.ensemble_config['voting_method']}")
        print(f"   Models: {len(self.model_configs)} algorithms")
        print(
            f"   Feature Selection: {self.feature_config['feature_selection_method']}"
        )
        print(f"   Online Learning: {self.adaptation_config['online_learning']}")

    def engineer_advanced_features(
        self, bars: list[Bar], indicators: dict
    ) -> np.ndarray:
        """
        Engineer comprehensive features using advanced techniques
        """
        if len(bars) < max(self.feature_config["lookback_periods"]):
            return np.array([])

        features = []
        feature_names = []

        # Extract price and volume data
        closes = np.array([float(bar.close) for bar in bars])
        highs = np.array([float(bar.high) for bar in bars])
        lows = np.array([float(bar.low) for bar in bars])
        volumes = np.array([float(bar.volume) for bar in bars])

        # Basic technical indicators
        if self.feature_config["use_technical_indicators"]:
            features.extend(
                [
                    indicators.get("rsi", 50),
                    indicators.get("adx", 25),
                    indicators.get("volatility", 0.02),
                    indicators.get("volume_ratio", 1.0),
                ]
            )
            feature_names.extend(["rsi", "adx", "volatility", "volume_ratio"])

        # Advanced price momentum features
        for period in self.feature_config["lookback_periods"]:
            if len(closes) >= period:
                # Price momentum and acceleration
                momentum = (closes[-1] - closes[-period]) / closes[-period]
                features.append(momentum)
                feature_names.append(f"momentum_{period}")

                # Volatility features
                returns = np.diff(closes[-period:]) / closes[-period:-1]
                volatility = np.std(returns) if len(returns) > 0 else 0
                features.append(volatility)
                feature_names.append(f"volatility_{period}")

                # Volume momentum and profile
                volume_momentum = (volumes[-1] - np.mean(volumes[-period:])) / np.mean(
                    volumes[-period:]
                )
                features.append(volume_momentum)
                feature_names.append(f"volume_momentum_{period}")

                # High-Low spread and range
                hl_spread = (max(highs[-period:]) - min(lows[-period:])) / np.mean(
                    closes[-period:]
                )
                features.append(hl_spread)
                feature_names.append(f"hl_spread_{period}")

                # Price acceleration (second derivative)
                if period >= 10:
                    recent_momentum = (closes[-1] - closes[-period // 2]) / closes[
                        -period // 2
                    ]
                    past_momentum = (closes[-period // 2] - closes[-period]) / closes[
                        -period
                    ]
                    acceleration = recent_momentum - past_momentum
                    features.append(acceleration)
                    feature_names.append(f"acceleration_{period}")
                else:
                    features.append(0)
                    feature_names.append(f"acceleration_{period}")

                # Trend strength using linear regression
                if period >= 5:
                    x = np.arange(period)
                    y = closes[-period:]
                    if len(y) > 1:
                        slope = np.polyfit(x, y, 1)[0]
                        trend_strength = slope / np.mean(y) if np.mean(y) > 0 else 0
                    else:
                        trend_strength = 0
                    features.append(trend_strength)
                    feature_names.append(f"trend_strength_{period}")
                else:
                    features.append(0)
                    feature_names.append(f"trend_strength_{period}")
            else:
                # Fill with zeros if insufficient data
                features.extend([0, 0, 0, 0, 0, 0])
                feature_names.extend(
                    [
                        f"momentum_{period}",
                        f"volatility_{period}",
                        f"volume_momentum_{period}",
                        f"hl_spread_{period}",
                        f"acceleration_{period}",
                        f"trend_strength_{period}",
                    ]
                )

        # Market microstructure features
        if self.feature_config["use_market_microstructure"]:
            # Bid-ask spread approximation
            spread_estimate = (highs[-1] - lows[-1]) / closes[-1]
            features.append(spread_estimate)
            feature_names.append("spread_estimate")

            # Volume profile analysis
            volume_profile = (
                np.percentile(volumes[-20:], [25, 50, 75])
                if len(volumes) >= 20
                else [0, 0, 0]
            )
            features.extend(
                [
                    volume_profile[0] / np.mean(volumes[-20:])
                    if len(volumes) >= 20
                    else 0,
                    volume_profile[2] / np.mean(volumes[-20:])
                    if len(volumes) >= 20
                    else 0,
                ]
            )
            feature_names.extend(["volume_q25", "volume_q75"])

            # Price impact
            price_impact = (
                abs(closes[-1] - closes[-2]) / volumes[-1] if volumes[-1] > 0 else 0
            )
            features.append(price_impact)
            feature_names.append("price_impact")

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
        feature_names.extend(
            ["fast_ma", "slow_ma", "trend_ma", "short_ma", "medium_ma", "long_ma"]
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
        feature_names.append("market_regime")

        # Advanced statistical features
        if len(closes) >= 30:
            # Skewness and kurtosis
            returns = np.diff(closes[-30:]) / closes[-30:-1]
            if len(returns) > 2:
                skewness = stats.skew(returns)
                kurtosis = stats.kurtosis(returns)
            else:
                skewness, kurtosis = 0, 0
            features.extend([skewness, kurtosis])
            feature_names.extend(["skewness", "kurtosis"])

            # Hurst exponent (trend persistence)
            hurst = self._calculate_hurst_exponent(closes[-30:])
            features.append(hurst)
            feature_names.append("hurst_exponent")
        else:
            features.extend([0, 0, 0])
            feature_names.extend(["skewness", "kurtosis", "hurst_exponent"])

        # Cross-asset features (if available)
        if "correlation_features" in indicators:
            features.extend(indicators["correlation_features"])
            feature_names.extend(
                [f"corr_{i}" for i in range(len(indicators["correlation_features"]))]
            )

        # Store feature names for interpretability
        self.feature_names = feature_names

        return np.array(features).reshape(1, -1)

    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent for trend persistence"""
        if len(prices) < 10:
            return 0.5

        try:
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]

            # Calculate cumulative sum
            cumsum = np.cumsum(returns)

            # Calculate range and standard deviation for different lags
            lags = range(2, min(20, len(returns) // 2))
            rs_values = []

            for lag in lags:
                # Split data into chunks
                chunks = len(returns) // lag
                if chunks < 2:
                    continue

                rs_chunk = []
                for i in range(chunks):
                    chunk = cumsum[i * lag : (i + 1) * lag]
                    if len(chunk) > 1:
                        R = np.max(chunk) - np.min(chunk)
                        S = np.std(returns[i * lag : (i + 1) * lag])
                        if S > 0:
                            rs_chunk.append(R / S)

                if rs_chunk:
                    rs_values.append(np.mean(rs_chunk))

            if len(rs_values) > 1:
                # Calculate Hurst exponent using log-log regression
                log_lags = np.log(lags[: len(rs_values)])
                log_rs = np.log(rs_values)

                if len(log_lags) > 1:
                    slope = np.polyfit(log_lags, log_rs, 1)[0]
                    return slope
                else:
                    return 0.5
            else:
                return 0.5
        except:
            return 0.5

    def prepare_enhanced_training_data(
        self, historical_data: list[tuple], lookback_days: int = 60
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare enhanced training data with better labeling
        """
        X, y = [], []

        for i in range(lookback_days, len(historical_data)):
            # Get historical bars and indicators
            bars = historical_data[i - lookback_days : i]
            current_indicators = bars[-1][1]  # Latest indicators

            # Engineer features
            features = self.engineer_advanced_features(
                [bar[0] for bar in bars], current_indicators
            )
            if features.size == 0:
                continue

            # Enhanced labeling strategy
            if i + 10 < len(historical_data):
                future_prices = [
                    float(historical_data[j][0].close)
                    for j in range(i, min(i + 10, len(historical_data)))
                ]
                current_price = float(bars[-1][0].close)

                # Calculate multiple future returns
                returns_5 = (
                    (future_prices[4] - current_price) / current_price
                    if len(future_prices) > 4
                    else 0
                )
                returns_10 = (
                    (future_prices[-1] - current_price) / current_price
                    if len(future_prices) > 9
                    else 0
                )

                # Multi-horizon labeling
                label_5 = 1 if returns_5 > 0.005 else 0  # 0.5% threshold for 5-period
                label_10 = 1 if returns_10 > 0.01 else 0  # 1% threshold for 10-period

                # Combined label (weighted average)
                label = 0.7 * label_5 + 0.3 * label_10
                label = 1 if label > 0.5 else 0
            else:
                label = 0

            X.append(features.flatten())
            y.append(label)

        return np.array(X), np.array(y)

    def train_enhanced_models(
        self, historical_data: list[tuple], validation_split: float = 0.2
    ):
        """
        Train enhanced ensemble of ML models
        """
        print("🤖 Training Enhanced ML models for signal confirmation...")

        # Prepare training data
        X, y = self.prepare_enhanced_training_data(historical_data)

        if len(X) < self.config.get("min_training_samples", 100):
            print(
                f"⚠️ Insufficient training data. Need at least {self.config.get('min_training_samples', 100)} samples."
            )
            return False

        # Split data (time series split)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Feature selection
        print("🔍 Performing feature selection...")
        X_train_selected, X_val_selected, selected_features = self._select_features(
            X_train, X_val, y_train
        )

        # Scale features
        scaler = RobustScaler()  # More robust to outliers
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)

        self.scalers["main"] = scaler
        self.feature_selectors["main"] = selected_features

        # Train individual models
        print("🏗️ Training individual models...")
        individual_models = {}

        for name, config in self.model_configs.items():
            try:
                print(f"   Training {name}...")
                model_class = self._get_model_class(name)
                model = model_class(**config)
                model.fit(X_train_scaled, y_train)
                individual_models[name] = model

                # Evaluate individual model
                y_pred = model.predict(X_val_scaled)
                accuracy = accuracy_score(y_val, y_pred)
                print(f"     {name} accuracy: {accuracy:.3f}")

            except Exception as e:
                print(f"     ⚠️ {name} training failed: {e}")
                continue

        if not individual_models:
            print("❌ No models trained successfully")
            return False

        # Create ensemble
        print("🎯 Creating ensemble model...")
        if len(individual_models) > 1:
            # Calculate model weights based on validation performance
            model_weights = {}
            for name, model in individual_models.items():
                y_pred = model.predict(X_val_scaled)
                accuracy = accuracy_score(y_val, y_pred)
                model_weights[name] = accuracy

            # Normalize weights
            total_weight = sum(model_weights.values())
            if total_weight > 0:
                model_weights = {k: v / total_weight for k, v in model_weights.items()}
            else:
                model_weights = {
                    k: 1.0 / len(model_weights) for k in model_weights.keys()
                }

            self.model_weights = model_weights

            # Create voting classifier
            estimators = [(name, model) for name, model in individual_models.items()]
            ensemble = VotingClassifier(
                estimators=estimators,
                voting=self.ensemble_config["voting_method"],
                weights=list(model_weights.values())
                if self.ensemble_config["use_weights"]
                else None,
            )

            # Train ensemble
            ensemble.fit(X_train_scaled, y_train)
            self.models["ensemble"] = ensemble

        else:
            # Single model case
            name = list(individual_models.keys())[0]
            self.models["ensemble"] = individual_models[name]
            self.model_weights = {name: 1.0}

        # Store individual models for analysis
        self.models.update(individual_models)

        # Evaluate ensemble
        self._evaluate_enhanced_models(X_val_scaled, y_val)

        # Store feature importance
        self._store_feature_importance(individual_models, selected_features)

        print("✅ Enhanced ML models trained successfully!")
        return True

    def _get_model_class(self, model_name: str):
        """Get model class by name"""
        model_classes = {
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
            "adaboost": AdaBoostClassifier,
            "extra_trees": ExtraTreesClassifier,
            "neural_network": MLPClassifier,
            "svm": SVC,
        }
        return model_classes.get(model_name, RandomForestClassifier)

    def _select_features(
        self, X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, object]:
        """Perform feature selection"""
        method = self.feature_config["feature_selection_method"]
        max_features = min(self.feature_config["max_features"], X_train.shape[1])

        if method == "mutual_info":
            selector = SelectKBest(score_func=f_classif, k=max_features)
        elif method == "rfe":
            base_model = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator=base_model, n_features_to_select=max_features)
        else:
            # No feature selection
            return X_train, X_val, None

        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)

        return X_train_selected, X_val_selected, selector

    def _evaluate_enhanced_models(self, X_val: np.ndarray, y_val: np.ndarray):
        """Evaluate enhanced model performance"""
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_val)
                y_pred_proba = (
                    model.predict_proba(X_val)[:, 1]
                    if hasattr(model, "predict_proba")
                    else None
                )

                metrics = {
                    "accuracy": accuracy_score(y_val, y_pred),
                    "precision": precision_score(y_val, y_pred, zero_division=0),
                    "recall": recall_score(y_val, y_pred, zero_division=0),
                    "f1": f1_score(y_val, y_pred, zero_division=0),
                }

                if y_pred_proba is not None:
                    metrics["roc_auc"] = roc_auc_score(y_val, y_pred_proba)

                self.performance_metrics[name] = metrics

                print(f"📊 {name} Performance:")
                for metric, value in metrics.items():
                    print(f"   {metric.capitalize()}: {value:.3f}")

            except Exception as e:
                print(f"⚠️ Error evaluating {name}: {e}")

    def _store_feature_importance(self, models: dict, feature_selector):
        """Store feature importance for interpretability"""
        if feature_selector is None:
            return

        for name, model in models.items():
            if hasattr(model, "feature_importances_"):
                # Map selected features back to original feature names
                if hasattr(feature_selector, "get_support"):
                    selected_indices = feature_selector.get_support()
                    if hasattr(self, "feature_names"):
                        selected_names = [
                            self.feature_names[i]
                            for i, selected in enumerate(selected_indices)
                            if selected
                        ]
                        importances = model.feature_importances_
                        self.feature_importance[name] = dict(
                            zip(selected_names, importances, strict=False)
                        )

    def confirm_signal_enhanced(
        self, bars: list[Bar], indicators: dict
    ) -> tuple[bool, float, dict]:
        """
        Enhanced signal confirmation with uncertainty quantification
        """
        if not self.models or "ensemble" not in self.models:
            return True, 1.0, {"ml_available": False}

        # Engineer features
        features = self.engineer_advanced_features(bars, indicators)
        if features.size == 0:
            return True, 1.0, {"ml_available": False}

        # Apply feature selection
        if (
            "main" in self.feature_selectors
            and self.feature_selectors["main"] is not None
        ):
            features = self.feature_selectors["main"].transform(features)

        # Scale features
        features_scaled = self.scalers["main"].transform(features)

        # Get ensemble prediction
        ensemble_model = self.models["ensemble"]
        ensemble_prediction = ensemble_model.predict(features_scaled)[0]
        ensemble_confidence = (
            ensemble_model.predict_proba(features_scaled)[0, 1]
            if hasattr(ensemble_model, "predict_proba")
            else 0.5
        )

        # Get individual model predictions for uncertainty quantification
        individual_predictions = {}
        individual_confidences = {}
        prediction_variance = 0.0

        for name, model in self.models.items():
            if name != "ensemble":
                try:
                    pred = model.predict(features_scaled)[0]
                    conf = (
                        model.predict_proba(features_scaled)[0, 1]
                        if hasattr(model, "predict_proba")
                        else 0.5
                    )

                    individual_predictions[name] = pred
                    individual_confidences[name] = conf

                    # Calculate prediction variance
                    prediction_variance += (pred - ensemble_prediction) ** 2

                except Exception as e:
                    print(f"⚠️ Error getting {name} prediction: {e}")

        # Normalize prediction variance
        if len(individual_predictions) > 1:
            prediction_variance /= len(individual_predictions)

        # Uncertainty quantification
        uncertainty = min(1.0, prediction_variance / 0.25)  # Normalize to [0, 1]

        # Signal confirmation logic with uncertainty
        confidence_threshold = self.confidence_config["confidence_threshold"]
        uncertainty_threshold = self.confidence_config["uncertainty_threshold"]

        # Adjust confidence based on uncertainty
        adjusted_confidence = ensemble_confidence * (1 - uncertainty)

        # Signal confirmation decision
        signal_confirmed = (
            ensemble_prediction > 0.5
            and adjusted_confidence > confidence_threshold
            and uncertainty < uncertainty_threshold
        )

        # Additional confidence boost for strong signals
        if adjusted_confidence > 0.8 and uncertainty < 0.1:
            signal_confirmed = True

        # Store adaptation data
        self._store_adaptation_data(
            ensemble_prediction, adjusted_confidence, uncertainty, signal_confirmed
        )

        result = {
            "ml_available": True,
            "ensemble_prediction": ensemble_prediction,
            "ensemble_confidence": ensemble_confidence,
            "adjusted_confidence": adjusted_confidence,
            "uncertainty": uncertainty,
            "individual_predictions": individual_predictions,
            "individual_confidences": individual_confidences,
            "prediction_variance": prediction_variance,
            "signal_confirmed": signal_confirmed,
            "confidence_threshold": confidence_threshold,
            "uncertainty_threshold": uncertainty_threshold,
            "model_weights": self.model_weights,
        }

        return signal_confirmed, adjusted_confidence, result

    def _store_adaptation_data(
        self, prediction: float, confidence: float, uncertainty: float, confirmed: bool
    ):
        """Store data for online adaptation"""
        self.adaptation_history.append(
            {
                "timestamp": datetime.now(UTC),
                "prediction": prediction,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "confirmed": confirmed,
            }
        )

        # Keep only recent history
        if len(self.adaptation_history) > 1000:
            self.adaptation_history = self.adaptation_history[-1000:]

        # Check if adaptation is needed
        if (
            self.adaptation_config["online_learning"]
            and len(self.adaptation_history)
            >= self.adaptation_config["adaptation_frequency"]
        ):
            self._adapt_models()

    def _adapt_models(self):
        """Adapt models based on recent performance"""
        if len(self.adaptation_history) < 100:
            return

        # Calculate recent performance
        recent_history = self.adaptation_history[-100:]
        recent_accuracy = sum(1 for h in recent_history if h["confirmed"]) / len(
            recent_history
        )

        if recent_accuracy < self.adaptation_config["performance_threshold"]:
            print("🔄 Model adaptation triggered due to poor performance")

            # Simple adaptation: adjust confidence threshold
            if recent_accuracy < 0.4:
                self.confidence_config["confidence_threshold"] = min(
                    0.9, self.confidence_config["confidence_threshold"] + 0.05
                )
            elif recent_accuracy < 0.6:
                self.confidence_config["confidence_threshold"] = min(
                    0.9, self.confidence_config["confidence_threshold"] + 0.02
                )

            print(
                f"   Adjusted confidence threshold to: {self.confidence_config['confidence_threshold']:.3f}"
            )

    def save_enhanced_models(self, filepath: str):
        """Save enhanced models and metadata"""
        if not self.models:
            print("⚠️ No models to save")
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save models
        for name, model in self.models.items():
            model_path = f"{filepath}_{name}.joblib"
            joblib.dump(model, model_path)
            print(f"💾 Saved {name} model to {model_path}")

        # Save scalers and feature selectors
        for name, scaler in self.scalers.items():
            scaler_path = f"{filepath}_{name}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            print(f"💾 Saved {name} scaler to {scaler_path}")

        for name, selector in self.feature_selectors.items():
            if selector is not None:
                selector_path = f"{filepath}_{name}_selector.joblib"
                joblib.dump(selector, selector_path)
                print(f"💾 Saved {name} selector to {selector_path}")

        # Save metadata
        metadata = {
            "feature_names": getattr(self, "feature_names", []),
            "model_weights": self.model_weights,
            "performance_metrics": self.performance_metrics,
            "feature_importance": self.feature_importance,
            "adaptation_history": self.adaptation_history[-100:],  # Keep last 100
        }

        metadata_path = f"{filepath}_metadata.joblib"
        joblib.dump(metadata, metadata_path)
        print(f"💾 Saved metadata to {metadata_path}")

    def load_enhanced_models(self, filepath: str):
        """Load enhanced models and metadata"""
        try:
            # Load models
            for name in [
                "ensemble",
                "random_forest",
                "gradient_boosting",
                "adaboost",
                "extra_trees",
                "neural_network",
                "svm",
            ]:
                model_path = f"{filepath}_{name}.joblib"
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                    print(f"📂 Loaded {name} model from {model_path}")

            # Load scalers
            scaler_path = f"{filepath}_main_scaler.joblib"
            if os.path.exists(scaler_path):
                self.scalers["main"] = joblib.load(scaler_path)
                print(f"📂 Loaded main scaler from {scaler_path}")

            # Load feature selectors
            selector_path = f"{filepath}_main_selector.joblib"
            if os.path.exists(selector_path):
                self.feature_selectors["main"] = joblib.load(selector_path)
                print(f"📂 Loaded main selector from {selector_path}")

            # Load metadata
            metadata_path = f"{filepath}_metadata.joblib"
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.feature_names = metadata.get("feature_names", [])
                self.model_weights = metadata.get("model_weights", {})
                self.performance_metrics = metadata.get("performance_metrics", {})
                self.feature_importance = metadata.get("feature_importance", {})
                self.adaptation_history = metadata.get("adaptation_history", [])
                print(f"📂 Loaded metadata from {metadata_path}")

            return len(self.models) > 0
        except Exception as e:
            print(f"❌ Error loading enhanced models: {e}")
            return False
