"""
ULTRA-ENHANCED Momentum Trading Strategy with ML & Advanced Risk Management

This strategy incorporates:
- Multi-timeframe moving average analysis
- RSI divergence and momentum confirmation
- Volume profile analysis and trend strength indicators
- Market regime detection (trending vs ranging)
- Dynamic position sizing based on market conditions
- Portfolio heat mapping and correlation analysis
- Machine Learning signal confirmation
- Advanced risk modeling (VaR, Kelly Criterion)
- Portfolio optimization algorithms
- Out-of-sample validation for unseen data
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Optional

import numpy as np

from ..data_models.market import Bar, MarketData
from ..data_models.trading import Portfolio, Position
from .base import Strategy

# Import advanced modules
try:
    from ..ml.signal_confirmation import MLSignalConfirmation
    from ..optimization.portfolio_optimizer import PortfolioOptimizer
    from ..risk.advanced_modeling import AdvancedRiskModeling

    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False
    print("⚠️ Advanced features not available - using basic strategy")


class UltraEnhancedMomentumStrategy(Strategy):
    """
    Ultra-enhanced momentum strategy with ML confirmation and advanced risk management
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

        # Core parameters
        self.fast_ma_period = config.get("fast_ma_period", 8)
        self.slow_ma_period = config.get("slow_ma_period", 21)
        self.trend_ma_period = config.get("trend_ma_period", 50)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_oversold = config.get("rsi_oversold", 35)
        self.rsi_overbought = config.get("rsi_overbought", 65)
        self.adx_period = config.get("adx_period", 14)
        self.adx_threshold = config.get("adx_threshold", 25)
        self.volume_multiplier = config.get("volume_multiplier", 1.2)
        self.position_size = config.get("position_size", 0.08)
        self.stop_loss_pct = config.get("stop_loss_pct", 0.015)
        self.take_profit_pct = config.get("take_profit_pct", 0.03)
        self.trailing_stop_pct = config.get("trailing_stop_pct", 0.01)

        # Advanced parameters
        self.atr_period = config.get("atr_period", 14)
        self.volatility_threshold = config.get("volatility_threshold", 0.02)
        self.correlation_threshold = config.get("correlation_threshold", 0.7)
        self.max_correlation_exposure = config.get("max_correlation_exposure", 0.3)

        # Multi-timeframe parameters
        self.short_ma_period = config.get("short_ma_period", 5)
        self.medium_ma_period = config.get("medium_ma_period", 13)
        self.long_ma_period = config.get("long_ma_period", 34)

        # ML and Advanced Risk parameters
        self.use_ml_confirmation = config.get("use_ml_confirmation", True)
        self.use_advanced_risk = config.get("use_advanced_risk", True)
        self.use_portfolio_optimization = config.get("use_portfolio_optimization", True)

        # Initialize advanced modules if available
        self.ml_confirmation = None
        self.risk_modeling = None
        self.portfolio_optimizer = None

        if ADVANCED_FEATURES_AVAILABLE:
            if self.use_ml_confirmation:
                ml_config = config.get("ml_config", {})
                self.ml_confirmation = MLSignalConfirmation(ml_config)
                print("🤖 ML Signal Confirmation initialized")

            if self.use_advanced_risk:
                risk_config = config.get("risk_config", {})
                self.risk_modeling = AdvancedRiskModeling(risk_config)
                print("📊 Advanced Risk Modeling initialized")

            if self.use_portfolio_optimization:
                opt_config = config.get("optimization_config", {})
                self.portfolio_optimizer = PortfolioOptimizer(opt_config)
                print("🎯 Portfolio Optimizer initialized")

        # Historical data for ML training and risk modeling
        self.historical_data = []
        self.historical_returns = {}

        print(f"🚀 ULTRA-ENHANCED STRATEGY INITIALIZED: {self.name}")
        print(f"   Advanced Features: {'✅' if ADVANCED_FEATURES_AVAILABLE else '❌'}")
        print(f"   ML Confirmation: {'✅' if self.ml_confirmation else '❌'}")
        print(f"   Advanced Risk: {'✅' if self.risk_modeling else '❌'}")
        print(f"   Portfolio Optimization: {'✅' if self.portfolio_optimizer else '❌'}")

    def update_historical_data(self, bars: list[Bar], indicators: dict):
        """Update historical data for ML training and risk modeling"""
        if not ADVANCED_FEATURES_AVAILABLE:
            return

        # Store historical data
        self.historical_data.append((bars, indicators))

        # Keep only last 1000 data points to prevent memory issues
        if len(self.historical_data) > 1000:
            self.historical_data = self.historical_data[-1000:]

        # Calculate and store returns
        if len(bars) >= 2:
            current_price = float(bars[-1].close)
            previous_price = float(bars[-2].close)
            returns = (current_price - previous_price) / previous_price

            symbol = bars[0].symbol if bars else "unknown"
            if symbol not in self.historical_returns:
                self.historical_returns[symbol] = []

            self.historical_returns[symbol].append(returns)

            # Keep only last 500 returns
            if len(self.historical_returns[symbol]) > 500:
                self.historical_returns[symbol] = self.historical_returns[symbol][-500:]

    def train_ml_models(self):
        """Train ML models for signal confirmation"""
        if not self.ml_confirmation or len(self.historical_data) < 100:
            return False

        print("🤖 Training ML models for signal confirmation...")

        # Convert historical data to format expected by ML module
        ml_data = []
        for bars, indicators in self.historical_data:
            if bars:
                ml_data.append((bars, indicators))

        success = self.ml_confirmation.train_models(ml_data)

        if success:
            print("✅ ML models trained successfully!")
            # Save models for future use
            self.ml_confirmation.save_models("models/signal_confirmation")

        return success

    def calculate_advanced_risk_metrics(self, portfolio: Portfolio) -> dict:
        """Calculate advanced risk metrics using VaR and Kelly Criterion"""
        if not self.risk_modeling:
            return {}

        risk_metrics = {}

        # Calculate portfolio VaR
        if self.historical_returns:
            portfolio_var = self.risk_modeling.calculate_portfolio_var(
                portfolio, self.historical_returns
            )
            risk_metrics["portfolio_var"] = portfolio_var

        # Calculate Kelly Criterion for position sizing
        if hasattr(self, "trade_history") and self.trade_history:
            # Calculate win rate and average win/loss from trade history
            wins = [t for t in self.trade_history if t["pnl"] > 0]
            losses = [t for t in self.trade_history if t["pnl"] < 0]

            if wins and losses:
                win_rate = len(wins) / len(self.trade_history)
                avg_win = np.mean([t["pnl"] for t in wins])
                avg_loss = abs(np.mean([t["pnl"] for t in losses]))

                kelly_result = self.risk_modeling.calculate_kelly_criterion(
                    win_rate, avg_win, avg_loss
                )
                risk_metrics["kelly_criterion"] = kelly_result

        # Perform stress testing
        if self.historical_returns:
            stress_test = self.risk_modeling.stress_test_portfolio(
                portfolio, self.historical_returns
            )
            risk_metrics["stress_test"] = stress_test

        return risk_metrics

    def optimize_portfolio_allocation(self, current_prices: dict[str, float]) -> dict:
        """Optimize portfolio allocation using advanced optimization algorithms"""
        if not self.portfolio_optimizer or not self.historical_returns:
            return {}

        print("🎯 Optimizing portfolio allocation...")

        # Try different optimization methods
        optimization_results = {}

        methods = [
            "efficient_frontier",
            "risk_parity",
            "maximum_sharpe",
            "minimum_variance",
        ]

        for method in methods:
            try:
                result = self.portfolio_optimizer.optimize_portfolio(
                    self.historical_returns, method=method
                )
                if "error" not in result:
                    optimization_results[method] = result
            except Exception as e:
                print(f"⚠️ {method} optimization failed: {e}")

        # Find best optimization result
        best_result = None
        best_sharpe = -np.inf

        for method, result in optimization_results.items():
            if "optimal_portfolio" in result:
                sharpe = result["optimal_portfolio"].get("sharpe_ratio", -np.inf)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_result = result
            elif "sharpe_ratio" in result:
                sharpe = result["sharpe_ratio"]
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_result = result

        if best_result:
            print(
                f"✅ Best optimization: {best_result.get('method', 'unknown')} - Sharpe: {best_sharpe:.3f}"
            )

            # Calculate rebalancing trades
            if "weights" in best_result:
                target_weights = {}
                symbols = best_result.get("symbols", [])
                weights = best_result["weights"]

                for i, symbol in enumerate(symbols):
                    if i < len(weights):
                        target_weights[symbol] = weights[i]

                # Create a mock portfolio for rebalancing calculation
                mock_portfolio = Portfolio(
                    equity=Decimal("10000"),
                    cash=Decimal("10000"),
                    positions=[],
                    timestamp=datetime.now(UTC),
                )

                rebalancing = self.portfolio_optimizer.rebalance_portfolio(
                    mock_portfolio, target_weights, current_prices
                )

                best_result["rebalancing"] = rebalancing

        return best_result if best_result else {}

    def _calculate_indicators(self, bars: list[Bar]) -> dict[str, float]:
        """Calculate all technical indicators"""
        if len(bars) < max(self.trend_ma_period, self.adx_period, self.rsi_period):
            return {}

        closes = [float(bar.close) for bar in bars]
        highs = [float(bar.high) for bar in bars]
        lows = [float(bar.low) for bar in bars]
        volumes = [float(bar.volume) for bar in bars]

        # Moving averages
        fast_ma = np.mean(closes[-self.fast_ma_period :])
        slow_ma = np.mean(closes[-self.slow_ma_period :])
        trend_ma = np.mean(closes[-self.trend_ma_period :])

        # Multi-timeframe MAs
        short_ma = np.mean(closes[-self.short_ma_period :])
        medium_ma = np.mean(closes[-self.medium_ma_period :])
        long_ma = np.mean(closes[-self.long_ma_period :])

        # RSI
        rsi = self._calculate_rsi(closes, self.rsi_period)

        # ADX (Trend strength)
        adx = self._calculate_adx(highs, lows, closes, self.adx_period)

        # ATR (Volatility)
        atr = self._calculate_atr(highs, lows, closes, self.atr_period)

        # Volume analysis
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        current_volume = volumes[-1] if volumes else 0
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        # Market regime detection
        market_regime = self._detect_market_regime(
            closes, short_ma, medium_ma, long_ma, adx
        )

        indicators = {
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "trend_ma": trend_ma,
            "short_ma": short_ma,
            "medium_ma": medium_ma,
            "long_ma": long_ma,
            "rsi": rsi,
            "adx": adx,
            "atr": atr,
            "volume_ratio": volume_ratio,
            "market_regime": market_regime,
            "current_price": closes[-1] if closes else 0,
            "volatility": atr / closes[-1] if closes and closes[-1] > 0 else 0,
        }

        # Update historical data for ML and risk modeling
        self.update_historical_data(bars, indicators)

        return indicators

    def _detect_market_regime(
        self,
        closes: list[float],
        short_ma: float,
        medium_ma: float,
        long_ma: float,
        adx: float,
    ) -> str:
        """Detect if market is trending or ranging"""
        if len(closes) < 20:
            return "unknown"

        # Check if MAs are aligned (trending)
        ma_aligned_up = short_ma > medium_ma > long_ma
        ma_aligned_down = short_ma < medium_ma < long_ma

        # Check price momentum
        recent_momentum = (
            (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        )

        # Strong trend if MAs aligned and ADX high
        if ma_aligned_up and adx > 25 and recent_momentum > 0.01:
            return "strong_uptrend"
        elif ma_aligned_down and adx > 25 and recent_momentum < -0.01:
            return "strong_downtrend"
        elif adx > 20 and (ma_aligned_up or ma_aligned_down):
            return "weak_trend"
        else:
            return "ranging"

    def _calculate_adaptive_entry_threshold(
        self, indicators: dict[str, float]
    ) -> float:
        """Calculate adaptive entry threshold based on market conditions"""
        base_threshold = 60  # Base threshold

        # Adjust for market regime
        if indicators.get("market_regime") == "strong_uptrend":
            base_threshold -= 10  # Easier entry in strong uptrends
        elif indicators.get("market_regime") == "strong_downtrend":
            base_threshold += 15  # Harder entry in strong downtrends
        elif indicators.get("market_regime") == "weak_trend":
            base_threshold -= 5  # Slightly easier in weak trends

        # Adjust for volatility
        volatility = indicators.get("volatility", 0)
        if volatility > self.volatility_threshold * 1.5:
            base_threshold -= 5  # Easier entry in high volatility (more opportunities)
        elif volatility < self.volatility_threshold * 0.5:
            base_threshold += 5  # Harder entry in low volatility (fewer opportunities)

        # Adjust for RSI extremes
        rsi = indicators.get("rsi", 50)
        if rsi < 25 or rsi > 75:
            base_threshold -= 5  # Easier entry at RSI extremes (reversal opportunities)

        return max(40, min(80, base_threshold))  # Keep between 40-80

    def _calculate_enhanced_position_size(
        self, indicators: dict[str, float], portfolio: Portfolio
    ) -> float:
        """Calculate enhanced position size with volatility and market regime adjustment"""
        base_size = self.position_size

        # Market regime adjustments
        if indicators.get("market_regime") == "strong_uptrend":
            base_size *= 1.3  # Increase size in strong uptrends
        elif indicators.get("market_regime") == "strong_downtrend":
            base_size *= 0.7  # Decrease size in strong downtrends
        elif indicators.get("market_regime") == "weak_trend":
            base_size *= 1.1  # Slightly increase in weak trends
        elif indicators.get("market_regime") == "ranging":
            base_size *= 0.9  # Decrease in ranging markets

        # Volatility adjustments
        volatility = indicators.get("volatility", 0)
        if volatility > self.volatility_threshold * 1.5:
            base_size *= 0.7  # Reduce size in very high volatility
        elif volatility > self.volatility_threshold:
            base_size *= 0.8  # Reduce size in high volatility
        elif volatility < self.volatility_threshold * 0.5:
            base_size *= 1.2  # Increase size in low volatility (more predictable)

        # RSI adjustments
        rsi = indicators.get("rsi", 50)
        if rsi < 30 or rsi > 70:
            base_size *= 0.8  # Reduce size at RSI extremes

        # Trend strength adjustments
        adx = indicators.get("adx", 25)
        if adx > 40:
            base_size *= 1.2  # Increase size in very strong trends
        elif adx < 15:
            base_size *= 0.8  # Decrease size in weak trends

        # Portfolio concentration adjustments
        current_exposure = self._calculate_portfolio_exposure(portfolio)
        if current_exposure > 0.5:  # If more than 50% exposed
            base_size *= 0.7  # Reduce size to manage risk

        # Kelly Criterion adjustment if available
        if self.risk_modeling and hasattr(self, "trade_history") and self.trade_history:
            try:
                wins = [t for t in self.trade_history if t["pnl"] > 0]
                losses = [t for t in self.trade_history if t["pnl"] < 0]

                if wins and losses:
                    win_rate = len(wins) / len(self.trade_history)
                    avg_win = np.mean([t["pnl"] for t in wins])
                    avg_loss = abs(np.mean([t["pnl"] for t in losses]))

                    kelly_result = self.risk_modeling.calculate_kelly_criterion(
                        win_rate, avg_win, avg_loss
                    )

                    kelly_fraction = kelly_result.get("kelly_fraction_capped", 0)
                    if kelly_fraction > 0:
                        base_size *= min(kelly_fraction, 1.0)  # Apply Kelly adjustment
            except Exception as e:
                print(f"⚠️ Kelly Criterion calculation failed: {e}")

        # Ensure position size is within bounds
        max_size = min(0.20, float(portfolio.cash) / float(portfolio.equity) * 0.6)
        min_size = 0.02  # Minimum 2% position
        return max(min_size, min(base_size, max_size))

    def _calculate_portfolio_exposure(self, portfolio: Portfolio) -> float:
        """Calculate current portfolio exposure to crypto assets"""
        if not portfolio.positions:
            return 0.0

        total_exposure = sum(abs(float(p.market_value)) for p in portfolio.positions)
        return total_exposure / float(portfolio.equity)

    def _check_portfolio_heat_map(
        self, symbol: str, portfolio: Portfolio
    ) -> dict[str, Any]:
        """Analyze portfolio heat map for better risk management"""
        heat_map = {
            "total_positions": len(portfolio.positions),
            "total_exposure": self._calculate_portfolio_exposure(portfolio),
            "crypto_exposure": 0.0,
            "correlation_risk": False,
            "max_position_size": 0.0,
            "position_concentration": 0.0,
        }

        if not portfolio.positions:
            return heat_map

        # Calculate crypto exposure
        crypto_positions = [p for p in portfolio.positions if "/" in p.symbol]
        heat_map["crypto_exposure"] = sum(
            abs(float(p.market_value)) for p in crypto_positions
        ) / float(portfolio.equity)

        # Check for position concentration
        if portfolio.positions:
            max_position = max(
                portfolio.positions, key=lambda p: abs(float(p.market_value))
            )
            heat_map["max_position_size"] = abs(
                float(max_position.market_value)
            ) / float(portfolio.equity)

        # Check correlation risk
        if heat_map["crypto_exposure"] > self.max_correlation_exposure:
            heat_map["correlation_risk"] = True

        # Check position concentration
        if heat_map["max_position_size"] > 0.3:  # If any position > 30%
            heat_map["position_concentration"] = heat_map["max_position_size"]

        return heat_map

    def _should_enter_with_heat_map(
        self, symbol: str, portfolio: Portfolio, heat_map: dict[str, Any]
    ) -> bool:
        """Determine if we should enter based on portfolio heat map analysis"""
        # Don't enter if portfolio is too concentrated
        if heat_map["position_concentration"] > 0.4:  # 40% max concentration
            print(
                f"⚠️ ULTRA-ENHANCED STRATEGY: {symbol} Portfolio too concentrated ({heat_map['position_concentration']:.1%})"
            )
            return False

        # Don't enter if total exposure is too high
        if heat_map["total_exposure"] > 0.8:  # 80% max total exposure
            print(
                f"⚠️ ULTRA-ENHANCED STRATEGY: {symbol} Total exposure too high ({heat_map['total_exposure']:.1%})"
            )
            return False

        # Don't enter if correlation risk is too high
        if heat_map["correlation_risk"]:
            print(
                f"⚠️ ULTRA-ENHANCED STRATEGY: {symbol} Correlation risk too high (crypto exposure: {heat_map['crypto_exposure']:.1%})"
            )
            return False

        return True

    def generate_signals(
        self, market_data: MarketData, portfolio: Portfolio
    ) -> dict[str, Any]:
        """Generate ultra-enhanced trading signals with ML confirmation and advanced risk management"""
        entry_signals = {}
        exit_signals = {}

        print(
            f"\n🚀 ULTRA-ENHANCED STRATEGY: Generating signals for {len(market_data.bars)} symbols"
        )

        # Calculate advanced risk metrics
        risk_metrics = {}
        if self.risk_modeling:
            risk_metrics = self.calculate_advanced_risk_metrics(portfolio)
            if risk_metrics:
                print(
                    f"📊 Advanced Risk Metrics: Portfolio VaR: {risk_metrics.get('portfolio_var', {}).get('portfolio_var_percentage', 0):.2%}"
                )

        # Optimize portfolio allocation
        current_prices = {}
        for symbol, bars in market_data.bars.items():
            if bars:
                current_prices[symbol] = float(bars[-1].close)

        portfolio_optimization = {}
        if self.portfolio_optimizer:
            portfolio_optimization = self.optimize_portfolio_allocation(current_prices)
            if portfolio_optimization:
                print(
                    f"🎯 Portfolio Optimization: {portfolio_optimization.get('method', 'unknown')} completed"
                )

        for symbol, bars in market_data.bars.items():
            if not bars or len(bars) < self.trend_ma_period:
                print(
                    f"❌ ULTRA-ENHANCED STRATEGY: {symbol} insufficient data ({len(bars)} bars)"
                )
                continue

            # Calculate indicators
            indicators = self._calculate_indicators(bars)
            if not indicators:
                continue

            current_price = indicators["current_price"]
            current_position = self._get_current_position(symbol, portfolio)

            print(f"\n📊 ULTRA-ENHANCED STRATEGY: {symbol} Analysis")
            print(f"   Price: ${current_price:.2f}")
            print(
                f"   Fast MA: ${indicators['fast_ma']:.2f}, Slow MA: ${indicators['slow_ma']:.2f}"
            )
            print(f"   RSI: {indicators['rsi']:.1f}, ADX: {indicators['adx']:.1f}")
            print(f"   Market Regime: {indicators['market_regime']}")
            print(f"   Volatility: {indicators['volatility']:.3f}")
            print(f"   Volume Ratio: {indicators['volume_ratio']:.2f}")

            # Entry signals with advanced filtering
            if not current_position:
                print(
                    f"🚀 ULTRA-ENHANCED STRATEGY: {symbol} has no position, checking entry conditions..."
                )

                # Check portfolio heat map first
                heat_map = self._check_portfolio_heat_map(symbol, portfolio)
                if not self._should_enter_with_heat_map(symbol, portfolio, heat_map):
                    continue

                print(
                    f"📊 ULTRA-ENHANCED STRATEGY: {symbol} Portfolio heat map - Total exposure: {heat_map['total_exposure']:.1%}, Crypto: {heat_map['crypto_exposure']:.1%}"
                )

                # Check correlation risk
                if self._check_correlation_risk(symbol, portfolio):
                    print(
                        f"⚠️ ULTRA-ENHANCED STRATEGY: {symbol} correlation risk too high, skipping"
                    )
                    continue

                # Calculate adaptive entry threshold
                adaptive_threshold = self._calculate_adaptive_entry_threshold(
                    indicators
                )
                print(
                    f"🎯 ULTRA-ENHANCED STRATEGY: {symbol} Adaptive entry threshold: {adaptive_threshold}/100"
                )

                # Advanced entry conditions
                entry_score = 0
                entry_reasons = []

                # 1. Multi-timeframe momentum alignment (40 points)
                if (
                    indicators["short_ma"]
                    > indicators["medium_ma"]
                    > indicators["long_ma"]
                ):
                    entry_score += 40
                    entry_reasons.append("Multi-timeframe uptrend")
                    print(
                        f"✅ ULTRA-ENHANCED STRATEGY: {symbol} Multi-timeframe uptrend confirmed"
                    )

                # 2. RSI momentum (20 points)
                if 30 < indicators["rsi"] < 70:
                    entry_score += 20
                    entry_reasons.append("RSI in healthy range")
                    print(
                        f"✅ ULTRA-ENHANCED STRATEGY: {symbol} RSI healthy: {indicators['rsi']:.1f}"
                    )

                # 3. Volume confirmation (15 points)
                if indicators["volume_ratio"] > 1.0:
                    entry_score += 15
                    entry_reasons.append("Above average volume")
                    print(
                        f"✅ ULTRA-ENHANCED STRATEGY: {symbol} Volume confirmed: {indicators['volume_ratio']:.2f}"
                    )

                # 4. Trend strength (15 points)
                if indicators["adx"] > 20:
                    entry_score += 15
                    entry_reasons.append("Strong trend")
                    print(
                        f"✅ ULTRA-ENHANCED STRATEGY: {symbol} Trend strength: {indicators['adx']:.1f}"
                    )

                # 5. Volatility opportunity (10 points)
                if indicators["volatility"] > self.volatility_threshold:
                    entry_score += 10
                    entry_reasons.append("Good volatility")
                    print(
                        f"✅ ULTRA-ENHANCED STRATEGY: {symbol} Volatility: {indicators['volatility']:.3f}"
                    )

                # ML Signal Confirmation
                ml_confirmed = True
                ml_confidence = 1.0
                ml_details = {"ml_available": False}

                if self.ml_confirmation and self.ml_confirmation.models:
                    (
                        ml_confirmed,
                        ml_confidence,
                        ml_details,
                    ) = self.ml_confirmation.confirm_signal(bars, indicators)

                    if ml_details.get("ml_available", False):
                        print(
                            f"🤖 ML Confirmation: {ml_confirmed}, Confidence: {ml_confidence:.3f}"
                        )

                        if ml_confirmed:
                            entry_score += 10  # Bonus points for ML confirmation
                            entry_reasons.append("ML confirmed")
                            print(
                                f"✅ ULTRA-ENHANCED STRATEGY: {symbol} ML confirmation bonus: +10 points"
                            )
                        else:
                            entry_score -= 20  # Penalty for ML rejection
                            print(
                                f"❌ ULTRA-ENHANCED STRATEGY: {symbol} ML rejection penalty: -20 points"
                            )

                # Entry decision using adaptive threshold
                if entry_score >= adaptive_threshold and ml_confirmed:
                    # Calculate enhanced position size
                    position_size = self._calculate_enhanced_position_size(
                        indicators, portfolio
                    )

                    entry_signals[symbol] = {
                        "side": "buy",
                        "price": current_price,
                        "quantity": self._calculate_position_size(
                            current_price, portfolio, position_size
                        ),
                        "reason": f"Ultra-Enhanced Entry Score: {entry_score}/{adaptive_threshold} - {' | '.join(entry_reasons)}",
                        "timestamp": datetime.now(UTC),
                        "entry_score": entry_score,
                        "position_size": position_size,
                        "adaptive_threshold": adaptive_threshold,
                        "ml_confirmed": ml_confirmed,
                        "ml_confidence": ml_confidence,
                        "ml_details": ml_details,
                    }
                    print(
                        f"🚀 ULTRA-ENHANCED STRATEGY: {symbol} BUY SIGNAL! Score: {entry_score}/{adaptive_threshold}"
                    )
                    print(
                        f"   Position Size: {position_size:.1%}, Quantity: {entry_signals[symbol]['quantity']:.4f}"
                    )
                    print(
                        f"   ML Confirmed: {ml_confirmed}, Confidence: {ml_confidence:.3f}"
                    )
                else:
                    print(
                        f"❌ ULTRA-ENHANCED STRATEGY: {symbol} Entry score too low: {entry_score}/{adaptive_threshold}"
                    )
                    if not ml_confirmed:
                        print(
                            f"   ML Rejection: Confidence {ml_confidence:.3f} below threshold"
                        )

            # Exit signals for existing positions
            elif current_position:
                print(
                    f"📉 ULTRA-ENHANCED STRATEGY: {symbol} has position, checking exit conditions..."
                )

                # Calculate current P&L
                pnl = (current_price - float(current_position.average_cost)) * float(
                    current_position.quantity
                )
                pnl_pct = pnl / (
                    float(current_position.average_cost)
                    * float(current_position.quantity)
                )

                print(f"   Current P&L: ${pnl:.2f} ({pnl_pct:.2%})")

                # Smart exit conditions for LONG positions
                should_exit = False
                exit_reason = ""

                # 1. Stop loss (tight)
                if pnl_pct <= -self.stop_loss_pct:
                    should_exit = True
                    exit_reason = f"Stop loss: {pnl_pct:.2%}"
                    print(
                        f"🛑 ULTRA-ENHANCED STRATEGY: {symbol} STOP LOSS triggered: {pnl_pct:.2%}"
                    )

                # 2. Take profit (better risk-reward)
                elif pnl_pct >= self.take_profit_pct:
                    should_exit = True
                    exit_reason = f"Take profit: {pnl_pct:.2%}"
                    print(
                        f"🎯 ULTRA-ENHANCED STRATEGY: {symbol} TAKE PROFIT triggered: {pnl_pct:.2%}"
                    )

                # 3. Trailing stop (protect profits)
                elif pnl_pct > 0.01:  # Only if we're in profit
                    # Calculate trailing stop based on highest price since entry
                    highest_price = max(
                        float(bar.high)
                        for bar in bars
                        if bar.timestamp >= current_position.entry_timestamp
                    )
                    trailing_stop_price = highest_price * (1 - self.trailing_stop_pct)

                    if current_price <= trailing_stop_price:
                        should_exit = True
                        exit_reason = f"Trailing stop: {pnl_pct:.2%}"
                        print(
                            f"📉 ULTRA-ENHANCED STRATEGY: {symbol} TRAILING STOP triggered: {pnl_pct:.2%}"
                        )

                # 4. Technical exit signals
                elif (
                    indicators["fast_ma"] < indicators["slow_ma"]
                    and indicators["rsi"] > 70
                    and indicators["adx"] > 25
                ):
                    should_exit = True
                    exit_reason = (
                        "Technical exit: MA crossover + RSI overbought + strong trend"
                    )
                    print(
                        f"📊 ULTRA-ENHANCED STRATEGY: {symbol} TECHNICAL EXIT: MA crossover + RSI overbought"
                    )

                # 5. Market regime change
                elif (
                    indicators["market_regime"] == "strong_downtrend"
                    and pnl_pct < 0.005
                ):
                    should_exit = True
                    exit_reason = "Market regime change to strong downtrend"
                    print(
                        f"🌊 ULTRA-ENHANCED STRATEGY: {symbol} MARKET REGIME EXIT: Strong downtrend detected"
                    )

                # 6. Advanced risk management exit
                if self.risk_modeling and risk_metrics:
                    portfolio_var = risk_metrics.get("portfolio_var", {})
                    if portfolio_var:
                        var_percentage = portfolio_var.get(
                            "portfolio_var_percentage", 0
                        )
                        if var_percentage > 0.05:  # If portfolio VaR > 5%
                            should_exit = True
                            exit_reason = f"Risk management: Portfolio VaR {var_percentage:.2%} > 5%"
                            print(
                                f"⚠️ ULTRA-ENHANCED STRATEGY: {symbol} RISK MANAGEMENT EXIT: VaR {var_percentage:.2%}"
                            )

                if should_exit:
                    exit_signals[symbol] = {
                        "side": "sell",
                        "price": current_price,
                        "quantity": abs(float(current_position.quantity)),
                        "reason": exit_reason,
                        "timestamp": datetime.now(UTC),
                    }
                    print(
                        f"📉 ULTRA-ENHANCED STRATEGY: {symbol} SELL SIGNAL: {exit_reason}"
                    )
                else:
                    print(
                        f"⏳ ULTRA-ENHANCED STRATEGY: {symbol} Holding position, no exit signal"
                    )

        print(
            f"\n🎯 ULTRA-ENHANCED STRATEGY: Generated {len(entry_signals)} entry signals, {len(exit_signals)} exit signals"
        )

        # Add advanced metrics to output
        result = {
            "entry_signals": entry_signals,
            "exit_signals": exit_signals,
            "advanced_metrics": {
                "risk_metrics": risk_metrics,
                "portfolio_optimization": portfolio_optimization,
                "ml_models_available": self.ml_confirmation is not None
                and self.ml_confirmation.models,
                "risk_modeling_available": self.risk_modeling is not None,
                "portfolio_optimization_available": self.portfolio_optimizer
                is not None,
            },
        }

        return result

    def _calculate_position_size(
        self, price: float, portfolio: Portfolio, position_size: float = None
    ) -> float:
        """Calculate position size in base currency"""
        if position_size is None:
            position_size = self.position_size

        # Calculate position size based on portfolio equity (convert Decimal to float)
        target_value = float(portfolio.equity) * position_size
        quantity = target_value / price

        # Ensure we don't exceed available cash (convert Decimal to float)
        max_quantity = float(portfolio.cash) / price
        return min(quantity, max_quantity)

    def _get_current_position(
        self, symbol: str, portfolio: Portfolio
    ) -> Optional[Position]:
        """Get current position for a symbol"""
        for position in portfolio.positions:
            if position.symbol == symbol:
                return position
        return None

    def _check_correlation_risk(self, symbol: str, portfolio: Portfolio) -> bool:
        """Check if adding this position would create correlation risk"""
        if not portfolio.positions:
            return False

        # Simple correlation check based on asset types
        # In a real implementation, you'd calculate actual correlation
        crypto_positions = [p for p in portfolio.positions if "/" in p.symbol]

        # Limit crypto exposure
        total_crypto_exposure = sum(abs(p.market_value) for p in crypto_positions)
        if total_crypto_exposure / portfolio.equity > self.max_correlation_exposure:
            return True

        return False

    def _calculate_rsi(self, prices: list[float], period: int) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gains = np.mean(gains[-period:])
        avg_losses = np.mean(losses[-period:])

        if avg_losses == 0:
            return 100.0

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_adx(
        self, highs: list[float], lows: list[float], closes: list[float], period: int
    ) -> float:
        """Calculate ADX indicator (simplified)"""
        if len(closes) < period + 1:
            return 25.0

        # Simplified ADX calculation
        tr_values = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            tr_values.append(tr)

        if len(tr_values) < period:
            return 25.0

        atr = np.mean(tr_values[-period:])
        current_tr = tr_values[-1] if tr_values else 0

        # Simplified ADX based on TR ratio
        adx = min(50, (current_tr / atr) * 25) if atr > 0 else 25
        return adx

    def _calculate_atr(
        self, highs: list[float], lows: list[float], closes: list[float], period: int
    ) -> float:
        """Calculate Average True Range"""
        if len(closes) < period + 1:
            return 0.0

        tr_values = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            tr_values.append(tr)

        if len(tr_values) < period:
            return 0.0

        return np.mean(tr_values[-period:])

    def should_enter(self, market_data: MarketData, portfolio: Portfolio) -> bool:
        """Check if we should enter new positions"""
        # Always allow entry in this ultra-enhanced strategy
        return True

    def should_exit(self, market_data: MarketData, portfolio: Portfolio) -> bool:
        """Check if we should exit existing positions"""
        # Always allow exit in this ultra-enhanced strategy
        return True
