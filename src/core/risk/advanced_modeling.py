"""
Advanced Risk Modeling Module

This module provides sophisticated risk management techniques:
- Value at Risk (VaR) calculations
- Kelly Criterion for optimal position sizing
- Expected Shortfall (Conditional VaR)
- Portfolio stress testing
- Risk-adjusted performance metrics
"""


import numpy as np
from scipy import stats

from ..data_models.trading import Portfolio


class AdvancedRiskModeling:
    """
    Advanced risk modeling and management system
    """

    def __init__(self, config: dict):
        self.config = config

        # VaR parameters
        self.var_confidence_level = config.get("var_confidence_level", 0.95)
        self.var_time_horizon = config.get("var_time_horizon", 1)  # days

        # Kelly Criterion parameters
        self.kelly_max_fraction = config.get(
            "kelly_max_fraction", 0.25
        )  # Max 25% of portfolio
        self.kelly_risk_free_rate = config.get(
            "kelly_risk_free_rate", 0.02
        )  # 2% annual

        # Stress testing parameters
        self.stress_scenarios = config.get(
            "stress_scenarios",
            {
                "market_crash": -0.20,  # 20% market decline
                "volatility_spike": 0.50,  # 50% volatility increase
                "correlation_breakdown": 0.8,  # 80% correlation increase
                "liquidity_crisis": -0.15,  # 15% liquidity reduction
            },
        )

        print("📊 Advanced Risk Modeling initialized")
        print(f"   VaR Confidence: {self.var_confidence_level:.1%}")
        print(f"   Kelly Max Fraction: {self.kelly_max_fraction:.1%}")
        print(f"   Stress Scenarios: {len(self.stress_scenarios)} configured")

    def calculate_var(
        self,
        returns: np.ndarray,
        confidence_level: float = None,
        time_horizon: int = None,
        method: str = "historical",
    ) -> dict:
        """
        Calculate Value at Risk using multiple methods

        Args:
            returns: Array of historical returns
            confidence_level: VaR confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            method: 'historical', 'parametric', or 'monte_carlo'
        """
        if confidence_level is None:
            confidence_level = self.var_confidence_level
        if time_horizon is None:
            time_horizon = self.var_time_horizon

        results = {}

        # Historical VaR
        if method in ["historical", "all"]:
            historical_var = self._calculate_historical_var(
                returns, confidence_level, time_horizon
            )
            results["historical"] = historical_var

        # Parametric VaR (assuming normal distribution)
        if method in ["parametric", "all"]:
            parametric_var = self._calculate_parametric_var(
                returns, confidence_level, time_horizon
            )
            results["parametric"] = parametric_var

        # Monte Carlo VaR
        if method in ["monte_carlo", "all"]:
            monte_carlo_var = self._calculate_monte_carlo_var(
                returns, confidence_level, time_horizon
            )
            results["monte_carlo"] = monte_carlo_var

        # Expected Shortfall (Conditional VaR)
        if method in ["expected_shortfall", "all"]:
            es = self._calculate_expected_shortfall(
                returns, confidence_level, time_horizon
            )
            results["expected_shortfall"] = es

        return results

    def _calculate_historical_var(
        self, returns: np.ndarray, confidence_level: float, time_horizon: int
    ) -> dict:
        """Calculate historical VaR"""
        # Sort returns in ascending order
        sorted_returns = np.sort(returns)

        # Find the percentile corresponding to the confidence level
        percentile = (1 - confidence_level) * 100
        var_percentile = np.percentile(sorted_returns, percentile)

        # Scale by time horizon (assuming square root of time)
        var_scaled = var_percentile * np.sqrt(time_horizon)

        return {
            "var": var_scaled,
            "percentile": percentile,
            "method": "historical",
            "confidence_level": confidence_level,
            "time_horizon": time_horizon,
        }

    def _calculate_parametric_var(
        self, returns: np.ndarray, confidence_level: float, time_horizon: int
    ) -> dict:
        """Calculate parametric VaR assuming normal distribution"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Z-score for confidence level
        z_score = stats.norm.ppf(confidence_level)

        # VaR calculation
        var = mean_return - z_score * std_return

        # Scale by time horizon
        var_scaled = var * time_horizon

        return {
            "var": var_scaled,
            "mean": mean_return,
            "std": std_return,
            "z_score": z_score,
            "method": "parametric",
            "confidence_level": confidence_level,
            "time_horizon": time_horizon,
        }

    def _calculate_monte_carlo_var(
        self,
        returns: np.ndarray,
        confidence_level: float,
        time_horizon: int,
        n_simulations: int = 10000,
    ) -> dict:
        """Calculate Monte Carlo VaR"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Generate random returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)

        # Scale by time horizon
        simulated_returns_scaled = simulated_returns * time_horizon

        # Calculate VaR
        var = np.percentile(simulated_returns_scaled, (1 - confidence_level) * 100)

        return {
            "var": var,
            "simulations": n_simulations,
            "method": "monte_carlo",
            "confidence_level": confidence_level,
            "time_horizon": time_horizon,
        }

    def _calculate_expected_shortfall(
        self, returns: np.ndarray, confidence_level: float, time_horizon: int
    ) -> dict:
        """Calculate Expected Shortfall (Conditional VaR)"""
        # First calculate VaR
        var_result = self._calculate_historical_var(
            returns, confidence_level, time_horizon
        )
        var_threshold = var_result["var"]

        # Find returns below VaR threshold
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            expected_shortfall = var_threshold
        else:
            expected_shortfall = np.mean(tail_returns)

        # Scale by time horizon
        es_scaled = expected_shortfall * np.sqrt(time_horizon)

        return {
            "expected_shortfall": es_scaled,
            "var_threshold": var_threshold,
            "tail_observations": len(tail_returns),
            "confidence_level": confidence_level,
            "time_horizon": time_horizon,
        }

    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        risk_free_rate: float = None,
    ) -> dict:
        """
        Calculate Kelly Criterion for optimal position sizing

        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount
            risk_free_rate: Risk-free rate of return
        """
        if risk_free_rate is None:
            risk_free_rate = self.kelly_risk_free_rate

        # Kelly formula: f = (bp - q) / b
        # where: b = odds received, p = probability of win, q = probability of loss
        b = avg_win / avg_loss if avg_loss != 0 else 1
        p = win_rate
        q = 1 - win_rate

        # Kelly fraction
        kelly_fraction = (b * p - q) / b if b != 0 else 0

        # Adjust for risk-free rate
        kelly_fraction_adjusted = (
            kelly_fraction - (risk_free_rate / avg_win)
            if avg_win != 0
            else kelly_fraction
        )

        # Cap at maximum fraction
        kelly_fraction_capped = min(kelly_fraction_adjusted, self.kelly_max_fraction)

        # Ensure non-negative
        kelly_fraction_final = max(kelly_fraction_capped, 0)

        return {
            "kelly_fraction": kelly_fraction,
            "kelly_fraction_adjusted": kelly_fraction_adjusted,
            "kelly_fraction_capped": kelly_fraction_final,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "odds_ratio": b,
            "risk_free_rate": risk_free_rate,
            "max_fraction": self.kelly_max_fraction,
        }

    def calculate_portfolio_var(
        self,
        portfolio: Portfolio,
        historical_returns: dict[str, np.ndarray],
        confidence_level: float = None,
    ) -> dict:
        """
        Calculate portfolio-level VaR considering correlations
        """
        if confidence_level is None:
            confidence_level = self.var_confidence_level

        if not portfolio.positions:
            return {"portfolio_var": 0, "positions": {}, "correlations": {}}

        position_vars = {}
        position_weights = {}
        total_portfolio_value = float(portfolio.equity)

        # Calculate individual position VaRs
        for position in portfolio.positions:
            symbol = position.symbol
            if symbol in historical_returns:
                returns = historical_returns[symbol]
                position_value = float(position.market_value)
                weight = position_value / total_portfolio_value

                # Calculate VaR for this position
                var_result = self.calculate_var(
                    returns, confidence_level, method="parametric"
                )
                position_var = var_result["parametric"]["var"] * position_value

                position_vars[symbol] = {
                    "var": position_var,
                    "weight": weight,
                    "position_value": position_value,
                    "var_percentage": position_var / position_value,
                }
                position_weights[symbol] = weight

        # Calculate portfolio VaR considering correlations
        if len(position_vars) > 1:
            portfolio_var = self._calculate_correlated_portfolio_var(
                position_vars, historical_returns
            )
        else:
            portfolio_var = sum(pos["var"] for pos in position_vars.values())

        return {
            "portfolio_var": portfolio_var,
            "portfolio_var_percentage": portfolio_var / total_portfolio_value,
            "positions": position_vars,
            "confidence_level": confidence_level,
            "total_portfolio_value": total_portfolio_value,
        }

    def _calculate_correlated_portfolio_var(
        self, position_vars: dict, historical_returns: dict[str, np.ndarray]
    ) -> float:
        """Calculate portfolio VaR considering correlations between positions"""
        symbols = list(position_vars.keys())
        n_positions = len(symbols)

        if n_positions < 2:
            return sum(pos["var"] for pos in position_vars.values())

        # Calculate correlation matrix
        correlation_matrix = np.zeros((n_positions, n_positions))
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Calculate correlation between returns
                    returns1 = historical_returns[symbol1]
                    returns2 = historical_returns[symbol2]

                    # Ensure same length
                    min_length = min(len(returns1), len(returns2))
                    if min_length > 10:
                        correlation = np.corrcoef(
                            returns1[:min_length], returns2[:min_length]
                        )[0, 1]
                        correlation_matrix[i, j] = (
                            correlation if not np.isnan(correlation) else 0
                        )
                    else:
                        correlation_matrix[i, j] = 0

        # Calculate portfolio VaR using correlation matrix
        weights = np.array([position_vars[symbol]["weight"] for symbol in symbols])
        position_vars_array = np.array(
            [position_vars[symbol]["var"] for symbol in symbols]
        )

        # Portfolio VaR = sqrt(w' * Σ * w) where Σ is the covariance matrix
        # For simplicity, we'll use a weighted sum with correlation adjustment
        portfolio_var = np.sqrt(
            np.dot(weights.T, np.dot(correlation_matrix, weights))
        ) * np.sum(position_vars_array)

        return portfolio_var

    def stress_test_portfolio(
        self, portfolio: Portfolio, historical_returns: dict[str, np.ndarray]
    ) -> dict:
        """
        Perform portfolio stress testing under various scenarios
        """
        if not portfolio.positions:
            return {"stress_test_results": {}, "scenarios": self.stress_scenarios}

        stress_results = {}

        for scenario_name, scenario_impact in self.stress_scenarios.items():
            scenario_result = self._apply_stress_scenario(
                portfolio, historical_returns, scenario_name, scenario_impact
            )
            stress_results[scenario_name] = scenario_result

        # Calculate worst-case scenario
        worst_case = min(stress_results.values(), key=lambda x: x["portfolio_value"])

        return {
            "stress_test_results": stress_results,
            "worst_case_scenario": worst_case,
            "scenarios": self.stress_scenarios,
            "portfolio_robustness": self._calculate_portfolio_robustness(
                stress_results
            ),
        }

    def _apply_stress_scenario(
        self,
        portfolio: Portfolio,
        historical_returns: dict[str, np.ndarray],
        scenario_name: str,
        scenario_impact: float,
    ) -> dict:
        """Apply a specific stress scenario to the portfolio"""
        original_value = float(portfolio.equity)
        stressed_value = original_value

        if scenario_name == "market_crash":
            # Apply market crash impact to all positions
            for position in portfolio.positions:
                position_value = float(position.market_value)
                stressed_value -= position_value * abs(scenario_impact)

        elif scenario_name == "volatility_spike":
            # Increase VaR due to volatility spike
            portfolio_var = self.calculate_portfolio_var(portfolio, historical_returns)
            var_increase = portfolio_var["portfolio_var"] * scenario_impact
            stressed_value -= var_increase

        elif scenario_name == "correlation_breakdown":
            # Assume correlations increase, leading to higher portfolio risk
            portfolio_var = self.calculate_portfolio_var(portfolio, historical_returns)
            correlation_impact = portfolio_var["portfolio_var"] * scenario_impact
            stressed_value -= correlation_impact

        elif scenario_name == "liquidity_crisis":
            # Reduce portfolio value due to liquidity constraints
            stressed_value *= 1 + scenario_impact

        return {
            "scenario_name": scenario_name,
            "scenario_impact": scenario_impact,
            "original_value": original_value,
            "stressed_value": stressed_value,
            "value_change": stressed_value - original_value,
            "value_change_percentage": (stressed_value - original_value)
            / original_value,
        }

    def _calculate_portfolio_robustness(self, stress_results: dict) -> dict:
        """Calculate portfolio robustness metrics"""
        value_changes = [
            result["value_change_percentage"] for result in stress_results.values()
        ]

        return {
            "min_stress_impact": min(value_changes),
            "max_stress_impact": max(value_changes),
            "avg_stress_impact": np.mean(value_changes),
            "stress_volatility": np.std(value_changes),
            "robustness_score": 1 - abs(np.mean(value_changes)),  # Higher is better
        }

    def calculate_risk_adjusted_metrics(
        self, returns: np.ndarray, risk_free_rate: float = None
    ) -> dict:
        """
        Calculate risk-adjusted performance metrics
        """
        if risk_free_rate is None:
            risk_free_rate = self.kelly_risk_free_rate

        if len(returns) == 0:
            return {}

        # Basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        total_return = np.prod(1 + returns) - 1

        # Risk-adjusted metrics
        sharpe_ratio = (
            (mean_return - risk_free_rate / 252) / std_return if std_return > 0 else 0
        )
        sortino_ratio = self._calculate_sortino_ratio(returns, risk_free_rate)
        calmar_ratio = self._calculate_calmar_ratio(returns)

        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # VaR and Expected Shortfall
        var_results = self.calculate_var(returns, method="all")

        return {
            "basic_stats": {
                "mean_return": mean_return,
                "std_return": std_return,
                "total_return": total_return,
                "min_return": np.min(returns),
                "max_return": np.max(returns),
            },
            "risk_adjusted": {
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
            },
            "risk_metrics": {
                "max_drawdown": max_drawdown,
                "var_95": var_results.get("historical", {}).get("var", 0),
                "expected_shortfall": var_results.get("expected_shortfall", {}).get(
                    "expected_shortfall", 0
                ),
            },
        }

    def _calculate_sortino_ratio(
        self, returns: np.ndarray, risk_free_rate: float
    ) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) == 0:
            return 0

        # Calculate downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float("inf") if np.mean(returns) > risk_free_rate / 252 else 0

        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0

        return (np.mean(returns) - risk_free_rate / 252) / downside_deviation

    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if len(returns) == 0:
            return 0

        # Annualized return
        annual_return = np.mean(returns) * 252

        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))

        if max_drawdown == 0:
            return 0

        return annual_return / max_drawdown
