"""
Performance metrics and reporting models for the Algo-Trading system.

Provides structured data models for:
- Trading performance metrics
- Risk metrics and analysis
- Portfolio performance tracking
- Backtesting results and reports
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Core performance metrics for trading strategies."""

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int

    def __post_init__(self):
        """Validate and calculate derived metrics."""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades

        if self.average_loss != 0:
            self.profit_factor = abs(self.average_win / self.average_loss)

        if self.max_drawdown != 0:
            self.calmar_ratio = self.annualized_return / abs(self.max_drawdown)


@dataclass
class RiskMetrics:
    """Risk management and analysis metrics."""

    var_95: float  # Value at Risk (95%)
    var_99: float  # Value at Risk (99%)
    cvar_95: float  # Conditional Value at Risk (95%)
    cvar_99: float  # Conditional Value at Risk (99%)
    downside_deviation: float
    beta: float
    alpha: float
    information_ratio: float
    treynor_ratio: float
    ulcer_index: float
    gain_to_pain_ratio: float

    def __post_init__(self):
        """Calculate derived risk metrics."""
        if self.beta != 0:
            self.treynor_ratio = self.annualized_return / self.beta


@dataclass
class PortfolioMetrics:
    """Portfolio-level performance and risk metrics."""

    portfolio_value: float
    cash_balance: float
    invested_amount: float
    margin_used: float
    free_margin: float
    margin_level: float
    leverage: float
    diversification_score: float
    concentration_risk: float
    sector_allocation: dict[str, float]
    asset_allocation: dict[str, float]

    def __post_init__(self):
        """Calculate derived portfolio metrics."""
        if self.portfolio_value > 0:
            self.margin_level = (self.portfolio_value / self.margin_used) if self.margin_used > 0 else float('inf')
            self.leverage = self.invested_amount / self.portfolio_value if self.portfolio_value > 0 else 0


@dataclass
class TradeRecord:
    """Individual trade record with performance data."""

    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_size: float
    side: str  # "LONG" or "SHORT"
    pnl: float
    pnl_pct: float
    holding_period: float  # in hours
    stop_loss: float | None = None
    take_profit: float | None = None
    exit_reason: str = ""
    strategy_name: str = ""
    tags: list[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

        # Calculate holding period in hours
        if self.entry_time and self.exit_time:
            self.holding_period = (self.exit_time - self.entry_time).total_seconds() / 3600

        # Calculate PnL percentage
        if self.entry_price != 0:
            if self.side == "LONG":
                self.pnl_pct = (self.exit_price - self.entry_price) / self.entry_price
            else:  # SHORT
                self.pnl_pct = (self.entry_price - self.exit_price) / self.entry_price


@dataclass
class BacktestResult:
    """Complete backtesting results and analysis."""

    strategy_name: str
    start_date: datetime
    end_date: datetime
    symbols: list[str]
    performance_metrics: PerformanceMetrics
    risk_metrics: RiskMetrics
    portfolio_metrics: PortfolioMetrics
    trades: list[TradeRecord]
    equity_curve: pd.DataFrame
    drawdown_curve: pd.DataFrame
    monthly_returns: pd.DataFrame
    config_hash: str
    execution_time: float

    def get_trade_summary(self) -> dict[str, Any]:
        """Get summary statistics for all trades."""
        if not self.trades:
            return {"total_trades": 0}

        summary = {
            "total_trades": len(self.trades),
            "winning_trades": len([t for t in self.trades if t.pnl > 0]),
            "losing_trades": len([t for t in self.trades if t.pnl < 0]),
            "break_even_trades": len([t for t in self.trades if t.pnl == 0]),
            "total_pnl": sum(t.pnl for t in self.trades),
            "average_pnl": np.mean([t.pnl for t in self.trades]),
            "pnl_std": np.std([t.pnl for t in self.trades]),
            "best_trade": max(t.pnl for t in self.trades) if self.trades else 0,
            "worst_trade": min(t.pnl for t in self.trades) if self.trades else 0,
            "average_winning_trade": np.mean([t.pnl for t in self.trades if t.pnl > 0]) if any(t.pnl > 0 for t in self.trades) else 0,
            "average_losing_trade": np.mean([t.pnl for t in self.trades if t.pnl < 0]) if any(t.pnl < 0 for t in self.trades) else 0,
            "symbols_traded": list({t.symbol for t in self.trades}),
            "strategies_used": list({t.strategy_name for t in self.trades if t.strategy_name})
        }

        return summary

    def get_monthly_performance(self) -> pd.DataFrame:
        """Get monthly performance breakdown."""
        if self.trades:
            # Create monthly returns DataFrame
            monthly_data = []
            for trade in self.trades:
                month_key = trade.exit_time.strftime("%Y-%m")
                monthly_data.append({
                    "month": month_key,
                    "pnl": trade.pnl,
                    "trades": 1
                })

            if monthly_data:
                monthly_df = pd.DataFrame(monthly_data)
                monthly_summary = monthly_df.groupby("month").agg({
                    "pnl": ["sum", "mean", "count"],
                    "trades": "sum"
                }).round(4)

                monthly_summary.columns = ["total_pnl", "avg_pnl", "trade_count", "total_trades"]
                return monthly_summary

        return pd.DataFrame()

    def validate_results(self) -> dict[str, Any]:
        """Validate backtesting results for consistency."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # Check for basic data integrity
        if not self.trades:
            validation_result["warnings"].append("No trades recorded in backtest")

        if self.start_date >= self.end_date:
            validation_result["errors"].append("Start date must be before end date")
            validation_result["valid"] = False

        # Check performance metrics consistency
        if self.performance_metrics.total_trades != len(self.trades):
            validation_result["errors"].append("Trade count mismatch between metrics and trade records")
            validation_result["valid"] = False

        # Check for extreme values
        if abs(self.performance_metrics.total_return) > 10:  # 1000% return
            validation_result["warnings"].append("Extremely high total return - verify calculations")

        if self.performance_metrics.max_drawdown > 1:  # 100% drawdown
            validation_result["warnings"].append("Maximum drawdown exceeds 100% - verify calculations")

        return validation_result


def calculate_performance_metrics(
    equity_curve: pd.DataFrame,
    risk_free_rate: float = 0.02,
    trade_data: pd.DataFrame = None
) -> PerformanceMetrics:
    """Calculate performance metrics from equity curve and trade data."""
    if equity_curve.empty:
        raise ValueError("Equity curve cannot be empty")

    # Calculate returns
    returns = equity_curve['equity'].pct_change().dropna()

    # Basic metrics
    total_return = (equity_curve['equity'].iloc[-1] / equity_curve['equity'].iloc[0]) - 1

    # Annualized return (assuming daily data)
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    annualized_return = ((1 + total_return) ** (365 / days)) - 1 if days > 0 else 0

    # Volatility
    volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0

    # Sharpe ratio
    excess_returns = returns - (risk_free_rate / 252)
    sharpe_ratio = (excess_returns.mean() * np.sqrt(252)) / volatility if volatility > 0 else 0

    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
    sortino_ratio = (excess_returns.mean() * np.sqrt(252)) / downside_deviation if downside_deviation > 0 else 0

    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Calculate trade metrics from actual trade data if available
    if trade_data is not None and not trade_data.empty:
        total_trades = len(trade_data)
        
        if 'trade_pnl' in trade_data.columns:
            winning_trades = len(trade_data[trade_data['trade_pnl'] > 0])
            losing_trades = len(trade_data[trade_data['trade_pnl'] < 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate average win/loss
            if winning_trades > 0:
                average_win = trade_data[trade_data['trade_pnl'] > 0]['trade_pnl'].mean()
            else:
                average_win = 0.0
                
            if losing_trades > 0:
                average_loss = abs(trade_data[trade_data['trade_pnl'] < 0]['trade_pnl'].mean())
            else:
                average_loss = 0.0
            
            # Calculate profit factor
            if average_loss > 0:
                profit_factor = average_win / average_loss
            else:
                profit_factor = 0.0
        else:
            # No PnL data available
            total_trades = 0
            winning_trades = 0
            losing_trades = 0
            win_rate = 0.0
            average_win = 0.0
            average_loss = 0.0
            profit_factor = 0.0
    else:
        # No trade data available, use defaults
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        win_rate = 0.0
        average_win = 0.0
        average_loss = 0.0
        profit_factor = 0.0

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        win_rate=win_rate,
        profit_factor=profit_factor,
        average_win=average_win,
        average_loss=average_loss,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades
    )


def create_backtest_report(
    backtest_result: BacktestResult,
    output_path: str = "reports/backtest_report.html"
) -> str:
    """Create an HTML report for backtesting results."""
    # This would generate a comprehensive HTML report
    # For now, return a placeholder
    return f"Backtest report would be generated at {output_path}"
