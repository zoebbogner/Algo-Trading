"""
Performance Analyzer for Backtest Results

Analyzes and visualizes backtest performance with comprehensive metrics
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from ..data_models.base import BaseEntity


class BacktestResult(BaseEntity):
    """
    Comprehensive results from a backtest run
    """

    # Strategy identification
    strategy_name: str

    # Time period
    start_date: Optional[datetime]
    end_date: Optional[datetime]

    # Performance metrics
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float

    # Trading statistics
    total_trades: int
    win_rate: float
    profit_factor: float

    # Detailed data
    equity_curve: list[dict[str, Any]]
    trades: list[dict[str, Any]]
    hourly_pnl: list[dict[str, Any]]  # New: Hourly P&L tracking
    portfolio_snapshots: list[dict[str, Any]]  # New: Portfolio state snapshots
    parameters: dict[str, Any]


class PerformanceAnalyzer:
    """
    Analyzes backtest results and generates comprehensive reports
    """

    def __init__(self, result: BacktestResult):
        self.result = result

    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report"""
        if not self.result.equity_curve:
            return "No equity curve data available for analysis"

        # Calculate additional metrics
        duration_days = (
            (self.result.end_date - self.result.start_date).days
            if self.result.start_date and self.result.end_date
            else 0
        )
        peak_equity = max(equity["equity"] for equity in self.result.equity_curve)

        # Performance rating
        rating, score = self._calculate_performance_rating()

        report = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    🚀 ALGORITHMIC TRADING BACKTEST REPORT 🚀                        ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

📋 EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════════════════════════════════════════
Strategy Name:     {self.result.strategy_name}
Test Period:       {self.result.start_date.strftime('%Y-%m-%d %H:%M UTC') if self.result.start_date else 'Unknown'} → {self.result.end_date.strftime('%Y-%m-%d %H:%M UTC') if self.result.end_date else 'Unknown'}
Duration:          {duration_days} days ({duration_days * 24} hours)
Test Date:         {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}

🏆 PERFORMANCE OVERVIEW
═══════════════════════════════════════════════════════════════════════════════════════════════════════
Initial Capital:   ${self.result.initial_capital:>15,.2f}
Final Portfolio:   ${self.result.final_capital:>15,.2f}
Total Return:      ${self.result.final_capital - self.result.initial_capital:>15,.2f} ({self.result.total_return:+.2%})
Annualized Return: {self.result.annualized_return:>15.2%}
Peak Portfolio:    ${peak_equity:>15,.2f}
Absolute Gain:     ${self.result.final_capital - self.result.initial_capital:>15,.2f}

📊 RISK METRICS
═══════════════════════════════════════════════════════════════════════════════════════════════════════
Max Drawdown:      {self.result.max_drawdown:>15.2%}
Sharpe Ratio:      {self.result.sharpe_ratio:>15.3f}
Volatility:        {self._calculate_volatility():>15.2%}
Risk-Adjusted Return: {self.result.sharpe_ratio * self.result.total_return:>15.2%}

🔄 TRADING STATISTICS
═══════════════════════════════════════════════════════════════════════════════════════════════════════
Total Trades:      {self.result.total_trades:>15}
Winning Trades:    {int(self.result.total_trades * self.result.win_rate):>15}
Losing Trades:     {int(self.result.total_trades * (1 - self.result.win_rate)):>15}
Win Rate:          {self.result.win_rate:>15.2%}
Profit Factor:     {self.result.profit_factor:>15.3f}
Average Win:       ${self._calculate_average_win():>15,.2f}
Average Loss:      ${self._calculate_average_loss():>15,.2f}

⚙️ STRATEGY PARAMETERS
═══════════════════════════════════════════════════════════════════════════════════════════════════════
"""

        for key, value in self.result.parameters.items():
            report += f"{key:<25}: {value}\n"

        report += f"""
🏅 PERFORMANCE RATING & INSIGHTS
═══════════════════════════════════════════════════════════════════════════════════════════════════════
Overall Rating:    {rating} ({score:.1f}/10)

{self._get_performance_insights()}

╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    📈 END OF SUMMARY REPORT 📈                                        ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

        return report

    def _calculate_performance_rating(self) -> tuple[str, float]:
        """Calculate overall performance rating"""
        score = 0.0

        # Return score (40% weight)
        if self.result.total_return > 0.20:  # >20%
            score += 4.0
        elif self.result.total_return > 0.10:  # >10%
            score += 3.0
        elif self.result.total_return > 0.05:  # >5%
            score += 2.0
        elif self.result.total_return > 0:  # >0%
            score += 1.0

        # Risk score (30% weight)
        if self.result.max_drawdown < 0.05:  # <5%
            score += 3.0
        elif self.result.max_drawdown < 0.10:  # <10%
            score += 2.0
        elif self.result.max_drawdown < 0.20:  # <20%
            score += 1.0

        # Consistency score (20% weight)
        if self.result.sharpe_ratio > 1.0:
            score += 2.0
        elif self.result.sharpe_ratio > 0.5:
            score += 1.5
        elif self.result.sharpe_ratio > 0:
            score += 1.0

        # Trading score (10% weight)
        if self.result.win_rate > 0.6:
            score += 1.0
        elif self.result.win_rate > 0.5:
            score += 0.5

        # Determine rating
        if score >= 8.0:
            rating = "A"
        elif score >= 6.0:
            rating = "B"
        elif score >= 4.0:
            rating = "C"
        elif score >= 2.0:
            rating = "D"
        else:
            rating = "F"

        return rating, score

    def _get_performance_insights(self) -> str:
        """Generate performance insights"""
        insights = []

        if self.result.total_return > 0:
            insights.append("Positive returns")
        else:
            insights.append("Negative returns")

        if self.result.max_drawdown < 0.10:
            insights.append("Low drawdown")
        else:
            insights.append("High drawdown")

        if self.result.sharpe_ratio > 1.0:
            insights.append("Excellent risk-adjusted returns")
        elif self.result.sharpe_ratio > 0.5:
            insights.append("Good risk-adjusted returns")
        else:
            insights.append("Poor risk-adjusted returns")

        if self.result.win_rate > 0.6:
            insights.append("High win rate")
        elif self.result.win_rate > 0.5:
            insights.append("Moderate win rate")
        else:
            insights.append("Low win rate")

        if self.result.profit_factor > 2.0:
            insights.append("Excellent profit factor")
        elif self.result.profit_factor > 1.5:
            insights.append("Good profit factor")
        else:
            insights.append("Poor profit factor")

        return f"Strengths: {', '.join(insights[:2])}\nAreas for Improvement: {', '.join(insights[2:])}"

    def _calculate_volatility(self) -> float:
        """Calculate portfolio volatility"""
        if not self.result.equity_curve or len(self.result.equity_curve) < 2:
            return 0.0

        returns = []
        for i in range(1, len(self.result.equity_curve)):
            prev_equity = self.result.equity_curve[i - 1]["equity"]
            curr_equity = self.result.equity_curve[i]["equity"]
            if prev_equity > 0:
                returns.append((curr_equity - prev_equity) / prev_equity)

        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return (variance**0.5) * (24**0.5) * (365**0.5)  # Annualized

    def _calculate_average_win(self) -> float:
        """Calculate average winning trade"""
        if not self.result.trades:
            return 0.0

        winning_trades = [t for t in self.result.trades if t.get("pnl", 0) > 0]
        if not winning_trades:
            return 0.0

        return sum(t.get("pnl", 0) for t in winning_trades) / len(winning_trades)

    def _calculate_average_loss(self) -> float:
        """Calculate average losing trade"""
        if not self.result.trades:
            return 0.0

        losing_trades = [t for t in self.result.trades if t.get("pnl", 0) < 0]
        if not losing_trades:
            return 0.0

        return sum(t.get("pnl", 0) for t in losing_trades) / len(losing_trades)

    def generate_hourly_breakdown(self) -> str:
        """Generate detailed hourly P&L breakdown"""
        if not self.result.hourly_pnl:
            return "No hourly P&L data available"

        # Calculate summary statistics
        total_hours = len(self.result.hourly_pnl)
        profitable_hours = len(
            [h for h in self.result.hourly_pnl if h["hourly_pnl"] > 0]
        )
        losing_hours = len([h for h in self.result.hourly_pnl if h["hourly_pnl"] < 0])
        flat_hours = total_hours - profitable_hours - losing_hours

        best_hour = max((h["hourly_pnl"] for h in self.result.hourly_pnl), default=0)
        worst_hour = min((h["hourly_pnl"] for h in self.result.hourly_pnl), default=0)

        # Calculate hourly statistics
        hourly_pnls = [h["hourly_pnl"] for h in self.result.hourly_pnl]
        avg_hourly_pnl = sum(hourly_pnls) / len(hourly_pnls) if hourly_pnls else 0

        report = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    🕐 HOURLY P&L BREAKDOWN REPORT 🕐                                ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

📋 REPORT OVERVIEW
═══════════════════════════════════════════════════════════════════════════════════════════════════════
Strategy:          {self.result.strategy_name}
Test Period:       {self.result.start_date.strftime('%Y-%m-%d %H:%M UTC') if self.result.start_date else 'Unknown'} → {self.result.end_date.strftime('%Y-%m-%d %H:%M UTC') if self.result.end_date else 'Unknown'}
Total Hours:       {total_hours}

📊 HOURLY PERFORMANCE SUMMARY
═══════════════════════════════════════════════════════════════════════════════════════════════════════
Profitable Hours:  {profitable_hours:>15} ({profitable_hours/total_hours*100:>6.1f}%)
Losing Hours:      {losing_hours:>15} ({losing_hours/total_hours*100:>6.1f}%)
Flat Hours:        {flat_hours:>15} ({flat_hours/total_hours*100:>6.1f}%)
Best Hour:         ${best_hour:>15,.2f}
Worst Hour:        ${worst_hour:>15,.2f}
Average Hour:      ${avg_hourly_pnl:>15,.2f}

🕐 DETAILED HOURLY BREAKDOWN
═══════════════════════════════════════════════════════════════════════════════════════════════════════
"""

        # Group by date for better readability
        current_date = None
        daily_summary = {}

        for hour_data in self.result.hourly_pnl:
            timestamp = hour_data["timestamp"]
            date_str = timestamp.strftime("%Y-%m-%d")
            time_str = timestamp.strftime("%H:%M")

            # Initialize daily summary
            if date_str not in daily_summary:
                daily_summary[date_str] = {
                    "hours": 0,
                    "total_pnl": 0,
                    "profitable_hours": 0,
                    "losing_hours": 0,
                    "trades": 0,
                }

            # Update daily summary
            daily_summary[date_str]["hours"] += 1
            daily_summary[date_str]["total_pnl"] += hour_data["hourly_pnl"]
            if hour_data["hourly_pnl"] > 0:
                daily_summary[date_str]["profitable_hours"] += 1
            elif hour_data["hourly_pnl"] < 0:
                daily_summary[date_str]["losing_hours"] += 1
            daily_summary[date_str]["trades"] += len(hour_data.get("trades", []))

            # Add date header
            if date_str != current_date:
                report += f"\n📅 {date_str} {'='*60}\n"
                current_date = date_str

            pnl = hour_data["hourly_pnl"]
            pnl_pct = hour_data.get("hourly_pnl_pct", 0)
            equity = hour_data["equity"]
            cash = hour_data["cash"]
            exposure = hour_data["exposure"]

            # Format P&L with indicators
            if pnl > 0:
                pnl_str = f"🟢 +${pnl:,.2f}"
                pnl_pct_str = f"+{pnl_pct:.2f}%"
            elif pnl < 0:
                pnl_str = f"🔴 -${abs(pnl):,.2f}"
                pnl_pct_str = f"-{abs(pnl_pct):.2f}%"
            else:
                pnl_str = f"⚪ ${pnl:,.2f}"
                pnl_pct_str = f"{pnl_pct:.2f}%"

            # Add trade details if any
            trades_summary = ""
            if hour_data.get("trades"):
                trades_summary = f" | 🎯 Trades: {len(hour_data['trades'])}"

            # Add position details if any
            positions_summary = ""
            if hour_data.get("positions"):
                positions_summary = f" | 📊 Positions: {len(hour_data['positions'])}"

            report += f"  {time_str} | {pnl_str} ({pnl_pct_str}) | 💰 ${equity:>10,.0f} | 💵 ${cash:>8,.0f} | 📈 ${exposure:>8,.0f}{trades_summary}{positions_summary}\n"

            # Show trade details for this hour
            for trade in hour_data.get("trades", []):
                trade_pnl = trade.get("pnl", 0)
                if trade_pnl > 0:
                    trade_pnl_str = f"🟢 +${trade_pnl:,.2f}"
                elif trade_pnl < 0:
                    trade_pnl_str = f"🔴 -${abs(trade_pnl):,.2f}"
                else:
                    trade_pnl_str = f"⚪ ${trade_pnl:,.2f}"

                report += f"    → {trade['side'].upper():>4} {trade['quantity']:>8.4f} {trade['symbol']:<10} @ ${trade['price']:>8,.2f} | P&L: {trade_pnl_str} | {trade.get('reason', 'No reason')}\n"

        # Add daily summary
        report += "\n📊 DAILY PERFORMANCE SUMMARY\n"
        report += f"{'='*80}\n"
        report += f"{'Date':<12} {'Hours':>6} {'P&L':>12} {'Profitable':>10} {'Losing':>8} {'Trades':>7}\n"
        report += f"{'-'*80}\n"

        for date_str, summary in sorted(daily_summary.items()):
            pnl_str = f"${summary['total_pnl']:+,.0f}"
            report += f"{date_str:<12} {summary['hours']:>6} {pnl_str:>12} {summary['profitable_hours']:>10} {summary['losing_hours']:>8} {summary['trades']:>7}\n"

        report += "\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
        report += "║                                    📊 END OF HOURLY BREAKDOWN 📊                                    ║\n"
        report += "╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝\n"

        return report

    def generate_monthly_returns(self) -> list[dict[str, Any]]:
        """Calculate monthly returns"""
        if not self.result.equity_curve:
            return []

        monthly_data = {}

        for equity_point in self.result.equity_curve:
            timestamp = equity_point["timestamp"]
            month_key = timestamp.strftime("%Y-%m")
            equity = equity_point["equity"]

            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    "month": month_key,
                    "start_equity": equity,
                    "end_equity": equity,
                    "peak_equity": equity,
                    "low_equity": equity,
                }
            else:
                monthly_data[month_key]["end_equity"] = equity
                monthly_data[month_key]["peak_equity"] = max(
                    monthly_data[month_key]["peak_equity"], equity
                )
                monthly_data[month_key]["low_equity"] = min(
                    monthly_data[month_key]["low_equity"], equity
                )

        # Calculate returns
        monthly_returns = []
        for month_data in monthly_data.values():
            start_equity = month_data["start_equity"]
            end_equity = month_data["end_equity"]

            if start_equity > 0:
                monthly_return = (end_equity - start_equity) / start_equity
                monthly_return_pct = monthly_return * 100
            else:
                monthly_return = 0
                monthly_return_pct = 0

            monthly_returns.append(
                {
                    "month": month_data["month"],
                    "start_equity": start_equity,
                    "end_equity": end_equity,
                    "return": monthly_return,
                    "return_pct": monthly_return_pct,
                    "peak_equity": month_data["peak_equity"],
                    "low_equity": month_data["low_equity"],
                    "drawdown": (month_data["peak_equity"] - month_data["low_equity"])
                    / month_data["peak_equity"]
                    if month_data["peak_equity"] > 0
                    else 0,
                }
            )

        return sorted(monthly_returns, key=lambda x: x["month"])

    def generate_drawdown_analysis(self) -> list[dict[str, Any]]:
        """Generate detailed drawdown analysis"""
        if not self.result.equity_curve:
            return []

        drawdowns = []
        current_drawdown_start = None
        current_peak = self.result.equity_curve[0]["equity"]

        for i, equity_point in enumerate(self.result.equity_curve):
            equity = equity_point["equity"]
            timestamp = equity_point["timestamp"]

            if equity > current_peak:
                # New peak - end current drawdown if exists
                if current_drawdown_start is not None:
                    # Find the previous equity point safely
                    if i > 0:
                        drawdown_end = self.result.equity_curve[i - 1]["timestamp"]
                        drawdown_end_equity = self.result.equity_curve[i - 1]["equity"]
                        drawdown_duration = (
                            drawdown_end - current_drawdown_start
                        ).total_seconds() / 3600  # hours

                        drawdowns.append(
                            {
                                "start_time": current_drawdown_start,
                                "end_time": drawdown_end,
                                "duration_hours": drawdown_duration,
                                "start_equity": current_peak,
                                "end_equity": drawdown_end_equity,
                                "drawdown_amount": current_peak - drawdown_end_equity,
                                "drawdown_pct": (current_peak - drawdown_end_equity)
                                / current_peak,
                            }
                        )

                current_peak = equity
                current_drawdown_start = None

            elif equity < current_peak and current_drawdown_start is None:
                # Start of new drawdown
                current_drawdown_start = timestamp

        # Handle final drawdown if still ongoing
        if current_drawdown_start is not None:
            final_equity = self.result.equity_curve[-1]["equity"]
            final_timestamp = self.result.equity_curve[-1]["timestamp"]
            drawdown_duration = (
                final_timestamp - current_drawdown_start
            ).total_seconds() / 3600

            drawdowns.append(
                {
                    "start_time": current_drawdown_start,
                    "end_time": final_timestamp,
                    "duration_hours": drawdown_duration,
                    "start_equity": current_peak,
                    "end_equity": final_equity,
                    "drawdown_amount": current_peak - final_equity,
                    "drawdown_pct": (current_peak - final_equity) / current_peak,
                }
            )

        # Sort by drawdown percentage (worst first)
        return sorted(drawdowns, key=lambda x: x["drawdown_pct"], reverse=True)

    def export_results(self, output_dir: str) -> list[str]:
        """Export all results to organized files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create timestamp for file organization
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        strategy_safe_name = (
            self.result.strategy_name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )

        exported_files = []

        # 1. MASTER SUMMARY REPORT (Main report)
        summary_file = (
            output_path / f"📊_{strategy_safe_name}_MASTER_SUMMARY_{timestamp}.txt"
        )
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(self.generate_summary_report())
        exported_files.append(str(summary_file))

        # 2. HOURLY BREAKDOWN REPORT
        hourly_file = (
            output_path / f"🕐_{strategy_safe_name}_HOURLY_BREAKDOWN_{timestamp}.txt"
        )
        with open(hourly_file, "w", encoding="utf-8") as f:
            f.write(self.generate_hourly_breakdown())
        exported_files.append(str(hourly_file))

        # 3. DETAILED JSON RESULTS (Raw data)
        detailed_file = (
            output_path / f"📈_{strategy_safe_name}_DETAILED_DATA_{timestamp}.json"
        )
        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(self.result.dict(), f, indent=2, default=str, ensure_ascii=False)
        exported_files.append(str(detailed_file))

        # 4. MONTHLY RETURNS ANALYSIS
        monthly_returns = self.generate_monthly_returns()
        monthly_file = (
            output_path / f"📅_{strategy_safe_name}_MONTHLY_RETURNS_{timestamp}.json"
        )
        with open(monthly_file, "w", encoding="utf-8") as f:
            json.dump(monthly_returns, f, indent=2, default=str, ensure_ascii=False)
        exported_files.append(str(monthly_file))

        # 5. DRAWDOWN ANALYSIS
        drawdown_analysis = self.generate_drawdown_analysis()
        drawdown_file = (
            output_path / f"📉_{strategy_safe_name}_DRAWDOWN_ANALYSIS_{timestamp}.json"
        )
        with open(drawdown_file, "w", encoding="utf-8") as f:
            json.dump(drawdown_analysis, f, indent=2, default=str, ensure_ascii=False)
        exported_files.append(str(drawdown_file))

        # 6. MASTER INDEX REPORT (File organization guide)
        index_file = (
            output_path / f"📋_{strategy_safe_name}_REPORTS_INDEX_{timestamp}.txt"
        )
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(self._generate_reports_index(timestamp, strategy_safe_name))
        exported_files.append(str(index_file))

        # 7. PERFORMANCE METRICS SUMMARY (CSV format for analysis)
        metrics_file = (
            output_path / f"📊_{strategy_safe_name}_PERFORMANCE_METRICS_{timestamp}.csv"
        )
        with open(metrics_file, "w", encoding="utf-8") as f:
            f.write(self._generate_metrics_csv())
        exported_files.append(str(metrics_file))

        return exported_files

    def _generate_reports_index(self, timestamp: str, strategy_name: str) -> str:
        """Generate a master index of all exported reports"""
        index = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    📋 REPORTS INDEX & ORGANIZATION 📋                                ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

📋 REPORT OVERVIEW
═══════════════════════════════════════════════════════════════════════════════════════════════════════
Strategy:          {self.result.strategy_name}
Test Date:         {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}
Report Timestamp:  {timestamp}
Total Files:       7

📁 FILE ORGANIZATION
═══════════════════════════════════════════════════════════════════════════════════════════════════════
1. 📊 MASTER SUMMARY REPORT
   File:           📊_{strategy_name}_MASTER_SUMMARY_{timestamp}.txt
   Purpose:        Executive summary with key performance metrics
   Content:        Performance overview, risk metrics, trading statistics, strategy parameters
   Best For:       Quick overview and decision making

2. 🕐 HOURLY BREAKDOWN REPORT
   File:           🕐_{strategy_name}_HOURLY_BREAKDOWN_{timestamp}.txt
   Purpose:        Detailed hourly P&L analysis with trade details
   Content:        Hour-by-hour performance, daily summaries, trade execution details
   Best For:       Detailed analysis and debugging

3. 📈 DETAILED DATA EXPORT
   File:           📈_{strategy_name}_DETAILED_DATA_{timestamp}.json
   Purpose:        Complete raw data for custom analysis
   Content:        All backtest data in JSON format
   Best For:       Data scientists, custom analysis, importing to other tools

4. 📅 MONTHLY RETURNS ANALYSIS
   File:           📅_{strategy_name}_MONTHLY_RETURNS_{timestamp}.json
   Purpose:        Monthly performance breakdown
   Content:        Month-by-month returns, peaks, and valleys
   Best For:       Long-term trend analysis

5. 📉 DRAWDOWN ANALYSIS
   File:           📉_{strategy_name}_DRAWDOWN_ANALYSIS_{timestamp}.json
   Purpose:        Risk analysis and drawdown periods
   Content:        Drawdown periods, duration, recovery analysis
   Best For:       Risk management and stress testing

6. 📊 PERFORMANCE METRICS CSV
   File:           📊_{strategy_name}_PERFORMANCE_METRICS_{timestamp}.csv
   Purpose:        Spreadsheet-friendly performance data
   Content:        Key metrics in CSV format for Excel/Google Sheets
   Best For:       Portfolio managers, analysts, reporting

7. 📋 THIS INDEX FILE
   File:           📋_{strategy_name}_REPORTS_INDEX_{timestamp}.txt
   Purpose:        File organization and navigation guide
   Content:        Overview of all generated reports
   Best For:       Understanding the complete reporting structure

🎯 RECOMMENDED READING ORDER
═══════════════════════════════════════════════════════════════════════════════════════════════════════
1. Start with:     📊 MASTER SUMMARY REPORT (executive overview)
2. Then review:    🕐 HOURLY BREAKDOWN REPORT (detailed analysis)
3. For analysis:   📊 PERFORMANCE METRICS CSV (spreadsheet analysis)
4. For deep dive:  📈 DETAILED DATA EXPORT (custom analysis)
5. For risk:       📉 DRAWDOWN ANALYSIS (risk assessment)

📊 KEY PERFORMANCE HIGHLIGHTS
═══════════════════════════════════════════════════════════════════════════════════════════════════════
Initial Capital:   ${self.result.initial_capital:>15,.2f}
Final Portfolio:   ${self.result.final_capital:>15,.2f}
Total Return:      {self.result.total_return:>15.2%}
Total Trades:      {self.result.total_trades:>15}
Win Rate:          {self.result.win_rate:>15.2%}
Max Drawdown:      {self.result.max_drawdown:>15.2%}

╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    🎯 END OF REPORTS INDEX 🎯                                        ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""
        return index

    def _generate_metrics_csv(self) -> str:
        """Generate CSV format performance metrics"""
        csv_header = "Metric,Value,Description\n"
        csv_data = [
            [
                "Strategy Name",
                self.result.strategy_name,
                "Name of the trading strategy",
            ],
            [
                "Start Date",
                self.result.start_date.strftime("%Y-%m-%d")
                if self.result.start_date
                else "Unknown",
                "Backtest start date",
            ],
            [
                "End Date",
                self.result.end_date.strftime("%Y-%m-%d")
                if self.result.end_date
                else "Unknown",
                "Backtest end date",
            ],
            [
                "Initial Capital",
                f"${self.result.initial_capital:,.2f}",
                "Starting portfolio value",
            ],
            [
                "Final Capital",
                f"${self.result.final_capital:,.2f}",
                "Ending portfolio value",
            ],
            [
                "Total Return",
                f"{self.result.total_return:+.2%}",
                "Total percentage return",
            ],
            [
                "Absolute Gain",
                f"${self.result.final_capital - self.result.initial_capital:+,.2f}",
                "Absolute dollar gain/loss",
            ],
            [
                "Annualized Return",
                f"{self.result.annualized_return:+.2%}",
                "Annualized return rate",
            ],
            [
                "Max Drawdown",
                f"{self.result.max_drawdown:.2%}",
                "Maximum portfolio drawdown",
            ],
            [
                "Sharpe Ratio",
                f"{self.result.sharpe_ratio:.3f}",
                "Risk-adjusted return metric",
            ],
            [
                "Total Trades",
                str(self.result.total_trades),
                "Total number of trades executed",
            ],
            ["Win Rate", f"{self.result.win_rate:.2%}", "Percentage of winning trades"],
            [
                "Profit Factor",
                f"{self.result.profit_factor:.3f}",
                "Ratio of gross profit to gross loss",
            ],
            [
                "Volatility",
                f"{self._calculate_volatility():.2%}",
                "Portfolio volatility (annualized)",
            ],
            [
                "Risk-Adjusted Return",
                f"{self.result.sharpe_ratio * self.result.total_return:.2%}",
                "Sharpe ratio × total return",
            ],
        ]

        csv_content = csv_header
        for row in csv_data:
            csv_content += f'"{row[0]}","{row[1]}","{row[2]}"\n'

        return csv_content
