"""
Performance Analyzer for Backtest Results

Analyzes and visualizes backtest performance with comprehensive metrics
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import json
import os
from pathlib import Path

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
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    hourly_pnl: List[Dict[str, Any]]  # New: Hourly P&L tracking
    portfolio_snapshots: List[Dict[str, Any]]  # New: Portfolio state snapshots
    parameters: Dict[str, Any]


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
        duration_days = (self.result.end_date - self.result.start_date).days if self.result.start_date and self.result.end_date else 0
        peak_equity = max(equity['equity'] for equity in self.result.equity_curve)
        
        # Performance rating
        rating, score = self._calculate_performance_rating()
        
        report = f"""
================================================================================
BACKTEST SUMMARY REPORT
================================================================================
Strategy: {self.result.strategy_name}
Period: {self.result.start_date.strftime('%Y-%m-%d') if self.result.start_date else 'Unknown'} to {self.result.end_date.strftime('%Y-%m-%d') if self.result.end_date else 'Unknown'}
Duration: {duration_days} days

📊 PERFORMANCE OVERVIEW
----------------------------------------
Initial Capital: ${self.result.initial_capital:,.2f}
Final Portfolio: ${self.result.final_capital:,.2f}
Total Return: ${self.result.final_capital - self.result.initial_capital:+,.2f} ({self.result.total_return:+.2%})
Annualized Return: {self.result.annualized_return:+.2%}
Peak Portfolio: ${peak_equity:,.2f}

⚠️ RISK METRICS
----------------------------------------
Max Drawdown: {self.result.max_drawdown:.2%}
Sharpe Ratio: {self.result.sharpe_ratio:.3f}
Final Cash: ${self.result.final_capital:,.2f}
Final Exposure: ${sum(abs(pos.get('market_value', 0)) for pos in self.result.hourly_pnl[-1]['positions']) if self.result.hourly_pnl else 0:,.2f}

🔄 TRADING STATISTICS
----------------------------------------
Total Trades: {self.result.total_trades}
Winning Trades: {int(self.result.total_trades * self.result.win_rate)}
Losing Trades: {int(self.result.total_trades * (1 - self.result.win_rate))}
Win Rate: {self.result.win_rate:.2%}
Profit Factor: {self.result.profit_factor:.3f}

⚙️ STRATEGY PARAMETERS
----------------------------------------
"""
        
        for key, value in self.result.parameters.items():
            report += f"{key}: {value}\n"
        
        report += f"""
🏆 PERFORMANCE RATING
----------------------------------------
Overall Rating: {rating} ({score:.1f}/10)
{self._get_performance_insights()}
================================================================================
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
    
    def generate_hourly_breakdown(self) -> str:
        """Generate detailed hourly P&L breakdown"""
        if not self.result.hourly_pnl:
            return "No hourly P&L data available"
        
        report = f"""
================================================================================
HOURLY P&L BREAKDOWN
================================================================================
Strategy: {self.result.strategy_name}
Period: {self.result.start_date.strftime('%Y-%m-%d %H:%M') if self.result.start_date else 'Unknown'} to {self.result.end_date.strftime('%Y-%m-%d %H:%M') if self.result.end_date else 'Unknown'}

📊 HOURLY PERFORMANCE SUMMARY
----------------------------------------
Total Hours: {len(self.result.hourly_pnl)}
Profitable Hours: {len([h for h in self.result.hourly_pnl if h['hourly_pnl'] > 0])}
Losing Hours: {len([h for h in self.result.hourly_pnl if h['hourly_pnl'] < 0])}
Best Hour: ${max((h['hourly_pnl'] for h in self.result.hourly_pnl), default=0):,.2f}
Worst Hour: ${min((h['hourly_pnl'] for h in self.result.hourly_pnl), default=0):,.2f}

🕐 DETAILED HOURLY BREAKDOWN
----------------------------------------
"""
        
        # Group by date for better readability
        current_date = None
        for hour_data in self.result.hourly_pnl:
            timestamp = hour_data['timestamp']
            date_str = timestamp.strftime('%Y-%m-%d')
            time_str = timestamp.strftime('%H:%M')
            
            if date_str != current_date:
                report += f"\n📅 {date_str}\n"
                current_date = date_str
            
            pnl = hour_data['hourly_pnl']
            pnl_pct = hour_data['hourly_pnl_pct']
            equity = hour_data['equity']
            cash = hour_data['cash']
            exposure = hour_data['exposure']
            
            # Format P&L with color indicators
            pnl_str = f"${pnl:+,.2f}" if pnl != 0 else "$0.00"
            pnl_pct_str = f"{pnl_pct:+.2f}%" if pnl_pct != 0 else "0.00%"
            
            # Add trade details if any
            trades_summary = ""
            if hour_data['trades']:
                trades_summary = f" | Trades: {len(hour_data['trades'])}"
            
            # Add position details if any
            positions_summary = ""
            if hour_data['positions']:
                positions_summary = f" | Positions: {len(hour_data['positions'])}"
            
            report += f"  {time_str}: {pnl_str} ({pnl_pct_str}) | Equity: ${equity:,.2f} | Cash: ${cash:,.2f} | Exposure: ${exposure:,.2f}{trades_summary}{positions_summary}\n"
            
            # Show trade details for this hour
            for trade in hour_data['trades']:
                trade_pnl = trade.get('pnl', 0)
                trade_pnl_str = f"${trade_pnl:+,.2f}" if trade_pnl != 0 else "$0.00"
                report += f"    → {trade['side'].upper()} {trade['quantity']} {trade['symbol']} @ ${trade['price']:,.2f} | P&L: {trade_pnl_str} | {trade.get('reason', 'No reason')}\n"
        
        report += "\n================================================================================\n"
        return report
    
    def generate_monthly_returns(self) -> List[Dict[str, Any]]:
        """Calculate monthly returns"""
        if not self.result.equity_curve:
            return []
        
        monthly_data = {}
        
        for equity_point in self.result.equity_curve:
            timestamp = equity_point['timestamp']
            month_key = timestamp.strftime('%Y-%m')
            equity = equity_point['equity']
            
            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    'month': month_key,
                    'start_equity': equity,
                    'end_equity': equity,
                    'peak_equity': equity,
                    'low_equity': equity
                }
            else:
                monthly_data[month_key]['end_equity'] = equity
                monthly_data[month_key]['peak_equity'] = max(monthly_data[month_key]['peak_equity'], equity)
                monthly_data[month_key]['low_equity'] = min(monthly_data[month_key]['low_equity'], equity)
        
        # Calculate returns
        monthly_returns = []
        for month_data in monthly_data.values():
            start_equity = month_data['start_equity']
            end_equity = month_data['end_equity']
            
            if start_equity > 0:
                monthly_return = (end_equity - start_equity) / start_equity
                monthly_return_pct = monthly_return * 100
            else:
                monthly_return = 0
                monthly_return_pct = 0
            
            monthly_returns.append({
                'month': month_data['month'],
                'start_equity': start_equity,
                'end_equity': end_equity,
                'return': monthly_return,
                'return_pct': monthly_return_pct,
                'peak_equity': month_data['peak_equity'],
                'low_equity': month_data['low_equity'],
                'drawdown': (month_data['peak_equity'] - month_data['low_equity']) / month_data['peak_equity'] if month_data['peak_equity'] > 0 else 0
            })
        
        return sorted(monthly_returns, key=lambda x: x['month'])
    
    def generate_drawdown_analysis(self) -> List[Dict[str, Any]]:
        """Generate detailed drawdown analysis"""
        if not self.result.equity_curve:
            return []
        
        drawdowns = []
        current_drawdown_start = None
        current_peak = self.result.equity_curve[0]['equity']
        
        for i, equity_point in enumerate(self.result.equity_curve):
            equity = equity_point['equity']
            timestamp = equity_point['timestamp']
            
            if equity > current_peak:
                # New peak - end current drawdown if exists
                if current_drawdown_start is not None:
                    # Find the previous equity point safely
                    if i > 0:
                        drawdown_end = self.result.equity_curve[i-1]['timestamp']
                        drawdown_end_equity = self.result.equity_curve[i-1]['equity']
                        drawdown_duration = (drawdown_end - current_drawdown_start).total_seconds() / 3600  # hours
                        
                        drawdowns.append({
                            'start_time': current_drawdown_start,
                            'end_time': drawdown_end,
                            'duration_hours': drawdown_duration,
                            'start_equity': current_peak,
                            'end_equity': drawdown_end_equity,
                            'drawdown_amount': current_peak - drawdown_end_equity,
                            'drawdown_pct': (current_peak - drawdown_end_equity) / current_peak
                        })
                
                current_peak = equity
                current_drawdown_start = None
            
            elif equity < current_peak and current_drawdown_start is None:
                # Start of new drawdown
                current_drawdown_start = timestamp
        
        # Handle final drawdown if still ongoing
        if current_drawdown_start is not None:
            final_equity = self.result.equity_curve[-1]['equity']
            final_timestamp = self.result.equity_curve[-1]['timestamp']
            drawdown_duration = (final_timestamp - current_drawdown_start).total_seconds() / 3600
            
            drawdowns.append({
                'start_time': current_drawdown_start,
                'end_time': final_timestamp,
                'duration_hours': drawdown_duration,
                'start_equity': current_peak,
                'end_equity': final_equity,
                'drawdown_amount': current_peak - final_equity,
                'drawdown_pct': (current_peak - final_equity) / current_peak
            })
        
        # Sort by drawdown percentage (worst first)
        return sorted(drawdowns, key=lambda x: x['drawdown_pct'], reverse=True)
    
    def export_results(self, output_dir: str) -> List[str]:
        """Export all results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        # Summary report
        summary_file = output_path / f"{self.result.strategy_name}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(self.generate_summary_report())
        exported_files.append(str(summary_file))
        
        # Hourly breakdown
        hourly_file = output_path / f"{self.result.strategy_name}_hourly.txt"
        with open(hourly_file, 'w') as f:
            f.write(self.generate_hourly_breakdown())
        exported_files.append(str(hourly_file))
        
        # Detailed JSON results
        detailed_file = output_path / f"{self.result.strategy_name}_detailed.json"
        with open(detailed_file, 'w') as f:
            json.dump(self.result.dict(), f, indent=2, default=str)
        exported_files.append(str(detailed_file))
        
        # Monthly returns
        monthly_returns = self.generate_monthly_returns()
        monthly_file = output_path / f"{self.result.strategy_name}_monthly.json"
        with open(monthly_file, 'w') as f:
            json.dump(monthly_returns, f, indent=2, default=str)
        exported_files.append(str(monthly_file))
        
        # Drawdown analysis
        drawdown_analysis = self.generate_drawdown_analysis()
        drawdown_file = output_path / f"{self.result.strategy_name}_drawdown.json"
        with open(drawdown_file, 'w') as f:
            json.dump(drawdown_analysis, f, indent=2, default=str)
        exported_files.append(str(drawdown_file))
        
        return exported_files
