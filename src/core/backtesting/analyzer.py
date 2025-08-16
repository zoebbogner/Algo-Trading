"""Performance analyzer for backtest results."""

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Any, Optional
from pathlib import Path

from .engine import BacktestResult
from ...utils.logging import logger


class PerformanceAnalyzer:
    """Analyzes and visualizes backtest performance."""
    
    def __init__(self, result: BacktestResult):
        """Initialize analyzer with backtest result.
        
        Args:
            result: BacktestResult to analyze
        """
        self.result = result
        logger.logger.info(f"Initialized performance analyzer for {result.strategy_name}")
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report.
        
        Returns:
            Formatted summary report string
        """
        report = []
        report.append("=" * 80)
        report.append(f"BACKTEST SUMMARY REPORT")
        report.append("=" * 80)
        report.append(f"Strategy: {self.result.strategy_name}")
        report.append(f"Period: {self.result.start_date.strftime('%Y-%m-%d')} to {self.result.end_date.strftime('%Y-%m-%d')}")
        report.append(f"Duration: {self.result.duration_days} days")
        report.append("")
        
        # Performance Overview
        report.append("📊 PERFORMANCE OVERVIEW")
        report.append("-" * 40)
        report.append(f"Initial Capital: ${self.result.final_portfolio_value - self.result.total_return:,.2f}")
        report.append(f"Final Portfolio: ${self.result.final_portfolio_value:,.2f}")
        report.append(f"Total Return: ${self.result.total_return:,.2f} ({self.result.total_return_pct:.2f}%)")
        report.append(f"Annualized Return: {self.result.annualized_return:.2f}%")
        report.append(f"Peak Portfolio: ${self.result.peak_portfolio_value:,.2f}")
        report.append("")
        
        # Risk Metrics
        report.append("⚠️ RISK METRICS")
        report.append("-" * 40)
        report.append(f"Max Drawdown: {self.result.max_drawdown_pct:.2f}%")
        report.append(f"Sharpe Ratio: {self.result.sharpe_ratio:.3f}")
        report.append(f"Final Cash: ${self.result.final_cash:,.2f}")
        report.append(f"Final Exposure: ${self.result.final_exposure:,.2f}")
        report.append("")
        
        # Trading Statistics
        report.append("🔄 TRADING STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Trades: {self.result.total_trades}")
        report.append(f"Winning Trades: {self.result.winning_trades}")
        report.append(f"Losing Trades: {self.result.losing_trades}")
        report.append(f"Win Rate: {self.result.win_rate:.2%}")
        report.append(f"Profit Factor: {self.result.profit_factor:.3f}")
        report.append("")
        
        # Trade Analysis
        if self.result.total_trades > 0:
            report.append("💰 TRADE ANALYSIS")
            report.append("-" * 40)
            report.append(f"Average Win: {self.result.avg_win:.2%}")
            report.append(f"Average Loss: {self.result.avg_loss:.2%}")
            report.append(f"Largest Win: {self.result.largest_win:.2%}")
            report.append(f"Largest Loss: {self.result.largest_loss:.2%}")
            report.append("")
        
        # Strategy Parameters
        if self.result.parameters:
            report.append("⚙️ STRATEGY PARAMETERS")
            report.append("-" * 40)
            for key, value in self.result.parameters.items():
                report.append(f"{key}: {value}")
            report.append("")
        
        # Performance Rating
        report.append("🏆 PERFORMANCE RATING")
        report.append("-" * 40)
        rating = self._calculate_performance_rating()
        report.append(f"Overall Rating: {rating['grade']} ({rating['score']:.1f}/10)")
        report.append(f"Strengths: {', '.join(rating['strengths'])}")
        report.append(f"Areas for Improvement: {', '.join(rating['improvements'])}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _calculate_performance_rating(self) -> Dict[str, Any]:
        """Calculate overall performance rating.
        
        Returns:
            Dictionary with rating details
        """
        score = 0
        strengths = []
        improvements = []
        
        # Return scoring
        if self.result.total_return_pct > 20:
            score += 2
            strengths.append("Strong returns")
        elif self.result.total_return_pct > 10:
            score += 1.5
            strengths.append("Good returns")
        elif self.result.total_return_pct > 0:
            score += 1
            strengths.append("Positive returns")
        else:
            improvements.append("Negative returns")
        
        # Risk scoring
        if self.result.max_drawdown_pct < 10:
            score += 2
            strengths.append("Low drawdown")
        elif self.result.max_drawdown_pct < 20:
            score += 1.5
            strengths.append("Moderate drawdown")
        elif self.result.max_drawdown_pct < 30:
            score += 1
            strengths.append("Acceptable drawdown")
        else:
            improvements.append("High drawdown")
        
        # Sharpe ratio scoring
        if self.result.sharpe_ratio > 1.5:
            score += 2
            strengths.append("Excellent risk-adjusted returns")
        elif self.result.sharpe_ratio > 1.0:
            score += 1.5
            strengths.append("Good risk-adjusted returns")
        elif self.result.sharpe_ratio > 0.5:
            score += 1
            strengths.append("Positive risk-adjusted returns")
        else:
            improvements.append("Poor risk-adjusted returns")
        
        # Win rate scoring
        if self.result.win_rate > 0.6:
            score += 1.5
            strengths.append("High win rate")
        elif self.result.win_rate > 0.5:
            score += 1
            strengths.append("Positive win rate")
        else:
            improvements.append("Low win rate")
        
        # Profit factor scoring
        if self.result.profit_factor > 2.0:
            score += 1.5
            strengths.append("Excellent profit factor")
        elif self.result.profit_factor > 1.5:
            score += 1
            strengths.append("Good profit factor")
        else:
            improvements.append("Poor profit factor")
        
        # Normalize score to 0-10
        score = min(10, max(0, score))
        
        # Determine grade
        if score >= 8.5:
            grade = "A+"
        elif score >= 7.5:
            grade = "A"
        elif score >= 6.5:
            grade = "B+"
        elif score >= 5.5:
            grade = "B"
        elif score >= 4.5:
            grade = "C+"
        elif score >= 3.5:
            grade = "C"
        elif score >= 2.5:
            grade = "D+"
        elif score >= 1.5:
            grade = "D"
        else:
            grade = "F"
        
        return {
            'score': score,
            'grade': grade,
            'strengths': strengths,
            'improvements': improvements
        }
    
    def generate_monthly_returns(self) -> Dict[str, float]:
        """Generate monthly returns breakdown.
        
        Returns:
            Dictionary mapping month to return percentage
        """
        monthly_returns = {}
        
        if not self.result.equity_curve:
            return monthly_returns
        
        # Group equity curve by month
        current_month = None
        month_start_equity = None
        
        for state in self.result.equity_curve:
            month_key = state['timestamp'].strftime('%Y-%m')
            
            if month_key != current_month:
                if current_month and month_start_equity:
                    # Calculate return for previous month
                    month_end_equity = state['equity']
                    month_return = ((month_end_equity - month_start_equity) / month_start_equity) * 100
                    monthly_returns[current_month] = month_return
                
                current_month = month_key
                month_start_equity = state['equity']
        
        # Calculate return for last month
        if current_month and month_start_equity and self.result.equity_curve:
            last_equity = self.result.equity_curve[-1]['equity']
            month_return = ((last_equity - month_start_equity) / month_start_equity) * 100
            monthly_returns[current_month] = month_return
        
        return monthly_returns
    
    def generate_drawdown_analysis(self) -> Dict[str, Any]:
        """Generate detailed drawdown analysis.
        
        Returns:
            Dictionary with drawdown statistics
        """
        if not self.result.equity_curve:
            return {}
        
        # Find drawdown periods
        drawdown_periods = []
        current_drawdown_start = None
        current_drawdown_end = None
        current_peak = 0
        
        for i, state in enumerate(self.result.equity_curve):
            equity = state['equity']
            timestamp = state['timestamp']
            
            if equity > current_peak:
                current_peak = equity
                # End of drawdown period
                if current_drawdown_start:
                    drawdown_periods.append({
                        'start': current_drawdown_start,
                        'end': current_drawdown_end,
                        'peak': current_peak,
                        'trough': min(state['equity'] for state in self.result.equity_curve[
                            self.result.equity_curve.index(current_drawdown_start):
                            self.result.equity_curve.index(current_drawdown_end) + 1
                        ]),
                        'duration_days': (current_drawdown_end - current_drawdown_start).days
                    })
                    current_drawdown_start = None
            elif equity < current_peak and not current_drawdown_start:
                # Start of drawdown period
                current_drawdown_start = timestamp
                current_drawdown_end = timestamp
        
        # Handle ongoing drawdown
        if current_drawdown_start:
            drawdown_periods.append({
                'start': current_drawdown_start,
                'end': self.result.equity_curve[-1]['timestamp'],
                'peak': current_peak,
                'trough': min(state['equity'] for state in self.result.equity_curve[
                    self.result.equity_curve.index(current_drawdown_start):
                ]),
                'duration_days': (self.result.equity_curve[-1]['timestamp'] - current_drawdown_start).days
            })
        
        # Calculate drawdown statistics
        if drawdown_periods:
            max_drawdown_period = max(drawdown_periods, key=lambda x: x['peak'] - x['trough'])
            avg_drawdown_duration = sum(d['duration_days'] for d in drawdown_periods) / len(drawdown_periods)
            
            return {
                'total_periods': len(drawdown_periods),
                'max_drawdown_period': max_drawdown_period,
                'avg_duration_days': avg_drawdown_duration,
                'all_periods': drawdown_periods
            }
        
        return {}
    
    def export_results(self, output_dir: str = "reports") -> Dict[str, str]:
        """Export backtest results to files.
        
        Args:
            output_dir: Directory to save reports
            
        Returns:
            Dictionary mapping report type to file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = self.result.strategy_name.replace(" ", "_").lower()
        
        exported_files = {}
        
        # Export summary report
        summary_path = output_path / f"backtest_summary_{strategy_name}_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(self.generate_summary_report())
        exported_files['summary'] = str(summary_path)
        
        # Export detailed results as JSON
        results_path = output_path / f"backtest_results_{strategy_name}_{timestamp}.json"
        
        # Convert Decimal objects to float for JSON serialization
        results_data = {
            'strategy_name': self.result.strategy_name,
            'start_date': self.result.start_date.isoformat(),
            'end_date': self.result.end_date.isoformat(),
            'duration_days': self.result.duration_days,
            'performance_metrics': {
                'total_return': float(self.result.total_return),
                'total_return_pct': float(self.result.total_return_pct),
                'annualized_return': float(self.result.annualized_return),
                'sharpe_ratio': float(self.result.sharpe_ratio),
                'max_drawdown': float(self.result.max_drawdown),
                'max_drawdown_pct': float(self.result.max_drawdown_pct),
                'win_rate': float(self.result.win_rate),
                'profit_factor': float(self.result.profit_factor)
            },
            'trading_statistics': {
                'total_trades': self.result.total_trades,
                'winning_trades': self.result.winning_trades,
                'losing_trades': self.result.losing_trades,
                'avg_win': float(self.result.avg_win),
                'avg_loss': float(self.result.avg_loss),
                'largest_win': float(self.result.largest_win),
                'largest_loss': float(self.result.largest_loss)
            },
            'portfolio_metrics': {
                'final_portfolio_value': float(self.result.final_portfolio_value),
                'peak_portfolio_value': float(self.result.peak_portfolio_value),
                'final_cash': float(self.result.final_cash),
                'final_exposure': float(self.result.final_exposure)
            },
            'strategy_parameters': self.result.parameters,
            'equity_curve': self.result.equity_curve,
            'trades': self.result.trades,
            'positions': self.result.positions
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        exported_files['detailed_results'] = str(results_path)
        
        # Export monthly returns
        monthly_returns = self.generate_monthly_returns()
        if monthly_returns:
            monthly_path = output_path / f"monthly_returns_{strategy_name}_{timestamp}.json"
            with open(monthly_path, 'w') as f:
                json.dump(monthly_returns, f, indent=2)
            exported_files['monthly_returns'] = str(monthly_path)
        
        # Export drawdown analysis
        drawdown_analysis = self.generate_drawdown_analysis()
        if drawdown_analysis:
            drawdown_path = output_path / f"drawdown_analysis_{strategy_name}_{timestamp}.json"
            with open(drawdown_path, 'w') as f:
                json.dump(drawdown_analysis, f, indent=2, default=str)
            exported_files['drawdown_analysis'] = str(drawdown_path)
        
        logger.logger.info(f"Exported backtest results to {output_dir}")
        return exported_files
    
    def print_summary(self) -> None:
        """Print summary to console."""
        print(self.generate_summary_report())
    
    def get_key_metrics(self) -> Dict[str, Any]:
        """Get key performance metrics.
        
        Returns:
            Dictionary with key metrics
        """
        return {
            'total_return_pct': float(self.result.total_return_pct),
            'annualized_return': float(self.result.annualized_return),
            'sharpe_ratio': float(self.result.sharpe_ratio),
            'max_drawdown_pct': float(self.result.max_drawdown_pct),
            'win_rate': float(self.result.win_rate),
            'profit_factor': float(self.result.profit_factor),
            'total_trades': self.result.total_trades
        }
