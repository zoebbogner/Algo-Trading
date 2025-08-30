"""
Trading Knowledge RAG System
Extracts concise insights from past trading performance for AI learning.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class TradingKnowledgeRAG:
    """RAG system for extracting and retrieving trading knowledge from past performance."""
    
    def __init__(self, knowledge_dir: str = "data/knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_file = self.knowledge_dir / "trading_knowledge.json"
        
    def extract_trading_insights(self, backtest_results: Dict) -> Dict:
        """Extract comprehensive trading insights from real backtest data."""
        try:
            insights = {
                'timestamp': datetime.now().isoformat(),
                'performance_summary': {},
                'key_patterns': [],
                'risk_insights': {},
                'actionable_lessons': [],
                'recent_trades': [],
                'trade_patterns': {},
                'portfolio_analysis': {},
                'execution_quality': {}
            }
            
            # Extract performance metrics from real data
            win_loss = backtest_results.get('win_loss_summary', {})
            if win_loss:
                total_trades = win_loss.get('total_trades', 0)
                winning_trades = win_loss.get('winning_trades', 0)
                losing_trades = win_loss.get('losing_trades', 0)
                open_trades = win_loss.get('open_trades', 0)
                
                if total_trades > 0:
                    hit_rate = winning_trades / total_trades
                    closed_rate = (winning_trades + losing_trades) / total_trades
                    
                    insights['performance_summary'] = {
                        'total_trades': total_trades,
                        'hit_rate': round(hit_rate, 3),
                        'closed_positions': round(closed_rate, 3),
                        'open_positions': open_trades
                    }
                    
                    # Extract comprehensive trade details from real backtest files
                    if 'all_backtests' in backtest_results:
                        all_backtests = backtest_results.get('all_backtests', [])
                        if all_backtests:
                            # Extract detailed trade data from actual files
                            detailed_trades = self._extract_comprehensive_trade_data(all_backtests)
                            insights['recent_trades'] = detailed_trades[:5]  # Keep last 5 trades
                            
                            # Analyze real trade patterns
                            insights['trade_patterns'] = self._analyze_real_trade_patterns(detailed_trades)
                            
                            # Extract portfolio analysis from real equity data
                            insights['portfolio_analysis'] = self._extract_portfolio_analysis(all_backtests)
                            
                            # Extract execution quality metrics
                            insights['execution_quality'] = self._extract_execution_quality(all_backtests)
                    
                    # Generate insights based on real data patterns
                    insights['key_patterns'] = self._generate_real_patterns(hit_rate, open_trades, total_trades, insights)
                    
                    # Risk insights based on actual performance
                    insights['risk_insights'] = self._generate_real_risk_insights(hit_rate, open_trades, total_trades, insights)
                    
                    # Actionable lessons based on real data
                    insights['actionable_lessons'] = self._generate_real_lessons(hit_rate, open_trades, total_trades, insights)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error extracting trading insights: {e}")
            return {}
    
    def _extract_trade_details(self, recent_performance_html: str) -> List[Dict]:
        """Extract trade details from recent performance HTML."""
        try:
            trades = []
            
            # Parse the HTML content to extract trade info
            lines = recent_performance_html.split('\n')
            
            for line_index, line in enumerate(lines):
                line = line.strip()
                if 'bt-cell-symbol' in line and 'span' in line:
                    # Extract symbol
                    symbol_start = line.find('>') + 1
                    symbol_end = line.find('<', symbol_start)
                    symbol = line[symbol_start:symbol_end].strip()
                    
                    # Look for the next few lines to get return, sharpe, and date
                    trade_data = {'symbol': symbol}
                    
                    # Find return, sharpe, and date in subsequent lines
                    for i in range(line_index + 1, min(line_index + 10, len(lines))):
                        next_line = lines[i].strip()
                        
                        if 'bt-cell-return' in next_line and 'total_return' not in trade_data:
                            return_start = next_line.find('>') + 1
                            return_end = next_line.find('<', return_start)
                            trade_data['total_return'] = next_line[return_start:return_end].strip()
                        
                        elif 'bt-cell-sharpe' in next_line and 'sharpe_ratio' not in trade_data:
                            sharpe_start = next_line.find('>') + 1
                            sharpe_end = next_line.find('<', sharpe_start)
                            trade_data['sharpe_ratio'] = next_line[sharpe_start:sharpe_end].strip()
                        
                        elif 'bt-cell-date' in next_line and 'date' not in trade_data:
                            date_start = next_line.find('>') + 1
                            date_end = next_line.find('<', date_start)
                            trade_data['date'] = next_line[date_start:date_end].strip()
                    
                    # Only add if we have at least symbol and return
                    if 'total_return' in trade_data:
                        trade_data['hit_rate'] = 'N/A'  # Not available in HTML
                        trade_data['status'] = 'open'  # Assume open for now
                        trades.append(trade_data)
            
            return trades
            
        except Exception as e:
            logger.error(f"Error extracting trade details: {e}")
            return []
    
    def _analyze_trade_patterns(self, trades: List[Dict]) -> Dict:
        """Analyze patterns in recent trades."""
        try:
            if not trades:
                return {}
            
            patterns = {
                'best_performing_symbol': '',
                'worst_performing_symbol': '',
                'avg_sharpe': 0.0,
                'symbol_distribution': {},
                'risk_level': 'medium'
            }
            
            # Analyze symbol performance
            symbol_returns = {}
            sharpe_values = []
            
            for trade in trades:
                symbol = trade.get('symbol', 'Unknown')
                total_return = trade.get('total_return', '0%')
                sharpe = trade.get('sharpe_ratio', '0')
                
                # Track symbol returns
                if symbol not in symbol_returns:
                    symbol_returns[symbol] = []
                symbol_returns[symbol].append(total_return)
                
                # Track Sharpe ratios
                try:
                    sharpe_float = float(sharpe)
                    sharpe_values.append(sharpe_float)
                except:
                    pass
            
            # Find best/worst performing symbols
            if symbol_returns:
                best_symbol = max(symbol_returns.keys(), key=lambda x: len(symbol_returns[x]))
                worst_symbol = min(symbol_returns.keys(), key=lambda x: len(symbol_returns[x]))
                patterns['best_performing_symbol'] = best_symbol
                patterns['worst_performing_symbol'] = worst_symbol
            
            # Calculate average Sharpe ratio
            if sharpe_values:
                patterns['avg_sharpe'] = round(sum(sharpe_values) / len(sharpe_values), 2)
            
            # Determine risk level based on Sharpe ratio
            if patterns['avg_sharpe'] > 1.0:
                patterns['risk_level'] = 'low'
            elif patterns['avg_sharpe'] < 0.5:
                patterns['risk_level'] = 'high'
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing trade patterns: {e}")
            return {}
    
    def _extract_comprehensive_trade_data(self, all_backtests: List) -> List[Dict]:
        """Extract comprehensive trade data from actual backtest files."""
        try:
            trades = []
            
            for file_path in all_backtests[:5]:  # Process first 5 files
                try:
                    if not str(file_path).endswith('metrics.json'):
                        continue
                        
                    with open(file_path, 'r') as f:
                        metrics_data = json.load(f)
                    
                    # Extract real metrics data
                    run_id = metrics_data.get('run_id', 'Unknown')
                    total_return_pct = metrics_data.get('total_return_pct', 0)
                    sharpe_ratio = metrics_data.get('sharpe_ratio', 0)
                    total_trades = metrics_data.get('total_trades', 0)
                    winning_trades = metrics_data.get('winning_trades', 0)
                    losing_trades = metrics_data.get('losing_trades', 0)
                    hit_rate = metrics_data.get('hit_rate', 0)
                    max_drawdown = metrics_data.get('max_drawdown', 0)
                    initial_equity = metrics_data.get('initial_equity', 0)
                    final_equity = metrics_data.get('final_equity', 0)
                    
                    # Calculate real P&L
                    total_pnl = float(final_equity - initial_equity)
                    pnl_pct = float((total_pnl / initial_equity) * 100 if initial_equity > 0 else 0)
                    
                    # Get final positions (real open positions)
                    final_positions = metrics_data.get('final_positions', {})
                    open_positions_count = len(final_positions)
                    
                    # Extract symbol from run directory name or use first position
                    symbol = list(final_positions.keys())[0] if final_positions else 'Unknown'
                    
                    trades.append({
                        'run_id': str(run_id),
                        'symbol': str(symbol),
                        'total_return': f"{float(total_return_pct):.2f}%",
                        'sharpe_ratio': f"{float(sharpe_ratio):.3f}",
                        'hit_rate': f"{float(hit_rate):.1%}",
                        'total_trades': int(total_trades),
                        'winning_trades': int(winning_trades),
                        'losing_trades': int(losing_trades),
                        'max_drawdown': f"{float(max_drawdown):.2%}",
                        'total_pnl': f"${total_pnl:,.2f}",
                        'pnl_pct': f"{pnl_pct:.2f}%",
                        'open_positions': int(open_positions_count),
                        'initial_equity': f"${float(initial_equity):,.2f}",
                        'final_equity': f"${float(final_equity):,.2f}",
                        'status': 'open' if open_positions_count > 0 else 'closed'
                    })
                
                except Exception as e:
                    logger.warning(f"Error processing backtest file {file_path}: {e}")
                    continue
            
            return trades
            
        except Exception as e:
            logger.error(f"Error extracting comprehensive trade data: {e}")
            return []
    
    def _analyze_real_trade_patterns(self, trades: List[Dict]) -> Dict:
        """Analyze patterns in real trade data."""
        try:
            if not trades:
                return {}
            
            patterns = {
                'best_performing_run': '',
                'worst_performing_run': '',
                'avg_sharpe': 0.0,
                'avg_drawdown': 0.0,
                'total_pnl': 0.0,
                'risk_level': 'medium',
                'position_management': 'unknown'
            }
            
            # Analyze performance patterns
            returns = []
            sharpe_values = []
            drawdown_values = []
            total_pnl_sum = 0.0
            
            for trade in trades:
                # Extract numeric values
                try:
                    return_pct = float(trade.get('total_return', '0%').rstrip('%'))
                    returns.append(return_pct)
                except:
                    pass
                
                try:
                    sharpe = float(trade.get('sharpe_ratio', '0'))
                    sharpe_values.append(sharpe)
                except:
                    pass
                
                try:
                    drawdown = float(trade.get('max_drawdown', '0%').rstrip('%'))
                    drawdown_values.append(drawdown)
                except:
                    pass
                
                try:
                    pnl = float(trade.get('pnl_pct', '0%').rstrip('%'))
                    total_pnl_sum += pnl
                except:
                    pass
            
            # Find best/worst performing runs
            if returns:
                best_index = returns.index(max(returns))
                worst_index = returns.index(min(returns))
                patterns['best_performing_run'] = trades[best_index].get('run_id', 'Unknown')[:8]
                patterns['worst_performing_run'] = trades[worst_index].get('run_id', 'Unknown')[:8]
            
            # Calculate averages
            if sharpe_values:
                patterns['avg_sharpe'] = float(round(sum(sharpe_values) / len(sharpe_values), 3))
            
            if drawdown_values:
                patterns['avg_drawdown'] = float(round(sum(drawdown_values) / len(drawdown_values), 2))
            
            patterns['total_pnl'] = float(round(total_pnl_sum, 2))
            
            # Determine risk level based on real Sharpe ratios
            if patterns['avg_sharpe'] > 1.0:
                patterns['risk_level'] = 'low'
            elif patterns['avg_sharpe'] < 0.5:
                patterns['risk_level'] = 'high'
            
            # Analyze position management
            open_positions = [t for t in trades if t.get('status') == 'open']
            if open_positions:
                patterns['position_management'] = f"{len(open_positions)} open positions"
            else:
                patterns['position_management'] = 'all positions closed'
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing real trade patterns: {e}")
            return {}
    
    def _extract_portfolio_analysis(self, all_backtests: List) -> Dict:
        """Extract portfolio analysis from real equity data."""
        try:
            portfolio_data = {
                'total_runs': len(all_backtests),
                'equity_evolution': [],
                'risk_metrics': {},
                'position_concentration': {}
            }
            
            total_initial_equity = 0
            total_final_equity = 0
            all_positions = {}
            
            for file_path in all_backtests[:3]:  # Analyze first 3 runs
                try:
                    if not str(file_path).endswith('metrics.json'):
                        continue
                        
                    with open(file_path, 'r') as f:
                        metrics = json.load(f)
                    
                    initial_equity = metrics.get('initial_equity', 0)
                    final_equity = metrics.get('final_equity', 0)
                    final_positions = metrics.get('final_positions', {})
                    
                    total_initial_equity += initial_equity
                    total_final_equity += final_equity
                    
                    # Track position concentration
                    for symbol, quantity in final_positions.items():
                        if symbol not in all_positions:
                            all_positions[symbol] = 0
                        all_positions[symbol] += quantity
                    
                    # Extract equity evolution if available
                    equity_file = file_path.parent / 'equity.csv'
                    if equity_file.exists():
                        try:
                            import pandas as pd
                            df = pd.read_csv(equity_file)
                            if not df.empty:
                                portfolio_data['equity_evolution'].append({
                                    'run_id': metrics.get('run_id', 'Unknown')[:8],
                                    'max_value': df['portfolio_value'].max(),
                                    'min_value': df['portfolio_value'].min(),
                                    'volatility': df['portfolio_value'].std()
                                })
                        except Exception as e:
                            logger.warning(f"Could not read equity data: {e}")
                
                except Exception as e:
                    logger.warning(f"Error processing portfolio data: {e}")
                    continue
            
            # Calculate portfolio metrics
            if total_initial_equity > 0:
                portfolio_data['risk_metrics'] = {
                    'total_return': f"{((total_final_equity - total_initial_equity) / total_initial_equity) * 100:.2f}%",
                    'total_equity_change': f"${total_final_equity - total_initial_equity:,.2f}",
                    'equity_efficiency': f"{total_final_equity / total_initial_equity:.2f}"
                }
            
            # Ensure all numeric values are JSON serializable
            for key, value in portfolio_data.items():
                if isinstance(value, (int, float)):
                    portfolio_data[key] = float(value)
            
            # Position concentration analysis
            if all_positions:
                portfolio_data['position_concentration'] = {
                    'total_symbols': len(all_positions),
                    'most_held': max(all_positions.items(), key=lambda x: x[1])[0] if all_positions else 'None',
                    'symbols': list(all_positions.keys())
                }
            
            return portfolio_data
            
        except Exception as e:
            logger.error(f"Error extracting portfolio analysis: {e}")
            return {}
    
    def _extract_execution_quality(self, all_backtests: List) -> Dict:
        """Extract execution quality metrics from real trade data."""
        try:
            execution_data = {
                'trade_execution': {},
                'timing_analysis': {},
                'cost_analysis': {}
            }
            
            total_execution_slippage = 0
            total_fees = 0
            trade_timing = []
            
            for file_path in all_backtests[:3]:  # Analyze first 3 runs
                try:
                    if not str(file_path).endswith('metrics.json'):
                        continue
                    
                    # Check for trades.csv file
                    trades_file = file_path.parent / 'trades.csv'
                    if trades_file.exists():
                        try:
                            import pandas as pd
                            df = pd.read_csv(trades_file)
                            
                            if not df.empty:
                                # Analyze execution quality
                                for _, trade in df.iterrows():
                                    # Calculate execution slippage
                                    if 'price' in trade and 'execution_price' in trade:
                                        slippage = abs(trade['execution_price'] - trade['price']) / trade['price']
                                        total_execution_slippage += slippage
                                    
                                    # Track fees
                                    if 'fees' in trade:
                                        total_fees += trade['fees']
                                    
                                    # Track timing
                                    if 'timestamp' in trade:
                                        trade_timing.append(pd.to_datetime(trade['timestamp']))
                        
                        except Exception as e:
                            logger.warning(f"Could not read trades data: {e}")
                
                except Exception as e:
                    logger.warning(f"Error processing execution data: {e}")
                    continue
            
            # Calculate execution metrics
            if total_execution_slippage > 0:
                execution_data['trade_execution'] = {
                    'avg_slippage': f"{total_execution_slippage:.4f}",
                    'total_fees': f"${total_fees:.2f}",
                    'execution_quality': 'high' if total_execution_slippage < 0.001 else 'medium' if total_execution_slippage < 0.005 else 'low'
                }
            
            # Timing analysis
            if trade_timing:
                try:
                    import pandas as pd
                    timing_df = pd.DataFrame({'timestamp': trade_timing})
                    timing_df['hour'] = timing_df['timestamp'].dt.hour
                    
                    execution_data['timing_analysis'] = {
                        'total_trades': len(trade_timing),
                        'peak_hour': timing_df['hour'].mode().iloc[0] if not timing_df.empty else 'Unknown',
                        'trading_hours': f"{timing_df['hour'].min()}-{timing_df['hour'].max()}"
                    }
                except Exception as e:
                    logger.warning(f"Could not analyze timing: {e}")
            
            return execution_data
            
        except Exception as e:
            logger.error(f"Error extracting execution quality: {e}")
            return {}
    
    def _generate_real_patterns(self, hit_rate: float, open_trades: int, total_trades: int, insights: Dict) -> List[str]:
        """Generate patterns based on real data analysis."""
        patterns = []
        
        # Performance patterns
        if hit_rate < 0.2:
            patterns.append("Critical: Extremely low win rate indicates fundamental strategy failure")
        elif hit_rate < 0.4:
            patterns.append("Low win rate suggests strategy needs immediate refinement")
        elif hit_rate > 0.6:
            patterns.append("Strong win rate indicates effective strategy execution")
        
        # Position management patterns
        if open_trades > total_trades * 0.7:
            patterns.append("High open position ratio suggests poor exit strategy or over-trading")
        elif open_trades > total_trades * 0.5:
            patterns.append("Moderate open position ratio indicates room for improvement in position management")
        
        # Portfolio patterns
        if insights.get('portfolio_analysis', {}).get('position_concentration', {}).get('total_symbols', 0) > 10:
            patterns.append("High symbol diversification may indicate lack of focus")
        elif insights.get('portfolio_analysis', {}).get('position_concentration', {}).get('total_symbols', 0) < 3:
            patterns.append("Low symbol diversification increases concentration risk")
        
        return patterns
    
    def _generate_real_risk_insights(self, hit_rate: float, open_trades: int, total_trades: int, insights: Dict) -> Dict:
        """Generate risk insights based on real performance data."""
        risk_insights = {}
        
        # Position management risk
        if open_trades > 0:
            open_ratio = open_trades / total_trades
            risk_insights['position_management'] = f"Open positions represent {open_ratio:.1%} of portfolio risk"
            risk_insights['unrealized_risk'] = f"Active exposure: {open_trades} positions with unrealized P&L"
        else:
            risk_insights['position_management'] = "All positions closed - no unrealized risk"
        
        # Performance risk
        risk_insights['win_rate_quality'] = f"Current hit rate: {hit_rate:.1%} - {'Critical' if hit_rate < 0.2 else 'High' if hit_rate < 0.4 else 'Moderate' if hit_rate < 0.6 else 'Low'} risk"
        
        # Portfolio risk
        portfolio_data = insights.get('portfolio_analysis', {})
        if portfolio_data.get('risk_metrics'):
            risk_insights['portfolio_risk'] = f"Portfolio return: {portfolio_data['risk_metrics'].get('total_return', 'Unknown')}"
        
        return risk_insights
    
    def _generate_real_lessons(self, hit_rate: float, open_trades: int, total_trades: int, insights: Dict) -> List[str]:
        """Generate actionable lessons based on real data analysis."""
        lessons = []
        
        # Performance lessons
        if hit_rate < 0.3:
            lessons.append("Immediate action required: Review and revise entry/exit criteria")
        elif hit_rate < 0.5:
            lessons.append("Strategy refinement needed: Analyze losing trades for common patterns")
        
        # Position management lessons
        if open_trades > total_trades * 0.5:
            lessons.append("Implement strict position management: Set stop-loss and take-profit levels")
        
        # Portfolio lessons
        portfolio_data = insights.get('portfolio_analysis', {})
        if portfolio_data.get('position_concentration', {}).get('total_symbols', 0) > 8:
            lessons.append("Consider reducing symbol count for better focus and management")
        
        # Execution lessons
        execution_data = insights.get('execution_quality', {})
        if execution_data.get('trade_execution', {}).get('execution_quality') == 'low':
            lessons.append("Improve execution quality: Review slippage and timing strategies")
        
        # Data quality lessons
        if total_trades > 50:
            lessons.append("Sufficient data for statistical analysis - patterns are reliable")
        elif total_trades > 20:
            lessons.append("Moderate data for analysis - continue collecting for better insights")
        else:
            lessons.append("Limited data available - continue trading for better pattern recognition")
        
        return lessons
    
    def _extract_from_backtest_files(self, all_backtests: List) -> List[Dict]:
        """Extract trade details from individual backtest files."""
        try:
            trades = []
            
            for file_path in all_backtests[:5]:  # Process first 5 files
                try:
                    with open(file_path, 'r') as f:
                        backtest_data = json.load(f)
                    
                    # Extract trade information
                    symbol = backtest_data.get('symbol', 'Unknown')
                    total_return = backtest_data.get('total_return_pct', 0)
                    sharpe_ratio = backtest_data.get('sharpe_ratio', 0)
                    winning_trades = backtest_data.get('winning_trades', 0)
                    losing_trades = backtest_data.get('losing_trades', 0)
                    total_trades = backtest_data.get('total_trades', 0)
                    
                    if total_trades > 0:
                        hit_rate = winning_trades / total_trades if winning_trades > 0 else 0
                        
                        trades.append({
                            'symbol': symbol,
                            'total_return': f"{total_return:.1%}" if isinstance(total_return, (int, float)) else str(total_return),
                            'sharpe_ratio': f"{sharpe_ratio:.2f}" if isinstance(sharpe_ratio, (int, float)) else str(sharpe_ratio),
                            'hit_rate': f"{hit_rate:.1%}",
                            'status': 'closed' if (winning_trades + losing_trades) == total_trades else 'open'
                        })
                
                except Exception as e:
                    logger.warning(f"Error processing backtest file {file_path}: {e}")
                    continue
            
            return trades
            
        except Exception as e:
            logger.error(f"Error extracting from backtest files: {e}")
            return []
    
    def save_knowledge(self, insights: Dict) -> bool:
        """Save trading insights to knowledge base."""
        try:
            # Load existing knowledge
            existing_knowledge = self.load_knowledge()
            
            # Add new insights
            existing_knowledge['insights'].append(insights)
            
            # Keep only last 50 insights to prevent bloat
            if len(existing_knowledge['insights']) > 50:
                existing_knowledge['insights'] = existing_knowledge['insights'][-50:]
            
            # Update summary
            existing_knowledge['last_updated'] = datetime.now().isoformat()
            existing_knowledge['total_insights'] = len(existing_knowledge['insights'])
            
            # Ensure all values are JSON serializable
            def make_json_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, (int, float)):
                    return float(obj)
                elif isinstance(obj, (str, bool, type(None))):
                    return obj
                else:
                    return str(obj)
            
            serializable_knowledge = make_json_serializable(existing_knowledge)
            
            # Save to file
            with open(self.knowledge_file, 'w') as f:
                json.dump(serializable_knowledge, f, indent=2)
            
            logger.info(f"Saved trading knowledge: {len(insights)} insights")
            return True
            
        except Exception as e:
            logger.error(f"Error saving trading knowledge: {e}")
            return False
    
    def load_knowledge(self) -> Dict:
        """Load existing trading knowledge."""
        try:
            if self.knowledge_file.exists():
                with open(self.knowledge_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading trading knowledge: {e}")
        
        # Return default structure
        return {
            'insights': [],
            'last_updated': datetime.now().isoformat(),
            'total_insights': 0
        }
    
    def get_relevant_context(self, query: str, max_insights: int = 5) -> str:
        """Get relevant trading context for GPT prompts with comprehensive real data."""
        try:
            knowledge = self.load_knowledge()
            if not knowledge['insights']:
                return "No trading history available for learning."
            
            # Get most recent insights
            recent_insights = knowledge['insights'][-max_insights:]
            
            context_parts = []
            
            # Add performance summary
            if recent_insights:
                latest = recent_insights[-1]
                perf = latest.get('performance_summary', {})
                if perf:
                    context_parts.append(
                        f"Recent Performance: {perf.get('total_trades', 0)} trades, "
                        f"{perf.get('hit_rate', 0):.1%} win rate, "
                        f"{perf.get('open_positions', 0)} open positions"
                    )
            
            # Add comprehensive trade details
            if latest.get('recent_trades'):
                trades_summary = []
                for trade in latest['recent_trades'][:3]:  # Show last 3 trades
                    symbol = trade.get('symbol', 'Unknown')
                    total_return = trade.get('total_return', '0%')
                    sharpe = trade.get('sharpe_ratio', '0')
                    pnl = trade.get('total_pnl', 'N/A')
                    trades_summary.append(f"{symbol}: {total_return} return, {sharpe} Sharpe, {pnl} P&L")
                
                if trades_summary:
                    context_parts.append(f"Recent Trades: {'; '.join(trades_summary)}")
            
            # Add enhanced trade patterns
            if latest.get('trade_patterns'):
                patterns = latest['trade_patterns']
                pattern_summary = []
                
                if patterns.get('best_performing_run') and patterns.get('worst_performing_run'):
                    pattern_summary.append(f"Best: {patterns['best_performing_run']}, Worst: {patterns['worst_performing_run']}")
                
                if patterns.get('avg_sharpe'):
                    pattern_summary.append(f"Avg Sharpe: {patterns['avg_sharpe']}")
                
                if patterns.get('risk_level'):
                    pattern_summary.append(f"Risk: {patterns['risk_level']}")
                
                if pattern_summary:
                    context_parts.append(f"Trade Patterns: {'; '.join(pattern_summary)}")
            
            # Add portfolio analysis
            if latest.get('portfolio_analysis'):
                portfolio = latest['portfolio_analysis']
                portfolio_summary = []
                
                if portfolio.get('risk_metrics', {}).get('total_return'):
                    portfolio_summary.append(f"Portfolio: {portfolio['risk_metrics']['total_return']}")
                
                if portfolio.get('position_concentration', {}).get('total_symbols'):
                    portfolio_summary.append(f"Symbols: {portfolio['position_concentration']['total_symbols']}")
                
                if portfolio_summary:
                    context_parts.append(f"Portfolio: {'; '.join(portfolio_summary)}")
            
            # Add execution quality
            if latest.get('execution_quality'):
                execution = latest['execution_quality']
                if execution.get('trade_execution', {}).get('execution_quality'):
                    context_parts.append(f"Execution: {execution['trade_execution']['execution_quality']} quality")
            
            # Add key patterns (most important)
            patterns = []
            for insight in recent_insights:
                patterns.extend(insight.get('key_patterns', [])[:2])  # Limit to 2 per insight
            
            if patterns:
                context_parts.append(f"Key Patterns: {'; '.join(patterns[:3])}")
            
            # Add actionable lessons
            lessons = []
            for insight in recent_insights:
                lessons.extend(insight.get('actionable_lessons', [])[:1])  # Limit to 1 per insight
            
            if lessons:
                context_parts.append(f"Lessons: {'; '.join(lessons[:2])}")
            
            return " | ".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return "Error retrieving trading context."
    
    def get_learning_summary(self) -> str:
        """Get a concise learning summary for GPT context."""
        try:
            knowledge = self.load_knowledge()
            if not knowledge['insights']:
                return "No trading history available."
            
            # Aggregate key metrics
            total_insights = len(knowledge['insights'])
            recent_insights = knowledge['insights'][-10:]  # Last 10
            
            # Calculate trends
            hit_rates = []
            for insight in recent_insights:
                perf = insight.get('performance_summary', {})
                if 'hit_rate' in perf:
                    hit_rates.append(perf['hit_rate'])
            
            if hit_rates:
                avg_hit_rate = sum(hit_rates) / len(hit_rates)
                trend = "improving" if len(hit_rates) > 1 and hit_rates[-1] > hit_rates[0] else "stable"
                
                return (
                    f"Trading History: {total_insights} insights | "
                    f"Recent Win Rate: {avg_hit_rate:.1%} | "
                    f"Trend: {trend}"
                )
            
            return f"Trading History: {total_insights} insights available"
            
        except Exception as e:
            logger.error(f"Error getting learning summary: {e}")
            return "Error retrieving learning summary."


def create_trading_prompt_with_context(
    base_prompt: str, 
    rag_system: TradingKnowledgeRAG,
    include_performance: bool = True
) -> str:
    """Create a GPT prompt with relevant trading context."""
    
    if include_performance:
        context = rag_system.get_relevant_context("trading performance", max_insights=3)
        learning_summary = rag_system.get_learning_summary()
        
        enhanced_prompt = f"""Based on this trading context:

{context}

{learning_summary}

{base_prompt}

Use the above context to provide more informed and relevant trading advice."""
    else:
        enhanced_prompt = base_prompt
    
    return enhanced_prompt


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = TradingKnowledgeRAG()
    
    # Example insights
    sample_results = {
        'win_loss_summary': {
            'total_trades': 270,
            'winning_trades': 0,
            'losing_trades': 13,
            'open_trades': 257
        }
    }
    
    # Extract and save insights
    insights = rag.extract_trading_insights(sample_results)
    rag.save_knowledge(insights)
    
    # Get context for GPT
    context = rag.get_relevant_context("trading performance")
    print(f"Trading Context: {context}")
    
    # Create enhanced prompt
    base_prompt = "What trading strategy adjustments would you recommend?"
    enhanced_prompt = create_trading_prompt_with_context(base_prompt, rag)
    print(f"\nEnhanced Prompt:\n{enhanced_prompt}")
