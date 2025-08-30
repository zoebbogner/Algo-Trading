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
        """Extract concise trading insights from backtest results."""
        try:
            insights = {
                'timestamp': datetime.now().isoformat(),
                'performance_summary': {},
                'key_patterns': [],
                'risk_insights': {},
                'actionable_lessons': [],
                'recent_trades': [],
                'trade_patterns': {}
            }
            
            # Extract performance metrics
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
                    
                    # Extract recent trade details from backtest results
                    if 'recent_performance_html' in backtest_results:
                        recent_performance = backtest_results.get('recent_performance_html', '')
                        if recent_performance:
                            # Parse recent trades from the HTML content
                            trade_details = self._extract_trade_details(recent_performance)
                            insights['recent_trades'] = trade_details[:5]  # Keep last 5 trades
                            
                            # Analyze trade patterns
                            insights['trade_patterns'] = self._analyze_trade_patterns(trade_details)
                    
                    # Also extract from individual backtest files if available
                    if 'all_backtests' in backtest_results:
                        all_backtests = backtest_results.get('all_backtests', [])
                        if all_backtests and not insights['recent_trades']:
                            # Extract from individual backtest files
                            trade_details = self._extract_from_backtest_files(all_backtests)
                            insights['recent_trades'] = trade_details[:5]
                            insights['trade_patterns'] = self._analyze_trade_patterns(trade_details)
                    
                    # Key patterns
                    if hit_rate < 0.3:
                        insights['key_patterns'].append("Low win rate suggests strategy needs refinement")
                    elif hit_rate > 0.6:
                        insights['key_patterns'].append("High win rate indicates strong strategy execution")
                    
                    if open_trades > total_trades * 0.5:
                        insights['key_patterns'].append("High open position ratio - consider position management")
                    
                    # Risk insights
                    insights['risk_insights'] = {
                        'position_management': "Open positions represent unrealized risk",
                        'win_rate_quality': f"Current hit rate: {hit_rate:.1%}",
                        'portfolio_exposure': f"Active exposure: {open_trades} positions"
                    }
                    
                    # Actionable lessons
                    if hit_rate < 0.4:
                        insights['actionable_lessons'].append("Review entry criteria and risk management")
                    if open_trades > 0:
                        insights['actionable_lessons'].append("Monitor open positions for exit opportunities")
                    if total_trades > 100:
                        insights['actionable_lessons'].append("Sufficient data for strategy validation")
            
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
            
            # Save to file
            with open(self.knowledge_file, 'w') as f:
                json.dump(existing_knowledge, f, indent=2)
            
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
        """Get relevant trading context for GPT prompts (concise format)."""
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
            
            # Add recent trade details
            if latest.get('recent_trades'):
                trades_summary = []
                for trade in latest['recent_trades'][:3]:  # Show last 3 trades
                    symbol = trade.get('symbol', 'Unknown')
                    total_return = trade.get('total_return', '0%')
                    sharpe = trade.get('sharpe_ratio', '0')
                    trades_summary.append(f"{symbol}: {total_return} return, {sharpe} Sharpe")
                
                if trades_summary:
                    context_parts.append(f"Recent Trades: {'; '.join(trades_summary)}")
            
            # Add trade patterns
            if latest.get('trade_patterns'):
                patterns = latest['trade_patterns']
                if patterns.get('best_performing_symbol') and patterns.get('worst_performing_symbol'):
                    context_parts.append(
                        f"Trade Patterns: Best {patterns['best_performing_symbol']}, "
                        f"Worst {patterns['worst_performing_symbol']}, "
                        f"Risk: {patterns.get('risk_level', 'medium')}"
                    )
            
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
