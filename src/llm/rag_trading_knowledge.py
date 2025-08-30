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
                'actionable_lessons': []
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
