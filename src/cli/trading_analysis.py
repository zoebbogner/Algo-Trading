#!/usr/bin/env python3
"""Trading Analysis CLI using LLM for market analysis and recommendations."""

import sys
import json
import click
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm import get_llm_client
from src.utils.logging import setup_logging, get_logger


@click.group()
@click.option('--log-level', default='INFO', help='Logging level')
def trading_analysis(log_level):
    """Trading Analysis CLI using LLM for intelligent market analysis."""
    setup_logging(log_level=log_level)


@trading_analysis.command()
@click.option('--symbol', required=True, help='Trading symbol (e.g., BTCUSDT)')
@click.option('--timeframe', default='1m', help='Analysis timeframe')
@click.option('--temperature', default=0.1, help='LLM temperature for consistency')
def analyze_symbol(symbol: str, timeframe: str, temperature: float):
    """Analyze a specific symbol using LLM-powered market analysis."""
    logger = get_logger(__name__)
    
    try:
        # Initialize LLM client
        logger.info(f"Initializing LLM client for {symbol} analysis")
        client = get_llm_client(
            run_id=f"trading_analysis_{symbol}_{datetime.now().isoformat()}",
            config={'temperature': temperature}
        )
        
        # Create analysis prompt
        analysis_prompt = f"""You are an expert quantitative trading analyst. Analyze the market data for {symbol} on {timeframe} timeframe and provide a trading recommendation.

Current Market Data:
- Symbol: {symbol}
- Timeframe: {timeframe}
- Price: $50,000 (placeholder)
- Volume: High
- Trend: Bullish
- RSI: 65 (neutral)
- MACD: Positive momentum

Provide a trading recommendation in valid JSON format only:

{{
  "version": "1.0.0",
  "intent": "propose_trade",
  "symbol": "{symbol}",
  "timeframe": "{timeframe}",
  "rationale": "Clear explanation of your analysis and reasoning",
  "action": {{
    "type": "hold|enter_long|enter_short|exit",
    "confidence": 0.0-1.0,
    "constraints": {{
      "max_position_size": 0.0-1.0,
      "stop_loss_pct": 0.0-0.5,
      "take_profit_pct": 0.0-2.0,
      "time_horizon": "intraday|swing|position"
    }},
    "regime_context": {{
      "regime_type": "trending|ranging|volatile|breakout|consolidation",
      "regime_confidence": 0.0-1.0,
      "regime_duration": "Estimated duration"
    }}
  }},
  "metadata": {{
    "analysis_timestamp": "{datetime.now().isoformat()}",
    "data_sources": ["Technical indicators", "Price action", "Volume analysis"],
    "indicators_used": ["RSI", "MACD", "Moving Averages"],
    "risk_factors": ["Market volatility", "Economic events", "Technical levels"]
  }}
}}"""
        
        logger.info(f"Generating LLM analysis for {symbol} on {timeframe} timeframe")
        
        # Generate analysis
        result = client.generate(
            analysis_prompt,
            json_mode=True,
            max_tokens=512
        )
        
        # Parse and validate response
        try:
            analysis = json.loads(result.text)
            logger.info("✅ LLM analysis generated successfully")
            
            # Display results
            _display_analysis_results(analysis, result.latency_ms)
            
            # Save analysis to file
            _save_analysis(analysis, symbol, timeframe)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.info(f"Raw response: {result.text}")
            click.echo("❌ LLM response is not valid JSON")
            return
        
        finally:
            client.close()
            
    except Exception as e:
        logger.error(f"Trading analysis failed: {e}")
        click.echo(f"❌ Analysis failed: {e}")
        sys.exit(1)


def _display_analysis_results(analysis: Dict[str, Any], latency_ms: int):
    """Display the analysis results in a formatted way."""
    click.echo(f"\n📊 Trading Analysis Results")
    click.echo(f"=" * 50)
    
    # Basic info
    click.echo(f"Symbol: {analysis.get('symbol', 'N/A')}")
    click.echo(f"Timeframe: {analysis.get('timeframe', 'N/A')}")
    click.echo(f"Intent: {analysis.get('intent', 'N/A')}")
    
    # Action details
    action = analysis.get('action', {})
    click.echo(f"Action: {action.get('type', 'N/A')}")
    
    # Handle confidence safely
    confidence = action.get('confidence', 'N/A')
    if isinstance(confidence, (int, float)):
        click.echo(f"Confidence: {confidence:.2f}")
    else:
        click.echo(f"Confidence: {confidence}")
    
    # Constraints
    constraints = action.get('constraints', {})
    if constraints:
        # Handle position size safely
        pos_size = constraints.get('max_position_size', 'N/A')
        if isinstance(pos_size, (int, float)):
            click.echo(f"Position Size: {pos_size:.2%}")
        else:
            click.echo(f"Position Size: {pos_size}")
        
        # Handle stop loss safely
        stop_loss = constraints.get('stop_loss_pct', 'N/A')
        if isinstance(stop_loss, (int, float)):
            click.echo(f"Stop Loss: {stop_loss:.2%}")
        else:
            click.echo(f"Stop Loss: {stop_loss}")
        
        # Handle take profit safely
        take_profit = constraints.get('take_profit_pct', 'N/A')
        if isinstance(take_profit, (int, float)):
            click.echo(f"Take Profit: {take_profit:.2%}")
        else:
            click.echo(f"Take Profit: {take_profit}")
    
    # Regime context
    regime = action.get('regime_context', {})
    if regime:
        click.echo(f"Market Regime: {regime.get('regime_type', 'N/A')}")
        
        # Handle regime confidence safely
        regime_conf = regime.get('regime_confidence', 'N/A')
        if isinstance(regime_conf, (int, float)):
            click.echo(f"Regime Confidence: {regime_conf:.2f}")
        else:
            click.echo(f"Regime Confidence: {regime_conf}")
    
    # Rationale
    click.echo(f"\n💭 Rationale:")
    click.echo(f"   {analysis.get('rationale', 'No rationale provided')}")
    
    # Performance
    click.echo(f"\n⚡ Performance:")
    click.echo(f"   Generation Time: {latency_ms}ms")
    click.echo(f"   Model: {analysis.get('metadata', {}).get('model', 'N/A')}")


def _save_analysis(analysis: Dict[str, Any], symbol: str, timeframe: str):
    """Save the analysis results to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_{symbol}_{timeframe}_{timestamp}.json"
    
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    filepath = logs_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    click.echo(f"💾 Analysis saved to: {filepath}")


if __name__ == '__main__':
    trading_analysis()
