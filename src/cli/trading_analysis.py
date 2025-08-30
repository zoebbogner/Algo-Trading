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
        client = _initialize_llm_client(symbol, temperature)
        market_data = _get_market_data(symbol, timeframe)
        analysis_prompt = _create_analysis_prompt(symbol, timeframe, market_data)
        
        logger.info(f"Generating LLM analysis for {symbol} on {timeframe} timeframe")
        result = _generate_analysis(client, analysis_prompt)
        
        _process_analysis_result(result, symbol, timeframe)
        
    except Exception as e:
        logger.error(f"Trading analysis failed: {e}")
        click.echo(f"❌ Analysis failed: {e}")
        sys.exit(1)
    
    finally:
        if 'client' in locals():
            client.close()


def _initialize_llm_client(symbol: str, temperature: float) -> Any:
    """Initialize the LLM client for analysis."""
    logger.info(f"Initializing LLM client for {symbol} analysis")
    return get_llm_client(
        run_id=f"trading_analysis_{symbol}_{datetime.now().isoformat()}",
        config={'temperature': temperature}
    )


def _get_market_data(symbol: str, timeframe: str) -> Dict[str, Any]:
    """Get market data for analysis (placeholder implementation)."""
    return {
        'current_price': 50000.0,
        'current_volume': 1000000,
        'price_change_24h': 0.025,
        'rsi_value': 65.5,
        'macd_value': 0.002,
        'ma_analysis': "Price above 20MA, below 50MA",
        's_r_levels': "Support: 48000, Resistance: 52000",
        'market_trend': "Bullish",
        'sector_performance': "Technology sector up 2%",
        'market_events': "Fed meeting this week, earnings season"
    }


def _create_analysis_prompt(symbol: str, timeframe: str, market_data: Dict[str, Any]) -> str:
    """Create the analysis prompt with market data."""
    return f"""You are an expert quantitative trading analyst. Analyze the market data for {symbol} on {timeframe} timeframe and provide a trading recommendation.

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


def _generate_analysis(client: Any, prompt: str) -> Any:
    """Generate analysis using the LLM client."""
    return client.generate(
        prompt,
        json_mode=True,
        max_tokens=512
    )


def _process_analysis_result(result: Any, symbol: str, timeframe: str) -> None:
    """Process and display the analysis result."""
    try:
        analysis = json.loads(result.text)
        logger.info("✅ LLM analysis generated successfully")
        
        _display_analysis_results(analysis, result.latency_ms)
        _save_analysis(analysis, symbol, timeframe)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.info(f"Raw response: {result.text}")
        click.echo("❌ LLM response is not valid JSON")


def _display_analysis_results(analysis: Dict[str, Any], latency_ms: int) -> None:
    """Display the analysis results in a formatted way."""
    click.echo(f"\n📊 Trading Analysis Results")
    click.echo(f"=" * 50)
    
    _display_basic_info(analysis)
    _display_action_details(analysis)
    _display_rationale(analysis)
    _display_performance(latency_ms, analysis)


def _display_basic_info(analysis: Dict[str, Any]) -> None:
    """Display basic analysis information."""
    click.echo(f"Symbol: {analysis.get('symbol', 'N/A')}")
    click.echo(f"Timeframe: {analysis.get('timeframe', 'N/A')}")
    click.echo(f"Intent: {analysis.get('intent', 'N/A')}")


def _display_action_details(analysis: Dict[str, Any]) -> None:
    """Display action details and constraints."""
    action = analysis.get('action', {})
    click.echo(f"Action: {action.get('type', 'N/A')}")
    
    confidence = action.get('confidence', 'N/A')
    if isinstance(confidence, (int, float)):
        click.echo(f"Confidence: {confidence:.2f}")
    else:
        click.echo(f"Confidence: {confidence}")
    
    _display_constraints(action.get('constraints', {}))
    _display_regime_context(action.get('regime_context', {}))


def _display_constraints(constraints: Dict[str, Any]) -> None:
    """Display trading constraints."""
    if not constraints:
        return
    
    _display_percentage_constraint(constraints, 'max_position_size', 'Position Size')
    _display_percentage_constraint(constraints, 'stop_loss_pct', 'Stop Loss')
    _display_percentage_constraint(constraints, 'take_profit_pct', 'Take Profit')


def _display_percentage_constraint(constraints: Dict[str, Any], key: str, label: str) -> None:
    """Display a percentage constraint."""
    value = constraints.get(key, 'N/A')
    if isinstance(value, (int, float)):
        click.echo(f"{label}: {value:.2%}")
    else:
        click.echo(f"{label}: {value}")


def _display_regime_context(regime: Dict[str, Any]) -> None:
    """Display market regime context."""
    if not regime:
        return
    
    click.echo(f"Market Regime: {regime.get('regime_type', 'N/A')}")
    
    regime_conf = regime.get('regime_confidence', 'N/A')
    if isinstance(regime_conf, (int, float)):
        click.echo(f"Regime Confidence: {regime_conf:.2f}")
    else:
        click.echo(f"Regime Confidence: {regime_conf}")


def _display_rationale(analysis: Dict[str, Any]) -> None:
    """Display the analysis rationale."""
    click.echo(f"\n💭 Rationale:")
    click.echo(f"   {analysis.get('rationale', 'No rationale provided')}")


def _display_performance(latency_ms: int, analysis: Dict[str, Any]) -> None:
    """Display performance metrics."""
    click.echo(f"\n⚡ Performance:")
    click.echo(f"   Generation Time: {latency_ms}ms")
    click.echo(f"   Model: {analysis.get('metadata', {}).get('model', 'N/A')}")


def _save_analysis(analysis: Dict[str, Any], symbol: str, timeframe: str) -> None:
    """Save the analysis results to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_{symbol}_{timeframe}_{timestamp}.json"
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    filepath = logs_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    click.echo(f"💾 Analysis saved to: {filepath}")


if __name__ == '__main__':
    trading_analysis()
