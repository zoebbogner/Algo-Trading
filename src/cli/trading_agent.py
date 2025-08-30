#!/usr/bin/env python3
"""Trading Agent CLI using Local LLM for intelligent market analysis and trading decisions."""

import sys
import json
import click
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables")
except Exception as e:
    print(f"⚠️ Failed to load .env file: {e}")

from src.utils.logging import setup_logging, get_logger
from src.llm import get_llm_client

logger = get_logger(__name__)


@click.group()
@click.option('--log-level', default='INFO', help='Logging level')
def trading_agent(log_level):
    """Trading Agent CLI using Local LLM for intelligent market analysis."""
    setup_logging(log_level=log_level)
    logger.info(f"🚀 Trading Agent CLI started with log level: {log_level}")


@trading_agent.command()
@click.option('--symbol', required=True, help='Trading symbol (e.g., BTCUSDT)')
@click.option('--timeframe', default='1h', help='Analysis timeframe')
@click.option('--temperature', default=0.1, help='LLM temperature for consistency')
@click.option('--max-tokens', default=150, help='Maximum tokens for response')
@click.option('--timeout', default=60, help='Timeout in seconds for LLM generation')
def analyze(symbol: str, timeframe: str, temperature: float, max_tokens: int, timeout: int):
    """Analyze a trading symbol and provide recommendations."""
    logger.info(f"🔍 Starting trading analysis: {symbol} on {timeframe}")
    
    try:
        # Initialize LLM client
        logger.info("🔧 Initializing LLM client...")
        client = _initialize_llm_client(symbol, temperature, max_tokens)
        logger.info("✅ LLM client initialized")
        
        # Get market data
        logger.info("📊 Collecting market data...")
        market_data = _get_market_data(symbol, timeframe)
        logger.info("✅ Market data collected")
        
        # Generate analysis prompt
        logger.info("📝 Creating analysis prompt...")
        prompt = _create_optimized_prompt(symbol, timeframe, market_data)
        logger.info(f"✅ Analysis prompt created ({len(prompt)} characters)")
        
        # Generate LLM analysis with timeout
        logger.info("🤖 Generating LLM analysis...")
        result = _generate_analysis_with_timeout(client, prompt, timeout, max_tokens)
        logger.info("✅ LLM analysis generated")
        
        # Process and display results
        logger.info("📋 Processing analysis results...")
        analysis = _process_analysis_result(result, symbol, timeframe)
        
        # Display results
        _display_analysis_results(analysis)
        
        # Save results
        logger.info("💾 Saving analysis results...")
        _save_analysis_results(analysis, symbol, timeframe)
        
        logger.info("🎉 Trading analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Trading analysis failed: {e}", exc_info=True)
        click.echo(f"❌ Analysis failed: {e}")
        sys.exit(1)
    
    finally:
        if 'client' in locals():
            logger.info("🧹 Cleaning up LLM client...")
            client.close()
            logger.info("✅ LLM client closed")


@trading_agent.command()
@click.option('--symbol', required=True, help='Trading symbol to backtest')
@click.option('--days', default=30, help='Number of days to backtest')
@click.option('--initial-capital', default=10000, help='Initial capital for backtest')
def backtest(symbol: str, days: int, initial_capital: float):
    """Run a backtest using LLM-generated trading signals."""
    logger.info(f"📊 Starting backtest: {symbol} for {days} days")
    
    try:
        # Initialize LLM client
        logger.info("🔧 Initializing LLM client...")
        client = _initialize_llm_client(symbol, temperature=0.1, max_tokens=100)
        logger.info("✅ LLM client initialized")
        
        # Generate trading strategy
        logger.info("🤖 Generating trading strategy...")
        strategy = _generate_trading_strategy(client, symbol, days, max_tokens=300)
        logger.info("✅ Trading strategy generated")
        
        # Run backtest simulation
        logger.info("📈 Running backtest simulation...")
        results = _run_backtest_simulation(strategy, symbol, days, initial_capital)
        logger.info("✅ Backtest completed")
        
        # Display results
        _display_backtest_results(results)
        
        # Save results
        _save_backtest_results(results, symbol, days)
        
        logger.info("🎉 Backtest completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}", exc_info=True)
        click.echo(f"❌ Backtest failed: {e}")
        sys.exit(1)
    
    finally:
        if 'client' in locals():
            client.close()


@trading_agent.command()
@click.option('--symbol', required=True, help='Trading symbol to monitor')
@click.option('--duration', default=10, help='Monitoring duration in minutes')
@click.option('--interval', default=1, help='Update interval in minutes')
def monitor(symbol: str, duration: int, interval: int):
    """Monitor a trading symbol in real-time with LLM analysis."""
    logger.info(f"👁️ Starting real-time monitoring: {symbol} for {duration} minutes")
    
    try:
        # Initialize LLM client
        logger.info("🔧 Initializing LLM client...")
        client = _initialize_llm_client(symbol, temperature=0.1, max_tokens=100)
        logger.info("✅ LLM client initialized")
        
        # Start monitoring loop
        logger.info("🔄 Starting monitoring loop...")
        _run_monitoring_loop(client, symbol, duration, interval)
        
        logger.info("🎉 Monitoring completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Monitoring failed: {e}", exc_info=True)
        click.echo(f"❌ Monitoring failed: {e}")
        sys.exit(1)
    
    finally:
        if 'client' in locals():
            client.close()


def _initialize_llm_client(symbol: str, temperature: float, max_tokens: int) -> Any:
    """Initialize the LLM client with optimized settings."""
    logger.info(f"🔧 Initializing LLM client for {symbol}")
    
    try:
        client = get_llm_client(
            run_id=f"trading_agent_{symbol}_{datetime.now().isoformat()}",
            config={
                'temperature': temperature,
                'max_tokens': max_tokens,
                'timeout_s': 60
            }
        )
        logger.info("✅ LLM client initialized successfully")
        return client
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM client: {e}")
        raise


def _get_market_data(symbol: str, timeframe: str) -> Dict[str, Any]:
    """Get market data for analysis."""
    logger.info(f"📊 Getting market data for {symbol} on {timeframe}")
    
    # Placeholder market data - in production, this would fetch real data
    market_data = {
        'symbol': symbol,
        'timeframe': timeframe,
        'current_price': 50000.0,
        'price_change_24h': 0.025,
        'volume_24h': 1000000,
        'rsi': 65.5,
        'macd': 0.002,
        'trend': 'bullish',
        'support_levels': [48000, 47000],
        'resistance_levels': [52000, 53000],
        'volatility': 0.15,
        'market_sentiment': 'positive'
    }
    
    logger.info("✅ Market data collected")
    return market_data


def _create_optimized_prompt(symbol: str, timeframe: str, market_data: Dict[str, Any]) -> str:
    """Create an optimized prompt for cost-effective GPT analysis."""
    logger.info("📝 Creating cost-optimized analysis prompt")
    
    # Ultra-concise prompt to minimize token usage
    prompt = f"""Analyze {symbol} {timeframe}: ${market_data['current_price']:,.0f}, RSI {market_data['rsi']}, {market_data['trend']} trend.

JSON: {{
  "action": "hold|buy|sell",
  "confidence": 0.1-0.95,
  "reason": "brief"
}}

Be decisive: high confidence (0.8+) for strong signals, medium (0.5-0.7) for moderate signals, low (0.1-0.4) for weak signals."""
    
    logger.info(f"✅ Cost-optimized prompt created ({len(prompt)} characters)")
    return prompt


def _generate_analysis_with_timeout(client: Any, prompt: str, timeout: int, max_tokens: int) -> Any:
    """Generate analysis with proper timeout handling."""
    logger.info(f"🤖 Generating analysis with {timeout}s timeout")
    
    import signal
    
    def timeout_handler(signum, frame):
        logger.warning(f"⏰ TIMEOUT: LLM generation exceeded {timeout} seconds")
        raise TimeoutError(f"LLM generation timed out after {timeout} seconds")
    
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        logger.info("🚀 Calling client.generate()...")
        result = client.generate(
            prompt,
            json_mode=True,
            max_tokens=max_tokens
        )
        
        signal.alarm(0)  # Cancel timeout
        logger.info("✅ LLM generation completed successfully")
        return result
        
    except TimeoutError:
        signal.alarm(0)
        raise
    except Exception as e:
        signal.alarm(0)
        logger.error(f"❌ LLM generation failed: {e}")
        raise


def _process_analysis_result(result: Any, symbol: str, timeframe: str) -> Dict[str, Any]:
    """Process and validate the LLM analysis result."""
    logger.info("📋 Processing analysis result")
    
    try:
        # Parse JSON response
        analysis = json.loads(result.text)
        logger.info("✅ JSON response parsed successfully")
        logger.info(f"📄 Parsed response: {analysis}")
        
        # Check if we have a nested structure (common with LLM responses)
        if 'result' in analysis:
            analysis = analysis['result']
            logger.info("📋 Extracted nested result structure")
        
        # Validate required fields with fallbacks
        validated_analysis = {
            'symbol': analysis.get('symbol', symbol),
            'timeframe': analysis.get('timeframe', timeframe),
            'action': analysis.get('action', 'hold'),
            'confidence': analysis.get('confidence', 0.5),
            'reason': analysis.get('reason', 'Analysis completed'),
            'price_target': analysis.get('price_target', 'N/A'),
            'stop_loss': analysis.get('stop_loss', 'N/A')
        }
        
        # Add metadata
        validated_analysis['timestamp'] = datetime.now().isoformat()
        validated_analysis['model'] = getattr(result, 'model', 'local_llama')
        validated_analysis['latency_ms'] = getattr(result, 'latency_ms', 0)
        validated_analysis['raw_response'] = result.text
        
        logger.info("✅ Analysis result validated and processed")
        logger.info(f"📋 Final analysis: {validated_analysis}")
        return validated_analysis
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse JSON response: {e}")
        logger.info(f"📄 Raw response: {result.text}")
        
        # Create fallback analysis with raw response
        fallback_analysis = {
            'symbol': symbol,
            'timeframe': timeframe,
            'action': 'hold',
            'confidence': 0.0,
            'reason': f'JSON parsing failed: {e}',
            'price_target': 'N/A',
            'stop_loss': 'N/A',
            'timestamp': datetime.now().isoformat(),
            'model': getattr(result, 'model', 'local_llama'),
            'latency_ms': getattr(result, 'latency_ms', 0),
            'raw_response': result.text,
            'parse_error': True
        }
        
        logger.warning("⚠️ Using fallback analysis due to JSON parsing failure")
        return fallback_analysis
        
    except Exception as e:
        logger.error(f"❌ Failed to process analysis result: {e}")
        raise


def _display_analysis_results(analysis: Dict[str, Any]) -> None:
    """Display the analysis results."""
    click.echo(f"\n📊 Trading Analysis Results")
    click.echo(f"=" * 50)
    click.echo(f"Symbol: {analysis.get('symbol', 'N/A')}")
    click.echo(f"Timeframe: {analysis.get('timeframe', 'N/A')}")
    click.echo(f"Action: {analysis.get('action', 'N/A').upper()}")
    click.echo(f"Confidence: {analysis.get('confidence', 0):.2f}")
    click.echo(f"Reason: {analysis.get('reason', 'N/A')}")
    click.echo(f"Price Target: {analysis.get('price_target', 'N/A')}")
    click.echo(f"Stop Loss: {analysis.get('stop_loss', 'N/A')}")
    
    if analysis.get('parse_error'):
        click.echo(f"⚠️ Note: This analysis had parsing issues")
        click.echo(f"📄 Raw response: {analysis.get('raw_response', 'N/A')}")


def _save_analysis_results(analysis: Dict[str, Any], symbol: str, timeframe: str) -> None:
    """Save analysis results to file."""
    logger.info("💾 Saving analysis results")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_{symbol}_{timeframe}_{timestamp}.json"
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    filepath = logs_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    click.echo(f"💾 Analysis saved to: {filepath}")
    logger.info(f"✅ Analysis results saved to {filepath}")


def _generate_trading_strategy(client: Any, symbol: str, days: int, max_tokens: int = 200) -> Dict[str, Any]:
    """Generate a trading strategy using the LLM."""
    logger.info("🤖 Generating trading strategy")
    
    prompt = f"""Create trading strategy for {symbol} {days}d. JSON response: {{
  "strategy_type": "trend_following|mean_reversion|breakout",
  "entry_rules": ["rule1", "rule2"],
  "exit_rules": ["rule1", "rule2"],
  "risk_per_trade": 0.02,
  "max_positions": 3,
  "stop_loss_pct": 0.05,
  "take_profit_pct": 0.15
}}"""
    
    result = _generate_analysis_with_timeout(client, prompt, 45, max_tokens)
    strategy = json.loads(result.text)
    
    logger.info("✅ Trading strategy generated")
    return strategy


def _run_backtest_simulation(strategy: Dict[str, Any], symbol: str, days: int, initial_capital: float) -> Dict[str, Any]:
    """Run a simple backtest simulation."""
    logger.info("📈 Running backtest simulation")
    
    # Simple simulation - in production, this would use real historical data
    results = {
        'symbol': symbol,
        'strategy': strategy,
        'initial_capital': initial_capital,
        'final_capital': initial_capital * 1.08,  # 8% return
        'total_return': 0.08,
        'total_trades': 12,
        'winning_trades': 8,
        'losing_trades': 4,
        'win_rate': 0.67,
        'max_drawdown': 0.03,
        'sharpe_ratio': 1.2,
        'simulation_days': days
    }
    
    logger.info("✅ Backtest simulation completed")
    return results


def _display_backtest_results(results: Dict[str, Any]) -> None:
    """Display backtest results."""
    click.echo(f"\n📊 Backtest Results")
    click.echo(f"=" * 50)
    click.echo(f"Symbol: {results['symbol']}")
    click.echo(f"Initial Capital: ${results['initial_capital']:,.2f}")
    click.echo(f"Final Capital: ${results['final_capital']:,.2f}")
    click.echo(f"Total Return: {results['total_return']:.1%}")
    click.echo(f"Total Trades: {results['total_trades']}")
    click.echo(f"Win Rate: {results['win_rate']:.1%}")
    click.echo(f"Max Drawdown: {results['max_drawdown']:.1%}")
    click.echo(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")


def _save_backtest_results(results: Dict[str, Any], symbol: str, days: int) -> None:
    """Save backtest results to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_{symbol}_{days}d_{timestamp}.json"
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    filepath = logs_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    click.echo(f"💾 Backtest results saved to: {filepath}")


def _run_monitoring_loop(client: Any, symbol: str, duration: int, interval: int) -> None:
    """Run the real-time monitoring loop."""
    logger.info(f"🔄 Starting monitoring loop for {duration} minutes with {interval} minute intervals")
    
    import time
    from datetime import datetime, timedelta
    
    end_time = datetime.now() + timedelta(minutes=duration)
    update_count = 0
    
    click.echo(f"\n👁️ Monitoring {symbol} - Updates every {interval} minute(s)")
    click.echo(f"⏰ Duration: {duration} minutes | End time: {end_time.strftime('%H:%M:%S')}")
    click.echo("=" * 60)
    
    while datetime.now() < end_time:
        update_count += 1
        current_time = datetime.now().strftime("%H:%M:%S")
        
        try:
            # Get current market data
            logger.info(f"📊 Update {update_count}: Getting market data...")
            market_data = _get_market_data(symbol, "1m")
            
            # Quick LLM analysis
            logger.info(f"🤖 Update {update_count}: Generating quick analysis...")
            prompt = f"{symbol}: ${market_data['current_price']:,.0f}, RSI {market_data['rsi']}, {market_data['trend']}. JSON: {{\"action\": \"hold|buy|sell\", \"confidence\": 0.1-0.95, \"reason\": \"brief\"}}. Be decisive: high confidence (0.8+) for strong signals, medium (0.5-0.7) for moderate signals, low (0.1-0.4) for weak signals."
            
            result = _generate_analysis_with_timeout(client, prompt, 15, 60)
            analysis = _process_analysis_result(result, symbol, "1m")
            
            # Display update
            click.echo(f"\n🕐 {current_time} - Update #{update_count}")
            click.echo(f"💰 Price: ${market_data['current_price']:,.0f} | RSI: {market_data['rsi']}")
            click.echo(f"📊 Action: {analysis['action'].upper()} | Confidence: {analysis['confidence']:.2f}")
            click.echo(f"💭 Reason: {analysis['reason']}")
            
            # Save monitoring data
            _save_monitoring_update(analysis, symbol, update_count)
            
        except Exception as e:
            logger.error(f"❌ Update {update_count} failed: {e}")
            click.echo(f"❌ Update #{update_count} failed: {e}")
        
        # Wait for next update
        if datetime.now() < end_time:
            logger.info(f"⏳ Waiting {interval} minute(s) for next update...")
            time.sleep(interval * 60)
    
    click.echo(f"\n✅ Monitoring completed! Total updates: {update_count}")
    logger.info(f"✅ Monitoring loop completed with {update_count} updates")


def _save_monitoring_update(analysis: Dict[str, Any], symbol: str, update_number: int) -> None:
    """Save a monitoring update to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"monitor_{symbol}_update{update_number}_{timestamp}.json"
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    filepath = logs_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    logger.info(f"💾 Monitoring update saved to {filepath}")


if __name__ == '__main__':
    trading_agent()
