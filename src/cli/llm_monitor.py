#!/usr/bin/env python3
"""LLM System Monitoring Dashboard for performance and usage analytics."""

import sys
import json
import click
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
import glob

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.llm import get_llm_client, healthcheck_all_backends
from src.utils.logging import setup_logging, get_logger


@click.group()
@click.option('--log-level', default='INFO', help='Logging level')
def llm_monitor(log_level):
    """LLM System Monitoring Dashboard."""
    setup_logging(log_level=log_level)


@llm_monitor.command()
def system_health():
    """Check overall system health."""
    click.echo("🏥 LLM System Health Check")
    click.echo("=" * 50)
    
    try:
        health_results = healthcheck_all_backends()
        overall_healthy = _evaluate_system_health(health_results)
        _display_health_summary(overall_healthy)
        
    except Exception as e:
        click.echo(f"❌ Health check failed: {e}")


def _evaluate_system_health(health_results: Dict[str, Any]) -> bool:
    """Evaluate overall system health from backend results."""
    overall_healthy = True
    
    for backend, health in health_results.items():
        status = "✅" if health['status'] == 'healthy' else "❌"
        click.echo(f"{status} {backend}: {health['status']}")
        
        if health['error']:
            click.echo(f"   Error: {health['error']}")
            overall_healthy = False
        
        if health['latency_ms']:
            click.echo(f"   Latency: {health['latency_ms']}ms")
    
    return overall_healthy


def _display_health_summary(overall_healthy: bool) -> None:
    """Display overall system health summary."""
    status_text = '✅ HEALTHY' if overall_healthy else '❌ UNHEALTHY'
    click.echo(f"\n📊 Overall Status: {status_text}")


@llm_monitor.command()
@click.option('--hours', default=24, help='Hours to look back for analysis')
def performance_metrics(hours: int):
    """Show performance metrics from recent analyses."""
    click.echo(f"📈 Performance Metrics (Last {hours} hours)")
    click.echo("=" * 50)
    
    try:
        analysis_files = _get_recent_analysis_files(hours)
        if not analysis_files:
            click.echo(f"❌ No analysis files found in last {hours} hours")
            return
        
        metrics = _extract_performance_metrics(analysis_files)
        _display_performance_metrics(metrics)
        
    except Exception as e:
        click.echo(f"❌ Performance analysis failed: {e}")


def _get_recent_analysis_files(hours: int) -> List[Path]:
    """Get analysis files from the last N hours."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return []
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    analysis_files = []
    
    for file_path in logs_dir.glob("analysis_*.json"):
        if file_path.stat().st_mtime > cutoff_time.timestamp():
            analysis_files.append(file_path)
    
    return analysis_files


def _extract_performance_metrics(analysis_files: List[Path]) -> Dict[str, Any]:
    """Extract performance metrics from analysis files."""
    metrics = {
        'latencies': [],
        'confidences': [],
        'symbols': set(),
        'actions': {}
    }
    
    for file_path in analysis_files:
        try:
            analysis = _load_analysis_file(file_path)
            _extract_metrics_from_analysis(analysis, metrics)
        except Exception as e:
            click.echo(f"⚠️  Error reading {file_path.name}: {e}")
    
    return metrics


def _load_analysis_file(file_path: Path) -> Dict[str, Any]:
    """Load and parse an analysis file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def _extract_metrics_from_analysis(analysis: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """Extract metrics from a single analysis."""
    # Extract latency
    if 'metadata' in analysis:
        metadata = analysis['metadata']
        if 'generation_latency_ms' in metadata:
            metrics['latencies'].append(metadata['generation_latency_ms'])
    
    # Extract confidence
    action = analysis.get('action', {})
    if 'confidence' in action:
        metrics['confidences'].append(action['confidence'])
    
    # Extract symbol
    if 'symbol' in analysis:
        metrics['symbols'].add(analysis['symbol'])
    
    # Extract action type
    action_type = action.get('type', 'unknown')
    metrics['actions'][action_type] = metrics['actions'].get(action_type, 0) + 1


def _display_performance_metrics(metrics: Dict[str, Any]) -> None:
    """Display performance metrics."""
    _display_latency_metrics(metrics['latencies'])
    _display_confidence_metrics(metrics['confidences'])
    _display_symbol_metrics(metrics['symbols'])
    _display_action_metrics(metrics['actions'])


def _display_latency_metrics(latencies: List[int]) -> None:
    """Display latency metrics."""
    if not latencies:
        return
    
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    click.echo(f"⏱️  Latency Metrics:")
    click.echo(f"   Average: {avg_latency:.0f}ms")
    click.echo(f"   Min: {min_latency}ms")
    click.echo(f"   Max: {max_latency}ms")
    click.echo(f"   Total Analyses: {len(latencies)}")


def _display_confidence_metrics(confidences: List[float]) -> None:
    """Display confidence metrics."""
    if not confidences:
        return
    
    avg_confidence = sum(confidences) / len(confidences)
    min_confidence = min(confidences)
    max_confidence = max(confidences)
    
    click.echo(f"\n🎯 Confidence Metrics:")
    click.echo(f"   Average: {avg_confidence:.2f}")
    click.echo(f"   Min: {min_confidence:.2f}")
    click.echo(f"   Max: {max_confidence:.2f}")


def _display_symbol_metrics(symbols: set) -> None:
    """Display symbol metrics."""
    if not symbols:
        return
    
    click.echo(f"\n📊 Symbols Analyzed:")
    click.echo(f"   Count: {len(symbols)}")
    click.echo(f"   List: {', '.join(sorted(symbols))}")


def _display_action_metrics(actions: Dict[str, int]) -> None:
    """Display action distribution metrics."""
    if not actions:
        return
    
    click.echo(f"\n🔄 Action Distribution:")
    total_actions = sum(actions.values())
    
    for action_type, count in sorted(actions.items()):
        percentage = (count / total_actions) * 100
        click.echo(f"   {action_type}: {count} ({percentage:.1f}%)")


@llm_monitor.command()
@click.option('--symbol', help='Filter by specific symbol')
@click.option('--action', help='Filter by action type')
@click.option('--min-confidence', type=float, help='Minimum confidence threshold')
def recent_analyses(symbol: str = None, action: str = None, min_confidence: float = None):
    """Show recent trading analyses."""
    click.echo("📋 Recent Trading Analyses")
    click.echo("=" * 50)
    
    try:
        analysis_files = _get_recent_analysis_files(24)  # Last 24 hours
        if not analysis_files:
            click.echo("❌ No analysis files found")
            return
        
        displayed_count = _display_filtered_analyses(analysis_files, symbol, action, min_confidence)
        _display_analysis_summary(displayed_count)
        
    except Exception as e:
        click.echo(f"❌ Recent analyses failed: {e}")


def _display_filtered_analyses(
    analysis_files: List[Path], 
    symbol: str = None, 
    action: str = None, 
    min_confidence: float = None
) -> int:
    """Display filtered analyses."""
    displayed_count = 0
    
    for file_path in analysis_files:
        try:
            analysis = _load_analysis_file(file_path)
            
            if not _passes_filters(analysis, symbol, action, min_confidence):
                continue
            
            _display_single_analysis(file_path, analysis)
            displayed_count += 1
            
        except Exception as e:
            click.echo(f"⚠️  Error reading {file_path.name}: {e}")
    
    return displayed_count


def _passes_filters(analysis: Dict[str, Any], symbol: str, action: str, min_confidence: float) -> bool:
    """Check if analysis passes all filters."""
    if symbol and analysis.get('symbol') != symbol:
        return False
    
    action_type = analysis.get('action', {}).get('type')
    if action and action_type != action:
        return False
    
    confidence = analysis.get('action', {}).get('confidence', 0)
    if min_confidence and confidence < min_confidence:
        return False
    
    return True


def _display_single_analysis(file_path: Path, analysis: Dict[str, Any]) -> None:
    """Display a single analysis."""
    click.echo(f"\n📊 {file_path.name}")
    click.echo(f"   Symbol: {analysis.get('symbol', 'N/A')}")
    click.echo(f"   Timeframe: {analysis.get('timeframe', 'N/A')}")
    click.echo(f"   Action: {analysis.get('action', {}).get('type', 'N/A')}")
    
    confidence = analysis.get('action', {}).get('confidence', 'N/A')
    if isinstance(confidence, (int, float)):
        click.echo(f"   Confidence: {confidence:.2f}")
    else:
        click.echo(f"   Confidence: {confidence}")
    
    # Show rationale (truncated)
    rationale = analysis.get('rationale', 'No rationale')
    if len(rationale) > 100:
        rationale = rationale[:100] + "..."
    click.echo(f"   Rationale: {rationale}")


def _display_analysis_summary(displayed_count: int) -> None:
    """Display analysis summary."""
    if displayed_count == 0:
        click.echo("❌ No analyses match the specified filters")
    else:
        click.echo(f"\n📈 Displayed {displayed_count} analyses")


@llm_monitor.command()
def system_info():
    """Show system configuration and information."""
    click.echo("⚙️  LLM System Information")
    click.echo("=" * 50)
    
    try:
        client = _initialize_monitor_client()
        _display_client_info(client)
        _display_config_info()
        _display_environment_info()
        _display_analytics_info()
        client.close()
        
    except Exception as e:
        click.echo(f"❌ System info failed: {e}")


def _initialize_monitor_client() -> Any:
    """Initialize LLM client for monitoring."""
    return get_llm_client(run_id="monitor_info")


def _display_client_info(client: Any) -> None:
    """Display client information."""
    click.echo(f"Backend: {client.backend}")
    click.echo(f"Model: {client.model}")
    click.echo(f"Temperature: {client.temperature}")
    click.echo(f"Max Tokens: {client.max_tokens}")
    click.echo(f"Timeout: {client.timeout_s}s")


def _display_config_info() -> None:
    """Display configuration information."""
    config_file = Path("configs/llm.yaml")
    click.echo(f"\n📁 Configuration:")
    
    if config_file.exists():
        click.echo(f"   Config File: {config_file} ✅")
        _display_config_contents(config_file)
    else:
        click.echo(f"   Configuration: No config file found")


def _display_config_contents(config_file: Path) -> None:
    """Display configuration file contents."""
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        click.echo(f"   Backend: {config.get('backend', 'N/A')}")
        click.echo(f"   Model: {config.get('model', 'N/A')}")
        click.echo(f"   Temperature: {config.get('temperature', 'N/A')}")
        click.echo(f"   JSON Mode: {config.get('json_mode', 'N/A')}")
        
    except Exception as e:
        click.echo(f"   Error reading config: {e}")


def _display_environment_info() -> None:
    """Display environment variable information."""
    import os
    click.echo(f"\n🌍 Environment:")
    
    llm_backend = os.getenv('LLM_BACKEND', 'Not set')
    llm_model_path = os.getenv('LLM_INSTRUCT_MODEL_PATH', 'Not set')
    
    click.echo(f"   LLM_BACKEND: {llm_backend}")
    click.echo(f"   LLM_INSTRUCT_MODEL_PATH: {'Set' if llm_model_path != 'Not set' else 'Not set'}")


def _display_analytics_info() -> None:
    """Display analytics information."""
    logs_dir = Path("logs")
    click.echo(f"\n📊 Analytics:")
    
    if logs_dir.exists():
        analysis_files = list(logs_dir.glob("analysis_*.json"))
        click.echo(f"   Logs Directory: {logs_dir} ✅")
        click.echo(f"   Analysis Files: {len(analysis_files)}")
    else:
        click.echo(f"   Analytics: No logs directory found")


if __name__ == '__main__':
    llm_monitor()
