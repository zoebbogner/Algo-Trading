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
        # Check all backends
        health_results = healthcheck_all_backends()
        
        overall_healthy = True
        for backend, health in health_results.items():
            status = "✅" if health['status'] == 'healthy' else "❌"
            click.echo(f"{status} {backend}: {health['status']}")
            
            if health['error']:
                click.echo(f"   Error: {health['error']}")
                overall_healthy = False
            
            if health['latency_ms']:
                click.echo(f"   Latency: {health['latency_ms']}ms")
        
        click.echo(f"\n📊 Overall Status: {'✅ HEALTHY' if overall_healthy else '❌ UNHEALTHY'}")
        
    except Exception as e:
        click.echo(f"❌ Health check failed: {e}")


@llm_monitor.command()
@click.option('--hours', default=24, help='Hours to look back for analysis')
def performance_metrics(hours: int):
    """Show performance metrics from recent analyses."""
    click.echo(f"📈 Performance Metrics (Last {hours} hours)")
    click.echo("=" * 50)
    
    try:
        # Find analysis files
        logs_dir = Path("logs")
        if not logs_dir.exists():
            click.echo("❌ No logs directory found")
            return
        
        # Get analysis files from last N hours
        cutoff_time = datetime.now() - timedelta(hours=hours)
        analysis_files = []
        
        for file_path in logs_dir.glob("analysis_*.json"):
            if file_path.stat().st_mtime > cutoff_time.timestamp():
                analysis_files.append(file_path)
        
        if not analysis_files:
            click.echo(f"❌ No analysis files found in last {hours} hours")
            return
        
        # Analyze performance
        latencies = []
        confidences = []
        symbols = set()
        actions = {}
        
        for file_path in analysis_files:
            try:
                with open(file_path, 'r') as f:
                    analysis = json.load(f)
                
                # Extract metrics
                if 'metadata' in analysis:
                    metadata = analysis['metadata']
                    if 'generation_latency_ms' in metadata:
                        latencies.append(metadata['generation_latency_ms'])
                
                # Extract confidence
                action = analysis.get('action', {})
                if 'confidence' in action:
                    confidences.append(action['confidence'])
                
                # Extract symbol
                if 'symbol' in analysis:
                    symbols.add(analysis['symbol'])
                
                # Extract action type
                action_type = action.get('type', 'unknown')
                actions[action_type] = actions.get(action_type, 0) + 1
                
            except Exception as e:
                click.echo(f"⚠️  Error reading {file_path.name}: {e}")
        
        # Display metrics
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            click.echo(f"⏱️  Latency Metrics:")
            click.echo(f"   Average: {avg_latency:.0f}ms")
            click.echo(f"   Min: {min_latency}ms")
            click.echo(f"   Max: {max_latency}ms")
            click.echo(f"   Total Analyses: {len(latencies)}")
        
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
            
            click.echo(f"\n🎯 Confidence Metrics:")
            click.echo(f"   Average: {avg_confidence:.2f}")
            click.echo(f"   Min: {min_confidence:.2f}")
            click.echo(f"   Max: {max_confidence:.2f}")
        
        if symbols:
            click.echo(f"\n📊 Symbols Analyzed:")
            click.echo(f"   Count: {len(symbols)}")
            click.echo(f"   List: {', '.join(sorted(symbols))}")
        
        if actions:
            click.echo(f"\n🔄 Action Distribution:")
            for action_type, count in sorted(actions.items()):
                percentage = (count / sum(actions.values())) * 100
                click.echo(f"   {action_type}: {count} ({percentage:.1f}%)")
        
    except Exception as e:
        click.echo(f"❌ Performance analysis failed: {e}")


@llm_monitor.command()
@click.option('--symbol', help='Filter by specific symbol')
@click.option('--action', help='Filter by action type')
@click.option('--min-confidence', type=float, help='Minimum confidence threshold')
def recent_analyses(symbol: str = None, action: str = None, min_confidence: float = None):
    """Show recent trading analyses."""
    click.echo("📋 Recent Trading Analyses")
    click.echo("=" * 50)
    
    try:
        # Find analysis files
        logs_dir = Path("logs")
        if not logs_dir.exists():
            click.echo("❌ No logs directory found")
            return
        
        # Get recent analysis files
        analysis_files = sorted(
            logs_dir.glob("analysis_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:10]  # Show last 10
        
        if not analysis_files:
            click.echo("❌ No analysis files found")
            return
        
        displayed_count = 0
        for file_path in analysis_files:
            try:
                with open(file_path, 'r') as f:
                    analysis = json.load(f)
                
                # Apply filters
                if symbol and analysis.get('symbol') != symbol:
                    continue
                
                action_type = analysis.get('action', {}).get('type')
                if action and action_type != action:
                    continue
                
                confidence = analysis.get('action', {}).get('confidence', 0)
                if min_confidence and confidence < min_confidence:
                    continue
                
                # Display analysis
                click.echo(f"\n📊 {file_path.name}")
                click.echo(f"   Symbol: {analysis.get('symbol', 'N/A')}")
                click.echo(f"   Timeframe: {analysis.get('timeframe', 'N/A')}")
                click.echo(f"   Action: {action_type or 'N/A'}")
                click.echo(f"   Confidence: {confidence:.2f}" if isinstance(confidence, (int, float)) else f"   Confidence: {confidence}")
                
                # Show rationale (truncated)
                rationale = analysis.get('rationale', 'No rationale')
                if len(rationale) > 100:
                    rationale = rationale[:100] + "..."
                click.echo(f"   Rationale: {rationale}")
                
                displayed_count += 1
                
            except Exception as e:
                click.echo(f"⚠️  Error reading {file_path.name}: {e}")
        
        if displayed_count == 0:
            click.echo("❌ No analyses match the specified filters")
        else:
            click.echo(f"\n📈 Displayed {displayed_count} analyses")
        
    except Exception as e:
        click.echo(f"❌ Recent analyses failed: {e}")


@llm_monitor.command()
def system_info():
    """Show system configuration and information."""
    click.echo("⚙️  LLM System Information")
    click.echo("=" * 50)
    
    try:
        # Get LLM client to check configuration
        client = get_llm_client(run_id="monitor_info")
        
        click.echo(f"Backend: {client.backend}")
        click.echo(f"Model: {client.model}")
        click.echo(f"Temperature: {client.temperature}")
        click.echo(f"Max Tokens: {client.max_tokens}")
        click.echo(f"Timeout: {client.timeout_s}s")
        
        # Check configuration files
        config_file = Path("configs/llm.yaml")
        if config_file.exists():
            click.echo(f"\n📁 Configuration:")
            click.echo(f"   Config File: {config_file} ✅")
            
            # Show config contents
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
        else:
            click.echo(f"\n📁 Configuration: No config file found")
        
        # Check environment variables
        import os
        click.echo(f"\n🌍 Environment:")
        llm_backend = os.getenv('LLM_BACKEND', 'Not set')
        llm_model_path = os.getenv('LLM_INSTRUCT_MODEL_PATH', 'Not set')
        
        click.echo(f"   LLM_BACKEND: {llm_backend}")
        click.echo(f"   LLM_INSTRUCT_MODEL_PATH: {'Set' if llm_model_path != 'Not set' else 'Not set'}")
        
        # Check logs directory
        logs_dir = Path("logs")
        if logs_dir.exists():
            analysis_files = list(logs_dir.glob("analysis_*.json"))
            click.echo(f"\n📊 Analytics:")
            click.echo(f"   Logs Directory: {logs_dir} ✅")
            click.echo(f"   Analysis Files: {len(analysis_files)}")
        else:
            click.echo(f"\n📊 Analytics: No logs directory found")
        
        client.close()
        
    except Exception as e:
        click.echo(f"❌ System info failed: {e}")


if __name__ == '__main__':
    llm_monitor()
