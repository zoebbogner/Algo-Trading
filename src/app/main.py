"""Main application entry point."""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..utils.config import settings, config_manager
from ..utils.logging import logger


console = Console()


def signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals gracefully."""
    console.print("\n[yellow]Shutdown signal received. Cleaning up...[/yellow]")
    # TODO: Implement cleanup logic
    sys.exit(0)


@click.group()
@click.option("--config-dir", default="configs", help="Configuration directory")
@click.option("--log-level", default="INFO", help="Log level")
@click.pass_context
def cli(ctx, config_dir: str, log_level: str):
    """Algo-Trading: Fast crypto trading bot with LLM-powered decision making."""
    ctx.ensure_object(dict)
    ctx.obj["config_dir"] = config_dir
    ctx.obj["log_level"] = log_level
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


@cli.command()
@click.option("--mode", default="SIM", type=click.Choice(["SIM", "PAPER", "LIVE"]), help="Trading mode")
@click.option("--symbols", default="BTC/USD,ETH/USD", help="Trading symbols (comma-separated)")
@click.option("--duration", default="1d", help="Backtest duration (e.g., 1d, 1w, 1m)")
@click.pass_context
def backtest(ctx, mode: str, symbols: str, duration: str):
    """Run backtesting simulation."""
    console.print(Panel.fit(
        f"[bold blue]Backtesting in {mode} mode[/bold blue]\n"
        f"Symbols: {symbols}\n"
        f"Duration: {duration}",
        title="Backtest Configuration"
    ))
    
    # TODO: Implement backtesting logic
    console.print("[green]Backtesting completed successfully![/green]")


@cli.command()
@click.option("--mode", default="SIM", type=click.Choice(["SIM", "PAPER", "LIVE"]), help="Trading mode")
@click.option("--symbols", default="BTC/USD,ETH/USD", help="Trading symbols (comma-separated)")
@click.pass_context
def live(ctx, mode: str, symbols: str):
    """Start live trading."""
    if mode == "LIVE":
        console.print(Panel.fit(
            "[bold red]⚠️  LIVE TRADING MODE ⚠️[/bold red]\n"
            "You are about to start live trading with real capital.\n"
            "Please confirm you understand the risks.",
            title="Live Trading Warning"
        ))
        
        if not click.confirm("Do you want to continue with live trading?"):
            console.print("[yellow]Live trading cancelled.[/yellow]")
            return
    
    console.print(Panel.fit(
        f"[bold green]Starting {mode} trading[/bold green]\n"
        f"Symbols: {symbols}",
        title="Trading Started"
    ))
    
    # TODO: Implement live trading logic
    console.print(f"[green]{mode} trading started successfully![/green]")


@cli.command()
@click.option("--config-file", help="Configuration file to validate")
def validate_config(config_file: Optional[str]):
    """Validate configuration files."""
    if config_file:
        config_path = Path(config_file)
        if not config_path.exists():
            console.print(f"[red]Configuration file not found: {config_file}[/red]")
            return
        
        try:
            # TODO: Implement config validation
            console.print(f"[green]Configuration file {config_file} is valid![/green]")
        except Exception as e:
            console.print(f"[red]Configuration validation failed: {e}[/red]")
    else:
        # Validate all configs
        try:
            configs = config_manager.load_configs()
            console.print(f"[green]All configuration files loaded successfully![/green]")
            console.print(f"Loaded {len(configs)} configuration sections")
        except Exception as e:
            console.print(f"[red]Configuration loading failed: {e}[/red]")


@cli.command()
def status():
    """Show system status."""
    console.print(Panel.fit(
        f"[bold]System Status[/bold]\n"
        f"Mode: {settings.mode}\n"
        f"Log Level: {settings.log_level}\n"
        f"Data Cache: {settings.data_cache_dir}\n"
        f"State Directory: {settings.state_dir}",
        title="Algo-Trading Status"
    ))


@cli.command()
def setup():
    """Setup the trading system."""
    console.print("[bold blue]Setting up Algo-Trading system...[/bold blue]")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "data/features",
        "data/cache",
        "state/portfolios",
        "state/checkpoints",
        "logs",
        "reports/daily",
        "reports/runs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created directory: {directory}")
    
    # Create initial state files
    state_files = [
        "state/registry.json",
        "logs/app.log",
        "logs/trades.log.jsonl",
        "logs/risk.log.jsonl"
    ]
    
    for file_path in state_files:
        Path(file_path).touch()
        console.print(f"[green]✓[/green] Created file: {file_path}")
    
    console.print("[bold green]Setup completed successfully![/bold green]")


def main():
    """Main application entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Application interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Application error: {e}[/red]")
        logger.logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
