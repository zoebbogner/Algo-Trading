#!/usr/bin/env python3
"""
System CLI for the Algo-Trading system.

Provides command-line interface for:
- System health monitoring
- Performance diagnostics
- Configuration management
- Maintenance tasks
"""

import sys
from pathlib import Path

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.base import load_config as load_base_config
from src.utils.logging import get_logger, setup_logging


@click.group()
@click.option('--config', '-c', default='configs/base.yaml',
              help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def system(ctx, config: str, verbose: bool):
    """System CLI for Algo-Trading system."""
    # Setup logging
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(log_level=log_level)

    # Load configuration
    base_config = load_base_config()
    ctx.obj = base_config


@system.command()
@click.pass_context
def health_check(ctx):
    """Check system health and status."""
    logger = get_logger(__name__)
    config = ctx.obj

    logger.info("🔍 Performing system health check...")

    # Check directories
    directories = [
        config['data_root'],
        config['reports_root'],
        config['logs_root']
    ]

    for directory in directories:
        dir_path = Path(directory)
        if dir_path.exists():
            logger.info(f"✅ {directory}: Exists")
        else:
            logger.warning(f"⚠️ {directory}: Missing")

    # Check configuration files
    config_files = [
        'configs/base.yaml',
        'configs/data.history.yaml',
        'configs/features.yaml'
    ]

    for config_file in config_files:
        file_path = Path(config_file)
        if file_path.exists():
            logger.info(f"✅ {config_file}: Exists")
        else:
            logger.warning(f"⚠️ {config_file}: Missing")

    logger.info("✅ System health check completed")


@system.command()
@click.pass_context
def run_tests(ctx):
    """Run the test suite."""
    logger = get_logger(__name__)

    logger.info("🧪 Running test suite...")

    try:
        import subprocess

        result = subprocess.run([
            'python3', '-m', 'pytest', '-v'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✅ All tests passed!")
        else:
            logger.error("❌ Some tests failed")
            logger.error(result.stdout)
            logger.error(result.stderr)

    except Exception as e:
        logger.error(f"Error running tests: {e}")


@system.command()
@click.pass_context
def generate_dashboard(ctx):
    """Generate the system dashboard."""
    logger = get_logger(__name__)

    logger.info("📊 Generating system dashboard...")

    try:
        from dashboard import main as generate_dashboard_main
        generate_dashboard_main()
        logger.info("✅ Dashboard generated successfully")

    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")


if __name__ == '__main__':
    system()

